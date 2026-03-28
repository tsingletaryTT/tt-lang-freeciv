# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Combat Positioning Kernel — advance/hold/retreat signals on P300C Blackhole hardware.

Drives military unit movement by comparing two TT-computed diffusion fields:

  threat_field   : high near enemy units (from threat_field_kernel.py)
  strength_field : high near own military concentrations (same diffusion, own units)

For each own military unit, the positioning signal is:
  signal[u] = own_strength[tile(u)] - threat[tile(u)]

  signal >  ADVANCE_THRESH : advance toward enemies  (set goto_tile = high-threat neighbour)
  signal < -RETREAT_THRESH : retreat toward safety   (set goto_tile = high-own-strength tile)
  otherwise                : hold position           (no goto change)

The "advance" goto tile is the map tile with the highest threat value reachable within
the unit's remaining move range (approximated as N_ADVANCE tiles away).
The "retreat" goto tile is the highest own-strength tile within N_RETREAT radius.
Both are identified by scanning the pre-computed fields on CPU.

TT hardware computes the own-strength diffusion field (same 8-pass smooth_height_map
as ThreatFieldModel).  Threat field is passed in from the turn cache to avoid
redundant TT calls.

Usage (standalone):
  source ~/code/tt-lang/build/env/activate
  python kernels/combat_pos_kernel.py

Usage (server integration):
  from combat_pos_kernel import CombatPosModel
  model = CombatPosModel(device)
  orders, ms = model.position(
      own_units  = [{'id': 7, 'tile': 42, 'moves': 2}],
      threat_flat = [...],   # from ThreatFieldModel or cache
      map_w=32, map_h=64)
  # orders: [{'unit_id': 7, 'action': 'advance'|'hold'|'retreat',
  #            'goto_tile': M}]
"""

import time

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

STRENGTH_GRID_SIZE = 512  # 512×512, same as threat field
N_PASSES_STRENGTH  = 8    # own-strength diffusion radius (matches threat radius)
DAMPING_STRENGTH   = 0.88 # same decay as threat field for comparable magnitudes
SEED_VALUE         = 800.0

# Positioning thresholds (signal = own_strength_norm - threat_norm, ∈ [-1, 1])
ADVANCE_THRESH  =  0.25   # clearly own side is stronger → advance
RETREAT_THRESH  =  0.25   # clearly enemy is stronger    → retreat
N_ADVANCE       = 6       # max tiles away to look for advance target
N_RETREAT       = 8       # max tiles away to look for retreat target


# ── CombatPosModel ─────────────────────────────────────────────────────────────

class CombatPosModel:
    """
    Computes advance/hold/retreat orders for own military units.

    Diffuses an own-strength field from own unit positions on TT hardware (same
    pattern as ThreatFieldModel).  Threat field is passed in from the cache so
    the same TT computation isn't duplicated.

    CPU then computes signal = own_strength - threat and picks action + goto_tile.
    """

    def __init__(self, device):
        self.device = device

    def position(
        self,
        own_units:   list,
        threat_flat: list,
        map_w:       int,
        map_h:       int,
    ) -> tuple[list, float]:
        """
        Compute positioning orders for all own military units.

        Args:
          own_units    : list of dicts {'id': int, 'tile': int, 'moves': int,
                                        'strength': float}
                         strength ∈ [1, 3] (unit attack power)
          threat_flat  : flat per-tile normalised threat [0,1] (from cache)
          map_w / map_h : FreeCiv map dimensions

        Returns:
          orders    : list of {'unit_id': int, 'action': str, 'goto_tile': int}
                      action ∈ 'advance' | 'hold' | 'retreat'
                      goto_tile = -1 for 'hold'
          kernel_ms : TT kernel time (own-strength diffusion only)
        """
        if not own_units or not threat_flat:
            return [], 0.0

        N         = STRENGTH_GRID_SIZE
        map_tiles = map_w * map_h
        t0        = time.perf_counter()

        # ── Seed own-strength field ────────────────────────────────────────
        field = torch.zeros(N, N, dtype=torch.bfloat16)
        for u in own_units:
            tile_idx = int(u.get('tile', -1))
            strength = float(u.get('strength', 1.0))
            if tile_idx < 0 or tile_idx >= map_tiles:
                continue
            cx = (tile_idx % map_w) * N // map_w
            cy = (tile_idx // map_w) * N // map_h
            cx = min(cx, N - 2)
            cy = min(cy, N - 2)
            seed = min(SEED_VALUE * strength, SEED_VALUE * 3.0)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < N and 0 <= rx < N:
                        cur = field[ry, rx].item()
                        field[ry, rx] = torch.tensor(
                            min(SEED_VALUE * 3.0, cur + seed),
                            dtype=torch.bfloat16
                        )

        # ── Diffuse own-strength on TT hardware ───────────────────────────
        for _ in range(N_PASSES_STRENGTH):
            field = self._diffuse_step(field)

        kernel_ms = (time.perf_counter() - t0) * 1000

        # ── Sample both fields to FreeCiv map ─────────────────────────────
        field_f = field.float()
        peak    = field_f.max().item()

        own_strength = []
        for t in range(map_tiles):
            row = t // map_w
            col = t % map_w
            src_row = row * N // map_h
            src_col = col * N // map_w
            val = field_f[src_row, src_col].item()
            own_strength.append(min(1.0, val / peak) if peak > 0 else 0.0)

        # ── CPU: compute signal and pick action per unit ───────────────────
        threat_t   = torch.tensor(threat_flat[:map_tiles],  dtype=torch.float32)
        own_str_t  = torch.tensor(own_strength[:map_tiles], dtype=torch.float32)

        orders = []
        for u in own_units:
            unit_id  = int(u['id'])
            tile_idx = int(u.get('tile', -1))
            moves    = int(u.get('moves', 1))
            if tile_idx < 0 or tile_idx >= map_tiles:
                continue

            t_val  = threat_t[tile_idx].item()
            s_val  = own_str_t[tile_idx].item()
            signal = s_val - t_val   # ∈ [-1, 1]

            if signal >= ADVANCE_THRESH:
                # Advance: find tile within N_ADVANCE radius with highest threat
                # (moving toward where enemies are concentrated)
                goto = self._find_best_in_radius(
                    threat_t, tile_idx, map_w, map_h,
                    radius=min(N_ADVANCE, moves + 2),
                    maximise=True,
                    exclude=tile_idx)
                orders.append({'unit_id': unit_id, 'action': 'advance',
                               'goto_tile': goto})

            elif signal <= -RETREAT_THRESH:
                # Retreat: find tile within N_RETREAT radius with highest own strength
                # (moving toward friendly concentrations)
                goto = self._find_best_in_radius(
                    own_str_t, tile_idx, map_w, map_h,
                    radius=N_RETREAT,
                    maximise=True,
                    exclude=tile_idx)
                orders.append({'unit_id': unit_id, 'action': 'retreat',
                               'goto_tile': goto})

            else:
                # Hold position — no goto assignment
                orders.append({'unit_id': unit_id, 'action': 'hold',
                               'goto_tile': -1})

        return orders, kernel_ms

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _diffuse_step(self, field: torch.Tensor) -> torch.Tensor:
        """One E-W + N-S diffusion pass using smooth_height_map on TT hardware."""
        d = DAMPING_STRENGTH

        f_h  = (field                       * (d / 2.0)).to(torch.bfloat16)
        fs_h = (torch.roll(field, 1, dims=1) * (d / 2.0)).to(torch.bfloat16)
        a_t  = ttnn.from_torch(f_h,  dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        b_t  = ttnn.from_torch(fs_h, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        ew_t = zeros_like_on_device(f_h, self.device)
        smooth_height_map(a_t, b_t, ew_t)

        ew    = ttnn.to_torch(ew_t)
        ew_h  = (ew                       * 0.5).to(torch.bfloat16)
        ews_h = (torch.roll(ew, 1, dims=0) * 0.5).to(torch.bfloat16)
        a2_t  = ttnn.from_torch(ew_h,  dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device)
        b2_t  = ttnn.from_torch(ews_h, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device)
        ns_t  = zeros_like_on_device(ew_h, self.device)
        smooth_height_map(a2_t, b2_t, ns_t)

        return ttnn.to_torch(ns_t)

    @staticmethod
    def _find_best_in_radius(
        field:    torch.Tensor,
        origin:   int,
        map_w:    int,
        map_h:    int,
        radius:   int,
        maximise: bool,
        exclude:  int,
    ) -> int:
        """Find tile within Manhattan radius of origin with max/min field value."""
        map_tiles = map_w * map_h
        origin_r  = origin // map_w
        origin_c  = origin % map_w
        best_val  = -1e9 if maximise else 1e9
        best_tile = origin   # fallback: stay put

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                r = (origin_r + dr) % map_h
                c = (origin_c + dc) % map_w
                t = r * map_w + c
                if t < 0 or t >= map_tiles or t == exclude:
                    continue
                val = field[t].item()
                if (maximise and val > best_val) or (not maximise and val < best_val):
                    best_val  = val
                    best_tile = t

        return best_tile


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    W, H = 32, 64
    print("=" * 60)
    print("Combat Positioning Kernel — Hardware Test")
    print(f"Map size: {W}×{H} = {W*H} tiles")
    print("=" * 60)

    map_tiles = W * H

    # Synthetic threat field: enemies in top-left quadrant
    enemy_tile = W // 4 + (H // 4) * W
    threat = []
    er, ec = enemy_tile // W, enemy_tile % W
    for i in range(map_tiles):
        r, c = i // W, i % W
        dist = abs(r - er) + abs(c - ec)
        threat.append(max(0.0, 1.0 - dist / 15.0))

    # Own units:
    #   Unit 1: near enemies → should ADVANCE
    #   Unit 2: surrounded by enemies, isolated → should RETREAT
    #   Unit 3: far from enemies, with own backup → should HOLD or ADVANCE
    near_enemy = (H // 4 + 2) * W + W // 4 + 2   # 2 tiles from enemy cluster
    surrounded  = (H // 4 + 1) * W + W // 4 + 1  # 1 tile from enemy
    safe_tile   = (H // 2) * W + W // 2           # map centre

    own_units = [
        {'id': 10, 'tile': near_enemy, 'moves': 2, 'strength': 2.0},
        {'id': 11, 'tile': surrounded,  'moves': 2, 'strength': 1.0},
        {'id': 12, 'tile': safe_tile,   'moves': 3, 'strength': 2.0},
    ]

    device = ttnn.open_device(device_id=0)
    try:
        model = CombatPosModel(device)

        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            model.position(own_units, threat, W, H)

        print("Timed run...")
        orders, ms = model.position(own_units, threat, W, H)

        print(f"\n  Kernel time: {ms:.3f} ms  ({len(own_units)} units)")
        print(f"\n  Combat orders:")
        for o in orders:
            gt  = o['goto_tile']
            gr, gc = (gt // W, gt % W) if gt >= 0 else (-1, -1)
            tid = next(u['tile'] for u in own_units if u['id'] == o['unit_id'])
            tr, tc = tid // W, tid % W
            print(f"    Unit #{o['unit_id']:2d} at ({tc:2d},{tr:2d}): "
                  f"{o['action']:8s}  → ({gc:2d},{gr:2d})  "
                  f"[tile={gt}]")

    finally:
        ttnn.close_device(device)

    print("\n✓ Combat positioning kernel complete!")


if __name__ == "__main__":
    main()
