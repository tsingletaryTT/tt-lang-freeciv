# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Threat Field Kernel — enemy military pressure diffusion on P300C Blackhole hardware.

Computes a per-tile "threat intensity" field by diffusing a signal outward from
every visible enemy military unit position.  High values indicate tiles under
military pressure from rival civilizations.

The threat field is the foundation for two higher-level kernels:
  - city_prod_kernel.py:  cities in high-threat zones should build military units
  - combat_pos_kernel.py: own units should advance into low-threat gaps or
                          retreat toward own strength concentrations

Technique: same smooth_height_map (eltwise-add with neighbour-shifted copy) as
city_influence_kernel.py and pathfind_kernel.py.  N_PASSES=8 covers an 8-tile
threat radius from each enemy unit — enough to warn cities before attackers arrive.

The returned per-tile threat is normalised to [0, 1.0]:
  0.0  = no enemy presence within diffusion radius
  1.0  = directly adjacent to a heavy enemy military concentration

Usage (standalone):
  source ~/code/tt-lang/build/env/activate
  python kernels/threat_field_kernel.py

Usage (server integration):
  from threat_field_kernel import ThreatFieldModel
  model = ThreatFieldModel(device)
  threat, stats = model.compute(
      enemy_units=[{'tile_idx': 42, 'strength': 2}],
      map_w=32, map_h=64)
  # threat: flat list len map_w*map_h, values in [0, 1]
  # stats:  {n_units, peak, kernel_ms}
"""

import time

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

THREAT_GRID_SIZE = 512   # 512×512 → 128 Tensix cores (full P300C chip)
N_PASSES         = 8     # diffusion passes → ~8-tile threat radius
DAMPING          = 0.88  # per-pass decay (faster fall-off than pathfind)
SEED_VALUE       = 800.0 # initial hotspot strength per enemy unit


# ── ThreatFieldModel ───────────────────────────────────────────────────────────

class ThreatFieldModel:
    """
    Computes a military threat intensity field from visible enemy unit positions.

    Stateless between turns — fully recomputed each call since units move.
    Identical diffusion pattern to PathfindModel but with smaller N_PASSES
    (8 vs 16) so the threat radius is tighter and more tactically accurate.
    """

    def __init__(self, device):
        self.device = device

    def compute(self, enemy_units: list,
                map_w: int, map_h: int) -> tuple[list, dict]:
        """
        Diffuse threat from all visible enemy military unit positions.

        Args:
          enemy_units : list of dicts {'tile_idx': int, 'strength': float}
                        strength ∈ [1, 3] scales seed intensity (unit power)
          map_w / map_h : FreeCiv map dimensions

        Returns:
          threat   : list[float] len=map_w*map_h, values in [0, 1]
          stats    : dict with n_units, peak, kernel_ms
        """
        N  = THREAT_GRID_SIZE
        t0 = time.perf_counter()

        # ── Seed threat field at each enemy unit ──────────────────────────
        field = torch.zeros(N, N, dtype=torch.bfloat16)

        n_units = 0
        for u in enemy_units:
            tile_idx = int(u.get('tile_idx', u.get('tile', -1)))
            strength = float(u.get('strength', 1.0))
            if tile_idx < 0 or tile_idx >= map_w * map_h:
                continue

            cx = (tile_idx % map_w) * N // map_w
            cy = (tile_idx // map_w) * N // map_h
            cx = min(cx, N - 2)
            cy = min(cy, N - 2)
            seed = min(SEED_VALUE * strength, SEED_VALUE * 3.0)

            # 3×3 hotspot so single-pixel units don't vanish in bfloat16 noise
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < N and 0 <= rx < N:
                        cur = field[ry, rx].item()
                        field[ry, rx] = torch.tensor(
                            min(SEED_VALUE * 3.0, cur + seed),
                            dtype=torch.bfloat16
                        )
            n_units += 1

        # ── Diffuse on TT hardware ─────────────────────────────────────────
        for _ in range(N_PASSES):
            field = self._diffuse_step(field)

        ms_kernel = (time.perf_counter() - t0) * 1000

        # ── Sample back to FreeCiv map ─────────────────────────────────────
        field_f = field.float()
        peak    = field_f.max().item()
        threat  = self._sample_to_map(field_f, map_w, map_h, N, peak)

        stats = {
            'n_units':   n_units,
            'peak':      round(peak, 1),
            'kernel_ms': round(ms_kernel, 2),
        }
        return threat, stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _diffuse_step(self, field: torch.Tensor) -> torch.Tensor:
        """One E-W + N-S diffusion pass using smooth_height_map on TT hardware."""
        d = DAMPING

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
    def _sample_to_map(field_f: torch.Tensor,
                       map_w: int, map_h: int, N: int, peak: float) -> list:
        """Nearest-neighbour downsample; normalise to [0, 1]."""
        if peak <= 0.0:
            return [0.0] * (map_w * map_h)
        threat = []
        for t in range(map_w * map_h):
            row = t // map_w
            col = t % map_w
            src_row = row * N // map_h
            src_col = col * N // map_w
            val = field_f[src_row, src_col].item()
            threat.append(min(1.0, val / peak))
        return threat


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    W, H = 32, 64
    print("=" * 60)
    print("Threat Field Kernel — Hardware Test")
    print(f"Map size: {W}×{H} = {W*H} tiles")
    print("=" * 60)

    # Two enemy "armies" at opposite sides of the map
    enemy_units = [
        {'tile_idx': W // 4,                   'strength': 2.0},  # top-left area
        {'tile_idx': W // 4 + 1,               'strength': 2.0},
        {'tile_idx': W // 4 + 2,               'strength': 1.5},
        {'tile_idx': (H - 2) * W + 3 * W // 4, 'strength': 2.0},  # bottom-right
        {'tile_idx': (H - 2) * W + 3 * W // 4 + 1, 'strength': 1.0},
    ]

    device = ttnn.open_device(device_id=0)
    try:
        model = ThreatFieldModel(device)

        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            model.compute(enemy_units, W, H)

        print("Timed run...")
        threat, stats = model.compute(enemy_units, W, H)

        print(f"\n  Enemy units:  {stats['n_units']}")
        print(f"  Kernel time:  {stats['kernel_ms']:.3f} ms")
        print(f"  Peak field:   {stats['peak']:.1f}")
        print(f"  Threat range: [{min(threat):.3f}, {max(threat):.3f}]")

        # Tiles near enemy should have high threat
        near_enemy = W // 4 + W           # one row below first army
        far_tile   = (H // 2) * W + W // 2  # map centre
        print(f"\n  Near-enemy tile #{near_enemy}: threat={threat[near_enemy]:.3f}")
        print(f"  Centre tile    #{far_tile}:  threat={threat[far_tile]:.3f}")
        assert threat[near_enemy] > threat[far_tile], \
            "Tile near enemy should have higher threat than map centre"

    finally:
        ttnn.close_device(device)

    print("\n✓ Threat field kernel complete!")


if __name__ == "__main__":
    main()
