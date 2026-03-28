# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit Evaluation Kernel — per-settler optimal destination on P300C Blackhole hardware.

For each idle AI settler, this kernel computes which tile it should move toward
by combining:
  1. The global attractiveness field (from pathfind_kernel.py) — already computed
     this turn, reused here.
  2. A per-settler "reach field" diffused from the settler's current position on
     TT hardware.  This decays with distance, so the combination rewards reachable
     high-value tiles over unreachable distant ones.

Algorithm per settler:
  a. Seed a 512×512 grid at the settler's position (value = SEED_VALUE).
  b. Run N_REACH_PASSES diffusion passes via smooth_height_map → reach field.
  c. CPU: combined[i] = pathfield[i] * reach_field_normalized[i]
  d. best_tile = argmax(combined[:map_tiles])

The N_REACH_PASSES=4 radius is deliberately small (~4 tiles), so settlers are
biased toward the best tile they can reach in a few moves rather than a globally
distant optimum they would never reach this turn.

TT hardware does the diffusion; CPU does the element-wise product and argmax.
Total TT kernel calls: 2 smooth_height_map × 4 passes × N settlers = 8N calls.
Each call: ~0.07ms → 8 calls ≈ 0.56ms per settler.

Usage (standalone):
  source ~/code/tt-lang/build/env/activate
  python kernels/unit_eval_kernel.py

Usage (server integration):
  from unit_eval_kernel import UnitEvalModel
  model = UnitEvalModel(device)
  recs, ms = model.evaluate(units, pathfield_flat, map_w=32, map_h=64)
  # recs: [{"unit_id": N, "best_tile": M}, ...]
"""

import time

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

EVAL_GRID_SIZE  = 512    # 512×512 — same grid as pathfind and city_influence
N_REACH_PASSES  = 4      # short diffusion → ~4-tile reach radius per settler
DAMPING_REACH   = 0.88   # per-pass decay for reach field (faster fall-off)
SEED_VALUE      = 800.0  # initial hotspot value at settler position


# ── UnitEvalModel ─────────────────────────────────────────────────────────────

class UnitEvalModel:
    """
    Computes per-settler optimal destination tile using TT hardware.

    Call evaluate() once per turn after PathfindModel.compute() has run.
    The pathfield is passed in (not recomputed) so we avoid redundant TT calls.

    The model is stateless between turns.
    """

    def __init__(self, device):
        self.device = device

    def evaluate(
        self,
        units:         list,
        pathfield_flat: list,
        map_w:         int,
        map_h:         int,
    ) -> tuple[list, float]:
        """
        Compute per-settler best destination tiles.

        Args:
          units          : list of dicts {'id': int, 'tile': int, 'is_settler': int}
          pathfield_flat : flat list of len map_w*map_h from PathfindModel.compute()
          map_w / map_h  : FreeCiv map dimensions

        Returns:
          recommendations : list of {'unit_id': int, 'best_tile': int}
          total_kernel_ms : total time spent in TT hardware kernels
        """
        if not units or not pathfield_flat:
            return [], 0.0

        N         = EVAL_GRID_SIZE
        map_tiles = map_w * map_h
        total_ms  = 0.0

        # Convert pathfield to a torch tensor for element-wise ops on CPU
        path_t = torch.tensor(pathfield_flat[:map_tiles], dtype=torch.float32)

        recommendations = []

        for u in units:
            if not u.get('is_settler', 1):
                continue  # skip non-settler units for now

            unit_id  = int(u['id'])
            tile_idx = int(u['tile'])

            if tile_idx < 0 or tile_idx >= map_tiles:
                continue

            # ── Seed reach field at settler position ─────────────────────
            field = torch.zeros(N, N, dtype=torch.bfloat16)
            cx = (tile_idx % map_w) * N // map_w
            cy = (tile_idx // map_w) * N // map_h
            cx = min(cx, N - 2)
            cy = min(cy, N - 2)
            # 3×3 hotspot
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < N and 0 <= rx < N:
                        field[ry, rx] = torch.tensor(SEED_VALUE,
                                                     dtype=torch.bfloat16)

            # ── Diffuse reach field on TT hardware ───────────────────────
            t0 = time.perf_counter()
            for _ in range(N_REACH_PASSES):
                field = self._diffuse_step(field)
            total_ms += (time.perf_counter() - t0) * 1000

            # ── CPU: sample reach field to FreeCiv map, normalise ────────
            field_f = field.float()
            peak    = field_f.max().item()
            if peak <= 0.0:
                continue

            reach_flat = []
            for t in range(map_tiles):
                row = t // map_w
                col = t % map_w
                src_row = row * N // map_h
                src_col = col * N // map_w
                val = field_f[src_row, src_col].item()
                reach_flat.append(val / peak)   # normalise to [0, 1]

            reach_t = torch.tensor(reach_flat, dtype=torch.float32)

            # ── CPU: combined score = pathfield * reach (element-wise) ───
            combined = path_t * reach_t

            # Exclude the settler's own tile (no point staying)
            combined[tile_idx] = -1.0

            best_tile = int(combined.argmax().item())
            if best_tile >= 0:
                recommendations.append({'unit_id': unit_id, 'best_tile': best_tile})

        return recommendations, total_ms

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _diffuse_step(self, field: torch.Tensor) -> torch.Tensor:
        """One E-W + N-S diffusion pass on TT hardware."""
        d = DAMPING_REACH

        # E-W pass
        f_h  = (field                       * (d / 2.0)).to(torch.bfloat16)
        fs_h = (torch.roll(field, 1, dims=1) * (d / 2.0)).to(torch.bfloat16)
        a_t  = ttnn.from_torch(f_h,  dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        b_t  = ttnn.from_torch(fs_h, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        ew_t = zeros_like_on_device(f_h, self.device)
        smooth_height_map(a_t, b_t, ew_t)

        # N-S pass
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


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    W, H = 32, 64

    print("=" * 60)
    print("Unit Eval Kernel — Hardware Test")
    print(f"Map size: {W}×{H} = {W*H} tiles")
    print("=" * 60)

    # Synthetic pathfield: strong attraction at a single prime tile
    map_tiles = W * H
    prime_tile = (H // 2) * W + W // 4    # quarter across, middle height
    pathfield = [0.0] * map_tiles
    # Set a gradient: decreasing with manhattan distance from prime_tile
    prime_row = prime_tile // W
    prime_col = prime_tile % W
    max_dist   = W + H
    for i in range(map_tiles):
        r, c  = i // W, i % W
        dist  = abs(r - prime_row) + abs(c - prime_col)
        pathfield[i] = max(0.0, 1.0 - dist / max_dist)

    # Two synthetic settlers at different locations
    settler_a = 0                            # top-left corner (far from prime)
    settler_b = (H - 1) * W + (W - 1)       # bottom-right corner (also far)
    units = [
        {'id': 1, 'tile': settler_a, 'is_settler': 1},
        {'id': 2, 'tile': settler_b, 'is_settler': 1},
    ]

    device = ttnn.open_device(device_id=0)
    try:
        model = UnitEvalModel(device)

        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            model.evaluate(units, pathfield, W, H)

        print("Timed run...")
        recs, ms = model.evaluate(units, pathfield, W, H)

        print(f"\n  TT kernel time: {ms:.3f} ms total ({len(units)} settlers)")
        print(f"  Per-settler:    {ms / len(units):.3f} ms")
        print(f"\n  Recommendations:")
        for r in recs:
            uid   = r['unit_id']
            best  = r['best_tile']
            br    = best // W
            bc    = best % W
            src   = units[uid - 1]['tile']
            sr    = src // W
            sc    = src % W
            print(f"    Settler #{uid}: ({sc},{sr}) → ({bc},{br})  "
                  f"[tile={best}, pf={pathfield[best]:.3f}]")

        # Verify: both settlers should be directed toward the prime tile
        for r in recs:
            best = r['best_tile']
            br, bc = best // W, best % W
            dist_best  = abs(br - prime_row) + abs(bc - prime_col)
            src        = units[r['unit_id'] - 1]['tile']
            sr, sc     = src // W, src % W
            dist_start = abs(sr - prime_row) + abs(sc - prime_col)
            assert dist_best < dist_start, \
                f"Settler #{r['unit_id']} should move closer to prime tile"
        print("\n  ✓ Both settlers directed toward prime tile")

    finally:
        ttnn.close_device(device)

    print("\n✓ Unit eval kernel complete!")


if __name__ == "__main__":
    main()
