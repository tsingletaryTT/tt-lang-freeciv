# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Pathfinding Kernel — attractiveness gradient diffusion on P300C Blackhole hardware.

Computes a global "attractiveness field" by diffusing a positive signal outward
from the top-K highest-scored tiles.  The resulting gradient guides AI settlers
toward high-value land even when they are far from any known prize tile.

Technique: same smooth_height_map (eltwise-add with a neighbour-shifted copy) as
city_influence_kernel.py.  Each pass spreads the signal by ~1 tile in each of the
four cardinal directions.  Running N_PASSES=16 passes covers a 16-tile radius from
each source tile.

The returned per-tile attractiveness is normalised to [0, MAX_ATTRACT]:
  0.0          = no high-value land reachable within the diffusion radius
  MAX_ATTRACT  = directly adjacent to a top-scored tile

Blended additively into the tile score cache by ttlang_server.py:
  final_score[i] = base_score[i] + PATHFIND_WEIGHT * attract[i]

This makes the tile score cache a globally coherent landscape gradient rather than
a purely local signal, so settlers far from good land are steered correctly.

Usage (standalone):
  source ~/code/tt-lang/build/env/activate
  python kernels/pathfind_kernel.py

Usage (server integration):
  from pathfind_kernel import PathfindModel
  model = PathfindModel(device)
  attract, stats = model.compute(top_tile_indices=[17, 42, 108], map_w=32, map_h=64)
  # attract is a flat list of len map_w*map_h, values in [0, MAX_ATTRACT]
"""

import time

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

PATHFIND_GRID_SIZE = 512    # 512×512 → 128 Tensix cores (same as city_influence)
N_PASSES           = 16     # diffusion passes; covers ~16-tile radius
DAMPING            = 0.90   # per-pass intensity decay (slower than city_influence)
MAX_ATTRACT        = 0.8    # maximum attractiveness boost (additive to base score)
SEED_VALUE         = 800.0  # initial intensity at each seed tile (bfloat16-safe)


# ── PathfindModel ──────────────────────────────────────────────────────────────

class PathfindModel:
    """
    Computes an attractiveness gradient field from a set of high-value tile positions.

    Like CityInfluenceModel, but the signal is *positive* (pull, not penalty)
    and seeds from the top-K TT-scored tiles rather than from city locations.

    The model is stateless between turns — fully recomputed each call since the
    top-scored tiles change every turn (weather, disasters, territorial shifts).
    """

    def __init__(self, device):
        self.device = device

    def compute(self, top_tile_indices: list,
                map_w: int, map_h: int) -> tuple[list, dict]:
        """
        Diffuse attractiveness from top-K tiles and return per-tile modifiers.

        Args:
          top_tile_indices : list of flat tile indices (from tile_scorer top_tiles)
          map_w / map_h    : FreeCiv map dimensions

        Returns:
          attract  : list[float]  len=map_w*map_h, values in [0, MAX_ATTRACT]
          stats    : dict with n_sources, peak_intensity, kernel_ms
        """
        N  = PATHFIND_GRID_SIZE
        t0 = time.perf_counter()

        # ── Seed attractiveness field ──────────────────────────────────────
        field = torch.zeros(N, N, dtype=torch.bfloat16)

        n_sources = 0
        for tile_idx in top_tile_indices:
            if tile_idx < 0 or tile_idx >= map_w * map_h:
                continue
            # Map flat tile index to pixel in N×N grid
            cx = (tile_idx % map_w) * N // map_w
            cy = (tile_idx // map_w) * N // map_h
            # Clamp to grid bounds (leave 1px margin for 3×3 hotspot)
            cx = min(cx, N - 2)
            cy = min(cy, N - 2)
            # 3×3 hotspot so source is not a single noisy pixel
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < N and 0 <= rx < N:
                        field[ry, rx] = torch.tensor(
                            min(SEED_VALUE,
                                field[ry, rx].item() + SEED_VALUE),
                            dtype=torch.bfloat16
                        )
            n_sources += 1

        # ── Diffuse on TT hardware ─────────────────────────────────────────
        for _ in range(N_PASSES):
            field = self._diffuse_step(field)

        ms_kernel = (time.perf_counter() - t0) * 1000

        # ── Sample to FreeCiv map ──────────────────────────────────────────
        field_f = field.float()
        peak    = field_f.max().item()
        attract = self._sample_to_map(field_f, map_w, map_h, N)

        stats = {
            'n_sources':  n_sources,
            'peak':       round(peak, 1),
            'kernel_ms':  round(ms_kernel, 2),
        }
        return attract, stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _diffuse_step(self, field: torch.Tensor) -> torch.Tensor:
        """One E-W + N-S diffusion pass using smooth_height_map on TT hardware."""
        d = DAMPING

        # E-W pass: output = d/2*field + d/2*roll(field,1,dims=1)
        f_h  = (field                       * (d / 2.0)).to(torch.bfloat16)
        fs_h = (torch.roll(field, 1, dims=1) * (d / 2.0)).to(torch.bfloat16)
        a_t  = ttnn.from_torch(f_h,  dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        b_t  = ttnn.from_torch(fs_h, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        ew_t = zeros_like_on_device(f_h, self.device)
        smooth_height_map(a_t, b_t, ew_t)

        # N-S pass: output = 0.5*ew + 0.5*roll(ew,1,dims=0)
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
                       map_w: int, map_h: int, N: int) -> list:
        """Nearest-neighbour downsample to FreeCiv map; normalise to [0, MAX_ATTRACT]."""
        peak = field_f.max().item()
        if peak <= 0.0:
            return [0.0] * (map_w * map_h)

        attract = []
        for tile_idx in range(map_w * map_h):
            row = tile_idx // map_w
            col = tile_idx % map_w
            src_row = row * N // map_h
            src_col = col * N // map_w
            val = field_f[src_row, src_col].item()
            # Normalise to [0, MAX_ATTRACT]
            attract.append(min(MAX_ATTRACT, (val / peak) * MAX_ATTRACT))

        return attract


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    W, H = 32, 64

    print("=" * 60)
    print("Pathfind Kernel — Hardware Test")
    print(f"Map size: {W}×{H} = {W*H} tiles")
    print("=" * 60)

    # Synthetic top tiles: four corners + centre of the map
    n_tiles = W * H
    top_tiles = [
        0,                              # top-left corner
        W - 1,                          # top-right corner
        (H // 2) * W + W // 2,          # centre
        (H - 1) * W,                    # bottom-left
        (H - 1) * W + W - 1,            # bottom-right
    ]

    device = ttnn.open_device(device_id=0)
    try:
        model = PathfindModel(device)

        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            model.compute(top_tiles, W, H)

        print("Timed run...")
        attract, stats = model.compute(top_tiles, W, H)

        print(f"\n  Sources:     {stats['n_sources']} seed tiles")
        print(f"  Kernel time: {stats['kernel_ms']:.3f} ms")
        print(f"  Peak field:  {stats['peak']:.1f}")
        print(f"  Attract range: [{min(attract):.3f}, {max(attract):.3f}]")
        print(f"  Expected max:  {MAX_ATTRACT:.3f}")

        # Verify: tiles near source should have high attract, far tiles lower
        centre = (H // 2) * W + W // 2
        near   = centre + 1                  # 1 tile from centre source
        far    = centre + W * (H // 4)       # H/4 tiles away
        print(f"\n  Centre tile #{centre}: attract={attract[centre]:.3f}")
        print(f"  Near tile   #{near}:   attract={attract[near]:.3f}")
        print(f"  Far tile    #{far}:    attract={attract[far]:.3f}")
        assert attract[centre] >= attract[far], \
            "Centre should attract more than distant tile"

    finally:
        ttnn.close_device(device)

    print("\n✓ Pathfind kernel complete!")


if __name__ == "__main__":
    main()
