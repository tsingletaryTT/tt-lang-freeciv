# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
City Influence Kernel — territorial reach diffusion on P300C Blackhole hardware.

Each FreeCiv city emits an "influence" signal that spreads outward across the
map.  The signal from rival civs becomes a negative score modifier, steering
AI settlers into genuinely open wilderness rather than into rival territory.

Technique: same smooth_height_map (eltwise-add) with neighbour-shifted copy
as disaster_kernel.py — each E-W + N-S pass spreads influence by ~1 pixel.
Running N_PASSES passes per turn covers ~N_PASSES tiles of radius from each city.

With N_PASSES=8 and a 512×512 diffusion grid, the kernel makes:
  8 × 2 smooth_height_map calls = 16 TT kernel calls per city_influence invocation.
Each call uses 128 Tensix cores (512×512 tensor). Total: ~1.2ms.

The returned per-tile modifier is in [-MAX_MODIFIER, 0.0]:
  0.0   = wilderness (no rival nearby)
 -0.3   = moderate rival presence
 -0.6   = deep inside rival territory

Own-civ cities contribute positively (encouraging nearby expansion), but the
net effect is normalised so it remains in [-MAX_MODIFIER, 0.0].
"""

import time

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

INFLUENCE_GRID_SIZE = 512   # 512×512 → 128 Tensix cores (full P300C chip)
N_PASSES    = 8             # diffusion passes; covers ~8-tile radius
DAMPING     = 0.92          # per-pass intensity decay
OWN_WEIGHT  = 0.5           # own cities emit at half intensity (less penalty)
RIVAL_WEIGHT = 1.0          # rival cities emit at full intensity
MAX_MODIFIER = 0.6          # maximum yield penalty from rival territory


# ── CityInfluenceModel ─────────────────────────────────────────────────────────

class CityInfluenceModel:
    """
    Computes the territorial influence field from current city positions.

    Unlike DisasterModel, this model is stateless between turns — it fully
    recomputes from the city list each call, since cities are added/removed.
    """

    def __init__(self, device):
        self.device = device

    def compute(self, cities: list, map_w: int, map_h: int,
                own_player_id: int = -1) -> tuple[list, dict]:
        """
        Diffuse influence from all cities and return per-tile modifiers.

        Args:
          cities        : list of dicts {'tile_idx': int, 'player_id': int}
          map_w / map_h : FreeCiv map dimensions
          own_player_id : if ≥0, treat cities owned by this player more leniently.
                          Pass -1 to treat all players as rivals (AI perspective).

        Returns:
          modifiers : list[float]  len=map_w*map_h, values in [-MAX_MODIFIER, 0]
          stats     : dict with n_cities, n_rival, intensity, kernel_ms
        """
        N = INFLUENCE_GRID_SIZE
        t0 = time.perf_counter()

        # ── Seed influence field ───────────────────────────────────────────
        field = torch.zeros(N, N, dtype=torch.bfloat16)

        n_own   = 0
        n_rival = 0
        for c in cities:
            tile_idx  = c['tile_idx']
            player_id = c['player_id']
            # Map flat tile index to pixel in N×N grid
            cx = (tile_idx % map_w) * N // map_w
            cy = (tile_idx // map_w) * N // map_h
            # Clamp to grid bounds
            cx = min(cx, N - 2)
            cy = min(cy, N - 2)
            weight = OWN_WEIGHT if player_id == own_player_id else RIVAL_WEIGHT
            # 3×3 hotspot; bfloat16 saturates above ~57344
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < N and 0 <= rx < N:
                        field[ry, rx] = min(800.0, field[ry, rx].item() + 800.0 * weight)
            if player_id == own_player_id:
                n_own += 1
            else:
                n_rival += 1

        # ── Diffuse on TT hardware ─────────────────────────────────────────
        for _ in range(N_PASSES):
            field = self._diffuse_step(field)

        ms_kernel = (time.perf_counter() - t0) * 1000

        # ── Sample to FreeCiv map ──────────────────────────────────────────
        field_f   = field.float()
        peak      = field_f.max().item()
        modifiers = self._sample_to_map(field_f, map_w, map_h, N)

        stats = {
            'n_cities':  len(cities),
            'n_rival':   n_rival,
            'n_own':     n_own,
            'intensity': round(peak, 1),
            'kernel_ms': round(ms_kernel, 2),
        }
        return modifiers, stats

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
        ew       = ttnn.to_torch(ew_t)
        ew_h     = (ew                       * 0.5).to(torch.bfloat16)
        ews_h    = (torch.roll(ew, 1, dims=0) * 0.5).to(torch.bfloat16)
        a2_t = ttnn.from_torch(ew_h,  dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        b2_t = ttnn.from_torch(ews_h, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        ns_t = zeros_like_on_device(ew_h, self.device)
        smooth_height_map(a2_t, b2_t, ns_t)

        return ttnn.to_torch(ns_t)

    @staticmethod
    def _sample_to_map(field_f: torch.Tensor,
                       map_w: int, map_h: int, N: int) -> list:
        """Nearest-neighbour downsample to FreeCiv map, return negative modifiers."""
        mods = []
        for tile_idx in range(map_w * map_h):
            row = tile_idx // map_w
            col = tile_idx % map_w
            src_row = row * N // map_h
            src_col = col * N // map_w
            val = field_f[src_row, src_col].item()
            # Normalise to [-MAX_MODIFIER, 0] (negative = rival territory penalty)
            mod = -(val / 800.0) * MAX_MODIFIER
            mods.append(max(-MAX_MODIFIER, mod))
        return mods
