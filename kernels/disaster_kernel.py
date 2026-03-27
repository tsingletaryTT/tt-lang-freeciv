# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Disaster Spread Kernel — 2D spatial diffusion on P300C Blackhole hardware.

Uses smooth_height_map (eltwise-add) as a spreading operator:

  Each turn (E-W pass):
    output = damping/2 * field  +  damping/2 * roll(field, +1, dims=1)
           = damping * (field + neighbor) / 2     ← east-neighbor average

  Then (N-S pass):
    output2 = 0.5 * spread_ew  +  0.5 * roll(spread_ew, +1, dims=0)
            = (spread_ew + south-neighbor) / 2    ← south-neighbor average

Two passes per turn = full 2D diffusion. The disaster peak spreads ~1 tile/turn
and decays by SPREAD_DAMPING each step, so it covers ~300 tiles before dying.

A new disaster is seeded every SEED_INTERVAL turns. Four disaster types rotate:
  plague   → food yields collapse near epicenter
  famine   → all yields reduced across wide area
  locusts  → fast-spreading, affects trade
  eruption → intense, localized, decays quickly

Returns negative score modifiers in [-MAX_MODIFIER, 0.0] for each map tile.
The DisasterModel is stateful — callers must keep it alive across turns.

Dependencies: smooth_height_map from height_map_smooth.py (same kernel as
weather and tile scoring).
"""

import math

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

DISASTER_GRID_SIZE = 512   # Full chip: (512//32//2) × (512//32) = 8×16 = 128 Tensix cores

# Disaster parameters vary by type; these are defaults for plague/famine.
# eruption uses a shorter interval and higher peak decay rate.
SEED_INTERVAL   = 20     # New event every this many turns (turn % interval == 1)
MAX_INTENSITY   = 800.0  # Peak value at epicenter on seeding turn
MAX_MODIFIER    = 0.50   # Maximum negative yield penalty (50%)
DAMPING         = 0.88   # Intensity multiplied by this each turn (governs lifespan)
AFFECT_THRESH   = 30.0   # Intensity below which a tile is considered unaffected

# Type-specific overrides applied on top of the defaults
_TYPE_PARAMS = {
    'plague':   {'damping': 0.90, 'max_mod': 0.40, 'spread': 1},  # slow, medium hit
    'famine':   {'damping': 0.85, 'max_mod': 0.35, 'spread': 1},  # wide, milder hit
    'locusts':  {'damping': 0.80, 'max_mod': 0.50, 'spread': 2},  # fast spread, heavy
    'eruption': {'damping': 0.70, 'max_mod': 0.70, 'spread': 1},  # intense, short-lived
}

DISASTER_TYPES = list(_TYPE_PARAMS.keys())


# ── DisasterModel ──────────────────────────────────────────────────────────────

class DisasterModel:
    """
    Stateful disaster spread model. One instance lives for the server session.

    Call update(turn, map_w, map_h, seed) once per turn (from tile_score handler).
    The model self-seeds at turn % SEED_INTERVAL == 1 and automatically expires
    when intensity drops below AFFECT_THRESH everywhere.

    The 512×512 field uses the same kernel (smooth_height_map) as weather and
    tile scoring, so no additional JIT compilation is needed after warm-up.
    """

    def __init__(self, device):
        self.device    = device
        N              = DISASTER_GRID_SIZE
        self._field    = torch.zeros(N, N, dtype=torch.bfloat16)
        self._active   = False
        self._seed_turn    = 0
        self._dtype_name   = 'plague'
        self._epicenter    = (N // 2, N // 2)
        self._params       = _TYPE_PARAMS['plague']

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, turn: int, map_w: int, map_h: int,
               seed: int = 42) -> tuple[list, dict]:
        """
        Advance the disaster one turn on P300C Blackhole hardware.

        Seeds a new disaster at turn % SEED_INTERVAL == 1 (turn 1, 21, 41 …).

        Returns:
          modifiers : list[float]  — one value per map tile in [-MAX_MODIFIER, 0.0]
          stats     : dict         — type, intensity, n_affected, age, epicenter, etc.
        """
        # Check whether to seed a new disaster this turn
        if turn % SEED_INTERVAL == 1:
            self._seed(turn, seed)

        # If no active disaster, return zeros
        if not self._active or self._field.max().item() < AFFECT_THRESH:
            self._active = False
            return ([0.0] * (map_w * map_h),
                    {'active': False, 'type': None, 'intensity': 0.0,
                     'n_affected': 0, 'turn_age': 0,
                     'epicenter': self._epicenter})

        # ── Diffuse on TT hardware ─────────────────────────────────────────
        self._field = self._diffuse(self._field)

        # ── Sample field to FreeCiv map coordinates ────────────────────────
        N         = DISASTER_GRID_SIZE
        field_f   = self._field.float()
        modifiers = self._sample_to_map(field_f, map_w, map_h, N)

        n_affected   = sum(1 for m in modifiers if m < -0.02)
        intensity    = field_f.max().item()
        turn_age     = turn - self._seed_turn

        stats = {
            'active':    True,
            'type':      self._dtype_name,
            'intensity': round(intensity, 1),
            'n_affected': n_affected,
            'turn_age':  turn_age,
            'epicenter': self._epicenter,
        }
        return modifiers, stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _seed(self, turn: int, seed: int) -> None:
        """Plant a fresh hotspot at a pseudorandom map location."""
        N = DISASTER_GRID_SIZE

        # Deterministic position: varies with turn and seed
        t = turn * 0.3 + seed * 0.07
        cx = int(abs(math.sin(t))          * (N - 80)) + 40
        cy = int(abs(math.cos(t * 0.618))  * (N - 80)) + 40

        # Pick disaster type from rotation
        self._dtype_name = DISASTER_TYPES[(turn // SEED_INTERVAL) % len(DISASTER_TYPES)]
        self._params     = _TYPE_PARAMS[self._dtype_name]

        # Initialize field with a 5×5 hotspot
        self._field = torch.zeros(N, N, dtype=torch.bfloat16)
        r = 3  # hotspot radius in pixels
        self._field[cy - r : cy + r + 1, cx - r : cx + r + 1] = MAX_INTENSITY

        self._active    = True
        self._seed_turn = turn
        self._epicenter = (cx, cy)

    def _diffuse(self, field: torch.Tensor) -> torch.Tensor:
        """
        Spread field one step in both axes using smooth_height_map on TT hardware.

        Two smooth_height_map calls total (E-W pass, then N-S pass).
        Each call uses 128 Tensix cores on the 512×512 tensor.

        E-W pass:  output_ew = d/2 * field  +  d/2 * roll(field, +1, dims=1)
        N-S pass:  output    = 0.5 * output_ew  +  0.5 * roll(output_ew, +1, dims=0)

        where d = damping coefficient for this disaster type.
        """
        d = self._params['damping']

        # ── E-W spread ────────────────────────────────────────────────────
        f_half   = (field                       * (d / 2.0)).to(torch.bfloat16)
        f_shift  = (torch.roll(field, 1, dims=1) * (d / 2.0)).to(torch.bfloat16)

        a_t  = ttnn.from_torch(f_half,  dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        b_t  = ttnn.from_torch(f_shift, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        ew_t = zeros_like_on_device(f_half, self.device)
        smooth_height_map(a_t, b_t, ew_t)

        # ── N-S spread ────────────────────────────────────────────────────
        ew_cpu    = ttnn.to_torch(ew_t)
        ew_half   = (ew_cpu                         * 0.5).to(torch.bfloat16)
        ew_shift  = (torch.roll(ew_cpu, 1, dims=0)   * 0.5).to(torch.bfloat16)

        a2_t  = ttnn.from_torch(ew_half,  dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device)
        b2_t  = ttnn.from_torch(ew_shift, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device)
        ns_t  = zeros_like_on_device(ew_half, self.device)
        smooth_height_map(a2_t, b2_t, ns_t)

        return ttnn.to_torch(ns_t)

    @staticmethod
    def _sample_to_map(field_f: torch.Tensor,
                       map_w: int, map_h: int, N: int) -> list:
        """Nearest-neighbour downsample from N×N disaster field to map_w×map_h."""
        max_mod   = MAX_MODIFIER
        max_inten = MAX_INTENSITY
        mods = []
        for tile_idx in range(map_w * map_h):
            row = tile_idx // map_w
            col = tile_idx % map_w
            src_row = row * N // map_h
            src_col = col * N // map_w
            val = field_f[src_row, src_col].item()
            # Negative modifier: 0 where clean, -max_mod at peak intensity
            mod = -(val / max_inten) * max_mod
            mods.append(max(-max_mod, mod))
        return mods
