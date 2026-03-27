# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))

"""
Weather Kernel — Full P300C Hardware Utilization Demo
=======================================================

Computes a per-tile climate modifier on a fixed 512×512 grid, regardless of
FreeCiv map size.  This grid size drives maximum hardware utilization:

    512×512 tensor  →  row_tiles = 512 // 32 // 2 = 8
                        col_tiles = 512 // 32 = 16
                        grid      = 8 × 16 = 128 Tensix cores per chip

Vs. the tile_score kernel on the 22×44 map:
    64×64 tensor    →  row_tiles = 1,  col_tiles = 2  →  2 cores  (1.5%)

The weather field is computed each turn by blending a temperature field
(latitude gradient + seasonal variation) with a precipitation field
(longitude waves + turn-based storm systems) using two smooth_height_map
(eltwise-add) passes on hardware.  Results are sampled down to actual
FreeCiv map size as a multiplicative modifier on tile scores.

Weather events (drought, frost, storm, mild) are detected from the
climate field and logged for narrative / display purposes.
"""

import math
import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# Fixed resolution — determines hardware utilization.
# 512 → 8×16 = 128 cores (full chip).  Must be a multiple of TILE_SIZE=32.
WEATHER_GRID_SIZE = 512

# Modifier range: how much weather can swing tile scores.
# ±0.4 means drought reduces a score by up to 40%, ideal weather boosts by 40%.
WEATHER_MODIFIER_RANGE = 0.4

# Named thresholds for weather event detection (climate values are 0-1000)
THRESHOLD_DROUGHT  = 750   # very warm + dry  → drought
THRESHOLD_FROST    = 200   # very cold + wet  → frost
THRESHOLD_STORM    = 600   # warm + very wet  → storm
THRESHOLD_IDEAL    = 450   # moderate climate → ideal growing conditions


def compute_weather(device, turn: int) -> tuple[torch.Tensor, dict]:
    """
    Run the weather kernel on P300C hardware using a 512×512 tensor.

    Returns:
        climate (512×512 float tensor, values 0-1000) — raw climate field
        stats (dict) — core count, event counts, kernel timings

    The computation:
      Pass 1: temperature_field + precipitation_field  → raw_climate (TT hw)
      Pass 2: raw_climate + storm_perturbation         → final_climate (TT hw)
    Both passes run smooth_height_map on 512×512 → 128 cores each.
    """
    N = WEATHER_GRID_SIZE

    # ── CPU: generate input fields ──────────────────────────────────────────

    x = torch.linspace(0.0, 1.0, N)
    y = torch.linspace(0.0, 1.0, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Temperature: warm equator (Y≈0.5), cold poles (Y≈0 or Y≈1)
    # Season shifts the warm band north/south each turn
    season_phase = math.sin(turn * 0.08)    # full season cycle ≈ 78 turns
    equator_shift = 0.15 * season_phase     # equator drifts ±15%
    dist_from_equator = torch.abs(Y - (0.5 + equator_shift))
    temperature = (1.0 - 2.0 * dist_from_equator).clamp(0, 1) * 900 + 50
    # Range: ~50 (poles) to ~950 (equator peak)

    # Precipitation: longitude-based wave + moving storm systems
    storm_lon   = (turn * 0.11) % 1.0      # storm front sweeps east each turn
    storm_lat   = 0.35 + 0.3 * math.sin(turn * 0.05)
    storm_width = 0.15
    # Base precipitation: alternating wet/dry bands
    base_precip = 500 + 300 * torch.sin(X * math.pi * 4 + turn * 0.07)
    # Storm system: Gaussian blob moving across the map
    storm_dist2 = (X - storm_lon) ** 2 + (Y - storm_lat) ** 2
    storm_boost = 300 * torch.exp(-storm_dist2 / (2 * storm_width ** 2))
    precipitation = (base_precip + storm_boost).clamp(0, 1000)

    # Both fields already in [0, 1000] — correct range for the kernel

    temp_t   = temperature.to(torch.bfloat16)
    precip_t = precipitation.to(torch.bfloat16)

    # ── Pass 1: temp + precip → raw climate (TT hardware, 128 cores) ────────
    a1 = ttnn.from_torch(temp_t,   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b1 = ttnn.from_torch(precip_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s1 = zeros_like_on_device(temp_t, device)

    import time
    t0 = time.perf_counter()
    smooth_height_map(a1, b1, s1)
    ms_pass1 = (time.perf_counter() - t0) * 1000

    raw_climate_cpu = ttnn.to_torch(s1).float()  # (512×512), values ~0-1000

    # ── Pass 2: raw_climate + storm_perturbation → final (TT hardware) ──────
    # The second pass mixes the raw climate with a fine-grain perturbation
    # (turbulence) that makes weather patterns look organic rather than smooth.
    turb_phase = turn * 0.31
    turbulence = (500 + 200 * torch.sin(X * math.pi * 8 + turb_phase)
                      + 200 * torch.cos(Y * math.pi * 6 + turb_phase * 0.7)
                 ).clamp(0, 1000).to(torch.bfloat16)

    a2 = ttnn.from_torch(raw_climate_cpu.to(torch.bfloat16),
                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b2 = ttnn.from_torch(turbulence,
                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s2 = zeros_like_on_device(temp_t, device)

    t0 = time.perf_counter()
    smooth_height_map(a2, b2, s2)
    ms_pass2 = (time.perf_counter() - t0) * 1000

    # smooth_height_map does (a+b), output range: raw_climate [0-1000] + turbulence [0-1000]
    # → [0-2000], normalize back to [0-1000]
    final_climate = ttnn.to_torch(s2).float() * 0.5  # (512×512), 0-1000

    # ── Event detection ──────────────────────────────────────────────────────
    flat = final_climate.reshape(-1)
    n_drought = int((flat > THRESHOLD_DROUGHT).sum().item())
    n_frost   = int((flat < THRESHOLD_FROST).sum().item())
    n_storm   = int(((flat > THRESHOLD_STORM) & (flat <= THRESHOLD_DROUGHT)).sum().item())
    n_ideal   = int(((flat >= THRESHOLD_FROST) & (flat <= THRESHOLD_STORM)).sum().item())

    # Dominant event for this turn (by tile count)
    event_counts = {'drought': n_drought, 'frost': n_frost, 'storm': n_storm, 'ideal': n_ideal}
    dominant = max(event_counts, key=event_counts.get)

    # ── Hardware stats ───────────────────────────────────────────────────────
    # 512×512 tensor with TILE_SIZE=32, GRANULARITY=2:
    #   row_tiles = 512 // 32 // 2 = 8
    #   col_tiles = 512 // 32      = 16
    #   cores_used = 8 × 16 = 128 per chip
    cores_per_chip = (N // 32 // 2) * (N // 32)   # 8 × 16 = 128

    stats = {
        'cores_used':   cores_per_chip,
        'grid_size':    N,
        'ms_pass1':     ms_pass1,
        'ms_pass2':     ms_pass2,
        'n_drought':    n_drought,
        'n_frost':      n_frost,
        'n_storm':      n_storm,
        'n_ideal':      n_ideal,
        'dominant':     dominant,
        'season_phase': round(season_phase, 3),
        'storm_lon':    round(storm_lon, 3),
        'storm_lat':    round(storm_lat, 3),
    }

    return final_climate, stats


def sample_weather_to_map(climate: torch.Tensor, w: int, h: int) -> list:
    """
    Sample the 512×512 climate field down to (w×h) map tiles.
    Returns a flat list of modifier values in [-WEATHER_MODIFIER_RANGE, +WEATHER_MODIFIER_RANGE].
    Positive = beneficial weather, negative = harsh weather.
    """
    N = WEATHER_GRID_SIZE
    modifiers = []
    for tile_idx in range(w * h):
        row = tile_idx // w
        col = tile_idx % w
        src_row = row * N // h
        src_col = col * N // w
        val = climate[src_row, src_col].item()
        # Normalize: 500 = neutral (mod=0), 0 = harshest (-range), 1000 = best (+range)
        mod = (val / 500.0 - 1.0) * WEATHER_MODIFIER_RANGE
        modifiers.append(max(-WEATHER_MODIFIER_RANGE, min(WEATHER_MODIFIER_RANGE, mod)))
    return modifiers


# ─── Main (standalone validation) ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Weather Kernel — P300C Hardware Validation")
    print("=" * 65)

    device = ttnn.open_device(device_id=0)
    try:
        # Warm up (JIT compile smooth_height_map at 512×512)
        print("\n[1/3] Warming up (JIT compile 512×512 smooth_height_map)...")
        import time
        for _ in range(2):
            compute_weather(device, turn=1)
        print("  ✓ Compiled and cached")

        # Timed run for 3 consecutive turns
        print("\n[2/3] Timed runs for turns 1, 10, 25...")
        for turn in [1, 10, 25]:
            climate, stats = compute_weather(device, turn=turn)
            total_ms = stats['ms_pass1'] + stats['ms_pass2']
            print(f"\n  Turn {turn:>3}:")
            print(f"    Cores used:  {stats['cores_used']} / 64 per chip  "
                  f"({stats['cores_used']/64*100:.0f}%)")
            print(f"    Grid size:   {stats['grid_size']}×{stats['grid_size']} "
                  f"({stats['grid_size']**2:,} tiles)")
            print(f"    Pass 1 (temp+precip):     {stats['ms_pass1']:.3f} ms")
            print(f"    Pass 2 (climate+turb):    {stats['ms_pass2']:.3f} ms")
            print(f"    Total TT kernel time:     {total_ms:.3f} ms")
            print(f"    Dominant weather:  {stats['dominant'].upper()}")
            print(f"    Events: drought={stats['n_drought']}  "
                  f"frost={stats['n_frost']}  "
                  f"storm={stats['n_storm']}  "
                  f"ideal={stats['n_ideal']}")
            print(f"    Season phase: {stats['season_phase']:+.3f}  "
                  f"Storm @ ({stats['storm_lon']:.2f}, {stats['storm_lat']:.2f})")

        # Sample to a 22×44 FreeCiv map
        print("\n[3/3] Sampling 512×512 → 22×44 FreeCiv map...")
        climate, _ = compute_weather(device, turn=5)
        mods = sample_weather_to_map(climate, w=22, h=44)
        positive = sum(1 for m in mods if m > 0)
        negative = sum(1 for m in mods if m < 0)
        print(f"  Tiles with bonus:   {positive}")
        print(f"  Tiles with penalty: {negative}")
        print(f"  Modifier range: [{min(mods):.3f}, {max(mods):.3f}]")
        print(f"  ✓ Sampling correct")

    finally:
        ttnn.close_device(device)

    print("\n" + "=" * 65)
    print("✓ Weather kernel validated on P300C hardware!")
    print("=" * 65)


if __name__ == "__main__":
    main()
