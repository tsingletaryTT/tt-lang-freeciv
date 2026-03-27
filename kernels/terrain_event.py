# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Terrain Event Kernel
====================

Generates per-tile "event intensity" values each game turn using TT hardware.
Tiles that exceed the threshold receive a bonus resource in FreeCiv.

Algorithm:
  1. CPU: generate a sine-wave intensity field seeded by (turn, phase)
     intensity[x,y] = 0.5*sin(π*freq_x*x + phase) + 0.5*sin(π*freq_y*y + phase*1.3)
     Scaled to [0, 1000] to pass through scale_height_map.

  2. TT hardware: run scale_height_map (the existing passthrough kernel)
     This is the hardware computation — confirms the values on-device,
     and in a production system would apply per-tile SFPU modifiers.

  3. CPU post-process: threshold at 900 → return tile indices that get events.

Each turn uses different frequencies (derived from turn number), so resources
appear in different locations each turn.

Turn-to-resource rotation:
  turn % 4 == 0  →  Gold
  turn % 4 == 1  →  Oil
  turn % 4 == 2  →  Pheasant
  turn % 4 == 3  →  Fish

This module is also used directly by ttlang_server.py.
"""

import math
import time
import torch

from height_map_simple import scale_height_map

TILE_SIZE  = 32
THRESHOLD  = 900.0   # tiles above this value (out of 1000) get a resource
EVENT_EXTRAS = ["Gold", "Oil", "Pheasant", "Fish"]


def _tile_aligned(n: int) -> int:
    return ((n + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def generate_events_hardware(
    w: int, h: int, turn: int, device
) -> tuple[list, float]:
    """
    Compute terrain events for one game turn on TT hardware.

    Parameters
    ----------
    w, h  : FreeCiv map dimensions in tiles
    turn  : current game turn (used as seed)
    device: open ttnn device

    Returns
    -------
    (events, kernel_ms)
      events     : list of {"tile": int, "extra": str}
      kernel_ms  : time spent in TT kernel
    """
    import ttnn

    map_tiles = w * h
    side      = max(TILE_SIZE, _tile_aligned(int(math.ceil(math.sqrt(map_tiles)))))

    # Derive per-turn frequencies for spatial variety
    freq_x = 2.0 + (turn % 5) * 0.5
    freq_y = 2.0 + (turn % 7) * 0.4
    phase  = turn * 0.618  # golden-ratio phase accumulation → uniform coverage

    x = torch.linspace(0, 1, side)
    y = torch.linspace(0, 1, side)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Sine-wave intensity field
    raw = (
        0.5 * torch.sin(X * math.pi * freq_x + phase) +
        0.5 * torch.sin(Y * math.pi * freq_y + phase * 1.3)
    )
    # Normalize to [0, 1000] (range of height_map kernel)
    intensity = ((raw + 1.0) / 2.0 * 1000.0).to(torch.bfloat16)

    # Run through TT hardware
    in_t  = ttnn.from_torch(intensity, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    out_t = ttnn.from_torch(torch.zeros_like(intensity),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device)

    t0 = time.perf_counter()
    scale_height_map(in_t, out_t)
    kernel_ms = (time.perf_counter() - t0) * 1000

    result = ttnn.to_torch(out_t).float()

    # Determine which extra to place this turn
    extra_name = EVENT_EXTRAS[turn % len(EVENT_EXTRAS)]

    # Identify tiles above threshold, sampling from square kernel tensor
    events = []
    for tile_idx in range(map_tiles):
        row = tile_idx // w
        col = tile_idx % w
        src_row = row * side // h
        src_col = col * side // w
        if result[src_row, src_col].item() > THRESHOLD:
            events.append({"tile": tile_idx, "extra": extra_name})

    return events, kernel_ms


# ── CPU reference for validation ───────────────────────────────────────────────

def generate_events_cpu(w: int, h: int, turn: int) -> list:
    """CPU reference implementation for comparison."""
    freq_x = 2.0 + (turn % 5) * 0.5
    freq_y = 2.0 + (turn % 7) * 0.4
    phase  = turn * 0.618

    map_tiles = w * h
    events    = []
    extra     = EVENT_EXTRAS[turn % len(EVENT_EXTRAS)]

    for tile_idx in range(map_tiles):
        row = tile_idx // w
        col = tile_idx % w
        x   = col / w
        y   = row / h
        raw = (0.5 * math.sin(math.pi * freq_x * x + phase) +
               0.5 * math.sin(math.pi * freq_y * y + phase * 1.3))
        val = (raw + 1.0) / 2.0 * 1000.0
        if val > THRESHOLD:
            events.append({"tile": tile_idx, "extra": extra})

    return events


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    W, H = 32, 64

    print("=" * 60)
    print("Terrain Event Kernel — Hardware Test")
    print(f"Map size: {W}×{H} = {W*H} tiles")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        # Warm up
        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            generate_events_hardware(W, H, turn=1, device=device)

        # Test several turns
        print("\nTesting turns 1-8:")
        print(f"{'Turn':<8} {'Events':>8} {'Extra':<12} {'Kernel ms':>10}")
        print("-" * 45)
        for turn in range(1, 9):
            events, ms = generate_events_hardware(W, H, turn=turn, device=device)
            cpu_events  = generate_events_cpu(W, H, turn=turn)
            # Check overlap (TT vs CPU — may differ slightly due to bfloat16)
            tt_tiles  = set(e["tile"] for e in events)
            cpu_tiles = set(e["tile"] for e in cpu_events)
            overlap   = len(tt_tiles & cpu_tiles)
            extra     = events[0]["extra"] if events else "-"
            print(f"  {turn:<6} {len(events):>8}  {extra:<12} {ms:>8.3f}ms"
                  f"   (CPU: {len(cpu_events)}, overlap: {overlap})")

    finally:
        ttnn.close_device(device)

    print("\n✓ Terrain event kernel complete!")


if __name__ == "__main__":
    main()
