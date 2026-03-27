# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tile Scorer Kernel
==================

Computes a settlement/movement desirability score for every map tile:

    score[i] = 2.0 * food[i] + 1.5 * shields[i] + 1.0 * trade[i]

This implements the classic Civilization tile-value formula on TT hardware.
High scores identify ideal city-founding locations — coastal grassland with
bonus resources scores highest.

Implementation:
  Two chained `smooth_height_map` (eltwise-add) passes on the hardware.
  CPU pre-scales food by 2.0 and shields by 1.5 so the kernel just adds.

  Pass 1:  sum1 = (food * 2.0) + (shields * 1.5)
  Pass 2:  score = sum1 + (trade * 1.0)

This module is also used directly by ttlang_server.py.
"""

import math
import time
import torch

from height_map_smooth import smooth_height_map, zeros_like_on_device

TILE_SIZE = 32


def _tile_aligned(n: int) -> int:
    return ((n + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def score_tiles_hardware(
    food_list:    list,
    shields_list: list,
    trade_list:   list,
    w: int,
    h: int,
    device,
) -> tuple[list, list, float]:
    """
    Score all tiles using two chained TT-Lang eltwise-add kernels.

    Parameters
    ----------
    food_list, shields_list, trade_list : flat lists of per-tile values
    w, h : map width and height in tiles
    device : open ttnn device

    Returns
    -------
    (scores_flat, top_tile_indices, kernel_ms)
    """
    import ttnn

    map_tiles = w * h
    side      = max(TILE_SIZE, _tile_aligned(int(math.ceil(math.sqrt(map_tiles)))))
    total     = side * side

    def pad(lst):
        return (lst + [0.0] * total)[:total]

    food_t    = torch.tensor(pad(food_list),    dtype=torch.bfloat16).reshape(side, side)
    shields_t = torch.tensor(pad(shields_list), dtype=torch.bfloat16).reshape(side, side)
    trade_t   = torch.tensor(pad(trade_list),   dtype=torch.bfloat16).reshape(side, side)

    # CPU pre-scale so the hardware additions give the right weighted sum
    food_2x       = (food_t    * 2.0).to(torch.bfloat16)
    shields_1pt5x = (shields_t * 1.5).to(torch.bfloat16)

    import ttnn
    a1 = ttnn.from_torch(food_2x,       dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
    b1 = ttnn.from_torch(shields_1pt5x, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
    s1 = zeros_like_on_device(food_t, device)

    t0 = time.perf_counter()

    # Pass 1: food_2x + shields_1pt5x
    smooth_height_map(a1, b1, s1)

    # Pass 2: sum1 + trade_1x
    sum1_cpu = ttnn.to_torch(s1)
    a2 = ttnn.from_torch(sum1_cpu, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
    b2 = ttnn.from_torch(trade_t,  dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
    s2 = zeros_like_on_device(food_t, device)
    smooth_height_map(a2, b2, s2)

    kernel_ms = (time.perf_counter() - t0) * 1000

    score_flat_t = ttnn.to_torch(s2).reshape(-1).float()
    scores       = score_flat_t[:map_tiles].tolist()
    top_k        = min(10, map_tiles)
    top_tiles    = torch.topk(score_flat_t[:map_tiles], top_k).indices.tolist()

    return scores, top_tiles, kernel_ms


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    """Test tile scorer on a synthetic 32×64 map (matching FreeCiv default size)."""
    import ttnn, torch

    W, H = 32, 64
    map_tiles = W * H

    print("=" * 60)
    print("Tile Scorer Kernel — Hardware Test")
    print(f"Map size: {W}×{H} = {map_tiles} tiles")
    print("=" * 60)

    # Synthetic tile data: coastal tiles have high food, hills have shields
    torch.manual_seed(7)
    food    = torch.rand(map_tiles) * 4    # 0-4 food
    shields = torch.rand(map_tiles) * 3    # 0-3 shields
    trade   = torch.rand(map_tiles) * 2    # 0-2 trade

    print("\nWarm-up (2 runs)...")
    device = ttnn.open_device(device_id=0)
    try:
        for _ in range(2):
            score_tiles_hardware(food.tolist(), shields.tolist(), trade.tolist(),
                                 W, H, device)

        print("Timed run...")
        scores, top, ms = score_tiles_hardware(
            food.tolist(), shields.tolist(), trade.tolist(), W, H, device)

        print(f"\n  Kernel time:  {ms:.3f} ms")
        print(f"  Score range:  [{min(scores):.2f}, {max(scores):.2f}]")
        print(f"  Top-5 tiles:  {top[:5]}")

        # Verify against CPU reference
        cpu_scores = (2.0 * food + 1.5 * shields + 1.0 * trade).tolist()
        import numpy as np
        corr = np.corrcoef(scores, cpu_scores)[0, 1]
        print(f"  CPU correlation: {corr:.4f}  (>0.99 expected)")

        # Show top tile details
        print(f"\n  Top tile breakdown (tile #{top[0]}):")
        t = top[0]
        print(f"    food={food[t]:.2f}  shields={shields[t]:.2f}  trade={trade[t]:.2f}")
        print(f"    TT score: {scores[t]:.2f}  CPU score: {cpu_scores[t]:.2f}")

    finally:
        ttnn.close_device(device)

    print("\n✓ Tile scorer complete!")


if __name__ == "__main__":
    main()
