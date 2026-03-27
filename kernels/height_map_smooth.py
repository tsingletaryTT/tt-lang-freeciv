# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Height map smoothing kernel - multi-kernel pipeline demo.

This kernel implements a 2D box-filter smoothing pass on the TT hardware.
Each output element is the weighted average of the corresponding input tile
and a spatially-shifted copy, giving terrain that looks naturally blended.

Pipeline:
  CPU generate → scale_height_map (pass-through) → smooth_height_map (horizontal)
              → smooth_height_map (vertical) → result

The smoothing is achieved by pre-multiplying each input by 0.5 on the CPU,
then using element-wise addition on the hardware:
    output[i,j] = 0.5 * center[i,j] + 0.5 * shifted[i,j]

Two such passes (horizontal + vertical) approximate a 3×3 box filter.
"""

import math
import time
import torch
import ttl
import ttnn

from height_map_simple import generate_height_map_cpu, scale_height_map

TILE_SIZE   = 32
GRANULARITY = 2


# ─── Kernel definition ────────────────────────────────────────────────────────

@ttl.kernel(grid="auto")
def smooth_height_map(
    input_a: ttnn.Tensor,   # 0.5 * center
    input_b: ttnn.Tensor,   # 0.5 * shifted neighbor
    output:  ttnn.Tensor,
) -> None:
    """
    Element-wise addition of two pre-weighted tensors.

    By passing (0.5 * center) and (0.5 * neighbor) as inputs, the result is
    the average of the two — a one-axis smoothing pass. Chain two calls for
    full 2D smoothing.
    """
    row_tiles = input_a.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = input_a.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb   = ttl.make_dataflow_buffer_like(input_a, shape=(GRANULARITY, 1), buffer_factor=2)
    b_dfb   = ttl.make_dataflow_buffer_like(input_b, shape=(GRANULARITY, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(output,  shape=(GRANULARITY, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            a_dfb.wait()   as a_blk,
                            b_dfb.wait()   as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            # Real hardware arithmetic: add the two half-weight tiles
                            out_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = ttl.copy(input_a[r0:r1, col:col + 1], a_blk)
                            tx_b = ttl.copy(input_b[r0:r1, col:col + 1], b_blk)
                            tx_a.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, output[r0:r1, col:col + 1])
                            tx.wait()


# ─── CPU helper ───────────────────────────────────────────────────────────────

def smooth_cpu_reference(height_map: torch.Tensor, shift_tiles: int = 1) -> torch.Tensor:
    """
    CPU reference: same 2-pass box-filter smoothing as the hardware pipeline.
    Shift is expressed in tiles (TILE_SIZE elements) to match hardware behavior.
    """
    shift = shift_tiles * TILE_SIZE

    # Horizontal pass
    shifted_h = torch.roll(height_map, shift, dims=1)
    h_pass    = (height_map * 0.5 + shifted_h * 0.5)

    # Vertical pass
    shifted_v = torch.roll(h_pass, shift, dims=0)
    result    = (h_pass * 0.5 + shifted_v * 0.5)

    return result.to(torch.bfloat16)


def to_device_half(tensor: torch.Tensor, device) -> ttnn.Tensor:
    """Transfer (0.5 * tensor) to device as a TILE_LAYOUT bfloat16 tensor."""
    half = (tensor * 0.5).to(torch.bfloat16)
    return ttnn.from_torch(half, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def zeros_like_on_device(tensor: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.zeros_like(tensor),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_smooth_pipeline(device, map_size=256, shift_tiles=1):
    """
    Full multi-kernel pipeline on hardware:
      1. scale_height_map   – passthrough (existing kernel)
      2. smooth_height_map  – horizontal averaging pass
      3. smooth_height_map  – vertical averaging pass

    Returns (raw_result, smooth_result, timings_dict).
    """
    shift = shift_tiles * TILE_SIZE

    # ── Step 1: generate on CPU, run pass-through kernel ──────────────────
    t_start = time.perf_counter()
    height_cpu = generate_height_map_cpu(map_size, seed=42)

    in_t  = ttnn.from_torch(height_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_t = zeros_like_on_device(height_cpu, device)
    t0 = time.perf_counter()
    scale_height_map(in_t, out_t)
    t_passthrough = (time.perf_counter() - t0) * 1000

    raw_cpu  = ttnn.to_torch(out_t)

    # ── Step 2: horizontal smoothing pass ─────────────────────────────────
    h_shifted_cpu = torch.roll(raw_cpu, shift, dims=1)

    a_h = to_device_half(raw_cpu,      device)
    b_h = to_device_half(h_shifted_cpu, device)
    h_smooth_t = zeros_like_on_device(height_cpu, device)

    t0 = time.perf_counter()
    smooth_height_map(a_h, b_h, h_smooth_t)
    t_smooth_h = (time.perf_counter() - t0) * 1000

    h_smooth_cpu = ttnn.to_torch(h_smooth_t)

    # ── Step 3: vertical smoothing pass ───────────────────────────────────
    v_shifted_cpu = torch.roll(h_smooth_cpu, shift, dims=0)

    a_v = to_device_half(h_smooth_cpu,  device)
    b_v = to_device_half(v_shifted_cpu, device)
    v_smooth_t = zeros_like_on_device(height_cpu, device)

    t0 = time.perf_counter()
    smooth_height_map(a_v, b_v, v_smooth_t)
    t_smooth_v = (time.perf_counter() - t0) * 1000

    smooth_result = ttnn.to_torch(v_smooth_t)
    t_total = (time.perf_counter() - t_start) * 1000

    timings = {
        "passthrough_ms": t_passthrough,
        "smooth_h_ms":    t_smooth_h,
        "smooth_v_ms":    t_smooth_v,
        "total_ms":       t_total,
    }
    return raw_cpu, smooth_result, timings


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    MAP_SIZE   = 256
    SHIFT_TILES = 1

    print("=" * 70)
    print("Multi-Kernel Pipeline: Height Map Generation + Smoothing")
    print("=" * 70)

    device = ttnn.open_device(device_id=0)
    try:
        # Warm up all kernel shapes so JIT cache is warm before measurements
        print("\n[1/4] Warming up kernels...")
        for _ in range(2):
            run_smooth_pipeline(device, MAP_SIZE, SHIFT_TILES)
        print("  ✓ Warm-up complete")

        # Timed run
        print("\n[2/4] Timed pipeline run...")
        raw, smoothed, timings = run_smooth_pipeline(device, MAP_SIZE, SHIFT_TILES)
        print(f"  Pass-through kernel:  {timings['passthrough_ms']:.3f} ms")
        print(f"  Smooth H kernel:      {timings['smooth_h_ms']:.3f} ms")
        print(f"  Smooth V kernel:      {timings['smooth_v_ms']:.3f} ms")
        print(f"  Total pipeline:       {timings['total_ms']:.2f} ms")

        # Correctness vs CPU reference
        print("\n[3/4] Validating against CPU reference...")
        ref = smooth_cpu_reference(raw, SHIFT_TILES)
        match = torch.allclose(smoothed, ref, rtol=0.02, atol=0.5)
        print(f"  CPU reference match: {'✓ PASSED' if match else '✗ FAILED'}")

        # Terrain statistics
        print("\n[4/4] Terrain statistics (256x256):")
        for label, t in [("Raw (passthrough)", raw), ("Smoothed (2-pass)", smoothed)]:
            print(f"\n  {label}:")
            print(f"    Min:  {t.min().item():.1f}")
            print(f"    Max:  {t.max().item():.1f}")
            print(f"    Mean: {t.mean().item():.1f}")
            print(f"    Std:  {t.float().std().item():.1f}")

        std_raw  = raw.float().std().item()
        std_smo  = smoothed.float().std().item()
        print(f"\n  Smoothing reduced std-dev by {(1 - std_smo/std_raw)*100:.1f}%")
        print("  (Lower std-dev = smoother terrain, fewer abrupt cliffs)")

        # Save smoothed map for FreeCiv export
        out_path = Path(__file__).parent.parent / "output" / "smoothed_height.npy"
        import numpy as np
        np.save(out_path, smoothed.float().numpy())
        print(f"\n  ✓ Smoothed map saved to {out_path}")

    finally:
        ttnn.close_device(device)

    print("\n" + "=" * 70)
    print("✓ Multi-kernel pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    from pathlib import Path
    main()
