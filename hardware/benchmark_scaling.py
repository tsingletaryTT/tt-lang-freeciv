#!/usr/bin/env python3
"""
Scaling benchmark with warm-up to eliminate JIT compilation overhead.

The previous test showed ~400ms for 128², 512², 1024² maps while 256² was
fast (0.6ms). Root cause: each test_hardware_basic() call opened a fresh
device, discarding the compiled kernel cache. The 256² test was fast only
because the basic_test (also 256²) ran first and seeded the cache.

Fix: keep the device open across all sizes and warm up each new shape before
measuring.
"""

import torch
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

print("Importing TT modules...")
import ttnn
from height_map_simple import generate_height_map_cpu, scale_height_map

WARMUP_RUNS = 2   # runs before timing begins
TIMED_RUNS  = 5   # runs averaged for the result
SIZES       = [128, 256, 512, 1024, 2048]


def run_kernel(device, map_size):
    """Transfer, execute, and retrieve one height map on the open device."""
    height_cpu = generate_height_map_cpu(map_size, seed=42)

    t0 = time.perf_counter()
    input_tensor = ttnn.from_torch(
        height_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.from_torch(
        torch.zeros_like(height_cpu),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    t_transfer_to = time.perf_counter() - t0

    t1 = time.perf_counter()
    scale_height_map(input_tensor, output_tensor)
    t_kernel = time.perf_counter() - t1

    t2 = time.perf_counter()
    result = ttnn.to_torch(output_tensor)
    t_transfer_from = time.perf_counter() - t2

    return result, t_transfer_to * 1000, t_kernel * 1000, t_transfer_from * 1000


def benchmark_size(device, map_size):
    """Warm up then time TIMED_RUNS repetitions for one map size."""
    print(f"\n  {map_size}x{map_size}  ", end="", flush=True)

    # Warm-up: compile and cache the kernel for this shape
    for _ in range(WARMUP_RUNS):
        run_kernel(device, map_size)
        print("W", end="", flush=True)

    # Timed runs
    times_kernel = []
    times_total  = []
    for _ in range(TIMED_RUNS):
        result, t_to, t_k, t_from = run_kernel(device, map_size)
        times_kernel.append(t_k)
        times_total.append(t_to + t_k + t_from)
        print(".", end="", flush=True)

    # Correctness (last result)
    height_cpu = generate_height_map_cpu(map_size, seed=42)
    correct = torch.allclose(result, height_cpu, rtol=1e-2, atol=1e-2)

    best_k = min(times_kernel)
    avg_k  = sum(times_kernel) / len(times_kernel)
    best_t = min(times_total)
    avg_t  = sum(times_total)  / len(times_total)

    tiles = (map_size // 32) ** 2

    return {
        "map_size":           map_size,
        "tiles":              tiles,
        "kernel_best_ms":     best_k,
        "kernel_avg_ms":      avg_k,
        "total_best_ms":      best_t,
        "total_avg_ms":       avg_t,
        "warmup_runs":        WARMUP_RUNS,
        "timed_runs":         TIMED_RUNS,
        "correctness":        bool(correct),
        "throughput_mtiles_s": tiles / (best_k / 1000) / 1e6,
    }


def main():
    print("=" * 70)
    print("TT-Lang Scaling Benchmark (warm-up edition)")
    print(f"Warm-up: {WARMUP_RUNS} run(s)  |  Timed: {TIMED_RUNS} run(s)")
    print("=" * 70)

    device = ttnn.open_device(device_id=0)
    try:
        results = []
        for size in SIZES:
            try:
                r = benchmark_size(device, size)
                results.append(r)
            except Exception as exc:
                print(f"\n  ERROR: {exc}")
                import traceback; traceback.print_exc()

    finally:
        ttnn.close_device(device)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("Benchmark Results (after warm-up)")
    print("=" * 80)
    print(f"{'Size':<10} {'Tiles':>8} {'Kernel best':>14} {'Kernel avg':>12} "
          f"{'Total best':>12} {'MTiles/s':>12} {'OK':>5}")
    print("-" * 80)
    for r in results:
        print(f"{r['map_size']}x{r['map_size']:<5} "
              f"{r['tiles']:>8} "
              f"{r['kernel_best_ms']:>12.3f}ms "
              f"{r['kernel_avg_ms']:>10.3f}ms "
              f"{r['total_best_ms']:>10.3f}ms "
              f"{r['throughput_mtiles_s']:>10.1f}M "
              f"{'✓' if r['correctness'] else '✗':>5}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "output" / "benchmark_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
