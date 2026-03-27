#!/usr/bin/env python3
"""
Test TT-Lang kernels on real Tenstorrent hardware

This script runs our height map generation kernel on P300C hardware
and measures performance compared to CPU baseline.
"""

import torch
import time
import sys
import json
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

print("Importing TT modules...")
import ttnn
from height_map_simple import generate_height_map_cpu, scale_height_map

def test_hardware_basic(device_id=0, map_size=256):
    """Test basic hardware execution"""
    print(f"\n{'='*70}")
    print(f"Test 1: Basic Hardware Execution (Device {device_id})")
    print(f"{'='*70}\n")

    device = ttnn.open_device(device_id=device_id)
    try:
        # Generate height map on CPU
        print(f"Generating {map_size}x{map_size} height map on CPU...")
        t_cpu_start = time.time()
        height_cpu = generate_height_map_cpu(map_size, seed=42)
        t_cpu_end = time.time()
        cpu_time = t_cpu_end - t_cpu_start

        print(f"  CPU generation time: {cpu_time*1000:.2f} ms")

        # Convert to device tensors
        print("\nTransferring to device...")
        t_transfer_start = time.time()

        input_tensor = ttnn.from_torch(
            height_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        output_torch = torch.zeros_like(height_cpu)
        output_tensor = ttnn.from_torch(
            output_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        t_transfer_mid = time.time()
        transfer_to_device = t_transfer_mid - t_transfer_start

        # Run kernel on hardware
        print("Running TT-Lang kernel on hardware...")
        t_kernel_start = time.time()
        scale_height_map(input_tensor, output_tensor)
        t_kernel_end = time.time()
        kernel_time = t_kernel_end - t_kernel_start

        # Get results back
        t_transfer_back_start = time.time()
        result = ttnn.to_torch(output_tensor)
        t_transfer_back_end = time.time()
        transfer_from_device = t_transfer_back_end - t_transfer_back_start

        # Verify correctness
        min_h = result.min().item()
        max_h = result.max().item()
        mean_h = result.mean().item()

        print(f"\n✓ Kernel executed successfully!")
        print(f"  Height map size: {map_size}x{map_size}")
        print(f"  Min:  {min_h:.1f}")
        print(f"  Max:  {max_h:.1f}")
        print(f"  Mean: {mean_h:.1f}")

        # Timing breakdown
        total_time = transfer_to_device + kernel_time + transfer_from_device

        print(f"\n  Timing Breakdown:")
        print(f"    CPU generation:       {cpu_time*1000:8.2f} ms")
        print(f"    Transfer to device:   {transfer_to_device*1000:8.2f} ms")
        print(f"    Kernel execution:     {kernel_time*1000:8.2f} ms")
        print(f"    Transfer from device: {transfer_from_device*1000:8.2f} ms")
        print(f"    Total hardware time:  {total_time*1000:8.2f} ms")

        # Check correctness
        matches = torch.allclose(result, height_cpu, rtol=1e-2, atol=1e-2)
        print(f"\n  Correctness: {'✓ PASSED' if matches else '✗ FAILED'}")

        return {
            "map_size": map_size,
            "cpu_time_ms": cpu_time * 1000,
            "transfer_to_ms": transfer_to_device * 1000,
            "kernel_time_ms": kernel_time * 1000,
            "transfer_from_ms": transfer_from_device * 1000,
            "total_time_ms": total_time * 1000,
            "correctness": matches,
            "device_id": device_id
        }

    finally:
        ttnn.close_device(device)


def test_hardware_scaling(device_id=0):
    """Test performance scaling with different map sizes"""
    print(f"\n{'='*70}")
    print(f"Test 2: Hardware Scaling (Device {device_id})")
    print(f"{'='*70}\n")

    sizes = [128, 256, 512, 1024]
    results = []

    for size in sizes:
        print(f"\n--- Testing {size}x{size} map ---")
        try:
            result = test_hardware_basic(device_id, size)
            results.append(result)
        except Exception as e:
            print(f"✗ Failed for size {size}: {e}")
            continue

    # Summary
    print(f"\n{'='*70}")
    print("Scaling Summary")
    print(f"{'='*70}\n")

    print(f"{'Size':<12} {'CPU (ms)':<12} {'Kernel (ms)':<12} {'Total (ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    for r in results:
        speedup = r['cpu_time_ms'] / r['total_time_ms'] if r['total_time_ms'] > 0 else 0
        print(f"{r['map_size']}x{r['map_size']:<6} "
              f"{r['cpu_time_ms']:>10.2f}  "
              f"{r['kernel_time_ms']:>10.2f}  "
              f"{r['total_time_ms']:>10.2f}  "
              f"{speedup:>8.2f}x")

    return results


def test_multi_device():
    """Test execution across multiple devices"""
    print(f"\n{'='*70}")
    print("Test 3: Multi-Device Execution")
    print(f"{'='*70}\n")

    # Check available devices
    try:
        num_devices = ttnn.GetNumAvailableDevices()
        print(f"Found {num_devices} available devices\n")
    except:
        print("Cannot query device count, assuming 4 devices")
        num_devices = 4

    map_size = 256
    results = []

    for device_id in range(min(num_devices, 4)):
        print(f"\n--- Device {device_id} ---")
        try:
            result = test_hardware_basic(device_id, map_size)
            results.append(result)
        except Exception as e:
            print(f"✗ Device {device_id} failed: {e}")
            continue

    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("Multi-Device Summary")
        print(f"{'='*70}\n")

        avg_kernel_time = sum(r['kernel_time_ms'] for r in results) / len(results)
        avg_total_time = sum(r['total_time_ms'] for r in results) / len(results)

        print(f"  Tested on {len(results)} devices")
        print(f"  Average kernel time: {avg_kernel_time:.2f} ms")
        print(f"  Average total time:  {avg_total_time:.2f} ms")
        print(f"  Performance variance: ", end="")

        kernel_times = [r['kernel_time_ms'] for r in results]
        variance = max(kernel_times) - min(kernel_times)
        print(f"{variance:.2f} ms ({variance/avg_kernel_time*100:.1f}%)")

    return results


def save_results(all_results, output_file="hardware_test_results.json"):
    """Save test results to JSON"""
    output_path = Path(__file__).parent.parent / "output" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


def main():
    print("="*70)
    print("TT-Lang Hardware Testing Suite")
    print("P300C Blackhole Architecture")
    print("="*70)

    all_results = {}

    # Test 1: Basic execution
    try:
        result = test_hardware_basic(device_id=0, map_size=256)
        all_results['basic_test'] = result
    except Exception as e:
        print(f"\n✗ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Scaling
    try:
        scaling_results = test_hardware_scaling(device_id=0)
        all_results['scaling_test'] = scaling_results
    except Exception as e:
        print(f"\n✗ Scaling test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Multi-device
    try:
        multi_device_results = test_multi_device()
        all_results['multi_device_test'] = multi_device_results
    except Exception as e:
        print(f"\n✗ Multi-device test failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    save_results(all_results)

    print("\n" + "="*70)
    print("✓ Hardware testing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
