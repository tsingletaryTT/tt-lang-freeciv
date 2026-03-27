# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple height map generation - proof of concept
"""

import torch
import ttl
import ttnn
import math

TILE_SIZE = 32
GRANULARITY = 2


@ttl.kernel(grid="auto")
def scale_height_map(
    input_map: ttnn.Tensor,
    output_map: ttnn.Tensor
) -> None:
    """Scale input values to FreeCiv height range [0, 1000]"""

    row_tiles = input_map.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = input_map.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    in_dfb = ttl.make_dataflow_buffer_like(input_map, shape=(GRANULARITY, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(output_map, shape=(GRANULARITY, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
                            # For now, just pass through
                            # In a real implementation, would scale here
                            out_blk.store(in_blk)

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
                        with in_dfb.reserve() as blk:
                            tx = ttl.copy(input_map[r0:r1, col:col + 1], blk)
                            tx.wait()

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
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, output_map[r0:r1, col:col + 1])
                            tx.wait()


def generate_height_map_cpu(map_size, seed=42):
    """Generate height map on CPU"""
    torch.manual_seed(seed)

    # Create coordinate grids
    x = torch.linspace(0, 1, map_size)
    y = torch.linspace(0, 1, map_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Multiple frequency components
    noise1 = torch.sin(X * math.pi * 4.0) + torch.sin(Y * math.pi * 4.0)
    noise2 = torch.sin(X * math.pi * 8.0 + 1.5) + torch.sin(Y * math.pi * 8.0 + 1.5)
    noise3 = torch.sin(X * math.pi * 16.0) * torch.sin(Y * math.pi * 16.0)

    # Combine and scale to [0, 1000]
    combined = noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2
    # Normalize from roughly [-3, 3] to [0, 1000]
    height_map = ((combined + 3.0) / 6.0) * 1000.0

    return height_map.to(torch.bfloat16)


def main():
    """Test height map generation"""
    device = ttnn.open_device(device_id=0)
    try:
        # Generate map
        map_size = 256

        print(f"Generating {map_size}x{map_size} height map on CPU...")
        height_cpu = generate_height_map_cpu(map_size, seed=42)

        # Convert to TT tensors
        input_tensor = ttnn.from_torch(
            height_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        output_torch = torch.zeros_like(height_cpu)
        output_tensor = ttnn.from_torch(
            output_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Process through TT kernel (passthrough for now)
        print("Processing through TT-Lang kernel...")
        scale_height_map(input_tensor, output_tensor)

        # Get results
        result = ttnn.to_torch(output_tensor)

        # Stats
        min_h = result.min().item()
        max_h = result.max().item()
        mean_h = result.mean().item()

        print(f"\n✓ Height map generated!")
        print(f"  Size: {map_size}x{map_size}")
        print(f"  Min:  {min_h:.1f}")
        print(f"  Max:  {max_h:.1f}")
        print(f"  Mean: {mean_h:.1f}")
        print(f"  Range: [0, 1000] (FreeCiv format)")

        assert min_h >= 0 and max_h <= 1000, f"Out of range!"
        assert torch.allclose(result, height_cpu, rtol=1e-2, atol=1e-2), "Mismatch!"

        print("\n✓ PASSED!")

        # Sample
        print("\nSample (top-left 8x8):")
        sample = result[:8, :8].to(torch.float32).numpy().astype(int)
        print(sample)

        return result

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    result = main()
