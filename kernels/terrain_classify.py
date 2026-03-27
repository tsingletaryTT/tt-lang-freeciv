#!/usr/bin/env python3
"""
TT-Lang kernel for terrain classification

Takes a height map [0, 1000] and classifies each tile as:
- 0: Ocean (height < shore_level)
- 1: Land (shore_level <= height < mountain_level)
- 2: Mountains (height >= mountain_level)

This demonstrates:
- Parallel classification
- Conditional logic in compute kernel
- Real game logic processing
"""

import torch
import ttnn

# Import ttl only when running as kernel (not when visualizing)
try:
    import ttl
    TTL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TTL_AVAILABLE = False
    print("Warning: ttl not available, using CPU implementation only")

TILE_SIZE = 32
GRANULARITY = 4  # Process 4x4 tiles at a time

# FreeCiv terrain thresholds
SHORE_LEVEL = 500
MOUNTAIN_LEVEL = 750


def classify_terrain_cpu(height_map: torch.Tensor) -> torch.Tensor:
    """
    CPU version: classify terrain based on height thresholds

    Args:
        height_map: Tensor of shape (H, W) with values [0, 1000]

    Returns:
        terrain: Tensor of shape (H, W) with values:
            0 = Ocean, 1 = Land, 2 = Mountains
    """
    terrain = torch.zeros_like(height_map, dtype=torch.int32)

    # Ocean: height < 500
    terrain[height_map < SHORE_LEVEL] = 0

    # Land: 500 <= height < 750
    terrain[(height_map >= SHORE_LEVEL) & (height_map < MOUNTAIN_LEVEL)] = 1

    # Mountains: height >= 750
    terrain[height_map >= MOUNTAIN_LEVEL] = 2

    return terrain


if TTL_AVAILABLE:
    @ttl.kernel(grid="auto")
    def classify_terrain_ttlang(height_map: ttnn.Tensor, terrain: ttnn.Tensor) -> None:
        """
        TT-Lang kernel: parallel terrain classification

        Each core processes a portion of the map in parallel.
        Demonstrates conditional classification logic.
        """
        # Calculate grid dimensions
        row_tiles = height_map.shape[0] // TILE_SIZE // GRANULARITY
        col_tiles = height_map.shape[1] // TILE_SIZE

        grid_cols, grid_rows = ttl.grid_size(dims=2)
        rows_per_node = -(-row_tiles // grid_rows)  # Ceiling division
        cols_per_node = -(-col_tiles // grid_cols)

        # Create dataflow buffers
        height_dfb = ttl.make_dataflow_buffer_like(
            height_map, shape=(GRANULARITY, 1), buffer_factor=2
        )
        terrain_dfb = ttl.make_dataflow_buffer_like(
            terrain, shape=(GRANULARITY, 1), buffer_factor=2
        )

        @ttl.compute()
        def compute():
            """
            Classify terrain based on height thresholds

            Note: TT-Lang works on tiles, so we do passthrough here
            and let CPU do the actual classification. In real hardware,
            we'd use SFPU ops for threshold comparison.
            """
            node_col, node_row = ttl.node(dims=2)

            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with height_dfb.wait() as height_blk, terrain_dfb.reserve() as terrain_blk:
                                # In real implementation, would use SFPU comparison ops
                                # For now, passthrough (classification done on CPU)
                                terrain_blk.store(height_blk)

        @ttl.datamovement()
        def read():
            """Read height map tiles from DRAM"""
            node_col, node_row = ttl.node(dims=2)

            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    start_row_tile = row * GRANULARITY
                    end_row_tile = (row + 1) * GRANULARITY

                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            start_col_tile = col
                            end_col_tile = col + 1

                            with height_dfb.reserve() as height_blk:
                                tx = ttl.copy(
                                    height_map[start_row_tile:end_row_tile, start_col_tile:end_col_tile],
                                    height_blk
                                )
                                tx.wait()

        @ttl.datamovement()
        def write():
            """Write classified terrain tiles to DRAM"""
            node_col, node_row = ttl.node(dims=2)

            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    start_row_tile = row * GRANULARITY
                    end_row_tile = (row + 1) * GRANULARITY

                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            start_col_tile = col
                            end_col_tile = col + 1

                            with terrain_dfb.wait() as terrain_blk:
                                tx = ttl.copy(
                                    terrain_blk,
                                    terrain[start_row_tile:end_row_tile, start_col_tile:end_col_tile]
                                )
                                tx.wait()


def classify_terrain(height_map: torch.Tensor, use_ttlang: bool = True) -> torch.Tensor:
    """
    Classify terrain from height map

    Args:
        height_map: Tensor of shape (H, W) with values [0, 1000]
        use_ttlang: If True, use TT-Lang kernel; otherwise CPU

    Returns:
        terrain: Classified terrain map (0=Ocean, 1=Land, 2=Mountains)
    """
    if not use_ttlang or not TTL_AVAILABLE:
        return classify_terrain_cpu(height_map)

    # TODO: TT-Lang implementation when we have SFPU threshold ops
    # For now, fall back to CPU
    print("Note: TT-Lang classification not yet implemented, using CPU")
    return classify_terrain_cpu(height_map)


# Demo / test
if __name__ == "__main__":
    import numpy as np

    def generate_test_height_map(map_size: int, seed: int = 42) -> torch.Tensor:
        """Generate simple test height map using CPU sine waves"""
        torch.manual_seed(seed)
        x = torch.linspace(0, 4 * np.pi, map_size)
        y = torch.linspace(0, 4 * np.pi, map_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        noise = (
            torch.sin(X) * torch.cos(Y) * 100 +
            torch.sin(X * 2.3) * torch.cos(Y * 1.7) * 50 +
            torch.sin(X * 5.1) * torch.cos(Y * 4.3) * 25
        )

        noise_min = noise.min()
        noise_max = noise.max()
        normalized = (noise - noise_min) / (noise_max - noise_min)
        return normalized * 1000

    print("="*60)
    print("Terrain Classification Demo")
    print("="*60)

    # Generate height map
    print("\n1. Generating 256x256 height map...")
    height_map = generate_test_height_map(256, seed=42)

    print(f"   Height range: [{height_map.min():.1f}, {height_map.max():.1f}]")
    print(f"   Mean height: {height_map.mean():.1f}")

    # Classify terrain
    print("\n2. Classifying terrain...")
    terrain = classify_terrain(height_map, use_ttlang=False)

    # Statistics
    total_tiles = terrain.numel()
    ocean_tiles = (terrain == 0).sum().item()
    land_tiles = (terrain == 1).sum().item()
    mountain_tiles = (terrain == 2).sum().item()

    print(f"\n3. Terrain Distribution:")
    print(f"   Ocean:     {ocean_tiles:6d} tiles ({ocean_tiles/total_tiles*100:5.1f}%)")
    print(f"   Land:      {land_tiles:6d} tiles ({land_tiles/total_tiles*100:5.1f}%)")
    print(f"   Mountains: {mountain_tiles:6d} tiles ({mountain_tiles/total_tiles*100:5.1f}%)")

    print(f"\n4. Sample (top-left 8x8):")
    print("   Terrain codes: 0=Ocean, 1=Land, 2=Mountains")
    print(terrain[:8, :8].int())

    print("\n" + "="*60)
    print("✓ Classification complete!")
    print("="*60)
