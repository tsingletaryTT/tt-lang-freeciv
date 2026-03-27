#!/usr/bin/env python3
"""
FreeCiv Export Bridge

Exports TT-Lang generated terrain to formats usable with FreeCiv:
- CSV format for easy viewing/editing
- NumPy binary format for fast loading
- Visualization for comparison with FreeCiv style
- Documentation for integration with FreeCiv source

FreeCiv Integration Points:
-------------------------
1. Direct C integration:
   - Modify server/generator/height_map.c
   - Replace make_random_hmap() or make_pseudofractal1_hmap()
   - Load height_map[] array from our exported data

2. Scenario file integration:
   - Parse existing .sav file
   - Modify [map] section terrain data
   - Save as custom scenario

3. External generator:
   - Export as standalone generator
   - Call from FreeCiv via custom mapgen script
"""

import torch
import numpy as np
import csv
from pathlib import Path
from typing import Tuple, Optional
import json


class FreeCivExporter:
    """Export terrain data in FreeCiv-compatible formats"""

    # FreeCiv terrain codes (from terrident in .sav files)
    TERRAIN_CODES = {
        'ocean': ' ',       # Deep water
        'lake': '+',        # Shallow water
        'grassland': 'g',   # Default land
        'plains': 'p',      # Flat land
        'hills': 'h',       # Hills
        'mountains': 'm',   # Mountains
        'forest': 'f',      # Forested
        'jungle': 'j',      # Jungle
        'desert': 'd',      # Desert
        'tundra': 't',      # Tundra
        'glacier': 'a',     # Frozen
        'swamp': 's',       # Swamp
    }

    def __init__(self, output_dir: str = "/home/ttuser/tt-lang-freeciv/output"):
        """Initialize exporter with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_height_map_csv(self, height_map: torch.Tensor, filename: str = "height_map.csv"):
        """
        Export height map as CSV

        Args:
            height_map: Tensor of shape (H, W) with values [0, 1000]
            filename: Output CSV filename
        """
        output_path = self.output_dir / filename

        # Convert to numpy
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.cpu().numpy()

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(height_map.astype(int))

        print(f"✓ Exported height map to {output_path}")
        print(f"  Dimensions: {height_map.shape}")
        print(f"  Format: CSV (integer values [0, 1000])")

        return output_path

    def export_height_map_npy(self, height_map: torch.Tensor, filename: str = "height_map.npy"):
        """
        Export height map as NumPy binary format

        Args:
            height_map: Tensor of shape (H, W) with values [0, 1000]
            filename: Output .npy filename
        """
        output_path = self.output_dir / filename

        # Convert to numpy
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.cpu().numpy()

        # Save as .npy
        np.save(output_path, height_map.astype(np.int32))

        print(f"✓ Exported height map to {output_path}")
        print(f"  Dimensions: {height_map.shape}")
        print(f"  Format: NumPy binary (.npy)")

        return output_path

    def export_terrain_map(self, terrain: torch.Tensor,
                          height_map: Optional[torch.Tensor] = None,
                          filename: str = "terrain_map.txt"):
        """
        Export classified terrain as FreeCiv-style text map

        Args:
            terrain: Tensor of shape (H, W) with values:
                0 = Ocean, 1 = Land, 2 = Mountains
            height_map: Optional height map for finer classification
            filename: Output text filename
        """
        output_path = self.output_dir / filename

        # Convert to numpy
        if isinstance(terrain, torch.Tensor):
            terrain = terrain.cpu().numpy()
        if height_map is not None and isinstance(height_map, torch.Tensor):
            height_map = height_map.cpu().numpy()

        # Map terrain codes to FreeCiv characters
        terrain_chars = np.empty(terrain.shape, dtype=str)

        # Basic classification
        terrain_chars[terrain == 0] = self.TERRAIN_CODES['ocean']
        terrain_chars[terrain == 1] = self.TERRAIN_CODES['grassland']
        terrain_chars[terrain == 2] = self.TERRAIN_CODES['mountains']

        # Finer classification using height if available
        if height_map is not None:
            # Hills (moderate elevation land)
            is_land = (terrain == 1)
            is_hilly = (height_map > 600) & (height_map < 750)
            terrain_chars[is_land & is_hilly] = self.TERRAIN_CODES['hills']

            # Shallow water (near shore)
            is_ocean = (terrain == 0)
            is_shallow = (height_map > 400) & (height_map < 500)
            terrain_chars[is_ocean & is_shallow] = self.TERRAIN_CODES['lake']

        # Write as text map
        with open(output_path, 'w') as f:
            f.write(f"# FreeCiv Terrain Map\n")
            f.write(f"# Generated by TT-Lang terrain generator\n")
            f.write(f"# Dimensions: {terrain.shape[0]} x {terrain.shape[1]}\n")
            f.write(f"#\n")
            f.write(f"# Terrain codes:\n")
            for name, code in self.TERRAIN_CODES.items():
                f.write(f"#   '{code}' = {name}\n")
            f.write(f"#\n")

            for row in terrain_chars:
                f.write(''.join(row) + '\n')

        print(f"✓ Exported terrain map to {output_path}")
        print(f"  Dimensions: {terrain.shape}")
        print(f"  Format: FreeCiv-style terrain codes")

        return output_path

    def export_c_array(self, height_map: torch.Tensor, filename: str = "height_map.c"):
        """
        Export height map as C array for direct FreeCiv integration

        Args:
            height_map: Tensor of shape (H, W) with values [0, 1000]
            filename: Output C source filename
        """
        output_path = self.output_dir / filename

        # Convert to numpy
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.cpu().numpy()

        H, W = height_map.shape
        size = H * W

        # Write C source file
        with open(output_path, 'w') as f:
            f.write(f"/* TT-Lang Generated Height Map */\n")
            f.write(f"/* Dimensions: {H} x {W} = {size} tiles */\n")
            f.write(f"/* Generated by tt-lang-freeciv project */\n\n")

            f.write(f"#define MAP_HEIGHT {H}\n")
            f.write(f"#define MAP_WIDTH {W}\n")
            f.write(f"#define MAP_SIZE {size}\n\n")

            f.write(f"/* Height map data [0, 1000] */\n")
            f.write(f"static int ttlang_height_map[MAP_SIZE] = {{\n")

            # Write array data (row-major order)
            flat = height_map.flatten().astype(int)
            for i in range(0, len(flat), 10):
                chunk = flat[i:i+10]
                f.write("    " + ", ".join(f"{val:4d}" for val in chunk))
                if i + 10 < len(flat):
                    f.write(",\n")
                else:
                    f.write("\n")

            f.write("};\n\n")

            f.write(f"/* Load function for FreeCiv integration */\n")
            f.write(f"void load_ttlang_height_map(int *height_map) {{\n")
            f.write(f"    int i;\n")
            f.write(f"    for (i = 0; i < MAP_SIZE; i++) {{\n")
            f.write(f"        height_map[i] = ttlang_height_map[i];\n")
            f.write(f"    }}\n")
            f.write(f"}}\n")

        print(f"✓ Exported C array to {output_path}")
        print(f"  Dimensions: {H} x {W}")
        print(f"  Format: C source code with static array")
        print(f"\n  Integration:")
        print(f"    1. Include this file in server/generator/height_map.c")
        print(f"    2. Call load_ttlang_height_map(height_map) in make_random_hmap()")

        return output_path

    def export_metadata(self, height_map: torch.Tensor, terrain: torch.Tensor,
                       filename: str = "metadata.json"):
        """
        Export metadata about the generated terrain

        Args:
            height_map: Height map tensor
            terrain: Terrain classification tensor
            filename: Output JSON filename
        """
        output_path = self.output_dir / filename

        # Calculate statistics
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.cpu()
        if isinstance(terrain, torch.Tensor):
            terrain = terrain.cpu()

        H, W = height_map.shape
        total_tiles = H * W

        ocean_count = (terrain == 0).sum().item()
        land_count = (terrain == 1).sum().item()
        mountain_count = (terrain == 2).sum().item()

        metadata = {
            "dimensions": {
                "height": int(H),
                "width": int(W),
                "total_tiles": int(total_tiles)
            },
            "height_statistics": {
                "min": float(height_map.min()),
                "max": float(height_map.max()),
                "mean": float(height_map.mean()),
                "std": float(height_map.std())
            },
            "terrain_distribution": {
                "ocean": {
                    "count": int(ocean_count),
                    "percentage": float(ocean_count / total_tiles * 100)
                },
                "land": {
                    "count": int(land_count),
                    "percentage": float(land_count / total_tiles * 100)
                },
                "mountains": {
                    "count": int(mountain_count),
                    "percentage": float(mountain_count / total_tiles * 100)
                }
            },
            "freeciv_compatibility": {
                "height_range": "[0, 1000]",
                "shore_level": 500,
                "mountain_level": 750,
                "format_version": "3.2.0"
            },
            "generation": {
                "method": "TT-Lang + CPU noise",
                "kernel": "height_map_simple.py",
                "classifier": "terrain_classify.py"
            }
        }

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Exported metadata to {output_path}")

        return output_path


# Demo
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/ttuser/tt-lang-freeciv/kernels')

    # Import generators (use standalone versions to avoid ttl import issues)
    import numpy as np

    def generate_test_terrain(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test height map and terrain classification"""
        torch.manual_seed(42)

        # Generate height map
        x = torch.linspace(0, 4 * np.pi, size)
        y = torch.linspace(0, 4 * np.pi, size)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        noise = (
            torch.sin(X) * torch.cos(Y) * 100 +
            torch.sin(X * 2.3) * torch.cos(Y * 1.7) * 50 +
            torch.sin(X * 5.1) * torch.cos(Y * 4.3) * 25
        )

        noise_min = noise.min()
        noise_max = noise.max()
        normalized = (noise - noise_min) / (noise_max - noise_min)
        height_map = normalized * 1000

        # Classify terrain
        terrain = torch.zeros_like(height_map, dtype=torch.int32)
        terrain[height_map < 500] = 0  # Ocean
        terrain[(height_map >= 500) & (height_map < 750)] = 1  # Land
        terrain[height_map >= 750] = 2  # Mountains

        return height_map, terrain

    print("="*60)
    print("FreeCiv Export Demo")
    print("="*60)

    # Generate terrain
    print("\n1. Generating 256x256 terrain...")
    height_map, terrain = generate_test_terrain(256)

    # Export in all formats
    print("\n2. Exporting to multiple formats...")
    exporter = FreeCivExporter()

    exporter.export_height_map_csv(height_map)
    print()
    exporter.export_height_map_npy(height_map)
    print()
    exporter.export_terrain_map(terrain, height_map)
    print()
    exporter.export_c_array(height_map)
    print()
    exporter.export_metadata(height_map, terrain)

    print("\n" + "="*60)
    print("✓ Export complete!")
    print("="*60)
    print(f"\nOutputs saved to: {exporter.output_dir}")
    print("\nFiles created:")
    for f in sorted(exporter.output_dir.iterdir()):
        print(f"  - {f.name}")
