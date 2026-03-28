#!/usr/bin/env python3
"""
Complete TT-Lang + FreeCiv Pipeline Demo

This script demonstrates the full workflow:
1. Generate height map (CPU noise, structured for TT-Lang acceleration)
2. Classify terrain (parallel classification logic)
3. Visualize results
4. Export to FreeCiv-compatible formats

This shows how TT-Lang can be used for game terrain generation,
with the CPU handling complex math (sine waves) and TT-Lang
handling parallel array operations (classification, filtering).
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, '/home/ttuser/tt-lang-freeciv/kernels')
sys.path.insert(0, '/home/ttuser/tt-lang-freeciv/bridge')

# Import our modules (suppress ttl import warnings)
import warnings
warnings.filterwarnings('ignore')

from terrain_classify import classify_terrain, SHORE_LEVEL, MOUNTAIN_LEVEL
from freeciv_export import FreeCivExporter


def generate_height_map_cpu(map_size: int, seed: int = 42) -> torch.Tensor:
    """Generate height map using CPU sine wave noise"""
    torch.manual_seed(seed)

    x = torch.linspace(0, 4 * np.pi, map_size)
    y = torch.linspace(0, 4 * np.pi, map_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Multi-frequency sine wave terrain
    noise = (
        torch.sin(X) * torch.cos(Y) * 100 +
        torch.sin(X * 2.3) * torch.cos(Y * 1.7) * 50 +
        torch.sin(X * 5.1) * torch.cos(Y * 4.3) * 25
    )

    # Normalize to [0, 1000] range (FreeCiv format)
    noise_min = noise.min()
    noise_max = noise.max()
    normalized = (noise - noise_min) / (noise_max - noise_min)
    height_map = normalized * 1000

    return height_map


def visualize_combined(height_map, terrain, save_path):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Convert to numpy
    if isinstance(height_map, torch.Tensor):
        height_map_np = height_map.cpu().numpy()
    else:
        height_map_np = height_map

    if isinstance(terrain, torch.Tensor):
        terrain_np = terrain.cpu().numpy()
    else:
        terrain_np = terrain

    # 1. Height map (raw)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(height_map_np, cmap='terrain', interpolation='bilinear')
    ax1.set_title('Height Map\n(Raw elevation data)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    plt.colorbar(im1, ax=ax1, label='Height [0-1000]', shrink=0.8)

    # 2. Terrain classification
    ax2 = fig.add_subplot(gs[0, 1])
    from matplotlib.colors import ListedColormap
    colors = ['#4040ff', '#40b040', '#808080']
    cmap = ListedColormap(colors)
    im2 = ax2.imshow(terrain_np, cmap=cmap, interpolation='nearest')
    ax2.set_title('Terrain Classification\n(Ocean/Land/Mountains)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Ocean'),
        Patch(facecolor=colors[1], label='Land'),
        Patch(facecolor=colors[2], label='Mountains')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    # 3. Height histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(height_map_np.flatten(), bins=50, color='#40b040', alpha=0.7, edgecolor='black')
    ax3.axvline(SHORE_LEVEL, color='blue', linestyle='--', linewidth=2, label=f'Shore Level ({SHORE_LEVEL})')
    ax3.axvline(MOUNTAIN_LEVEL, color='red', linestyle='--', linewidth=2, label=f'Mountain Level ({MOUNTAIN_LEVEL})')
    ax3.set_title('Height Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Height')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. FreeCiv-style view (zoomed section)
    ax4 = fig.add_subplot(gs[1, 0])
    zoom_size = 64
    zoom_height = height_map_np[:zoom_size, :zoom_size]
    im4 = ax4.imshow(zoom_height, cmap='terrain', interpolation='nearest')
    ax4.set_title('Zoomed Section (64x64)\n(FreeCiv tile view)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Y coordinate')
    plt.colorbar(im4, ax=ax4, label='Height', shrink=0.8)

    # 5. Statistics table
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    total_tiles = terrain_np.size
    ocean_count = (terrain_np == 0).sum()
    land_count = (terrain_np == 1).sum()
    mountain_count = (terrain_np == 2).sum()

    stats_text = f"""
    Terrain Generation Statistics
    {'='*40}

    Map Size: {height_map_np.shape[0]} x {height_map_np.shape[1]}
    Total Tiles: {total_tiles:,}

    Height Statistics:
      Min:  {height_map_np.min():.1f}
      Max:  {height_map_np.max():.1f}
      Mean: {height_map_np.mean():.1f}
      Std:  {height_map_np.std():.1f}

    Terrain Distribution:
      Ocean:     {ocean_count:6,} tiles ({ocean_count/total_tiles*100:5.1f}%)
      Land:      {land_count:6,} tiles ({land_count/total_tiles*100:5.1f}%)
      Mountains: {mountain_count:6,} tiles ({mountain_count/total_tiles*100:5.1f}%)

    FreeCiv Compatibility:
      ✓ Height range: [0, 1000]
      ✓ Shore level: {SHORE_LEVEL}
      ✓ Mountain level: {MOUNTAIN_LEVEL}
      ✓ Format: Ready for integration

    Generation Method:
      - CPU: Multi-frequency sine waves
      - Classification: Threshold-based
      - Future: TT-Lang hardware acceleration
    """

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 6. Export formats info
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    export_text = """
    Export Formats Available
    ========================

    ✓ CSV Format
      - Plain text, human-readable
      - 256x256 grid of integers
      - Easy to edit/inspect

    ✓ NumPy Binary (.npy)
      - Fast loading
      - Preserves data types
      - Python-friendly

    ✓ FreeCiv Terrain Map
      - Character codes (g/m/ /+)
      - Direct terrain representation
      - Viewable in text editor

    ✓ C Source Code
      - Static array declaration
      - Drop-in FreeCiv integration
      - Includes load function

    ✓ JSON Metadata
      - Statistics and parameters
      - Generation details
      - Compatibility info

    Integration Path:
    1. Use C array in FreeCiv source
    2. Modify height_map.c
    3. Load data in mapgen
    4. Generate game map!
    """

    ax6.text(0.05, 0.95, export_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('TT-Lang + FreeCiv Terrain Generation Pipeline',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive visualization to {save_path}")


def main():
    print("="*70)
    print("TT-Lang + FreeCiv Terrain Generation Pipeline")
    print("="*70)

    # Configuration
    MAP_SIZE = 256
    SEED = 42
    OUTPUT_DIR = Path("/home/ttuser/tt-lang-freeciv/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate height map
    print("\n" + "="*70)
    print("STEP 1: Generate Height Map")
    print("="*70)
    print(f"  Map size: {MAP_SIZE}x{MAP_SIZE}")
    print(f"  Seed: {SEED}")
    print(f"  Method: Multi-frequency sine waves (CPU)")

    height_map = generate_height_map_cpu(MAP_SIZE, seed=SEED)

    print(f"\n  Height Statistics:")
    print(f"    Min:  {height_map.min():.1f}")
    print(f"    Max:  {height_map.max():.1f}")
    print(f"    Mean: {height_map.mean():.1f}")
    print(f"    Std:  {height_map.std():.1f}")
    print(f"\n  ✓ Height map generated!")

    # Step 2: Classify terrain
    print("\n" + "="*70)
    print("STEP 2: Classify Terrain")
    print("="*70)
    print(f"  Shore level: {SHORE_LEVEL}")
    print(f"  Mountain level: {MOUNTAIN_LEVEL}")

    terrain = classify_terrain(height_map, use_ttlang=False)

    total_tiles = terrain.numel()
    ocean_tiles = (terrain == 0).sum().item()
    land_tiles = (terrain == 1).sum().item()
    mountain_tiles = (terrain == 2).sum().item()

    print(f"\n  Terrain Distribution:")
    print(f"    Ocean:     {ocean_tiles:6,} tiles ({ocean_tiles/total_tiles*100:5.1f}%)")
    print(f"    Land:      {land_tiles:6,} tiles ({land_tiles/total_tiles*100:5.1f}%)")
    print(f"    Mountains: {mountain_tiles:6,} tiles ({mountain_tiles/total_tiles*100:5.1f}%)")
    print(f"\n  ✓ Terrain classified!")

    # Step 3: Visualize
    print("\n" + "="*70)
    print("STEP 3: Visualize Results")
    print("="*70)

    viz_path = OUTPUT_DIR / "pipeline_complete.png"
    visualize_combined(height_map, terrain, viz_path)

    # Step 4: Export to all formats
    print("\n" + "="*70)
    print("STEP 4: Export to FreeCiv Formats")
    print("="*70)

    exporter = FreeCivExporter(output_dir=str(OUTPUT_DIR))

    print("\n  Exporting to multiple formats...")
    exporter.export_height_map_csv(height_map, "pipeline_height.csv")
    print()
    exporter.export_height_map_npy(height_map, "pipeline_height.npy")
    print()
    exporter.export_terrain_map(terrain, height_map, "pipeline_terrain.txt")
    print()
    exporter.export_c_array(height_map, "pipeline_height.c")
    print()
    exporter.export_metadata(height_map, terrain, "pipeline_metadata.json")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.name.startswith('pipeline_'):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name:<30} ({size_kb:>7.1f} KB)")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("""
  1. View visualization: pipeline_complete.png
  2. Inspect terrain map: pipeline_terrain.txt (text-viewable)
  3. Integrate with FreeCiv:
     - Copy pipeline_height.c to FreeCiv source
     - Modify server/generator/height_map.c
     - Call load_ttlang_height_map() in map generation
  4. Hardware deployment:
     - Run on Tenstorrent hardware for larger maps
     - Compare CPU vs hardware performance
     - Scale to 512x512, 1024x1024, or larger
    """)

    print("="*70)
    print("✓ Demo complete! Terrain ready for FreeCiv integration.")
    print("="*70)


if __name__ == "__main__":
    main()
