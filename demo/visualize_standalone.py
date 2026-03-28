#!/usr/bin/env python3
"""
Standalone visualization of height map (no TT-Lang dependencies)
Generates CPU noise and visualizes terrain classification
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def generate_height_map_cpu(map_size: int, seed: int = 42) -> torch.Tensor:
    """Generate height map using CPU sine wave noise"""
    torch.manual_seed(seed)

    # Generate coordinate grids
    x = torch.linspace(0, 4 * np.pi, map_size)
    y = torch.linspace(0, 4 * np.pi, map_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Multi-frequency sine wave terrain
    noise = (
        torch.sin(X) * torch.cos(Y) * 100 +           # Large features
        torch.sin(X * 2.3) * torch.cos(Y * 1.7) * 50 + # Medium features
        torch.sin(X * 5.1) * torch.cos(Y * 4.3) * 25    # Small features
    )

    # Normalize to [0, 1000] range (FreeCiv format)
    noise_min = noise.min()
    noise_max = noise.max()
    normalized = (noise - noise_min) / (noise_max - noise_min)
    height_map = normalized * 1000

    return height_map


def visualize_height_map(height_map, title="Height Map", save_path=None):
    """Visualize a height map with terrain-like colors"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Convert to numpy
    if isinstance(height_map, torch.Tensor):
        height_map = height_map.to(torch.float32).numpy()

    # Plot 1: Height map with terrain colors
    im1 = ax1.imshow(height_map, cmap='terrain', interpolation='bilinear')
    ax1.set_title(f'{title}\n({height_map.shape[0]}x{height_map.shape[1]} tiles)')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    plt.colorbar(im1, ax=ax1, label='Height [0-1000]')

    # Plot 2: Terrain classification
    # Simulate FreeCiv terrain types based on height thresholds
    terrain = np.zeros_like(height_map)
    shore_level = 500  # FreeCiv typical shore level
    mountain_level = 750  # Mountain threshold

    terrain[height_map < shore_level] = 0  # Ocean
    terrain[(height_map >= shore_level) & (height_map < mountain_level)] = 1  # Land
    terrain[height_map >= mountain_level] = 2  # Mountains

    colors = ['#4040ff', '#40b040', '#808080']  # Ocean, Land, Mountains
    labels = ['Ocean', 'Land', 'Mountains']

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)

    im2 = ax2.imshow(terrain, cmap=cmap, interpolation='nearest')
    ax2.set_title('Terrain Classification\n(FreeCiv-style)')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")

    # Statistics
    ocean_pct = (terrain == 0).sum() / terrain.size * 100
    land_pct = (terrain == 1).sum() / terrain.size * 100
    mountain_pct = (terrain == 2).sum() / terrain.size * 100

    print(f"\nTerrain Distribution:")
    print(f"  Ocean:     {ocean_pct:5.1f}%")
    print(f"  Land:      {land_pct:5.1f}%")
    print(f"  Mountains: {mountain_pct:5.1f}%")

    return terrain


if __name__ == "__main__":
    print("Generating height maps with CPU noise (FreeCiv-compatible)...")

    # Generate different map sizes
    sizes = [128, 256, 512]

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Generating {size}x{size} map...")
        print('='*60)

        height_map = generate_height_map_cpu(size, seed=42)

        print(f"Height map statistics:")
        print(f"  Min:  {height_map.min().item():.1f}")
        print(f"  Max:  {height_map.max().item():.1f}")
        print(f"  Mean: {height_map.mean().item():.1f}")

        save_path = f"/home/ttuser/tt-lang-freeciv/demo/height_map_{size}x{size}.png"
        visualize_height_map(height_map, title=f"Height Map ({size}x{size})", save_path=save_path)

    print("\n" + "="*60)
    print("✓ All visualizations complete!")
    print("="*60)
    print("\nGenerated files:")
    for size in sizes:
        print(f"  - demo/height_map_{size}x{size}.png")
