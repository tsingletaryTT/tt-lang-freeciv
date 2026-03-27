# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Terrain Colorize Kernel
========================

Maps a height-map tensor to a biome-colored output tensor using a TT-Lang
eltwise-multiply kernel.  Every pixel's color is the product of:

  (biome_base_color) × (TT genome color bias)

The biome_base_color tensor encodes the "ground truth" terrain color —
ocean is blue, plains are green-yellow, mountains are gray, etc.
The genome color bias is a per-element multiplier generated from the
civilization genome pipeline (smooth_height_map), making each game load
render terrain in subtly different hues.

Kernel pipeline (all on TT hardware):
  1. smooth_height_map(biome_a, genome_shift_b, shifted_biome)
     — shifts base biome colors toward genome-derived palette
  2. scale_height_map(shifted_biome, output)
     — normalizes the result back to [0, 1000] range

CPU roles:
  - Discretize height_map values into biome buckets
  - Build the biome_base_color tensor (one color per tile, tiled to TILE_SIZE)
  - Apply inverse-transform to bring TT output into [0, 255] pixel range

This produces "infinite worlds" where ocean is always recognizably blue
but the exact shade — deep cobalt, turquoise, stormy slate — varies per seed.
Same for every other terrain type.

Biome color bases (before genome shift):
  Ocean      [0,   150]:  (40,  90, 180)  — deep blue
  Coast      [150, 250]:  (80, 130, 200)  — medium blue
  Plains     [250, 400]:  (165, 165, 95)  — yellow-green
  Grassland  [400, 550]:  (80,  150, 65)  — green
  Hills      [550, 650]:  (120, 105, 75)  — brown-green
  Forest     [550, 650 with forest flag]:  (35,  95, 40) — dark green (applied post)
  Mountains  [650, 850]:  (130, 115, 95)  — brown-gray
  Tundra     [850, 920]:  (175, 185, 195) — blue-gray
  Arctic     [920,1000]:  (225, 235, 250) — near-white blue
"""

import math
import time
import torch
import ttl
import ttnn

from height_map_simple import scale_height_map
from height_map_smooth  import smooth_height_map, to_device_half, zeros_like_on_device

TILE_SIZE   = 32
GRANULARITY = 2

# ── Biome definitions ─────────────────────────────────────────────────────────
# Each biome: (height_min, height_max, R, G, B)
# Heights are in the 0-1000 TT scale.  RGB are 0-255.
# Order matters: first matching biome wins.

BIOMES = [
    (  0, 150,  40,  90, 180),   # Ocean
    (150, 250,  80, 130, 200),   # Coast
    (250, 400, 165, 165,  95),   # Plains
    (400, 550,  80, 150,  65),   # Grassland
    (550, 650, 120, 105,  75),   # Hills
    (650, 850, 130, 115,  95),   # Mountains
    (850, 920, 175, 185, 195),   # Tundra
    (920,1001, 225, 235, 250),   # Arctic
]


def _height_to_biome_rgb(height: float) -> tuple[int, int, int]:
    """Return the base biome RGB color for a given height value (0-1000)."""
    for h_min, h_max, r, g, b in BIOMES:
        if h_min <= height < h_max:
            return (r, g, b)
    return (225, 235, 250)   # Arctic fallback


def build_biome_tensor(
    height_map_flat: list,
    rows: int,
    cols: int,
    channel: int,   # 0=R, 1=G, 2=B
) -> torch.Tensor:
    """
    Build a (rows × cols) bfloat16 tensor encoding one color channel
    of the biome base color for each tile, scaled to [0, 1000].

    Parameters
    ----------
    height_map_flat : flat list of height values (0-1000), length = rows * cols
    rows, cols      : tensor dimensions (must be tile-aligned)
    channel         : 0=red, 1=green, 2=blue

    Returns
    -------
    bfloat16 tensor of shape (rows, cols), values in [0, 1000]
    """
    data = []
    for i in range(rows * cols):
        h = height_map_flat[i] if i < len(height_map_flat) else 500.0
        rgb = _height_to_biome_rgb(float(h))
        # Scale from [0, 255] to [0, 1000]
        data.append(float(rgb[channel]) * (1000.0 / 255.0))

    return torch.tensor(data, dtype=torch.bfloat16).reshape(rows, cols)


def colorize_terrain_hardware(
    height_map_flat: list,
    genome_color_offset: torch.Tensor,   # (3,) R/G/B offset, values in [0, 1]
    w: int,
    h: int,
    device,
) -> tuple[torch.Tensor, float]:
    """
    Apply TT hardware to colorize a height map into RGB terrain tiles.

    For each channel (R, G, B):
      1. Build biome_base tensor on CPU (encodes terrain type per tile)
      2. Build genome_shift tensor on CPU (genome color offset, broadcast)
      3. Run smooth_height_map(biome_base * 0.85, genome_shift * 0.15)
         on TT hardware — the addition blends base color with genome shift
      4. Run scale_height_map for normalization on TT hardware
      5. CPU converts TT output back to [0, 255] pixel values

    Parameters
    ----------
    height_map_flat    : flat list of height values (0-1000)
    genome_color_offset: (3,) float32 tensor; each value in [0, 1]
                         represents the genome's per-channel color bias
                         (from color_hue, color_sat, color_val genome params)
    w, h               : map dimensions
    device             : open ttnn device

    Returns
    -------
    (rgb_tensor, kernel_ms)
        rgb_tensor  : float32 tensor of shape (h, w, 3), values in [0, 255]
        kernel_ms   : total time in TT kernels (6 kernel calls, 2 per channel)
    """
    map_tiles = w * h
    rows = max(TILE_SIZE, ((h + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)
    cols = max(TILE_SIZE, ((w + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)

    # Pad height map to (rows × cols)
    hm_padded = (height_map_flat + [500.0] * (rows * cols))[:rows * cols]

    rgb_channels = []
    total_ms = 0.0

    for ch in range(3):   # R, G, B
        # ── CPU: build biome base and genome shift tensors ───────────────────
        biome_base = build_biome_tensor(hm_padded, rows, cols, ch)

        # Genome shift: broadcast a single per-channel value across all tiles
        # The shift is proportional to how much the genome wants to push this color
        genome_val  = genome_color_offset[ch].item()   # 0-1
        # Shift in [0, 150] range — enough to noticeably tint without washing out
        shift_value = genome_val * 150.0
        genome_shift = torch.full((rows, cols), shift_value, dtype=torch.bfloat16)

        # Blend weights: 85% biome base + 15% genome shift
        # This preserves terrain recognizability while adding per-load variety
        a_half = (biome_base * 0.85).to(torch.bfloat16)
        b_half = (genome_shift * 0.15).to(torch.bfloat16)

        # ── TT Stage 1: smooth_height_map blends biome base with genome shift ─
        a_t   = to_device_half(biome_base * 0.85, device)
        b_t   = to_device_half(genome_shift * 0.15, device)
        mid_t = zeros_like_on_device(biome_base, device)

        t0 = time.perf_counter()
        smooth_height_map(a_t, b_t, mid_t)

        # ── TT Stage 2: scale_height_map normalizes the blended result ────────
        out_t = zeros_like_on_device(biome_base, device)
        scale_height_map(mid_t, out_t)
        total_ms += (time.perf_counter() - t0) * 1000

        # ── CPU: extract channel values [0, 255] ────────────────────────────
        channel_result = ttnn.to_torch(out_t).float()[:h, :w]
        # TT output is in [0, 1000]; convert back to [0, 255]
        channel_pixels = (channel_result * (255.0 / 1000.0)).clamp(0.0, 255.0)
        rgb_channels.append(channel_pixels)

    # Stack into (h, w, 3) RGB tensor
    rgb_tensor = torch.stack(rgb_channels, dim=2)   # (h, w, 3)
    return rgb_tensor, total_ms


def colorize_to_image(
    height_map_flat: list,
    genome: torch.Tensor,
    w: int,
    h: int,
    device,
) -> tuple["Image.Image", float]:
    """
    Convenience: colorize height map → PIL Image.

    Extracts genome color offsets from [16, 17, 18] (hue→R offset, sat→G offset,
    val→B offset after normalization) and calls colorize_terrain_hardware.

    Parameters
    ----------
    height_map_flat : flat list of height values (0-1000)
    genome          : (64,) float32 genome vector
    w, h            : map dimensions
    device          : open ttnn device

    Returns
    -------
    (PIL_Image, kernel_ms)
    """
    try:
        from PIL import Image as _PIL_Image
    except ImportError:
        raise ImportError("Pillow required for image output: pip install Pillow")

    # Extract per-channel genome color offsets from the color genome params.
    # genome[16] = hue → biases red channel (warm colors)
    # genome[17] = saturation → biases green channel (nature/life colors)
    # genome[18] = brightness → biases blue channel (cool colors)
    genome_color_offset = torch.tensor([
        genome[16].item(),   # R bias (hue)
        genome[17].item(),   # G bias (saturation)
        genome[18].item(),   # B bias (brightness)
    ], dtype=torch.float32)

    rgb_tensor, ms = colorize_terrain_hardware(
        height_map_flat, genome_color_offset, w, h, device
    )

    # Convert to PIL
    rgb_uint8 = rgb_tensor.byte().numpy()
    img = _PIL_Image.fromarray(rgb_uint8, mode="RGB")
    return img, ms


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    """
    Test terrain colorization on P300C hardware.
    Generates a height map, colorizes it with TT hardware, and saves a PNG.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from kernels.height_map_simple import generate_height_map_cpu
    from kernels.civ_genome        import generate_genomes_hardware

    W, H = 64, 64
    SEED = 42
    N    = 1

    print("=" * 60)
    print("Terrain Colorize Kernel — Hardware Test")
    print(f"Map size: {W}×{H}  seed={SEED}")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        # Warm up
        print("\nWarm-up...")
        for _ in range(2):
            generate_genomes_hardware(N, SEED, device)

        # Generate height map on CPU
        map_size = max(TILE_SIZE, ((max(W, H) + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)
        hm_cpu   = generate_height_map_cpu(map_size, seed=SEED)
        # Sample to W×H
        hm_flat = []
        for row in range(H):
            for col in range(W):
                sr = row * map_size // H
                sc = col * map_size // W
                hm_flat.append(hm_cpu[sr, sc].item())

        # Generate genome for color bias
        genomes, g_ms = generate_genomes_hardware(N, SEED, device)
        print(f"Genome gen: {g_ms:.3f}ms")

        # Colorize
        print("\nColorizing terrain on TT hardware...")
        img, c_ms = colorize_to_image(hm_flat, genomes[0], W, H, device)
        print(f"Colorize: {c_ms:.3f}ms total ({c_ms/6:.3f}ms per kernel call)")
        print(f"  (6 TT kernel calls: 2 per color channel × 3 channels)")

        # Save
        out_path = __import__("pathlib").Path("output/terrain_colorized.png")
        out_path.parent.mkdir(exist_ok=True)
        img.save(out_path)
        print(f"\nSaved: {out_path}")

        # Show biome distribution
        from collections import Counter
        biome_counts = Counter()
        for h_val in hm_flat:
            for h_min, h_max, r, g, b in BIOMES:
                if h_min <= h_val < h_max:
                    biome_counts[(h_min, h_max)] += 1
                    break
        biome_names = ["Ocean", "Coast", "Plains", "Grassland", "Hills", "Mountains",
                       "Tundra", "Arctic"]
        print("\nBiome distribution:")
        for name, (entry, *_) in zip(biome_names, BIOMES):
            count = biome_counts.get((entry[0], entry[1]), 0)
            pct   = 100 * count / (W * H)
            print(f"  {name:<12} {count:4d} tiles ({pct:.1f}%)")

    finally:
        ttnn.close_device(device)

    print("\n✓ Terrain colorization complete!")


if __name__ == "__main__":
    main()
