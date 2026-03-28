# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
flag_gen.py — PIL flag generator driven by TT genome color/pattern parameters
==============================================================================

All color and pattern parameters come from the civilization genome computed on
P300C Blackhole hardware.  CPU renders the flag image using PIL.

Genome parameters used:
  [16] color_hue    — primary hue (scaled to 0-360)
  [17] color_sat    — primary saturation (0-1)
  [18] color_val    — primary brightness (0-1)
  [19] color2_hue   — secondary hue
  [20] color2_sat   — secondary saturation
  [21] color2_val   — secondary brightness
  [22] flag_pattern — layout type (0-7 → stripe, cross, diagonal, etc.)
  [23] flag_emblem  — emblem type (0-5 → star, circle, triangle, diamond, etc.)
  [24] flag_complexity — detail level (0=simple, 1=complex)

Output:
  64×40 PNG flag — matches FreeCiv's expected small flag dimensions.
  The flag is also suitable for scaling to the 128×80 large size.

FreeCiv uses flags from data/flags/ directory.  We write:
  <output_dir>/flags/<nation_slug>.png  (64×40)
  <output_dir>/flags/<nation_slug>-large.png  (128×80)
"""

import math
import colorsys
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import torch

# ── Flag dimensions ──────────────────────────────────────────────────────────

FLAG_W      = 64
FLAG_H      = 40
FLAG_W_LARGE = 128
FLAG_H_LARGE = 80

# ── Helpers ──────────────────────────────────────────────────────────────────

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV (all 0-1) to RGB tuple (0-255)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _genome_to_color(genome: torch.Tensor, hue_idx: int) -> Tuple[int, int, int]:
    """
    Extract an RGB color from genome parameters at hue_idx, hue_idx+1, hue_idx+2.
    Genome values are in [0, 1]; hue is scaled to [0, 1] (full circle).
    Saturation is boosted to [0.55, 1.0] to avoid washed-out flags.
    Value (brightness) is mapped to [0.4, 1.0].
    """
    hue = genome[hue_idx].item()                          # 0-1 → 0-360°
    sat = 0.55 + genome[hue_idx + 1].item() * 0.45       # [0.55, 1.0]
    val = 0.4  + genome[hue_idx + 2].item() * 0.60       # [0.40, 1.0]
    return _hsv_to_rgb(hue, sat, val)


def _contrasting_outline(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return black or white depending on which contrasts better with rgb."""
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


# ── Pattern renderers ─────────────────────────────────────────────────────────
# Each receives an ImageDraw, the flag Image, primary color, secondary color,
# and pattern_detail (0.0-1.0 from genome[24]).

def _pattern_horizontal_stripes(draw: "ImageDraw.Draw", img: "Image.Image",
                                  c1, c2, detail):
    """Two or three horizontal stripes."""
    n_stripes = 3 if detail > 0.5 else 2
    stripe_h = FLAG_H // n_stripes
    for i in range(n_stripes):
        color = c1 if i % 2 == 0 else c2
        draw.rectangle([0, i * stripe_h, FLAG_W, (i + 1) * stripe_h - 1], fill=color)
    # Fill any remaining pixels at bottom
    draw.rectangle([0, n_stripes * stripe_h, FLAG_W, FLAG_H], fill=c1)


def _pattern_vertical_stripes(draw, img, c1, c2, detail):
    """Two or three vertical stripes."""
    n_stripes = 3 if detail > 0.5 else 2
    stripe_w = FLAG_W // n_stripes
    for i in range(n_stripes):
        color = c1 if i % 2 == 0 else c2
        draw.rectangle([i * stripe_w, 0, (i + 1) * stripe_w - 1, FLAG_H], fill=color)
    draw.rectangle([n_stripes * stripe_w, 0, FLAG_W, FLAG_H], fill=c1)


def _pattern_cross(draw, img, c1, c2, detail):
    """Scandinavian-style asymmetric cross."""
    img.paste(c1, [0, 0, FLAG_W, FLAG_H])  # background
    cross_w = max(4, FLAG_H // 5)
    cx = FLAG_W // 3   # cross vertical position (offset left like Nordic flags)
    draw.rectangle([cx - cross_w // 2, 0, cx + cross_w // 2, FLAG_H], fill=c2)
    mid_y = FLAG_H // 2
    draw.rectangle([0, mid_y - cross_w // 2, FLAG_W, mid_y + cross_w // 2], fill=c2)


def _pattern_diagonal(draw, img, c1, c2, detail):
    """Diagonal split (top-left triangle + bottom-right triangle)."""
    img.paste(c1, [0, 0, FLAG_W, FLAG_H])
    # Bottom-right triangle in c2
    draw.polygon([(FLAG_W, 0), (FLAG_W, FLAG_H), (0, FLAG_H)], fill=c2)


def _pattern_quartered(draw, img, c1, c2, detail):
    """Four quarters alternating two colors."""
    hw, hh = FLAG_W // 2, FLAG_H // 2
    draw.rectangle([0,  0,  hw, hh], fill=c1)
    draw.rectangle([hw, 0,  FLAG_W, hh], fill=c2)
    draw.rectangle([0,  hh, hw, FLAG_H], fill=c2)
    draw.rectangle([hw, hh, FLAG_W, FLAG_H], fill=c1)


def _pattern_canton(draw, img, c1, c2, detail):
    """Solid background with upper-left canton (box) in secondary color."""
    img.paste(c1, [0, 0, FLAG_W, FLAG_H])
    canton_w, canton_h = FLAG_W // 3, FLAG_H // 2
    draw.rectangle([0, 0, canton_w, canton_h], fill=c2)


def _pattern_bend(draw, img, c1, c2, detail):
    """Diagonal band (bend) across the flag."""
    img.paste(c1, [0, 0, FLAG_W, FLAG_H])
    band = max(6, FLAG_H // 3)
    # Draw the diagonal band as a filled polygon
    draw.polygon([
        (0, 0),
        (FLAG_W, 0),
        (FLAG_W, band),
        (0, FLAG_H),
        (0, FLAG_H - band),
    ], fill=c2)


def _pattern_chevron(draw, img, c1, c2, detail):
    """V-shape (chevron) on left side."""
    img.paste(c1, [0, 0, FLAG_W, FLAG_H])
    mid_y   = FLAG_H // 2
    tip_x   = FLAG_W // 3
    draw.polygon([
        (0, 0),
        (tip_x, mid_y),
        (0, FLAG_H),
    ], fill=c2)


_PATTERNS = [
    _pattern_horizontal_stripes,
    _pattern_vertical_stripes,
    _pattern_cross,
    _pattern_diagonal,
    _pattern_quartered,
    _pattern_canton,
    _pattern_bend,
    _pattern_chevron,
]


# ── Emblem renderers ──────────────────────────────────────────────────────────

def _draw_star(draw, cx, cy, r, fill, outline):
    """Draw a 5-pointed star centered at (cx,cy) with outer radius r."""
    points = []
    inner_r = r * 0.4
    for i in range(10):
        angle = math.pi / 2 + math.pi * i / 5
        radius = r if i % 2 == 0 else inner_r
        points.append((
            cx + radius * math.cos(angle),
            cy - radius * math.sin(angle),
        ))
    draw.polygon(points, fill=fill, outline=outline)


def _draw_circle(draw, cx, cy, r, fill, outline):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)


def _draw_triangle(draw, cx, cy, r, fill, outline):
    h = int(r * 0.87)
    points = [(cx, cy - h), (cx - r, cy + h // 2), (cx + r, cy + h // 2)]
    draw.polygon(points, fill=fill, outline=outline)


def _draw_diamond(draw, cx, cy, r, fill, outline):
    points = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    draw.polygon(points, fill=fill, outline=outline)


def _draw_cross_emblem(draw, cx, cy, r, fill, outline):
    """Small cross emblem (not the full flag cross pattern)."""
    arm = max(2, r // 3)
    draw.rectangle([cx - arm, cy - r, cx + arm, cy + r], fill=fill, outline=outline)
    draw.rectangle([cx - r, cy - arm, cx + r, cy + arm], fill=fill, outline=outline)


def _draw_sun(draw, cx, cy, r, fill, outline):
    """Simple sun with rays."""
    draw.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2],
                 fill=fill, outline=outline)
    for i in range(8):
        angle = math.pi * i / 4
        x1 = cx + int((r // 2 + 1) * math.cos(angle))
        y1 = cy + int((r // 2 + 1) * math.sin(angle))
        x2 = cx + int(r * math.cos(angle))
        y2 = cy + int(r * math.sin(angle))
        draw.line([x1, y1, x2, y2], fill=fill, width=1)


_EMBLEMS = [
    _draw_star,
    _draw_circle,
    _draw_triangle,
    _draw_diamond,
    _draw_cross_emblem,
    _draw_sun,
]


# ── Main generator ────────────────────────────────────────────────────────────

def generate_flag(genome: torch.Tensor) -> "Image.Image":
    """
    Render a 64×40 PIL flag image from a single civ's genome vector.

    Parameters
    ----------
    genome : (64,) float32 tensor, values in [0, 1]

    Returns
    -------
    PIL Image (RGB, 64×40)
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow required: pip install Pillow")

    c1 = _genome_to_color(genome, 16)          # primary color
    c2 = _genome_to_color(genome, 19)          # secondary color
    pattern_idx  = int(genome[22].item() * 7.99)   # 0-7
    emblem_idx   = int(genome[23].item() * 5.99)   # 0-5
    detail_level = genome[24].item()               # 0-1

    img  = Image.new("RGB", (FLAG_W, FLAG_H), color=c1)
    draw = ImageDraw.Draw(img)

    # Draw the background pattern
    _PATTERNS[pattern_idx](draw, img, c1, c2, detail_level)

    # Draw emblem in center (slightly left of center like many real flags)
    cx = FLAG_W // 3
    cy = FLAG_H // 2
    # Emblem radius scales with detail: 6-10 pixels
    emblem_r = 6 + int(detail_level * 4)
    # Emblem color: pick whichever of c1/c2 contrasts better with background at center
    # Sample the pixel at the center after background rendering
    center_pixel = img.getpixel((cx, cy))
    emblem_fill  = c2 if center_pixel == c1 else c1
    emblem_outline = _contrasting_outline(emblem_fill)

    _EMBLEMS[emblem_idx](draw, cx, cy, emblem_r, emblem_fill, emblem_outline)

    return img


def generate_flags(
    genomes: torch.Tensor,
    nation_slugs: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Generate flag PNGs for all civs and save to output_dir/flags/.

    Parameters
    ----------
    genomes      : (n_civs, 64) float32 tensor
    nation_slugs : list of slug strings (lowercase, no spaces) e.g. ["romai", "grecia"]
    output_dir   : base output directory

    Returns
    -------
    dict mapping slug → path to the small (64×40) PNG
    """
    flags_dir = output_dir / "flags"
    flags_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for i, slug in enumerate(nation_slugs):
        genome = genomes[i]
        img    = generate_flag(genome)

        small_path = flags_dir / f"{slug}.png"
        img.save(small_path)

        large_img  = img.resize((FLAG_W_LARGE, FLAG_H_LARGE), Image.NEAREST)
        large_path = flags_dir / f"{slug}-large.png"
        large_img.save(large_path)

        paths[slug] = small_path

    return paths


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, "..")
    from kernels.civ_genome import generate_genomes_hardware
    import ttnn
    from pathlib import Path

    N = 6
    SEED = 42

    print("=" * 60)
    print("Flag Generator — Hardware Test")
    print(f"Generating {N} civilization flags (seed={SEED})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        genomes, ms = generate_genomes_hardware(N, SEED, device)
    finally:
        ttnn.close_device(device)

    print(f"TT genome time: {ms:.3f}ms")

    slugs = [f"testciv{i+1}" for i in range(N)]
    out   = Path("output/civtest")
    paths = generate_flags(genomes, slugs, out)

    for slug, path in paths.items():
        print(f"  {slug}: {path}")

    print("\n✓ Flag generation complete!")
