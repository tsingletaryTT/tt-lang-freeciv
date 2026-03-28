# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
sprite_recolor.py — Unit sprite recoloring from TT-generated civ colors
=========================================================================

FreeCiv uses a "player color" system where unit sprites are rendered with
palette substitution: a canonical player-color pixel (pure blue #0000FF)
is replaced by the civilization's actual color.

This module:
  1. Reads stock FreeCiv unit sprites from the tileset directory
  2. For each civilization, replaces the player-color pixels with the
     civ's primary color (from the TT genome)
  3. Writes the recolored sprites to the output ruleset directory

Genome parameters used:
  [16] color_hue   — primary hue (0-1)
  [17] color_sat   — primary saturation
  [18] color_val   — primary brightness

FreeCiv player colors (the pixels we replace):
  Primary:   (0, 0, 255)    = pure blue
  Secondary: (0, 0, 128)    = dark blue (shadow areas)
  Highlight: (128, 128, 255) = light blue (highlight areas)

If the FreeCiv tileset directory is not found, this module generates simple
32×32 colored blocks as placeholder unit icons instead.
"""

import colorsys
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import torch

# ── FreeCiv player-color pixels (the ones we substitute) ─────────────────────

# These are the canonical player-color markers used in FreeCiv trident tileset
PLAYER_COLOR_PRIMARY   = (0,   0, 255)   # main unit color
PLAYER_COLOR_SECONDARY = (0,   0, 128)   # shadow/darker
PLAYER_COLOR_HIGHLIGHT = (128, 128, 255) # highlight/lighter

# Tolerance for approximate color matching (some tilesets use near-values)
COLOR_TOLERANCE = 20

# FreeCiv unit sprite names we recolor (subset most visible in gameplay)
UNIT_SPRITES = [
    "u.warriors",
    "u.archers",
    "u.phalanx",
    "u.legion",
    "u.musketeers",
    "u.riflemen",
    "u.settlers",
    "u.workers",
    "u.diplomat",
    "u.caravan",
    "u.trireme",
    "u.galleon",
    "u.frigate",
    "u.knights",
    "u.cavalry",
    "u.armor",
    "u.artillery",
    "u.catapult",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    import colorsys as cs
    r, g, b = cs.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _genome_to_primary_color(genome: torch.Tensor) -> Tuple[int, int, int]:
    """Extract the primary civ color from genome parameters [16, 17, 18]."""
    hue = genome[16].item()
    sat = 0.55 + genome[17].item() * 0.45
    val = 0.5  + genome[18].item() * 0.50
    return _hsv_to_rgb(hue, sat, val)


def _darken(rgb: Tuple[int,int,int], factor: float = 0.5) -> Tuple[int,int,int]:
    """Return a darkened version of an RGB color."""
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))


def _lighten(rgb: Tuple[int,int,int], factor: float = 1.5) -> Tuple[int,int,int]:
    """Return a lightened version of an RGB color."""
    return (
        min(255, int(rgb[0] * factor)),
        min(255, int(rgb[1] * factor)),
        min(255, int(rgb[2] * factor)),
    )


def _colors_close(c1: Tuple[int,int,int], c2: Tuple[int,int,int], tol: int) -> bool:
    """Return True if two colors are within tolerance on all channels."""
    return all(abs(c1[i] - c2[i]) <= tol for i in range(3))


def _recolor_image(
    img: "Image.Image",
    primary: Tuple[int,int,int],
    secondary: Optional[Tuple[int,int,int]] = None,
    highlight: Optional[Tuple[int,int,int]] = None,
) -> "Image.Image":
    """
    Replace player-color pixels in img with the given civ colors.

    Works on both RGBA and RGB images. Preserves alpha channel.
    """
    if secondary is None:
        secondary = _darken(primary, 0.5)
    if highlight is None:
        highlight = _lighten(primary, 1.5)

    mode   = img.mode
    has_alpha = mode == "RGBA"
    rgb_img = img.convert("RGBA")
    pixels  = rgb_img.load()
    w, h    = rgb_img.size

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            pixel_rgb   = (r, g, b)

            if _colors_close(pixel_rgb, PLAYER_COLOR_PRIMARY, COLOR_TOLERANCE):
                pixels[x, y] = (*primary, a)
            elif _colors_close(pixel_rgb, PLAYER_COLOR_SECONDARY, COLOR_TOLERANCE):
                pixels[x, y] = (*secondary, a)
            elif _colors_close(pixel_rgb, PLAYER_COLOR_HIGHLIGHT, COLOR_TOLERANCE):
                pixels[x, y] = (*highlight, a)

    return rgb_img.convert(mode) if not has_alpha else rgb_img


def _make_placeholder_sprite(
    color: Tuple[int,int,int],
    label: str,
    size: int = 32,
) -> "Image.Image":
    """
    Generate a simple placeholder unit sprite when FreeCiv tileset is unavailable.

    Creates a colored square with a contrasting border, representing a unit
    token in the civ's color.
    """
    from PIL import ImageDraw
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Colored background square
    inner = size // 8
    draw.rectangle([inner, inner, size - inner, size - inner], fill=(*color, 255))

    # Border (dark or light depending on color brightness)
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    border_color = (20, 20, 20, 255) if lum > 100 else (200, 200, 200, 255)
    draw.rectangle([inner, inner, size - inner, size - inner],
                   outline=border_color)

    # Small symbol in center (darker version of civ color)
    sym_color = _darken(color, 0.5)
    cx = size // 2
    cy = size // 2
    # Draw a small diamond
    draw.polygon([
        (cx, cy - inner), (cx + inner, cy),
        (cx, cy + inner), (cx - inner, cy),
    ], fill=(*sym_color, 255))

    return img


def recolor_units(
    genomes: torch.Tensor,
    nation_slugs: list[str],
    output_dir: Path,
    tileset_dir: Optional[Path] = None,
) -> dict[str, dict[str, Path]]:
    """
    Recolor unit sprites for all civilizations.

    For each civ, creates recolored versions of stock FreeCiv unit sprites
    in the civ's primary color (derived from TT hardware genome).

    Parameters
    ----------
    genomes      : (n_civs, 64) float32 tensor from TT hardware
    nation_slugs : list of slug strings (one per civ)
    output_dir   : base output directory; sprites go to output_dir/sprites/<slug>/
    tileset_dir  : path to FreeCiv tileset directory containing unit PNGs.
                   If None or not found, placeholder sprites are generated.

    Returns
    -------
    Nested dict: {slug: {sprite_name: path_to_png}}
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow required: pip install Pillow")

    sprites_base = output_dir / "sprites"
    sprites_base.mkdir(parents=True, exist_ok=True)

    # Determine if we have actual FreeCiv sprites to recolor
    # Common tileset locations
    tileset_candidates = [
        tileset_dir,
        Path("/usr/share/freeciv/trident"),
        Path("/home/ttuser/code/freeciv/data/trident"),
        Path("/usr/local/share/freeciv/trident"),
    ]
    found_tileset = None
    for candidate in tileset_candidates:
        if candidate and candidate.exists():
            found_tileset = candidate
            break

    all_paths: dict[str, dict[str, Path]] = {}

    for civ_idx, slug in enumerate(nation_slugs):
        genome  = genomes[civ_idx]
        primary = _genome_to_primary_color(genome)
        civ_dir = sprites_base / slug
        civ_dir.mkdir(exist_ok=True)

        civ_paths: dict[str, Path] = {}

        for sprite_name in UNIT_SPRITES:
            out_path = civ_dir / f"{sprite_name}.png"

            if found_tileset:
                # Try to find the sprite file in the tileset
                sprite_file = None
                candidates = [
                    found_tileset / f"{sprite_name}.png",
                    found_tileset / f"{sprite_name.replace('.', '/')}.png",
                    # Some tilesets use different name format
                    found_tileset / f"units/{sprite_name.split('.')[1]}.png",
                ]
                for sf in candidates:
                    if sf.exists():
                        sprite_file = sf
                        break

                if sprite_file:
                    source_img = Image.open(sprite_file)
                    recolored  = _recolor_image(source_img, primary)
                    recolored.save(out_path)
                    civ_paths[sprite_name] = out_path
                    continue

            # No tileset found or sprite not found — use placeholder
            placeholder = _make_placeholder_sprite(primary, sprite_name)
            placeholder.save(out_path)
            civ_paths[sprite_name] = out_path

        all_paths[slug] = civ_paths

    return all_paths


def write_color_ini(
    genomes: torch.Tensor,
    nation_slugs: list[str],
    output_dir: Path,
) -> Path:
    """
    Write a colors.tilespec snippet mapping each civ slug to its RGB color.

    This tells FreeCiv which color to use for this nation's units when
    rendering the actual in-game sprites (the server-side color used for
    the player color indicator in the status bar, city outlines, etc).

    Parameters
    ----------
    genomes      : (n_civs, 64) float32 tensor
    nation_slugs : list of slug strings
    output_dir   : output directory

    Returns
    -------
    Path to the written colors.tilespec file
    """
    lines = [
        "; Auto-generated by civgen/sprite_recolor.py",
        "; TT-hardware generated civilization colors",
        "; (c) 2025 Tenstorrent AI ULC",
        "",
        "[spec]",
        "options = \"+Freeciv-tilespec-Devel-2023-Feb-13\"",
        "files = \"data/misc/colors.spec\"",
        "",
        "[player_colors]",
        "colors =",
    ]

    color_entries = []
    for i, slug in enumerate(nation_slugs):
        genome = genomes[i]
        r, g, b = _genome_to_primary_color(genome)
        color_entries.append(f"  {{ r={r}, g={g}, b={b} }}  ; {slug}")

    lines.append(",\n".join(color_entries))

    out_path = output_dir / "colors.tilespec"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from kernels.civ_genome import generate_genomes_hardware
    import ttnn
    from pathlib import Path

    N = 6
    SEED = 42

    print("=" * 60)
    print("Sprite Recolor — Hardware Test")
    print(f"Generating sprites for {N} civilizations (seed={SEED})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        genomes, ms = generate_genomes_hardware(N, SEED, device)
    finally:
        ttnn.close_device(device)

    print(f"TT genome time: {ms:.3f}ms")

    slugs = [f"testciv{i+1}" for i in range(N)]
    out   = Path("output/civtest")

    all_paths = recolor_units(genomes, slugs, out)
    col_path  = write_color_ini(genomes, slugs, out)

    print(f"\nColor INI: {col_path}")
    for slug, sprite_paths in all_paths.items():
        r, g, b = _genome_to_primary_color(genomes[list(all_paths.keys()).index(slug)])
        print(f"  {slug}: rgb({r},{g},{b})  — {len(sprite_paths)} sprites")

    print("\n✓ Sprite recolor complete!")
