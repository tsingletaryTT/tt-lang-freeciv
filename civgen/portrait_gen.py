# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
portrait_gen.py — Pixel-art leader portrait generator
=======================================================

Generates leader portraits using parameters produced by P300C Blackhole hardware.
All numeric decisions (skin tone, hair, face shape, expression, garment, headgear)
come from the civilization genome computed on TT hardware.  CPU renders pixel art
using PIL.

Genome parameters used:
  [29] portrait_skin    — skin tone (0=very light, 1=very dark)
  [30] portrait_hair    — hair darkness (0=blonde/white, 1=black)
  [31] portrait_face    — face shape (0=narrow/angular, 1=round/broad)
  [32] portrait_expr    — expression type (0=stern, 0.33=warm, 0.66=fierce, 1=noble)
  [33] portrait_headgear — headgear (0=none, 0.2=helm, 0.4=crown, 0.6=hat, 0.8=turban)
  [34] portrait_garment — garment (0=armor, 0.25=robe, 0.5=tunic, 0.75=fur)
  [35] portrait_beard   — beard (0=none, 0.33=stubble, 0.66=full, 1=epic)

Output: 64×80 pixel PNG (portrait orientation, pixel-art style)
FreeCiv leader portraits are displayed in the diplomacy screen.
"""

import colorsys
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import torch

# ── Portrait dimensions ───────────────────────────────────────────────────────

PORT_W = 64
PORT_H = 80

# ── Skin tone palette (from lightest to darkest) ──────────────────────────────
# Each entry: (r, g, b) for skin, shadow, highlight
_SKIN_TONES = [
    ((255, 224, 196), (210, 175, 140), (255, 240, 220)),  # 0: very fair
    ((240, 200, 168), (192, 154, 116), (252, 220, 190)),  # 1: fair
    ((210, 168, 130), (165, 120,  80), (230, 190, 152)),  # 2: medium-fair
    ((180, 134,  90), (136,  90,  50), (200, 158, 112)),  # 3: medium
    ((150, 105,  65), (105,  65,  30), (172, 128,  88)),  # 4: medium-dark
    ((120,  80,  45), ( 80,  45,  15), (142,  98,  60)),  # 5: dark
    ( (90,  55,  25), ( 60,  28,   8), (110,  72,  38)),  # 6: very dark
    ( (70,  40,  15), ( 45,  20,   4), ( 88,  54,  25)),  # 7: deepest
]

# ── Hair color from darkness parameter ───────────────────────────────────────
_HAIR_COLORS = [
    (240, 220, 160),  # 0.0: blonde
    (200, 170, 100),  # 0.14: dirty blonde
    (160, 110,  50),  # 0.28: auburn/light brown
    (100,  60,  20),  # 0.43: brown
    ( 60,  35,  10),  # 0.57: dark brown
    ( 35,  20,   5),  # 0.71: very dark brown
    ( 20,  10,   2),  # 0.86: near black
    ( 10,   5,   0),  # 1.0: black
]

# ── Expression adjustments (mouth shape via y-offset, eye shape) ──────────────
# (mouth_curve, eye_open) — used to offset pixel placement
_EXPRESSIONS = [
    (-1,  2),   # 0: stern — mouth slightly down, squinting
    ( 2,  4),   # 1: warm  — smile, wide eyes
    (-2,  1),   # 2: fierce — frown, narrow eyes
    ( 0,  3),   # 3: noble — neutral mouth, medium eyes
]

# ── Background colors by garment type ────────────────────────────────────────
_GARMENT_COLORS = [
    ( 80,  80, 110),  # 0: armor — steel blue-gray
    ( 90,  50, 130),  # 1: robe  — deep purple
    (130,  90,  50),  # 2: tunic — warm brown
    (100,  60,  30),  # 3: fur   — dark fur brown
]

# ── Headgear colors ────────────────────────────────────────────────────────────
_HEADGEAR_COLORS = [
    None,                   # 0: none
    (150, 155, 160),        # 1: helm — silver
    (220, 190,  50),        # 2: crown — gold
    ( 50,  30,  20),        # 3: hat — dark felt
    (200, 180, 140),        # 4: turban — cream/tan
]


def _pick_from_list(lst, value_0_1):
    """Pick an item from lst using a 0-1 value (uniformly sampled)."""
    idx = min(len(lst) - 1, int(value_0_1 * len(lst)))
    return lst[idx]


def _blend(c1: Tuple[int,int,int], c2: Tuple[int,int,int], t: float) -> Tuple[int,int,int]:
    """Linear interpolation between two RGB colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def generate_portrait(genome: torch.Tensor) -> "Image.Image":
    """
    Render a 64×80 pixel-art leader portrait from a single civ's genome.

    Parameters
    ----------
    genome : (64,) float32 tensor, values in [0, 1]

    Returns
    -------
    PIL Image (RGB, 64×80)
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow required: pip install Pillow")

    # Extract genome parameters
    skin_t     = genome[29].item()   # 0-1
    hair_t     = genome[30].item()   # 0-1
    face_t     = genome[31].item()   # 0=narrow, 1=round
    expr_t     = genome[32].item()   # 0-1
    headgear_t = genome[33].item()   # 0-1
    garment_t  = genome[34].item()   # 0-1
    beard_t    = genome[35].item()   # 0=none, 1=epic

    # Map parameters to palette entries
    skin_idx    = min(7, int(skin_t * 8))
    hair_idx    = min(7, int(hair_t * 8))
    expr_idx    = min(3, int(expr_t * 4))
    garment_idx = min(3, int(garment_t * 4))
    headgear_idx = min(4, int(headgear_t * 5))

    skin_color, skin_shadow, skin_highlight = _SKIN_TONES[skin_idx]
    hair_color = _HAIR_COLORS[hair_idx]
    expr       = _EXPRESSIONS[expr_idx]
    garment_color = _GARMENT_COLORS[garment_idx]
    headgear_color = _HEADGEAR_COLORS[headgear_idx]

    # Background gradient (top lighter → bottom darker matching garment)
    img  = Image.new("RGB", (PORT_W, PORT_H), color=(20, 20, 32))
    draw = ImageDraw.Draw(img)

    # ── Background (gradient-ish via banded rectangles) ──────────────────────
    bg_top    = _blend(garment_color, (30, 30, 50), 0.3)
    bg_bottom = _blend(garment_color, (10, 10, 20), 0.6)
    for y in range(PORT_H):
        t   = y / PORT_H
        col = _blend(bg_top, bg_bottom, t)
        draw.line([(0, y), (PORT_W, y)], fill=col)

    # ── Face geometry ─────────────────────────────────────────────────────────
    # Face center: upper-center of the portrait
    face_cx = PORT_W // 2
    face_cy = PORT_H // 3

    # Face width/height based on face_t (0=narrow, 1=round/broad)
    face_w = int(18 + face_t * 8)   # 18-26 px
    face_h = int(22 + face_t * 6)   # 22-28 px

    # Face base (ellipse)
    fl = face_cx - face_w // 2
    ft = face_cy - face_h // 2
    fr = face_cx + face_w // 2
    fb = face_cy + face_h // 2
    draw.ellipse([fl, ft, fr, fb], fill=skin_color, outline=skin_shadow)

    # ── Hair ─────────────────────────────────────────────────────────────────
    # Hair cap: arc on top of face + sides
    hair_top = ft - 4
    hair_l   = fl - 2
    hair_r   = fr + 2
    draw.ellipse([hair_l, hair_top, hair_r, face_cy], fill=hair_color)

    # Side hair flowing down
    draw.rectangle([hair_l, ft, fl + 2, face_cy + 6], fill=hair_color)
    draw.rectangle([fr - 2, ft, hair_r, face_cy + 6], fill=hair_color)

    # ── Eyes ─────────────────────────────────────────────────────────────────
    eye_y     = face_cy - 2
    eye_open  = expr[1]   # 1-4 pixels tall
    left_ex   = face_cx - face_w // 4
    right_ex  = face_cx + face_w // 4
    eye_white = (240, 240, 240)
    eye_iris  = _blend(hair_color, (50, 80, 180), 0.4)  # eye color near hair
    pupil     = (15, 10, 5)

    for ex in [left_ex, right_ex]:
        # White of eye
        draw.rectangle([ex - 3, eye_y, ex + 3, eye_y + eye_open], fill=eye_white)
        # Iris
        draw.rectangle([ex - 1, eye_y, ex + 1, eye_y + eye_open - 1], fill=eye_iris)
        # Pupil (1px)
        draw.point((ex, eye_y + eye_open // 2), fill=pupil)
        # Eyelid (top)
        draw.line([(ex - 3, eye_y), (ex + 3, eye_y)], fill=skin_shadow)

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    brow_y = eye_y - 3
    # Stern/fierce: flat or angled down toward center; warm/noble: slight arch
    for ex in [left_ex, right_ex]:
        angle = -1 if expr_idx in (0, 2) else 1  # -1=inner up, 1=inner down
        offset = angle if ex == left_ex else -angle
        draw.line([(ex - 4, brow_y + offset), (ex + 4, brow_y - offset)],
                  fill=hair_color, width=1)

    # ── Nose ─────────────────────────────────────────────────────────────────
    nose_y = face_cy + 2
    draw.line([(face_cx, eye_y + eye_open + 2), (face_cx, nose_y)],
              fill=skin_shadow, width=1)
    # Nostril dots
    draw.point((face_cx - 2, nose_y), fill=skin_shadow)
    draw.point((face_cx + 2, nose_y), fill=skin_shadow)

    # ── Mouth ─────────────────────────────────────────────────────────────────
    mouth_y     = face_cy + 8
    mouth_curve = expr[0]  # -2 to +2
    lip_color   = _blend(skin_color, (180, 60, 60), 0.3)
    # Upper lip arc
    mouth_l = face_cx - 5
    mouth_r = face_cx + 5
    if mouth_curve > 0:
        # Smile: corners up
        draw.arc([mouth_l, mouth_y - mouth_curve, mouth_r, mouth_y + mouth_curve * 2],
                 start=0, end=180, fill=lip_color)
    else:
        # Frown or neutral
        frown = abs(mouth_curve)
        draw.arc([mouth_l, mouth_y, mouth_r, mouth_y + max(2, frown * 2)],
                 start=180, end=360, fill=lip_color)

    # ── Beard ─────────────────────────────────────────────────────────────────
    if beard_t > 0.1:
        beard_start = face_cy + 10
        beard_end   = fb + int(beard_t * 12)  # longer for more epic beard
        beard_width = int(face_w * 0.7 * beard_t)
        beard_color = _blend(hair_color, (20, 12, 5), 0.2)
        draw.ellipse([
            face_cx - beard_width // 2, beard_start,
            face_cx + beard_width // 2, beard_end,
        ], fill=beard_color)
        # Stubble dots for low beard values
        if beard_t < 0.35:
            for dx in range(-face_w // 3, face_w // 3, 3):
                for dy in range(0, 6, 2):
                    draw.point((face_cx + dx, face_cy + 10 + dy), fill=beard_color)

    # ── Neck ─────────────────────────────────────────────────────────────────
    neck_top = fb
    neck_bot = fb + 8
    neck_w   = face_w // 3
    draw.rectangle([face_cx - neck_w, neck_top, face_cx + neck_w, neck_bot],
                   fill=skin_color, outline=skin_shadow)

    # ── Garment (shoulders and chest) ────────────────────────────────────────
    chest_top = neck_bot - 2
    chest_bot = PORT_H
    draw.rectangle([0, chest_top, PORT_W, chest_bot], fill=garment_color)

    # Shoulder pauldrons / collar detail
    collar_color = _blend(garment_color, (220, 200, 160), 0.3)
    draw.rectangle([face_cx - neck_w - 4, chest_top,
                    face_cx + neck_w + 4, chest_top + 4], fill=collar_color)

    # Garment detail lines (simple vertical seam)
    seam_color = _blend(garment_color, (0, 0, 0), 0.3)
    draw.line([(face_cx, chest_top + 4), (face_cx, chest_bot)], fill=seam_color)

    # ── Headgear ─────────────────────────────────────────────────────────────
    if headgear_color is not None:
        hg = headgear_idx
        hg_top = hair_top - 6

        if hg == 1:   # Helm
            draw.ellipse([fl - 3, hg_top, fr + 3, face_cy - 4],
                         fill=headgear_color, outline=(100, 110, 120))
            # Nose guard
            draw.rectangle([face_cx - 1, face_cy - 8, face_cx + 1, face_cy],
                           fill=headgear_color)

        elif hg == 2:  # Crown
            # Band
            draw.rectangle([fl, hg_top + 6, fr, hg_top + 10], fill=headgear_color)
            # Points (3 spikes)
            for sx in [fl + 4, face_cx, fr - 4]:
                draw.polygon([(sx - 3, hg_top + 6), (sx + 3, hg_top + 6), (sx, hg_top)],
                              fill=headgear_color)
            # Gems: tiny colored dots on crown band
            gem_color = (200, 50, 50)
            for sx in [fl + 8, face_cx - 4, face_cx + 4, fr - 8]:
                draw.point((sx, hg_top + 8), fill=gem_color)

        elif hg == 3:  # Hat (wide brim + tall top)
            brim_y = hg_top + 8
            draw.rectangle([fl - 6, brim_y, fr + 6, brim_y + 2], fill=headgear_color)
            draw.rectangle([fl + 2, hg_top, fr - 2, brim_y], fill=headgear_color)

        elif hg == 4:  # Turban (wrapped cloth)
            # Multi-layer wrap
            for layer, y_off in enumerate(range(0, 12, 3)):
                tc = _blend(headgear_color, (160, 130, 90), layer * 0.15)
                draw.ellipse([fl - layer, hg_top + y_off,
                              fr + layer, face_cy - 4 + y_off // 2], outline=tc)
            draw.ellipse([fl, hg_top + 2, fr, face_cy - 2],
                         fill=headgear_color)

    # ── Face highlight (subtle) ────────────────────────────────────────────────
    # Small bright patch on forehead
    hl_x, hl_y = face_cx - face_w // 6, ft + 4
    draw.ellipse([hl_x - 3, hl_y - 2, hl_x + 3, hl_y + 2], fill=skin_highlight)

    return img


def generate_portraits(
    genomes: torch.Tensor,
    nation_slugs: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Generate leader portrait PNGs for all civs and save to output_dir/portraits/.

    Parameters
    ----------
    genomes      : (n_civs, 64) float32 tensor
    nation_slugs : list of slug strings
    output_dir   : base output directory

    Returns
    -------
    dict mapping slug → path to the portrait PNG
    """
    portraits_dir = output_dir / "portraits"
    portraits_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for i, slug in enumerate(nation_slugs):
        genome = genomes[i]
        img    = generate_portrait(genome)

        path = portraits_dir / f"{slug}_leader.png"
        img.save(path)
        paths[slug] = path

    return paths


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
    print("Portrait Generator — Hardware Test")
    print(f"Generating {N} leader portraits (seed={SEED})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        genomes, ms = generate_genomes_hardware(N, SEED, device)
    finally:
        ttnn.close_device(device)

    print(f"TT genome time: {ms:.3f}ms")

    slugs = [f"testciv{i+1}" for i in range(N)]
    out   = Path("output/civtest")
    paths = generate_portraits(genomes, slugs, out)

    for slug, path in paths.items():
        print(f"  {slug}: {path}")

    print("\n✓ Portrait generation complete!")
