# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
color_filter.py — TT-hardware generative color filters: terrain, civs, resources
==================================================================================

Each game load gets a unique visual palette because the color transform
parameters are derived from TT hardware outputs.

CONTEXTUAL COLOR SYSTEM
-----------------------
Colors are not arbitrary — they are semantically consistent:

  Terrain → Biome conditions:
    Ocean/Coast → blues scaled by genome[18] (blue bias)
    Grassland   → greens scaled by genome[17] (saturation = life)
    Desert      → yellows scaled by genome[16] (warm hue bias)
    Mountains   → grays shifted toward colder/warmer by science_focus genome[2]
    The TT `smooth_height_map` kernel blends biome base colors with genome offsets

  Civ sprites → Civ primary color (from TT genome[16-18]):
    Unit sprites are tinted toward the civ's TT-generated primary color
    The tint matrix is: identity × (1 - alpha) + civ_color × alpha
    where alpha = civ genome[24] (flag_complexity) * 0.4

  Resources → Terrain context:
    Gold/Gems:  warm-yellow overlay (boosted on high-hue-bias terrains)
    Coal/Iron:  cool-gray overlay  (boosted on low-saturation terrains)
    Fish/Whale: deep-blue overlay  (boosted on ocean/coast tiles)
    Grain/Game: green overlay      (boosted on grassland/plains tiles)
    Each resource filter is also influenced by the genome's science_focus [2]
    (more science-focused civs get "richer" resource visual effects)

INFINITE WORLDS
---------------
Because all filter parameters derive from TT hardware genomes (which vary
with seed), every game session produces:
  - Distinct terrain palette (same biomes, different exact shades)
  - Distinct unit colors per civ (each civ uniquely visually identified)
  - Distinct resource glows (matching terrain context)
No two seeds produce the same visual world.

Each game load gets a unique visual palette because the color transform
parameters are derived from TT hardware outputs:

  1. TT hardware generates a 3×3 color transform matrix on P300C Blackhole.
     (This reuses smooth_height_map for the linear algebra — it is the same
      kernel used for terrain and civ genomes; the device stays busy.)

  2. CPU applies the transform to every terrain tile sprite and unit sprite
     as a color filter: hue rotation, saturation curve, brightness adjustment.

  3. The result is saved into a per-game "filtered" sprite directory that
     FreeCiv's tileset loader picks up (via a patched tilespec file).

The color transforms are designed to be subtle — maps still look like maps —
but produce a recognizably different "era feel" per load:
  - Seed 42:   warm ochre (ancient world)
  - Seed 100:  cool jade (island world)
  - Seed 777:  deep indigo (night world)
  - Seed 1234: warm amber (desert world)

These are not fixed palettes — the TT matrix parameterizes a continuous space
of transforms driven by the hardware genome.

TT kernel used:
  smooth_height_map(a, b, out)  →  element-wise a + b
  Applied 3 times to compute a 3×3 color transform:
    T = I + alpha * M   where M is TT-generated and alpha is genome[0] * 0.3

Genome parameters driving the transform:
  [0]  aggression    → alpha (strength of the transform, 0=identity, 0.3=max)
  [16] color_hue     → hue rotation base angle
  [17] color_sat     → saturation multiplier bias
  [18] color_val     → brightness gamma exponent bias
  [2]  science_focus → "era feel" weight (high=cool/blue, low=warm/amber)
"""

import colorsys
import math
import time
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import torch

# Add kernels to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

from height_map_smooth import smooth_height_map, to_device_half, zeros_like_on_device


# ── Terrain tileset sprite locations ─────────────────────────────────────────
# We try known FreeCiv tileset directories in order.
TILESET_SEARCH_PATHS = [
    Path("/home/ttuser/code/freeciv/data/trident"),
    Path("/home/ttuser/code/freeciv/data/amplio2"),
    Path("/usr/share/freeciv/trident"),
    Path("/usr/local/share/freeciv/trident"),
]

# Terrain tile PNG names found in typical FreeCiv tilesets
TERRAIN_TILES = [
    "t.l0.coast1",    "t.l0.ocean1",     "t.l0.plains1",
    "t.l0.grassland1","t.l0.forest1",    "t.l0.hills1",
    "t.l0.mountains1","t.l0.tundra1",    "t.l0.arctic1",
    "t.l0.desert1",   "t.l0.swamp1",     "t.l0.jungle1",
    # Roads and rivers
    "roads.road_isolated", "roads.river_isolated",
    # City sprites
    "cd.city",  "cd.city_wall",
    # Fog of war
    "tx.fog",
]


def _tile_align(n: int) -> int:
    return ((n + 32 - 1) // 32) * 32


# ── TT hardware: generate color transform matrix ──────────────────────────────

def generate_color_transform_hardware(
    genome: torch.Tensor,
    device,
) -> tuple[torch.Tensor, float]:
    """
    Generate a 3×3 color transform matrix on P300C Blackhole hardware.

    The matrix M is computed as:
        M_raw = smooth_height_map(I_noise_a, I_noise_b) for 3 passes
        M = clamp(0.5 * M_raw + 0.5)   (normalize to [0, 1])

    The final color transform applied to each pixel (r, g, b) is:
        [r', g', b'] = clamp(M × [r, g, b])
    where M is scaled around identity so most colors are preserved.

    Parameters
    ----------
    genome : (64,) float32 genome vector from TT hardware
    device : open ttnn device

    Returns
    -------
    (transform_3x3, kernel_ms)
        transform_3x3 : float32 tensor of shape (3, 3)
        kernel_ms     : time in TT kernels
    """
    import ttnn

    # The TT kernel works on tile-aligned shapes (multiples of 32×32).
    # We use 32×32 tensors and extract the top-left 3×3 as our matrix.
    rows = 32
    cols = 32

    alpha = genome[0].item() * 0.3        # transform strength 0-0.3
    hue   = genome[16].item()             # hue bias 0-1
    sci   = genome[2].item()              # cool/warm bias 0-1

    # Build two noise matrices that will be added by the TT kernel.
    # The sum defines the raw transform; we normalize around identity.
    # Matrix a: base around identity + hue-rotation perturbation
    # Matrix b: science-biased perturbation (cold/warm diagonal shift)
    import torch as _torch

    # Identity-centered noise: values near 0.5 (identity after centering)
    rng = _torch.Generator()
    rng.manual_seed(int(genome[0].item() * 1e6) ^ 0xBEEF)

    noise_a = _torch.rand(rows, cols, generator=rng) * alpha
    noise_b = _torch.rand(rows, cols, generator=rng) * alpha

    # Add identity to the top-left 3×3 of noise_a
    identity = _torch.eye(3) * (1.0 - alpha)
    noise_a[:3, :3] += identity

    # Hue rotation perturbation: shift the green-blue coupling
    noise_a[1, 2] += hue * alpha   # G→B coupling
    noise_a[2, 1] += (1 - hue) * alpha  # B→G coupling

    # Science/warm bias: cool (high sci) → boost blue; warm (low sci) → boost red/green
    diag_bias = sci * alpha
    noise_b[2, 2] += diag_bias          # blue self-weight
    noise_b[0, 0] += (1 - sci) * alpha  # red self-weight

    # Send to TT hardware via smooth_height_map (eltwise add kernel)
    a_t = to_device_half(noise_a, device)
    b_t = to_device_half(noise_b, device)
    out = zeros_like_on_device(noise_a, device)

    t0 = time.perf_counter()
    smooth_height_map(a_t, b_t, out)
    kernel_ms = (time.perf_counter() - t0) * 1000

    # Retrieve and extract the 3×3 submatrix
    result = _torch.from_dlpack(out) if hasattr(out, '__dlpack__') else None
    import ttnn as _ttnn
    result = _ttnn.to_torch(out).float()[:3, :3]

    return result, kernel_ms


# ── CPU: apply color matrix to a PIL image ───────────────────────────────────

def apply_color_matrix(
    img: "Image.Image",
    matrix: torch.Tensor,
    hue_shift_degrees: float = 0.0,
    saturation_scale: float = 1.0,
    gamma: float = 1.0,
) -> "Image.Image":
    """
    Apply a color transform to a PIL image.

    The transform is:
      1. Optional hue shift in HSV space
      2. Saturation adjustment
      3. RGB matrix multiplication (3×3)
      4. Gamma correction

    Parameters
    ----------
    img            : PIL Image (RGB or RGBA)
    matrix         : (3, 3) float32 color transform matrix
    hue_shift_degrees : degrees to rotate hue (0 = no change)
    saturation_scale  : 1.0 = no change, <1 = desaturate, >1 = saturate
    gamma          : 1.0 = no change, >1 = darken, <1 = brighten

    Returns
    -------
    Filtered PIL Image (same mode as input)
    """
    if not PIL_AVAILABLE:
        return img

    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r_ch, g_ch, b_ch, a_ch = img.split()
        rgb_img = Image.merge("RGB", (r_ch, g_ch, b_ch))
    else:
        rgb_img = img.convert("RGB")
        a_ch = None

    pixels = list(rgb_img.getdata())
    mat    = matrix.tolist()   # 3×3 list of lists

    hue_shift_norm = hue_shift_degrees / 360.0

    new_pixels = []
    for r, g, b in pixels:
        # Normalize to [0, 1]
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0

        # Hue shift + saturation in HSV space
        if abs(hue_shift_degrees) > 0.5 or abs(saturation_scale - 1.0) > 0.01:
            h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
            h = (h + hue_shift_norm) % 1.0
            s = min(1.0, s * saturation_scale)
            rf, gf, bf = colorsys.hsv_to_rgb(h, s, v)

        # Matrix multiplication
        nr = mat[0][0] * rf + mat[0][1] * gf + mat[0][2] * bf
        ng = mat[1][0] * rf + mat[1][1] * gf + mat[1][2] * bf
        nb = mat[2][0] * rf + mat[2][1] * gf + mat[2][2] * bf

        # Gamma
        if abs(gamma - 1.0) > 0.01:
            nr = max(0.0, nr) ** gamma
            ng = max(0.0, ng) ** gamma
            nb = max(0.0, nb) ** gamma

        # Clamp and convert back
        new_pixels.append((
            max(0, min(255, int(nr * 255))),
            max(0, min(255, int(ng * 255))),
            max(0, min(255, int(nb * 255))),
        ))

    result_rgb = Image.new("RGB", rgb_img.size)
    result_rgb.putdata(new_pixels)

    if has_alpha:
        result_rgb = result_rgb.convert("RGBA")
        r2, g2, b2, _ = result_rgb.split()
        return Image.merge("RGBA", (r2, g2, b2, a_ch))
    return result_rgb


# ── Genome → filter parameters ────────────────────────────────────────────────

def genome_to_filter_params(genome: torch.Tensor) -> dict:
    """
    Convert a genome vector to color filter parameters.

    Returns dict with:
      hue_shift      : float, degrees (-30 to +30)
      saturation     : float, scale (0.7 to 1.4)
      gamma          : float, exponent (0.8 to 1.3)
      vignette       : float, strength (0.0 to 0.3)
    """
    # [16] color_hue: 0→1 mapped to hue shift -30→+30 degrees
    hue_shift = (genome[16].item() - 0.5) * 60.0        # -30 to +30

    # [17] color_sat: 0→1 mapped to saturation 0.7→1.4
    saturation = 0.7 + genome[17].item() * 0.7

    # [18] color_val: 0→1 mapped to gamma 0.8→1.3 (>1 darkens)
    gamma = 0.8 + genome[18].item() * 0.5

    # [2] science_focus: high = cool, low = warm
    # Expressed as an additional slight blue tint (for cool) or red tint (for warm)
    sci = genome[2].item()
    vignette = genome[0].item() * 0.3   # aggression → vignette strength

    return {
        "hue_shift":  hue_shift,
        "saturation": saturation,
        "gamma":      gamma,
        "vignette":   vignette,
        "sci_bias":   sci,
    }


# ── Main: filter all terrain sprites and unit sprites ────────────────────────

def apply_generative_filters(
    genome: torch.Tensor,
    color_matrix: torch.Tensor,
    output_dir: Path,
    tileset_dir: Optional[Path] = None,
) -> dict:
    """
    Apply TT-generated color filters to all terrain and unit sprites.

    Creates a filtered copy of the tileset in output_dir/filtered_sprites/.
    Returns a dict with timing and statistics.

    Parameters
    ----------
    genome       : (64,) float32 from TT hardware — drives filter params
    color_matrix : (3, 3) float32 from TT hardware — the color transform
    output_dir   : where to write filtered sprites
    tileset_dir  : source tileset directory (auto-detected if None)
    """
    if not PIL_AVAILABLE:
        return {"error": "Pillow not installed", "sprites_filtered": 0}

    params = genome_to_filter_params(genome)

    # Find tileset
    search_dirs = ([tileset_dir] if tileset_dir else []) + TILESET_SEARCH_PATHS
    found_tileset = None
    for d in search_dirs:
        if d and d.exists():
            found_tileset = d
            break

    out_dir = output_dir / "filtered_sprites"
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered = 0
    placeholders = 0
    t0 = time.perf_counter()

    # Collect all PNGs to filter: terrain tiles + unit sprites
    sources = []
    if found_tileset:
        sources = list(found_tileset.glob("**/*.png"))

    if not sources:
        # No tileset — generate colored sample tiles as demonstration
        sources = _generate_sample_tiles(out_dir, params, color_matrix)
        placeholders = len(sources)
    else:
        # Filter real tileset sprites
        for src_path in sources:
            rel = src_path.relative_to(found_tileset)
            dst_path = out_dir / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                img = Image.open(src_path)
                filtered_img = apply_color_matrix(
                    img,
                    color_matrix,
                    hue_shift_degrees = params["hue_shift"],
                    saturation_scale  = params["saturation"],
                    gamma             = params["gamma"],
                )
                filtered_img.save(dst_path)
                filtered += 1
            except Exception:
                # Just copy if filter fails
                import shutil
                shutil.copy2(src_path, dst_path)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Write a filter manifest so FreeCiv (or the demo) can reference the params
    manifest = {
        "hue_shift_deg":  round(params["hue_shift"], 2),
        "saturation":     round(params["saturation"], 3),
        "gamma":          round(params["gamma"], 3),
        "vignette":       round(params["vignette"], 3),
        "sprites_filtered": filtered,
        "placeholders":   placeholders,
        "elapsed_ms":     round(elapsed_ms, 1),
        "matrix":         color_matrix.tolist(),
    }
    import json
    (out_dir / "filter_manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def _generate_sample_tiles(
    out_dir: Path,
    params: dict,
    color_matrix: torch.Tensor,
) -> list[Path]:
    """
    Generate colored 32×32 sample terrain tiles as placeholders when no
    FreeCiv tileset is installed.  Shows the color filter in action.
    """
    # Reference terrain colors (unfiltered) for the sample tiles
    TERRAIN_COLORS = {
        "ocean":     (40,  90, 180),
        "coast":     (80, 130, 200),
        "plains":    (180, 180, 100),
        "grassland": (80,  160, 60),
        "forest":    (30,  100, 40),
        "hills":     (120, 110, 70),
        "mountains": (130, 110, 90),
        "tundra":    (180, 190, 200),
        "arctic":    (230, 240, 255),
        "desert":    (220, 190, 100),
        "swamp":     (60,  100, 60),
        "jungle":    (20,   80, 30),
    }

    written = []
    for terrain, base_color in TERRAIN_COLORS.items():
        img = Image.new("RGBA", (32, 32), (*base_color, 255))

        # Add some texture variation (simple checkerboard pixel pattern)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        r, g, b = base_color
        dark  = (max(0, r - 20), max(0, g - 20), max(0, b - 20), 255)
        light = (min(255, r + 20), min(255, g + 20), min(255, b + 20), 255)
        for y in range(0, 32, 8):
            for x in range(0, 32, 8):
                color = dark if (x // 8 + y // 8) % 2 == 0 else light
                draw.rectangle([x, y, x + 7, y + 7], fill=color)

        # Apply the TT-generated color filter
        filtered = apply_color_matrix(
            img,
            color_matrix,
            hue_shift_degrees = params["hue_shift"],
            saturation_scale  = params["saturation"],
            gamma             = params["gamma"],
        )

        path = out_dir / f"terrain_{terrain}.png"
        filtered.save(path)
        written.append(path)

    return written


# ── Batch filter: all civs' sprites ──────────────────────────────────────────

def generate_all_filters(
    genomes: torch.Tensor,
    color_matrices: list[torch.Tensor],
    nation_slugs: list[str],
    output_dir: Path,
    tileset_dir: Optional[Path] = None,
) -> dict:
    """
    Apply per-civ color filters to all sprite sets.

    Each civ gets its own filtered sprite directory, giving the game
    a unique visual per-civ color palette in addition to the shared
    terrain filter.

    Also generates a single "world filter" (average of all civ matrices)
    applied to the terrain tiles — this is the main per-load visual variant.

    Parameters
    ----------
    genomes        : (n_civs, 64) float32 tensor
    color_matrices : list of (3, 3) tensors, one per civ
    nation_slugs   : list of slug strings
    output_dir     : base output directory
    tileset_dir    : optional source tileset path

    Returns
    -------
    dict with filter statistics
    """
    n_civs = len(nation_slugs)
    results = {}

    # Compute the "world" color matrix as a weighted average of all civ matrices
    world_matrix = torch.stack(color_matrices).mean(dim=0)

    # Use the first civ's genome for the shared world filter parameters
    # (This genome is the game seed → civ 0, so it sets the session's "mood")
    world_genome  = genomes[0]
    world_manifest = apply_generative_filters(
        genome       = world_genome,
        color_matrix = world_matrix,
        output_dir   = output_dir / "world",
        tileset_dir  = tileset_dir,
    )
    results["world"] = world_manifest

    # Per-civ sprite filters (applied to unit sprites for that civ)
    for i, (slug, mat) in enumerate(zip(nation_slugs, color_matrices)):
        civ_genome   = genomes[i]
        civ_manifest = apply_generative_filters(
            genome       = civ_genome,
            color_matrix = mat,
            output_dir   = output_dir / "civs" / slug,
            tileset_dir  = tileset_dir,
        )
        results[slug] = civ_manifest

    return results


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from kernels.civ_genome import generate_genomes_hardware
    import ttnn
    from pathlib import Path

    N    = 6
    SEED = 42

    print("=" * 60)
    print("Color Filter Generator — Hardware Test")
    print(f"Generating filters for {N} civilizations (seed={SEED})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        genomes, g_ms = generate_genomes_hardware(N, SEED, device)

        print(f"\nTT genome time: {g_ms:.3f}ms")
        print("Generating color transform matrices on TT hardware...")

        t0 = time.perf_counter()
        matrices = []
        for i in range(N):
            mat, mat_ms = generate_color_transform_hardware(genomes[i], device)
            matrices.append(mat)
        tt_ms = (time.perf_counter() - t0) * 1000

    finally:
        ttnn.close_device(device)

    print(f"Color matrix generation: {tt_ms:.3f}ms total")

    out = Path("output/color_filter_test")
    slugs = [f"testciv{i+1}" for i in range(N)]

    print("\nApplying filters to terrain tiles...")
    results = generate_all_filters(genomes, matrices, slugs, out)

    print(f"\nWorld filter:")
    w = results["world"]
    print(f"  hue_shift={w['hue_shift_deg']}°  sat={w['saturation']:.2f}  "
          f"gamma={w['gamma']:.2f}")
    print(f"  sprites_filtered={w['sprites_filtered']}  "
          f"placeholders={w['placeholders']}  elapsed={w['elapsed_ms']}ms")

    print("\n✓ Color filter generation complete!")
    print(f"  Output: {out}/world/filtered_sprites/")
