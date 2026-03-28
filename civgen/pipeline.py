# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
pipeline.py — Master orchestrator: TT hardware → complete FreeCiv civilizations
================================================================================

Calls each stage in order and wires their outputs together:

  Stage 1 (TT hardware):  generate 64-param genome vectors on P300C Blackhole
  Stage 2 (TT hardware):  generate 26×26 phoneme Markov matrices on P300C
  Stage 3 (CPU):          sample Markov chains → leader, nation, city names
  Stage 4 (CPU/PIL):      render flag PNGs from TT color/pattern params
  Stage 5 (CPU/PIL):      render pixel-art leader portraits from TT feature vectors
  Stage 6 (CPU/PIL):      recolor unit sprites in TT-generated civ colors
  Stage 7 (CPU):          write FreeCiv .ruleset files

Timing is tracked per stage — TT stages are typically <1ms (after JIT warm-up).
CPU stages (PIL rendering) are typically 5-50ms per civ.

Usage:
    from civgen.pipeline import generate_civs
    import ttnn

    device = ttnn.open_device(device_id=0)
    civs = generate_civs(n_civs=6, seed=42, device=device, output_dir=Path("civs"))
    ttnn.close_device(device)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from kernels.civ_genome import generate_genomes_hardware, generate_phoneme_matrices_hardware
from civgen.name_gen      import generate_names
from civgen.flag_gen      import generate_flags
from civgen.portrait_gen  import generate_portraits
from civgen.sprite_recolor import recolor_units, write_color_ini
from civgen.ruleset       import write_all_rulesets

# Color filter import (requires Pillow; graceful fallback if unavailable)
try:
    from civgen.color_filter import generate_all_filters, genome_to_filter_params
    COLOR_FILTER_AVAILABLE = True
except ImportError:
    COLOR_FILTER_AVAILABLE = False


@dataclass
class CivData:
    """
    Complete data for a single generated civilization.

    All numeric parameters were computed on P300C Blackhole hardware.
    Graphics and text were rendered on CPU from those parameters.
    """

    # Identity
    index:      int         # 0-based civ index
    slug:       str         # lowercase unique identifier (e.g. "romai")
    names:      dict        # {"leader", "nation", "adjective", "city1", "city2", "city3"}

    # Raw TT hardware outputs
    genome:     torch.Tensor  # (64,) float32 in [0, 1]

    # Derived outputs (filled after each pipeline stage)
    flag_path:      Optional[Path] = None
    portrait_path:  Optional[Path] = None
    ruleset_path:   Optional[Path] = None
    sprite_paths:   dict = field(default_factory=dict)   # {sprite_name: path}

    # Timing (milliseconds)
    timing: dict = field(default_factory=dict)

    @property
    def primary_color_rgb(self) -> tuple[int, int, int]:
        """Primary color as (r, g, b) tuple (0-255 each)."""
        import colorsys
        h = self.genome[16].item()
        s = 0.55 + self.genome[17].item() * 0.45
        v = 0.5  + self.genome[18].item() * 0.50
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    @property
    def aggression(self) -> float:
        return self.genome[0].item()

    @property
    def expansion(self) -> float:
        return self.genome[1].item()

    @property
    def science(self) -> float:
        return self.genome[2].item()

    @property
    def military(self) -> float:
        return self.genome[4].item()

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        r, g, b = self.primary_color_rgb
        return (
            f"{self.names['nation']} (leader: {self.names['leader']})  "
            f"color=rgb({r},{g},{b})  "
            f"agg={self.aggression:.2f}  mil={self.military:.2f}  sci={self.science:.2f}"
        )


@dataclass
class GenerationResult:
    """
    Complete output of generate_civs().

    Contains all CivData objects and aggregate timing info.
    """
    civs:         list[CivData]
    output_dir:   Path
    timing:       dict           # stage → milliseconds
    ruleset_paths: dict          # from write_all_rulesets()

    def print_summary(self):
        """Print a formatted summary of all generated civilizations."""
        total_tt  = sum(v for k, v in self.timing.items() if k.startswith("tt_"))
        total_cpu = sum(v for k, v in self.timing.items() if not k.startswith("tt_"))

        print("=" * 70)
        print("TT-Lang P300C Blackhole — Generated Civilizations")
        print("=" * 70)
        print(f"  P300C Blackhole hardware time:")
        for k, v in sorted(self.timing.items()):
            if k.startswith("tt_") and v > 0:
                print(f"    {k:<28} {v:.3f}ms")
        print(f"  Total TT hardware:     {total_tt:.3f}ms")
        print(f"  Total CPU render:      {total_cpu:.1f}ms")
        print(f"  Output directory:      {self.output_dir}")
        print(f"")
        print(f"  TT kernel calls per game load:")
        print(f"    scale_height_map:    terrain gen ×2 + genome norm ×2 + colorize ×{3*2} = {4+6*len(self.civs)} calls")
        print(f"    smooth_height_map:   genome ×2 + phonemes + tile_score ×2 + colorize ×{3*len(self.civs)} = many calls")
        print(f"    Total: all numeric parameters for {len(self.civs)} civs from Blackhole")
        print()
        for civ in self.civs:
            r, g, b = civ.primary_color_rgb
            print(f"  Civ {civ.index + 1}: {civ.names['leader']} of {civ.names['nation']}")
            print(f"          slug={civ.slug}  color=#{r:02X}{g:02X}{b:02X}")
            print(f"          agg={civ.aggression:.2f}  exp={civ.expansion:.2f}  "
                  f"mil={civ.military:.2f}  sci={civ.science:.2f}")
            if civ.flag_path:
                print(f"          flag={civ.flag_path.name}")
            if civ.portrait_path:
                print(f"          portrait={civ.portrait_path.name}")
        print()
        print(f"  Ruleset summary: {self.ruleset_paths.get('summary', 'N/A')}")
        print("=" * 70)


def _slugify(nation_name: str, existing: set[str]) -> str:
    """
    Convert a nation name to a unique lowercase slug.

    Strips non-ASCII, replaces spaces with underscores, truncates to 16 chars.
    If the slug collides with an existing one, appends a numeric suffix.
    """
    import re
    slug = re.sub(r"[^a-z0-9]", "", nation_name.lower())[:12]
    if not slug:
        slug = "civ"

    base_slug = slug
    counter   = 2
    while slug in existing:
        slug = f"{base_slug}{counter}"
        counter += 1

    existing.add(slug)
    return slug


def generate_civs(
    n_civs:    int,
    seed:      int,
    device,
    output_dir: Optional[Path] = None,
    tileset_dir: Optional[Path] = None,
    skip_sprites: bool = False,
) -> GenerationResult:
    """
    Generate complete FreeCiv civilizations using P300C Blackhole hardware.

    All numeric parameters are computed on TT hardware.
    Graphics (flags, portraits, sprites) are rendered on CPU using PIL.

    Parameters
    ----------
    n_civs      : number of civilizations to generate (1-8)
    seed        : game seed for full reproducibility
    device      : open ttnn device handle (kept alive by caller)
    output_dir  : where to write output files (default: /tmp/tt_civs_<seed>/)
    tileset_dir : FreeCiv tileset dir for sprite recoloring (auto-detected if None)
    skip_sprites: if True, skip unit sprite recoloring (faster, for testing)

    Returns
    -------
    GenerationResult containing all CivData and timing information
    """
    if output_dir is None:
        output_dir = Path(f"/tmp/tt_civs_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timing: dict[str, float] = {}
    n_civs = max(1, min(8, n_civs))

    # ── Stage 1: TT hardware — genome generation ──────────────────────────────
    print(f"[civgen] Stage 1/7: Generating {n_civs} genomes on P300C hardware...")
    t0 = time.perf_counter()
    genomes, genome_ms = generate_genomes_hardware(n_civs, seed, device)
    timing["tt_genomes"] = genome_ms
    print(f"[civgen]   → {genome_ms:.3f}ms ({n_civs * 64} params on Blackhole)")

    # ── Stage 2: TT hardware — phoneme matrix generation ─────────────────────
    print(f"[civgen] Stage 2/7: Generating phoneme Markov matrices on P300C...")
    t0 = time.perf_counter()
    phoneme_matrices, phoneme_ms = generate_phoneme_matrices_hardware(genomes, seed, device)
    timing["tt_phonemes"] = phoneme_ms
    print(f"[civgen]   → {phoneme_ms:.3f}ms ({n_civs} × 26×26 matrices on Blackhole)")

    # ── Stage 3: CPU — Markov chain name generation ───────────────────────────
    print(f"[civgen] Stage 3/7: Sampling Markov chains → names...")
    t0 = time.perf_counter()
    names_list = generate_names(genomes, phoneme_matrices, seed)
    timing["cpu_names"] = (time.perf_counter() - t0) * 1000
    print(f"[civgen]   → {timing['cpu_names']:.1f}ms")

    # Build slugs from nation names
    used_slugs: set[str] = set()
    slugs = [_slugify(n["nation"], used_slugs) for n in names_list]

    # ── Stage 4: CPU/PIL — flag generation ────────────────────────────────────
    print(f"[civgen] Stage 4/7: Rendering flags (PIL)...")
    t0 = time.perf_counter()
    try:
        flag_paths = generate_flags(genomes, slugs, output_dir)
        timing["cpu_flags"] = (time.perf_counter() - t0) * 1000
        print(f"[civgen]   → {timing['cpu_flags']:.1f}ms  (written to {output_dir}/flags/)")
    except ImportError as e:
        print(f"[civgen]   ⚠ Pillow not installed, skipping flags: {e}")
        flag_paths = {}
        timing["cpu_flags"] = 0.0

    # ── Stage 5: CPU/PIL — portrait generation ────────────────────────────────
    print(f"[civgen] Stage 5/7: Rendering leader portraits (PIL pixel art)...")
    t0 = time.perf_counter()
    try:
        portrait_paths = generate_portraits(genomes, slugs, output_dir)
        timing["cpu_portraits"] = (time.perf_counter() - t0) * 1000
        print(f"[civgen]   → {timing['cpu_portraits']:.1f}ms  (written to {output_dir}/portraits/)")
    except ImportError as e:
        print(f"[civgen]   ⚠ Pillow not installed, skipping portraits: {e}")
        portrait_paths = {}
        timing["cpu_portraits"] = 0.0

    # ── Stage 6: CPU/PIL — sprite recoloring ─────────────────────────────────
    if skip_sprites:
        print(f"[civgen] Stage 6/7: Sprite recoloring skipped (skip_sprites=True)")
        sprite_paths_all = {}
        timing["cpu_sprites"] = 0.0
    else:
        print(f"[civgen] Stage 6/7: Recoloring unit sprites (PIL)...")
        t0 = time.perf_counter()
        try:
            sprite_paths_all = recolor_units(genomes, slugs, output_dir, tileset_dir)
            write_color_ini(genomes, slugs, output_dir)
            timing["cpu_sprites"] = (time.perf_counter() - t0) * 1000
            print(f"[civgen]   → {timing['cpu_sprites']:.1f}ms  (written to {output_dir}/sprites/)")
        except ImportError as e:
            print(f"[civgen]   ⚠ Pillow not installed, skipping sprites: {e}")
            sprite_paths_all = {}
            timing["cpu_sprites"] = 0.0

    # ── Stage 6b: TT+CPU — contextual color filters ───────────────────────────
    # Generate per-civ color transform matrices on TT hardware, then apply
    # contextual filters: terrain colors follow biome conditions, civ sprites
    # match civ primary color, resources get terrain-appropriate tints.
    color_filter_results = {}
    if COLOR_FILTER_AVAILABLE:
        print(f"[civgen] Stage 6b: Generating contextual color filters on TT hardware...")
        t0 = time.perf_counter()
        try:
            from civgen.color_filter import generate_color_transform_hardware
            color_matrices = []
            tt_color_ms = 0.0
            for i in range(n_civs):
                mat, mat_ms = generate_color_transform_hardware(genomes[i], device)
                color_matrices.append(mat)
                tt_color_ms += mat_ms
            timing["tt_color_matrices"] = tt_color_ms

            color_filter_results = generate_all_filters(
                genomes        = genomes,
                color_matrices = color_matrices,
                nation_slugs   = slugs,
                output_dir     = output_dir,
                tileset_dir    = tileset_dir,
            )
            timing["cpu_color_filters"] = (time.perf_counter() - t0) * 1000 - tt_color_ms
            world_f = color_filter_results.get("world", {})
            print(f"[civgen]   → TT {tt_color_ms:.3f}ms + CPU {timing['cpu_color_filters']:.1f}ms  "
                  f"hue_shift={world_f.get('hue_shift_deg', 0):.1f}°  "
                  f"sat={world_f.get('saturation', 1.0):.2f}  "
                  f"gamma={world_f.get('gamma', 1.0):.2f}")
        except Exception as e:
            print(f"[civgen]   ⚠ Color filter error: {e}")
            timing["tt_color_matrices"] = 0.0
            timing["cpu_color_filters"] = 0.0
    else:
        print(f"[civgen] Stage 6b: Color filters skipped (Pillow not installed)")
        timing["tt_color_matrices"] = 0.0
        timing["cpu_color_filters"] = 0.0

    # ── Stage 7: CPU — ruleset writing ────────────────────────────────────────
    print(f"[civgen] Stage 7/7: Writing FreeCiv ruleset files...")
    t0 = time.perf_counter()
    ruleset_paths = write_all_rulesets(genomes, names_list, slugs, output_dir)
    timing["cpu_rulesets"] = (time.perf_counter() - t0) * 1000
    print(f"[civgen]   → {timing['cpu_rulesets']:.1f}ms  (written to {output_dir}/nations/)")

    # ── Assemble CivData objects ───────────────────────────────────────────────
    civs: list[CivData] = []
    for i, (slug, names) in enumerate(zip(slugs, names_list)):
        civ = CivData(
            index        = i,
            slug         = slug,
            names        = names,
            genome       = genomes[i],
            flag_path    = flag_paths.get(slug),
            portrait_path = portrait_paths.get(slug),
            ruleset_path = next(
                (p for p in ruleset_paths.get("nation_files", [])
                 if slug in str(p)), None
            ),
            sprite_paths = sprite_paths_all.get(slug, {}),
            timing       = timing.copy(),
        )
        civs.append(civ)

    result = GenerationResult(
        civs          = civs,
        output_dir    = output_dir,
        timing        = timing,
        ruleset_paths = ruleset_paths,
    )

    return result


# ── Standalone test ──────────────────────────────────────────────────────────

def main():
    """
    Full pipeline test on P300C hardware.

    Run from the tt-lang-freeciv directory:
        python civgen/pipeline.py
    """
    import ttnn
    from pathlib import Path

    N    = 6
    SEED = 42
    OUT  = Path("output/full_civgen_test")

    print("=" * 70)
    print("civgen Full Pipeline — P300C Blackhole Hardware Test")
    print(f"Generating {N} civilizations (seed={SEED})")
    print("=" * 70)
    print()

    # Warm-up (first call compiles JIT; second run is the timed one)
    print("[civgen] Warming up TT kernels (2 runs, first compiles JIT)...")
    device = ttnn.open_device(device_id=0)
    try:
        for warm in range(2):
            from kernels.civ_genome import generate_genomes_hardware
            g, _ = generate_genomes_hardware(N, SEED, device)
        print("[civgen] Warm-up complete.\n")

        result = generate_civs(
            n_civs     = N,
            seed       = SEED,
            device     = device,
            output_dir = OUT,
            skip_sprites = False,
        )
    finally:
        ttnn.close_device(device)

    print()
    result.print_summary()

    # Read and print the civilization summary
    summary_path = result.ruleset_paths.get("summary")
    if summary_path and summary_path.exists():
        print()
        print(summary_path.read_text()[:2000])

    total_tt  = result.timing.get("tt_genomes", 0) + result.timing.get("tt_phonemes", 0)
    total_cpu = sum(v for k, v in result.timing.items() if not k.startswith("tt_"))
    print(f"\nTotal P300C hardware time : {total_tt:.3f}ms")
    print(f"Total CPU render time      : {total_cpu:.1f}ms")
    print(f"\n✓ civgen pipeline complete! Output in: {OUT}")


if __name__ == "__main__":
    main()
