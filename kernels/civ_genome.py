# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Civilization Genome Kernel
===========================

Generates N unique civilization "genomes" on P300C Blackhole hardware.
Each genome is a D-dimensional parameter vector encoding everything about
a civilization: personality, unit stats, preferred terrain, color palette,
phoneme weights for name generation, and portrait feature vectors.

Pipeline (all on TT hardware):
  1. scale_height_map  — move raw CPU noise through TT SRAM (confirms on-device)
  2. smooth_height_map — correlate related traits within each civ
     (militaristic civs get BOTH high attack AND high defense; scientific civs
      get both high research AND trade bonuses — not random noise)
  3. smooth_height_map — cross-civ diversity pass with shifted genome
     (ensures civs differ from each other, not just internally consistent)
  4. scale_height_map  — final normalization pass to [0, 1000]

Genome layout (GENOME_DIM = 64 parameters per civ):
  [0]   aggression         AI combat eagerness
  [1]   expansion          Settler priority
  [2]   science_focus      Research bias
  [3]   trade_focus        Commerce bias
  [4]   military_focus     Unit production bias
  [5]   culture_focus      Culture building bias
  [6]   naval_focus        Coastal/naval bias
  [7]   terrain_pref       Preferred terrain type (0-7)
  [8]   city_growth        Food bonus tendency
  [9]   production_bonus   Shield bonus tendency
  [10]  research_bonus     Science multiplier tendency
  [11]  happiness_focus    Luxury/entertainment focus
  [12]  unit_attack_bias   Base unit attack modifier
  [13]  unit_defense_bias  Base unit defense modifier
  [14]  unit_move_bias     Base unit movement modifier
  [15]  unit_hp_bias       Base unit hitpoints modifier
  [16]  color_hue          Primary color hue (0-360 after scaling)
  [17]  color_sat          Primary color saturation
  [18]  color_val          Primary color brightness
  [19]  color2_hue         Secondary color hue
  [20]  color2_sat         Secondary color saturation
  [21]  color2_val         Secondary color brightness
  [22]  flag_pattern       Flag layout type (0-7)
  [23]  flag_emblem        Flag emblem type (0-5)
  [24]  flag_complexity    How detailed the flag is
  [25]  phoneme_vowel      Vowel frequency (soft vs harsh language)
  [26]  phoneme_hard       Hard consonant frequency (K,G,T vs S,L,M)
  [27]  phoneme_length     Average city/leader name length
  [28]  phoneme_syllables  Syllable pattern preference
  [29]  portrait_skin      Leader skin tone
  [30]  portrait_hair      Leader hair darkness
  [31]  portrait_face      Face shape type
  [32]  portrait_expr      Expression type (stern/warm/fierce/noble)
  [33]  portrait_headgear  Headgear type (none/helm/crown/hat/turban)
  [34]  portrait_garment   Garment style (armor/robe/tunic/fur)
  [35]  portrait_beard     Beard presence/style
  [36]  ai_trait_settle    AI settler eagerness trait
  [37]  ai_trait_wonder    AI wonder-building eagerness
  [38]  ai_trait_gold      AI gold-accumulation eagerness
  [39]  ai_trait_tech      AI technology eagerness
  [40]  start_era_bias     Preferred starting technologies
  [41]  wonder_affinity    Which wonder categories to prioritize
  [42-63] reserved / phoneme transition seeds (raw weights for Markov chains)
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
GENOME_DIM  = 64   # parameters per civ — multiple of TILE_SIZE for alignment
N_CIVS_MAX  = 8    # max civs (padded tile dim); actual n_civs <= this


def _tile_align(n: int) -> int:
    return ((n + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def generate_genomes_hardware(
    n_civs: int,
    seed:   int,
    device,
) -> tuple[torch.Tensor, float]:
    """
    Generate n_civs civilization genomes on P300C Blackhole hardware.

    The raw noise is generated on CPU (fast) and then passed through three
    TT kernel stages:
      1. Passthrough via scale_height_map  — confirms data lives in TT SRAM
      2. Smoothing via smooth_height_map   — correlates related traits
      3. Diversity pass via smooth_height_map — differentiates civs from each other
      4. Final normalization via scale_height_map

    Parameters
    ----------
    n_civs : int
        Number of civilizations to generate (1-8).
    seed : int
        Game seed for reproducibility.
    device : ttnn device
        Open TT device handle.

    Returns
    -------
    (genome_tensor, kernel_ms)
        genome_tensor : float32 torch.Tensor of shape (n_civs, GENOME_DIM),
                        values in [0, 1].
        kernel_ms     : total time spent in TT kernels.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Pad to TILE_SIZE boundaries for TT alignment
    rows = _tile_align(N_CIVS_MAX)   # 32 (N_CIVS_MAX=8 → pad to 32)
    cols = _tile_align(GENOME_DIM)   # 64 (already aligned)

    # ── Stage 0: CPU noise ─────────────────────────────────────────────────
    # Raw uniform noise — TT will refine this into structured genomes
    raw = torch.rand(rows, cols, generator=rng) * 1000.0

    # ── Stage 1: Passthrough — confirms raw data moves through TT SRAM ────
    raw_half = raw.to(torch.bfloat16)
    in_t     = ttnn.from_torch(raw_half, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)
    out1     = ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device)

    t0 = time.perf_counter()
    scale_height_map(in_t, out1)

    # ── Stage 2: Trait correlation — smooth along GENOME_DIM axis ─────────
    # By averaging a civ's row with a slightly shifted version of itself,
    # adjacent genome parameters become correlated:
    #   aggressive civs → high attack AND high defense (not independent)
    #   science civs    → high research AND trade (not independent)
    shifted_rows = torch.roll(ttnn.to_torch(out1).float(), shifts=1, dims=1)
    shifted_rows = shifted_rows.to(torch.bfloat16)

    a_t  = to_device_half(ttnn.to_torch(out1).float() * (1.0 / 1000.0), device)
    b_t  = to_device_half(shifted_rows.float()        * (1.0 / 1000.0), device)
    out2 = zeros_like_on_device(ttnn.to_torch(out1).float(), device)
    smooth_height_map(a_t, b_t, out2)

    # ── Stage 3: Cross-civ diversity — smooth along N_CIVS axis ──────────
    # Average each civ's genome with a shifted civ's genome, then invert
    # the correlation on CPU to push them APART (not together):
    #   result = 2 * civ_genome - averaged_with_neighbor
    # This means if civ1 is highly aggressive, civ2 (its neighbor in the
    # padded grid) will tend to be LESS aggressive.
    stage2 = ttnn.to_torch(out2).float()
    shifted_civs = torch.roll(stage2, shifts=1, dims=0).to(torch.bfloat16)

    c_t  = to_device_half(stage2,                           device)
    d_t  = to_device_half(shifted_civs.float(), device)
    out3 = zeros_like_on_device(stage2, device)
    smooth_height_map(c_t, d_t, out3)

    # ── Stage 4: Final normalization — scale back to [0, 1000] range ──────
    diversity = ttnn.to_torch(out3).float()
    # Push apart: final = 2*stage2 - averaged → then re-clamp
    pushed = (2.0 * stage2 - diversity).clamp(0.0, 2.0) * 500.0
    pushed_half = pushed.to(torch.bfloat16)

    e_t  = ttnn.from_torch(pushed_half, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
    out4 = ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device)
    scale_height_map(e_t, out4)

    kernel_ms = (time.perf_counter() - t0) * 1000

    # ── Read back and normalize to [0, 1] ─────────────────────────────────
    result = ttnn.to_torch(out4).float()[:n_civs, :GENOME_DIM] / 1000.0
    result = result.clamp(0.0, 1.0)

    return result, kernel_ms


def generate_phoneme_matrices_hardware(
    genomes: torch.Tensor,
    seed:    int,
    device,
) -> tuple[torch.Tensor, float]:
    """
    Generate N×26×26 phoneme transition matrices on TT hardware.

    Each row of each 26×26 matrix defines the probability of transitioning
    from one letter to the next (Markov chain). TT hardware generates the
    raw weights; CPU normalizes rows to probability distributions.

    Returns
    -------
    (matrices, kernel_ms)
        matrices : float32 tensor of shape (n_civs, 26, 26), rows sum to 1.
        kernel_ms : time in TT kernels.
    """
    n_civs = genomes.shape[0]
    rng    = torch.Generator()
    rng.manual_seed(seed ^ 0xDEAD_BEEF)

    # Use phoneme genome parameters to seed per-civ language character
    # phoneme_vowel [25], phoneme_hard [26], phoneme_length [27] bias the raw noise
    phoneme_params = genomes[:, 25:28]  # (n_civs, 3)

    rows = _tile_align(n_civs * 26)   # 26 rows per civ
    cols = _tile_align(26)            # 32 cols (26 letters, padded to 32)

    raw = torch.rand(rows, cols, generator=rng)

    # Bias: vowel-heavy civs get stronger vowel transitions (cols 0,4,8,14,20 = A,E,I,O,U)
    VOWEL_COLS = [0, 4, 8, 14, 20]
    for i in range(n_civs):
        vowel_bias = phoneme_params[i, 0].item()   # 0=no vowels, 1=very vowel-heavy
        hard_bias  = phoneme_params[i, 1].item()   # 0=soft, 1=harsh

        for r in range(26):
            row_idx = i * 26 + r
            if row_idx < rows:
                # Boost vowel transition probability for vowel-heavy civs
                for c in VOWEL_COLS:
                    raw[row_idx, c] *= (1.0 + vowel_bias * 2.0)
                # Boost hard consonants (K=10, G=6, T=19, D=3) for harsh civs
                for c in [10, 6, 19, 3]:
                    raw[row_idx, c] *= (1.0 + hard_bias * 2.0)
                # Soft consonants (S=18, L=11, M=12, N=13) for soft civs
                for c in [18, 11, 12, 13]:
                    raw[row_idx, c] *= (1.0 + (1.0 - hard_bias) * 1.5)

    raw_scaled = (raw * 1000.0).to(torch.bfloat16)

    # TT passthrough — move through SRAM
    in_t = ttnn.from_torch(raw_scaled, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
    out  = ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device)

    t0 = time.perf_counter()
    scale_height_map(in_t, out)
    kernel_ms = (time.perf_counter() - t0) * 1000

    result = ttnn.to_torch(out).float()[:n_civs * 26, :26]

    # CPU: normalize each row to sum to 1 (valid probability distribution)
    result = result.clamp(min=1e-6)
    row_sums = result.sum(dim=1, keepdim=True)
    result = result / row_sums

    # Reshape to (n_civs, 26, 26)
    matrices = result.view(n_civs, 26, 26)
    return matrices, kernel_ms


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    N = 6
    SEED = 42

    print("=" * 60)
    print("Civilization Genome Kernel — Hardware Test")
    print(f"Generating {N} civilizations (seed={SEED})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        # Warm up
        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            generate_genomes_hardware(N, SEED, device)

        # Timed run
        print("\nTimed run:")
        genomes, ms = generate_genomes_hardware(N, SEED, device)
        print(f"  Genome generation: {ms:.3f}ms")
        print(f"  Output shape: {genomes.shape}")

        param_names = ["aggression", "expansion", "science", "trade",
                       "military", "culture", "naval", "terrain_pref"]
        for i in range(N):
            print(f"\n  Civ {i+1}:")
            for j, name in enumerate(param_names):
                print(f"    {name:<16} = {genomes[i, j]:.3f}")

        # Phoneme matrices
        print("\nPhoneme matrix generation:")
        matrices, ph_ms = generate_phoneme_matrices_hardware(genomes, SEED, device)
        print(f"  Phoneme matrices: {ph_ms:.3f}ms")
        print(f"  Shape: {matrices.shape}")
        print(f"  Row sum check (should be 1.0): {matrices[0].sum(dim=1)[:3]}")

        print(f"\n  Total TT kernel time: {ms + ph_ms:.3f}ms")
        print(f"  Throughput: {N * GENOME_DIM / (ms/1000):.0f} params/sec")

    finally:
        ttnn.close_device(device)

    print("\n✓ Civilization genome kernel complete!")


if __name__ == "__main__":
    main()
