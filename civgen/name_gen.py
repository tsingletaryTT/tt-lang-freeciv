# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
name_gen.py — Markov-chain name generator driven by TT phoneme matrices
========================================================================

The 26×26 phoneme transition matrices were computed on P300C Blackhole hardware
in civ_genome.generate_phoneme_matrices_hardware().  Each row defines the
probability of the next letter given the current letter.

CPU-side we:
  1. Sample names using the Markov chain (weighted random walk through the matrix)
  2. Apply genome parameters to control length and vowel density
  3. Capitalize and clean up the result
  4. Derive nation/adjective/city names from the leader name seed

Genome parameters used:
  [25] phoneme_vowel   — 0=all consonants, 1=very vowel-heavy
  [26] phoneme_hard    — 0=soft (S,L,M,N), 1=harsh (K,G,T,D)
  [27] phoneme_length  — average character count (scaled to 4-14)
  [28] phoneme_syllables — syllable pattern (CV, CVC, VC, etc.)
"""

import random
import torch

# ── Constants ────────────────────────────────────────────────────────────────

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS   = set("aeiou")
CONSONANTS = set(ALPHABET) - VOWELS

# Minimum leader name length (avoids 1-2 char garbage)
MIN_NAME_LEN = 4
MAX_NAME_LEN = 14

# Nation name suffixes by terrain/personality archetype (index from genome[7])
_TERRAIN_SUFFIXES = [
    ["ia", "land", "ar"],        # 0: plains
    ["ia", "um", "os"],          # 1: grassland
    ["or", "heim", "gar"],       # 2: forest
    ["axa", "uur", "kh"],        # 3: tundra
    ["ara", "oasis", "an"],      # 4: desert
    ["ia", "um", "mar"],         # 5: ocean/coastal
    ["peak", "heim", "gard"],    # 6: mountains
    ["marsh", "fen", "moor"],    # 7: swamp
]

# City name suffixes (different from nation name suffixes)
_CITY_SUFFIXES = [
    "polis", "burg", "ville", "ton", "ford",
    "haven", "port", "holm", "grad", "abad",
    "shire", "keep", "gate", "hold", "fall",
]


def _sample_name(
    matrix: torch.Tensor,
    length: int,
    rng: random.Random,
    start_char: str | None = None,
) -> str:
    """
    Walk the 26×26 Markov chain to produce a name of given length.

    Parameters
    ----------
    matrix : (26, 26) float32 tensor, rows summing to 1
    length : target character count
    rng    : CPU RNG (seeded per-civ for reproducibility)
    start_char : if given, force the first character

    Returns
    -------
    Lowercase string of approximately `length` characters.
    """
    name = []

    # Choose starting character — weight toward vowels at word start
    if start_char:
        idx = ord(start_char) - ord('a')
    else:
        # Slightly bias toward vowel-start for pronounceability
        vowel_idxs = [ord(c) - ord('a') for c in "aeiou"]
        if rng.random() < 0.45:
            idx = rng.choice(vowel_idxs)
        else:
            idx = rng.randint(0, 25)

    name.append(ALPHABET[idx])

    for _ in range(length - 1):
        row = matrix[idx].tolist()
        # Weighted random choice from the probability row
        r = rng.random()
        cumulative = 0.0
        next_idx = 0
        for j, p in enumerate(row):
            cumulative += p
            if r <= cumulative:
                next_idx = j
                break
        idx = next_idx
        name.append(ALPHABET[idx])

    return "".join(name)


def _clean_name(raw: str) -> str:
    """
    Post-process a raw Markov-sampled string into something pronounceable.

    Rules:
    - No more than 3 consecutive consonants
    - No more than 2 consecutive identical letters
    - Must have at least 1 vowel
    """
    cleaned = []
    consecutive_consonants = 0
    last_char = ""
    last_last_char = ""

    for ch in raw:
        # Break up runs of 3+ consecutive consonants by inserting a vowel
        if ch not in VOWELS and last_char not in VOWELS and last_last_char not in VOWELS:
            cleaned.append("a")
            consecutive_consonants = 0

        # No triple letters
        if ch == last_char == last_last_char:
            continue

        cleaned.append(ch)
        last_last_char = last_char
        last_char = ch

    result = "".join(cleaned)

    # Ensure at least one vowel
    if not any(c in VOWELS for c in result):
        mid = len(result) // 2
        result = result[:mid] + "a" + result[mid:]

    return result


def generate_names(
    genomes: torch.Tensor,
    phoneme_matrices: torch.Tensor,
    seed: int,
) -> list[dict]:
    """
    Generate civilization names for all civs.

    Parameters
    ----------
    genomes : (n_civs, 64) float32 — genome vectors from TT hardware
    phoneme_matrices : (n_civs, 26, 26) float32 — Markov chain matrices from TT
    seed : int — game seed for reproducibility

    Returns
    -------
    List of dicts, one per civ:
      {
        "leader"  : "Augustus",
        "nation"  : "Roma",
        "adjective": "Roman",
        "city1"   : "Rome",
        "city2"   : "Carthago",
        "city3"   : "Alexandria",
      }
    """
    n_civs = genomes.shape[0]
    results = []

    for i in range(n_civs):
        # Seed per-civ RNG from game seed + civ index for full reproducibility
        rng = random.Random(seed ^ (i * 0x9E3779B9))

        mat    = phoneme_matrices[i]       # (26, 26)
        genome = genomes[i]                # (64,)

        # Genome parameters control name character
        vowel_bias    = genome[25].item()  # 0-1
        length_bias   = genome[27].item()  # 0-1 → mapped to [4, 14]
        terrain_type  = int(genome[7].item() * 7.99)  # 0-7

        # Leader name: 5-12 characters depending on genome
        leader_len = int(MIN_NAME_LEN + length_bias * (10))
        leader_len = max(MIN_NAME_LEN, min(MAX_NAME_LEN, leader_len))

        raw_leader = _sample_name(mat, leader_len, rng)
        leader_name = _clean_name(raw_leader).capitalize()

        # Nation name: shorter, uses same phoneme character but different start
        nation_len = max(3, leader_len - 2)
        nation_seed_char = ALPHABET[rng.randint(0, 25)]
        raw_nation = _sample_name(mat, nation_len, rng, start_char=nation_seed_char)
        nation_base = _clean_name(raw_nation).capitalize()

        # Append a terrain-appropriate suffix
        suffixes = _TERRAIN_SUFFIXES[terrain_type]
        nation_suffix = rng.choice(suffixes)
        nation_name = nation_base + nation_suffix

        # Adjective: usually nation name with small modification
        # Common patterns: add "n", "an", "ian", "ese", "ish"
        adj_patterns = ["n", "an", "ian", "ese", "ish", "ic", "ine"]
        adj_suffix = rng.choice(adj_patterns)
        # Strip trailing vowel before adding vowel-starting suffix to avoid double-vowels
        adj_base = nation_base.rstrip("aeiouAEIOU") if adj_suffix[0] in "aeiou" else nation_base
        adjective = adj_base + adj_suffix

        # City names: 3 cities with city-style suffixes
        city_suffix = rng.choice(_CITY_SUFFIXES)
        city1_len = max(3, nation_len - 1)
        raw_city1 = _sample_name(mat, city1_len, rng)
        city1 = _clean_name(raw_city1).capitalize() + city_suffix

        raw_city2 = _sample_name(mat, city1_len, rng)
        city2 = _clean_name(raw_city2).capitalize() + rng.choice(_CITY_SUFFIXES)

        raw_city3 = _sample_name(mat, city1_len, rng)
        city3 = _clean_name(raw_city3).capitalize() + rng.choice(_CITY_SUFFIXES)

        results.append({
            "leader":    leader_name,
            "nation":    nation_name,
            "adjective": adjective.capitalize(),
            "city1":     city1,
            "city2":     city2,
            "city3":     city3,
        })

    return results


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from kernels.civ_genome import generate_genomes_hardware, generate_phoneme_matrices_hardware
    import ttnn

    N = 6
    SEED = 42

    print("=" * 60)
    print("Name Generator — Hardware Test")
    print(f"Generating {N} civilization names (seed={SEED})")
    print("=" * 60)

    device = ttnn.open_device(device_id=0)
    try:
        genomes, g_ms = generate_genomes_hardware(N, SEED, device)
        matrices, p_ms = generate_phoneme_matrices_hardware(genomes, SEED, device)
    finally:
        ttnn.close_device(device)

    print(f"\nTT kernel time: {g_ms + p_ms:.3f}ms")
    print()

    names = generate_names(genomes, matrices, SEED)
    for i, n in enumerate(names):
        print(f"Civ {i+1}: {n['leader']} of {n['nation']} ({n['adjective']})")
        print(f"        Cities: {n['city1']}, {n['city2']}, {n['city3']}")
