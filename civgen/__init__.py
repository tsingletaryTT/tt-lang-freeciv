# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
civgen — TT-Hardware Civilization Generator
============================================

Generates complete FreeCiv civilizations whose core parameters are computed
on P300C Blackhole hardware.  CPU handles only graphics rendering (PIL) and
file I/O; every numeric decision is made on-device.

Pipeline (called by pipeline.py):
  1. civ_genome.generate_genomes_hardware()       → (N, 64) float genome vectors
  2. civ_genome.generate_phoneme_matrices_hardware() → (N, 26, 26) Markov matrices
  3. name_gen.generate_names()                    → leader & nation names
  4. flag_gen.generate_flags()                    → PNG flag images
  5. portrait_gen.generate_portraits()            → PNG leader portraits
  6. sprite_recolor.recolor_units()               → recolored unit PNGs
  7. ruleset.write_ruleset()                      → FreeCiv .ruleset files

Entry point:
  from civgen.pipeline import generate_civs
  civs = generate_civs(n_civs=6, seed=42, device=tt_device)
"""

from .pipeline import generate_civs, CivData

__all__ = ["generate_civs", "CivData"]
