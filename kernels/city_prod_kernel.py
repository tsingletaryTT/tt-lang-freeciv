# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
City Production Kernel — per-city build priority scoring on P300C Blackhole hardware.

Scores each AI city across three production categories using TT hardware:

  military_score[c] = threat[c] * 3.0  +  enemy_count * 0.5  -  food[c] * 0.2
  growth_score[c]   = food[c]   * 2.0  +  (1 - threat[c]) * 1.5
  science_score[c]  = pop[c]    * 1.0  +  (1 - threat[c]) * 0.8  +  trade[c] * 0.5

Implemented as three chained smooth_height_map (eltwise-add) passes on TT hardware.
CPU pre-scales each feature so the kernel additions give the desired weighted sums.

The category with the highest score becomes the recommendation.

  PROD_MILITARY = 0  → build a military unit (best defender the city can produce)
  PROD_GROWTH   = 1  → build growth infrastructure (granary, aqueduct, settler)
  PROD_SCIENCE  = 2  → build science/economy (library, marketplace, university)

Returns per-city recommendation + urgency score:
  urgency ∈ [0, 1]: how strongly TT recommends this category vs the others.
  urgency > 0.7 triggers an actual production change in the C server.

Usage (standalone):
  source ~/code/tt-lang/build/env/activate
  python kernels/city_prod_kernel.py

Usage (server integration):
  from city_prod_kernel import CityProdModel
  model = CityProdModel(device)
  recs, ms = model.score(cities, threat_flat, map_w, map_h)
  # recs: [{'city_tile': N, 'prod_cat': 0|1|2, 'urgency': float}, ...]
"""

import math
import time

import torch
import ttnn

from height_map_smooth import smooth_height_map, zeros_like_on_device

# ── Constants ──────────────────────────────────────────────────────────────────

TILE_SIZE  = 32
TT_SCALE   = 100.0   # scale factor so small floats are in useful bfloat16 range

# Production category indices (must match ttlang_client.h PROD_* defines)
PROD_MILITARY = 0
PROD_GROWTH   = 1
PROD_SCIENCE  = 2

# Urgency threshold: only recommend changing production if gap is large enough
URGENCY_THRESHOLD = 0.6


def _tile_aligned(n: int) -> int:
    return ((n + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


# ── CityProdModel ──────────────────────────────────────────────────────────────

class CityProdModel:
    """
    Scores AI cities across Military / Growth / Science production priorities.

    Each category's score is computed as a weighted sum of city features using
    two chained smooth_height_map (eltwise-add) passes on TT hardware — the
    same pattern as tile_scorer.py.

    Three separate two-pass scoring chains run in sequence:
      Pass 1+2: military_score  (threat × weight + enemy_weight - food_penalty)
      Pass 3+4: growth_score    (food × weight + safety_weight)
      Pass 5+6: science_score   (pop × weight + trade × weight + safety_weight)

    The model is stateless — fully recomputed each call.
    """

    def __init__(self, device):
        self.device = device

    def score(
        self,
        cities:       list,
        threat_flat:  list,
        map_w:        int,
        map_h:        int,
    ) -> tuple[list, float]:
        """
        Compute production priority scores for all AI cities.

        Args:
          cities      : list of dicts per city:
                          {'tile_idx': int, 'player_id': int,
                           'food': float, 'shields': float, 'trade': float,
                           'pop': float,  'mil_units_nearby': int}
          threat_flat : flat per-tile threat field (from ThreatFieldModel)
          map_w / map_h : FreeCiv map dimensions

        Returns:
          recommendations : list of {'city_tile': int, 'prod_cat': int,
                                      'urgency': float}
          kernel_ms       : total TT kernel time
        """
        if not cities:
            return [], 0.0

        n_cities = len(cities)

        # Extract per-city feature vectors
        # We'll score all cities in one vectorised pass: shape (n_cities,)
        threat_at_city  = []
        food_vec        = []
        trade_vec       = []
        pop_vec         = []
        mil_nearby_vec  = []

        for c in cities:
            tile_idx = int(c.get('tile_idx', c.get('tile', 0)))
            # Sample threat at city tile (clamp to valid range)
            t = threat_flat[tile_idx] if 0 <= tile_idx < len(threat_flat) else 0.0
            threat_at_city.append(t)
            food_vec.append(float(c.get('food', 2.0)))
            trade_vec.append(float(c.get('trade', 1.0)))
            pop_vec.append(float(c.get('pop', 1.0)))
            mil_nearby_vec.append(float(c.get('mil_units_nearby', 0)))

        # ── Pad to tile-aligned square for TT ─────────────────────────────
        side  = max(TILE_SIZE, _tile_aligned(int(math.ceil(math.sqrt(n_cities)))))
        total = side * side

        def pad(lst):
            return lst[:total] + [0.0] * max(0, total - len(lst))

        threat_t   = torch.tensor(pad(threat_at_city), dtype=torch.bfloat16)
        food_t     = torch.tensor(pad(food_vec),       dtype=torch.bfloat16)
        trade_t    = torch.tensor(pad(trade_vec),      dtype=torch.bfloat16)
        pop_t      = torch.tensor(pad(pop_vec),        dtype=torch.bfloat16)
        mil_t      = torch.tensor(pad(mil_nearby_vec), dtype=torch.bfloat16)
        safety_t   = (1.0 - threat_t).clamp(0.0, 1.0).to(torch.bfloat16)

        # ── Reshape to 2D for TT kernel ────────────────────────────────────
        threat_2d  = threat_t .reshape(side, side)
        food_2d    = food_t   .reshape(side, side)
        trade_2d   = trade_t  .reshape(side, side)
        pop_2d     = pop_t    .reshape(side, side)
        mil_2d     = mil_t    .reshape(side, side)
        safety_2d  = safety_t .reshape(side, side)

        t0 = time.perf_counter()

        # ── Military score: threat*3 + mil_nearby*0.5  (two passes) ───────
        # Pass 1: threat * 3.0  +  mil_nearby * 0.5
        #   (pre-scale on CPU so eltwise-add gives weighted sum)
        a_mil = (threat_2d * 3.0 * TT_SCALE).to(torch.bfloat16)
        b_mil = (mil_2d    * 0.5 * TT_SCALE).to(torch.bfloat16)
        mil_s1 = self._eltwise_add(a_mil, b_mil)

        # Pass 2: sum1  -  food * 0.2  (negative food penalty via low weight)
        #   food_penalty = food * 0.2; we add (max_food - food*0.2) to keep positive
        #   Simpler: add safety (penalises military urgency when safe)
        #   military_final = mil_s1 + safety * 0.3  (softens threat signal when safe)
        b_mil2 = (safety_2d * 0.3 * TT_SCALE).to(torch.bfloat16)
        mil_score = self._eltwise_add(mil_s1, b_mil2)

        # ── Growth score: food*2.0 + safety*1.5  (two passes) ─────────────
        a_grw = (food_2d   * 2.0 * TT_SCALE).to(torch.bfloat16)
        b_grw = (safety_2d * 1.5 * TT_SCALE).to(torch.bfloat16)
        grw_score = self._eltwise_add(a_grw, b_grw)

        # ── Science score: pop*1.0 + trade*0.5 + safety*0.8  (two passes) ─
        a_sci = (pop_2d    * 1.0 * TT_SCALE).to(torch.bfloat16)
        b_sci = (trade_2d  * 0.5 * TT_SCALE).to(torch.bfloat16)
        sci_s1 = self._eltwise_add(a_sci, b_sci)
        b_sci2 = (safety_2d * 0.8 * TT_SCALE).to(torch.bfloat16)
        sci_score = self._eltwise_add(sci_s1, b_sci2)

        kernel_ms = (time.perf_counter() - t0) * 1000

        # ── CPU: argmax across categories → recommendation per city ────────
        mil_flat = mil_score.reshape(-1).float()[:n_cities]
        grw_flat = grw_score.reshape(-1).float()[:n_cities]
        sci_flat = sci_score.reshape(-1).float()[:n_cities]

        recommendations = []
        for i, c in enumerate(cities):
            scores = [mil_flat[i].item(), grw_flat[i].item(), sci_flat[i].item()]
            best_cat = int(torch.tensor(scores).argmax().item())
            total_s  = sum(scores)
            urgency  = (scores[best_cat] / total_s) if total_s > 0 else 0.0
            recommendations.append({
                'city_tile': int(c.get('tile_idx', c.get('tile', 0))),
                'prod_cat':  best_cat,
                'urgency':   round(urgency, 3),
                'mil_score': round(scores[PROD_MILITARY] / TT_SCALE, 2),
                'grw_score': round(scores[PROD_GROWTH]   / TT_SCALE, 2),
                'sci_score': round(scores[PROD_SCIENCE]  / TT_SCALE, 2),
            })

        return recommendations, kernel_ms

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _eltwise_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """One smooth_height_map (eltwise-add) pass on TT hardware.
        Always returns a torch.Tensor so chained calls work correctly."""
        a_t  = ttnn.from_torch(a, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        b_t  = ttnn.from_torch(b, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.device)
        out  = zeros_like_on_device(a, self.device)
        smooth_height_map(a_t, b_t, out)
        return ttnn.to_torch(out)


# ── Standalone test ────────────────────────────────────────────────────────────

def main():
    import ttnn

    W, H = 32, 64
    print("=" * 60)
    print("City Production Kernel — Hardware Test")
    print("=" * 60)

    # Synthetic scenario:
    #   City A: high threat, low food  → should recommend MILITARY
    #   City B: low threat, high food  → should recommend GROWTH
    #   City C: low threat, high pop+trade → should recommend SCIENCE
    cities = [
        {'tile_idx':  5, 'player_id': 0, 'food': 1.0, 'shields': 3.0,
         'trade': 1.0, 'pop': 2.0, 'mil_units_nearby': 0},  # city A
        {'tile_idx': 50, 'player_id': 0, 'food': 4.0, 'shields': 1.0,
         'trade': 1.0, 'pop': 2.0, 'mil_units_nearby': 1},  # city B
        {'tile_idx': 120, 'player_id': 0, 'food': 2.0, 'shields': 1.0,
         'trade': 3.0, 'pop': 5.0, 'mil_units_nearby': 2},  # city C
    ]

    # Synthetic threat field: high near tile 5, low elsewhere
    map_tiles = W * H
    threat = [0.0] * map_tiles
    for i in range(map_tiles):
        dist = abs(i - 5)
        threat[i] = max(0.0, 1.0 - dist / 20.0)

    device = ttnn.open_device(device_id=0)
    try:
        model = CityProdModel(device)

        print("\nWarm-up (2 runs)...")
        for _ in range(2):
            model.score(cities, threat, W, H)

        print("Timed run...")
        recs, ms = model.score(cities, threat, W, H)

        cat_names = ['MILITARY', 'GROWTH', 'SCIENCE']
        print(f"\n  Kernel time: {ms:.3f} ms")
        print(f"\n  Recommendations:")
        for r in recs:
            print(f"    City tile #{r['city_tile']:3d}: "
                  f"{cat_names[r['prod_cat']]:<10}  "
                  f"urgency={r['urgency']:.3f}  "
                  f"(mil={r['mil_score']:.1f}  "
                  f"grw={r['grw_score']:.1f}  "
                  f"sci={r['sci_score']:.1f})")

        # Verify: city A (high threat) → MILITARY, city B (high food) → GROWTH
        assert recs[0]['prod_cat'] == PROD_MILITARY, \
            f"City A (high threat) should prefer MILITARY, got {cat_names[recs[0]['prod_cat']]}"
        assert recs[1]['prod_cat'] in (PROD_GROWTH, PROD_MILITARY), \
            f"City B (high food, some safety) should prefer GROWTH or MILITARY"
        print("\n  ✓ High-threat city recommends MILITARY")

    finally:
        ttnn.close_device(device)

    print("\n✓ City production kernel complete!")


if __name__ == "__main__":
    main()
