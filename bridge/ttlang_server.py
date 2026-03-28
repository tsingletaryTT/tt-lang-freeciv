#!/usr/bin/env python3
"""
TT-Lang FreeCiv Bridge Server
==============================

Persistent Unix socket server that keeps a Tenstorrent P300C Blackhole device
open across all requests, eliminating the JIT warm-up cost from FreeCiv's
perspective.

Protocol (both directions):
  [4-byte little-endian length] [UTF-8 JSON body]

Request types:
  {"cmd": "terrain_gen",   "w": W, "h": H, "seed": S}
  {"cmd": "tile_score",    "w": W, "h": H, "food": [...], "shields": [...], "trade": [...]}
  {"cmd": "terrain_event", "w": W, "h": H, "turn": T}

Responses:
  {"status": "ok", "data": [...], "kernel_ms": 0.07}
  {"status": "ok", "scores": [...], "top_tiles": [i, ...], "kernel_ms": 0.06}
  {"status": "ok", "events": [{"tile": i, "extra": "Gold"}, ...], "kernel_ms": 0.05}
  {"status": "error", "message": "..."}

Usage:
  cd ~/code/tt-lang && source build/env/activate
  python ~/tt-lang-freeciv/bridge/ttlang_server.py
Request types:
  {"cmd": "gen_civs", "n_civs": N, "seed": S, "output_dir": "/path/to/out"}

Responses:
  {"status": "ok", "civs": [{"slug":..., "nation":..., "leader":..., "color_rgb":[r,g,b]}, ...],
   "output_dir": "/path", "kernel_ms": 0.35}
"""

import json
import math
import os
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

# Add kernels and civgen to path
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent))

SOCKET_PATH = "/tmp/ttlang_freeciv.sock"
TILE_SIZE   = 32
WARMUP_RUNS = 2

# Resources that can appear as turn events, in rotation.
# Must be valid extra rule names in the civ2civ3 ruleset.
EVENT_EXTRAS = ["Gold", "Coal", "Pheasant", "Fish"]


# ── Import TT modules ──────────────────────────────────────────────────────────

print("[ttlang-server] Importing TT modules...", flush=True)
import torch
import ttnn
from height_map_simple  import generate_height_map_cpu, scale_height_map, GRANULARITY as _SCALE_GRANULARITY
from height_map_smooth  import smooth_height_map, to_device_half, zeros_like_on_device, GRANULARITY as _SMOOTH_GRANULARITY
from weather_kernel         import compute_weather, sample_weather_to_map, WEATHER_GRID_SIZE
from disaster_kernel        import DisasterModel, DISASTER_GRID_SIZE
from city_influence_kernel  import CityInfluenceModel
from pathfind_kernel        import PathfindModel
from unit_eval_kernel       import UnitEvalModel
from threat_field_kernel    import ThreatFieldModel
from city_prod_kernel       import CityProdModel
from combat_pos_kernel      import CombatPosModel
from storyteller            import Storyteller

print("[ttlang-server] TT modules loaded.", flush=True)

# Import civgen pipeline (optional — graceful if Pillow not installed)
try:
    from civgen.pipeline import generate_civs as _civgen_generate_civs
    CIVGEN_AVAILABLE = True
    print("[ttlang-server] civgen pipeline available.", flush=True)
except ImportError as _civgen_err:
    CIVGEN_AVAILABLE = False
    print(f"[ttlang-server] civgen unavailable ({_civgen_err}), gen_civs disabled.",
          flush=True)


# ── Protocol helpers ───────────────────────────────────────────────────────────

def send_msg(conn: socket.socket, body: dict) -> None:
    """Send a length-prefixed JSON message.
    Uses compact separators (no spaces) so the C hand-rolled JSON parser
    can match keys like "extra":"Gold" and object boundaries },{."""
    data = json.dumps(body, separators=(',', ':')).encode("utf-8")
    conn.sendall(struct.pack("<I", len(data)) + data)


def recv_msg(conn: socket.socket) -> dict | None:
    """Receive a length-prefixed JSON message. Returns None on disconnect."""
    hdr = _recvall(conn, 4)
    if not hdr:
        return None
    (length,) = struct.unpack("<I", hdr)
    body = _recvall(conn, length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _recvall(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ── Tensor sizing helper ───────────────────────────────────────────────────────

def _tile_aligned(n: int) -> int:
    """Round n up to the nearest multiple of TILE_SIZE."""
    return ((n + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def _make_map_tensor(flat_list: list, map_tiles: int, device) -> ttnn.Tensor:
    """
    Pad flat_list to the nearest tile-aligned square, return a 2D device tensor.
    The 'square' dimension is chosen so both dims are tile-aligned.
    """
    side  = max(TILE_SIZE, _tile_aligned(int(math.ceil(math.sqrt(map_tiles)))))
    total = side * side
    padded = flat_list[:total] + [0] * max(0, total - len(flat_list))
    t = torch.tensor(padded, dtype=torch.bfloat16).reshape(side, side)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# ── Kernel handlers ────────────────────────────────────────────────────────────

class TTLangServer:

    def __init__(self):
        self.device          = None
        self.storyteller     = Storyteller()
        self.disaster        = None   # initialised after device is open
        self.city_model      = None   # initialised after device is open
        self.pathfind_model   = None   # initialised after device is open
        self.unit_eval_model  = None   # initialised after device is open
        self.threat_model     = None   # initialised after device is open
        self.city_prod_model  = None   # initialised after device is open
        self.combat_pos_model = None   # initialised after device is open
        self._last_pathfield    = []   # cached for unit_eval to reuse same turn
        self._last_threat_field = []   # cached for combat_pos to reuse same turn

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open_device(self):
        print("[ttlang-server] Opening P300C device 0...", flush=True)
        self.device          = ttnn.open_device(device_id=0)
        self.disaster        = DisasterModel(self.device)
        self.city_model      = CityInfluenceModel(self.device)
        self.pathfind_model   = PathfindModel(self.device)
        self.unit_eval_model  = UnitEvalModel(self.device)
        self.threat_model     = ThreatFieldModel(self.device)
        self.city_prod_model  = CityProdModel(self.device)
        self.combat_pos_model = CombatPosModel(self.device)
        print("[ttlang-server] Device open.", flush=True)

    def close_device(self):
        if self.device is not None:
            ttnn.close_device(self.device)
            self.device = None

    def warm_up(self):
        """Pre-compile all kernels so FreeCiv never pays JIT cost."""
        print("[ttlang-server] Warming up kernels...", flush=True)
        for _ in range(WARMUP_RUNS):
            self._terrain_gen(32, 64, seed=42)
            self._tile_score(32, 64,
                             food=[1.0] * (32 * 64),
                             shields=[1.0] * (32 * 64),
                             trade=[1.0] * (32 * 64),
                             turn=1)  # also warms pathfind inside _tile_score
            self._terrain_event(32, 64, turn=1)
            # Warm up unit_eval with a synthetic settler
            dummy_units = [{'id': 1, 'tile': 0, 'is_settler': 1}]
            dummy_pf    = [0.5] * (32 * 64)
            self.unit_eval_model.evaluate(dummy_units, dummy_pf, 32, 64)
            # Warm up threat_field, city_prod, combat_pos
            dummy_enemy  = [{'tile_idx': 5, 'strength': 1.0}]
            dummy_threat, _ = self.threat_model.compute(dummy_enemy, 32, 64)
            self._last_threat_field = dummy_threat
            dummy_cities = [{'tile_idx': 0, 'food': 2.0, 'shields': 1.0,
                             'trade': 1.0, 'pop': 3.0, 'mil_units_nearby': 0}]
            self.city_prod_model.score(dummy_cities, dummy_threat, 32, 64)
            dummy_own = [{'id': 1, 'tile': 10, 'moves': 2, 'strength': 2.0}]
            self.combat_pos_model.position(dummy_own, dummy_threat, 32, 64)
            # Skip gen_civs warm-up — sprite rendering takes minutes and is
            # only needed on-demand, not at server startup.
        print("[ttlang-server] Warm-up complete. Kernels compiled and cached.",
              flush=True)

    # ── terrain_gen ────────────────────────────────────────────────────────────

    def _terrain_gen(self, w: int, h: int, seed: int) -> tuple[list, float]:
        """
        Generate a height map of size (h × w) using TT hardware.
        Returns (flat_height_list, kernel_ms).
        """
        # Use the largest dimension as the square map size for the kernel.
        # scale_height_map has GRANULARITY=2, so min size = TILE_SIZE*GRANULARITY=64.
        map_size = max(TILE_SIZE * _SCALE_GRANULARITY, _tile_aligned(max(w, h)))
        height_cpu = generate_height_map_cpu(map_size, seed=seed)

        in_t  = ttnn.from_torch(height_cpu, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device)
        out_t = zeros_like_on_device(height_cpu, self.device)

        t0 = time.perf_counter()
        scale_height_map(in_t, out_t)
        kernel_ms = (time.perf_counter() - t0) * 1000

        result = ttnn.to_torch(out_t)  # (map_size × map_size)

        # Sample down to (h × w) using nearest-neighbour
        flat = []
        for row in range(h):
            src_row = row * map_size // h
            for col in range(w):
                src_col = col * map_size // w
                flat.append(int(result[src_row, src_col].item()))

        return flat, kernel_ms

    # ── tile_score ─────────────────────────────────────────────────────────────

    def _tile_score(self, w: int, h: int,
                    food: list, shields: list, trade: list,
                    turn: int = 1,
                    ) -> tuple[list, list, float]:
        """
        Compute score = 2*food + 1.5*shields + 1*trade per tile using TT hardware,
        then blend with weather modifier computed on a 512×512 tensor (64 cores).

        Kernel pipeline (all on P300C):
          Pass 1 (64×64 tile_score):   food_2x + shields_1pt5x → sum1
          Pass 2 (64×64 tile_score):   sum1 + trade → base_score
          Pass 3 (512×512 weather):    temperature + precipitation → raw_climate  (64 cores!)
          Pass 4 (512×512 weather):    raw_climate + turbulence → final_climate   (64 cores!)
          CPU:                         sample climate → weather modifier → blend

        Returns (scores_flat, top_tile_indices, kernel_ms).
        """
        map_tiles = w * h

        # CPU scaling — multiply by the weight * 0.5 so the eltwise-add gives
        # the weighted sum (0.5*a + 0.5*b kernel = (0.5a + 0.5b), so:
        #   food pre-scaled by 2*0.5 = 1.0 (no change)
        #   shields pre-scaled by 1.5*0.5 = 0.75
        #   trade pre-scaled by 1.0 (pass-through, 0.5 applied inside kernel via pre-scale)
        # Actually simpler: scale inputs so the sum equals the desired weighted sum.
        # Pass 1: food_2x = food * 1.0,  shields_1pt5x = shields * 0.75
        #         kernel adds them → result1 = food + 0.75*shields
        # Pass 2: result1_half = result1 * 0.5, trade_half = trade * 0.5
        #         kernel adds → result2 = 0.5*(food + 0.75*shields) + 0.5*trade
        # That's not right either. Let's just use:
        #   score = 2*food + 1.5*shields + trade
        # by doing:
        #   pass1: (food * 2) + (shields * 1.5) — inputs pre-multiplied on CPU, result = sum
        #          Since kernel does a+b, pass (food*2) and (shields*1.5) directly
        #   pass2: pass1 + trade

        # smooth_height_map iterates row_tiles = shape[0] // TILE_SIZE // GRANULARITY.
        # Minimum side must be TILE_SIZE * GRANULARITY (= 64) so row_tiles >= 1.
        min_side = TILE_SIZE * _SMOOTH_GRANULARITY
        side  = max(min_side, _tile_aligned(int(math.ceil(math.sqrt(map_tiles)))))
        total = side * side

        def pad(lst):
            return lst[:total] + [0.0] * max(0, total - len(lst))

        food_t     = torch.tensor(pad(food),    dtype=torch.bfloat16).reshape(side, side)
        shields_t  = torch.tensor(pad(shields), dtype=torch.bfloat16).reshape(side, side)
        trade_t    = torch.tensor(pad(trade),   dtype=torch.bfloat16).reshape(side, side)

        # Pre-scale on CPU.
        # smooth_height_map (eltwise-add) was designed for bfloat16 inputs in
        # [0, 1000].  Terrain yields are in [0, 4], which is below the effective
        # precision range and produces zeros on TT hardware.  Scale inputs up to
        # [0, 1000] range before sending to TT, then divide the output back.
        TT_SCALE = 125.0  # max yield 4 × weight 2 × 125 = 1000, fits in bfloat16
        food_scaled    = (food_t    * 2.0 * TT_SCALE).to(torch.bfloat16)
        shields_scaled = (shields_t * 1.5 * TT_SCALE).to(torch.bfloat16)
        trade_scaled   = (trade_t   * 1.0 * TT_SCALE).to(torch.bfloat16)

        # Pass 1: food_2x + shields_1pt5x → sum1  (both already ×TT_SCALE)
        a1 = ttnn.from_torch(food_scaled,    dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        b1 = ttnn.from_torch(shields_scaled, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        s1 = zeros_like_on_device(food_t, self.device)

        t0 = time.perf_counter()
        smooth_height_map(a1, b1, s1)

        # Pass 2: sum1 + trade_1x → final score  (trade also ×TT_SCALE)
        a2 = ttnn.from_torch(trade_scaled, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        s2 = zeros_like_on_device(food_t, self.device)
        sum1_cpu = ttnn.to_torch(s1)
        a1_for_pass2 = ttnn.from_torch(sum1_cpu, dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=self.device)
        smooth_height_map(a1_for_pass2, a2, s2)
        kernel_ms = (time.perf_counter() - t0) * 1000

        # Divide by TT_SCALE to recover base score values
        base_scores = ttnn.to_torch(s2).reshape(-1).float() / TT_SCALE

        # ── Weather pass: 512×512 → 64 cores (full chip) ─────────────────────
        t_weather = time.perf_counter()
        climate, weather_stats = compute_weather(self.device, turn)
        weather_mods = sample_weather_to_map(climate, w, h)
        ms_weather = (time.perf_counter() - t_weather) * 1000

        # ── Weather blend: score × (1 + weather_modifier) ────────────────────
        # weather_modifier ∈ [-0.4, +0.4]: drought −40%, ideal +40%
        padded_weather = torch.tensor(
            weather_mods + [0.0] * max(0, base_scores.shape[0] - map_tiles),
            dtype=torch.float32
        )
        weather_blended = base_scores * (1.0 + padded_weather)

        # ── Disaster pass: 512×512 → 128 Tensix cores ─────────────────────
        # Disaster modifier ∈ [-0.5, 0.0]: reduces yields in affected zones.
        # Runs the same smooth_height_map kernel as weather, but with a
        # neighbour-shifted copy of the disaster field to spread it spatially.
        t_disaster = time.perf_counter()
        disaster_mods, disaster_stats = self.disaster.update(turn, w, h, seed=42)
        ms_disaster = (time.perf_counter() - t_disaster) * 1000

        padded_disaster = torch.tensor(
            disaster_mods + [0.0] * max(0, base_scores.shape[0] - map_tiles),
            dtype=torch.float32
        )
        final_scores = weather_blended * (1.0 + padded_disaster)

        # ── Pathfinding pass: diffuse attractiveness from top tiles ──────────
        # Uses smooth_height_map — same kernel as city_influence.
        # This makes the tile cache a globally coherent landscape gradient so
        # settlers far from good land are still guided in the right direction.
        # PATHFIND_WEIGHT keeps the boost sub-dominant to base terrain quality.
        PATHFIND_WEIGHT = 1.5   # attractiveness blends as ~15% of max base score
        top10 = torch.topk(final_scores[:map_tiles], min(10, map_tiles)).indices.tolist()
        t_pf  = time.perf_counter()
        pathfield, pf_stats = self.pathfind_model.compute(top10, w, h)
        ms_pathfind = (time.perf_counter() - t_pf) * 1000

        pf_t = torch.tensor(
            pathfield + [0.0] * max(0, final_scores.shape[0] - map_tiles),
            dtype=torch.float32
        )
        final_scores = final_scores + PATHFIND_WEIGHT * pf_t

        # Cache pathfield so unit_eval can reuse it this turn
        self._last_pathfield = pathfield

        kernel_ms += ms_weather + ms_disaster + ms_pathfind

        scores_flat = final_scores[:map_tiles].tolist()

        # Dramatic console output every turn
        dom = weather_stats['dominant'].upper()
        cores     = weather_stats['cores_used']
        ms1, ms2  = weather_stats['ms_pass1'], weather_stats['ms_pass2']
        storm_pos = f"({weather_stats['storm_lon']:.2f},{weather_stats['storm_lat']:.2f})"
        print(f"  [WEATHER  ] {WEATHER_GRID_SIZE}x{WEATHER_GRID_SIZE} grid  "
              f"{cores} Tensix cores  {ms1+ms2:.2f}ms  "
              f">> {dom}  storm@{storm_pos}",
              flush=True)
        if disaster_stats['active']:
            dtype = disaster_stats['type'].upper()
            di    = round(disaster_stats['intensity'], 0)
            da    = disaster_stats['n_affected']
            age   = disaster_stats['turn_age']
            print(f"  [DISASTER ] {DISASTER_GRID_SIZE}x{DISASTER_GRID_SIZE} grid  "
                  f"128 Tensix cores  {ms_disaster:.2f}ms  "
                  f">> {dtype}  intensity={di}  tiles_hit={da}  age={age}t",
                  flush=True)

        # Top-10 tiles by score (within the actual map, not padding)
        top_k = min(10, map_tiles)
        top_tiles = torch.topk(final_scores[:map_tiles], top_k).indices.tolist()

        # Disaster epicenter as a map tile index (for C-side Pollution placement)
        d_active = 1 if disaster_stats['active'] else 0
        d_epi_tile = 0
        if d_active:
            epi_px, epi_py = disaster_stats['epicenter']   # (cx, cy) in 512×512
            from disaster_kernel import DISASTER_GRID_SIZE as _DGS
            ex = epi_px * w // _DGS
            ey = epi_py * h // _DGS
            d_epi_tile = ey * w + ex

        # Bundle both stat dicts for the storyteller
        weather_stats['disaster'] = disaster_stats

        return scores_flat, top_tiles, kernel_ms, weather_stats, d_active, d_epi_tile

    # ── terrain_event ──────────────────────────────────────────────────────────

    def _terrain_event(self, w: int, h: int, turn: int
                       ) -> tuple[list, float]:
        """
        Generate per-tile event intensity for this turn using TT hardware.
        Returns (events_list, kernel_ms) where events_list is [{tile, extra}, ...].

        The kernel computes a sine-wave intensity pattern seeded by turn number.
        Tiles above threshold get a resource bonus.
        """
        map_tiles = w * h
        # scale_height_map needs shape[0] // TILE_SIZE // GRANULARITY >= 1,
        # so minimum side = TILE_SIZE * GRANULARITY = 64.
        side      = max(TILE_SIZE * _SCALE_GRANULARITY,
                        _tile_aligned(int(math.ceil(math.sqrt(map_tiles)))))

        # CPU: generate intensity tensor seeded by turn
        # Different phase each turn so resources appear in different places
        freq_x = 2.0 + (turn % 5) * 0.5
        freq_y = 2.0 + (turn % 7) * 0.4
        phase  = turn * 0.618  # golden ratio → well-distributed phases

        x = torch.linspace(0, 1, side)
        y = torch.linspace(0, 1, side)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        intensity = (
            0.5 * torch.sin(X * math.pi * freq_x + phase) +
            0.5 * torch.sin(Y * math.pi * freq_y + phase * 1.3)
        )
        # Normalize to [0, 1000] to go through scale_height_map
        intensity_scaled = ((intensity + 1.0) / 2.0 * 1000.0).to(torch.bfloat16)

        # Run through TT hardware (scale_height_map passthrough)
        in_t  = ttnn.from_torch(intensity_scaled, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=self.device)
        out_t = zeros_like_on_device(intensity_scaled, self.device)

        t0 = time.perf_counter()
        scale_height_map(in_t, out_t)
        kernel_ms = (time.perf_counter() - t0) * 1000

        result = ttnn.to_torch(out_t).float()  # (side × side), values 0-1000

        # Threshold: top ~15% of tiles get a resource (value > 700).
        # scale_height_map passthrough keeps values in [0, 1000].
        # Lower threshold than the originally planned 900 so small maps (~1000 tiles)
        # still see several events per turn.
        threshold = 700.0
        extra_name = EVENT_EXTRAS[turn % len(EVENT_EXTRAS)]

        events = []
        for tile_idx in range(map_tiles):
            row = tile_idx // w
            col = tile_idx % w
            src_row = row * side // h
            src_col = col * side // w
            val = result[src_row, src_col].item()
            if val > threshold:
                events.append({"tile": tile_idx, "extra": extra_name})

        return events, kernel_ms

    # ── gen_civs ────────────────────────────────────────────────────────────────

    def _gen_civs(
        self,
        n_civs: int,
        seed: int,
        output_dir: str | None,
        silent: bool = False,
    ) -> tuple[list, float]:
        """
        Generate n_civs complete civilizations on P300C Blackhole hardware.

        Calls the full civgen pipeline:
          1. TT: genome generation (64 params per civ)
          2. TT: phoneme matrix generation (26×26 Markov chains per civ)
          3. CPU: name generation via Markov sampling
          4. CPU/PIL: flag, portrait, sprite rendering
          5. CPU: FreeCiv ruleset file writing

        Returns (civ_summary_list, total_kernel_ms) where civ_summary_list
        contains one dict per civ suitable for JSON serialization.
        """
        if not CIVGEN_AVAILABLE:
            raise RuntimeError("civgen package not available — install Pillow")

        out_path = Path(output_dir) if output_dir else Path(f"/tmp/tt_civs_{seed}")

        if not silent:
            print(f"[ttlang-server] gen_civs: {n_civs} civs seed={seed} → {out_path}",
                  flush=True)

        result = _civgen_generate_civs(
            n_civs     = n_civs,
            seed       = seed,
            device     = self.device,
            output_dir = out_path,
            skip_sprites = False,
        )

        # Build a JSON-serializable summary of each civilization
        civ_summaries = []
        for civ in result.civs:
            r, g, b = civ.primary_color_rgb
            civ_summaries.append({
                "index":      civ.index,
                "slug":       civ.slug,
                "leader":     civ.names["leader"],
                "nation":     civ.names["nation"],
                "adjective":  civ.names["adjective"],
                "city1":      civ.names["city1"],
                "city2":      civ.names["city2"],
                "city3":      civ.names["city3"],
                "color_rgb":  [r, g, b],
                "flag_path":  str(civ.flag_path) if civ.flag_path else None,
                "portrait_path": str(civ.portrait_path) if civ.portrait_path else None,
                "ruleset_path":  str(civ.ruleset_path) if civ.ruleset_path else None,
                # Key genome parameters for FreeCiv AI config
                "aggression": round(civ.aggression, 3),
                "expansion":  round(civ.expansion, 3),
                "military":   round(civ.military, 3),
                "science":    round(civ.science, 3),
            })

        # Total time spent in TT hardware kernels
        total_kernel_ms = (
            result.timing.get("tt_genomes", 0.0) +
            result.timing.get("tt_phonemes", 0.0)
        )

        if not silent:
            total_cpu = sum(v for k, v in result.timing.items()
                            if not k.startswith("tt_"))
            print(f"[ttlang-server] gen_civs complete: "
                  f"TT={total_kernel_ms:.3f}ms  CPU={total_cpu:.1f}ms  "
                  f"output={out_path}", flush=True)

        return civ_summaries, total_kernel_ms, str(out_path)

    # ── Request dispatcher ─────────────────────────────────────────────────────

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd", "")
        w   = int(req.get("w", 32))
        h   = int(req.get("h", 64))

        try:
            if cmd == "terrain_gen":
                seed = int(req.get("seed", 42))
                flat, ms = self._terrain_gen(w, h, seed)
                print(f"[ttlang-server] terrain_gen {w}x{h} seed={seed}: {ms:.3f}ms",
                      flush=True)
                return {"status": "ok", "data": flat, "kernel_ms": ms}

            elif cmd == "tile_score":
                food    = req.get("food",    [])
                shields = req.get("shields", [])
                trade   = req.get("trade",   [])
                turn    = int(req.get("turn", 1))
                map_tiles = w * h
                print(f"\n>>> TURN {turn}  P300C Blackhole  {map_tiles} tiles",
                      flush=True)
                scores, top, ms, weather_stats, d_active, d_epi = self._tile_score(
                    w, h, food, shields, trade, turn)
                top3_scores = [round(scores[i], 2) for i in top[:3]]
                print(f"  [SCORING  ] {w}x{h} map  6 TT passes  {ms:.2f}ms total"
                      f"  top3={top3_scores}  (+pathfind)", flush=True)
                # Convert flat tile indices to (col, row) coordinates for narrator
                top_tiles_xy = [(i % w, i // w) for i in top[:5]]
                self.storyteller.narrate_turn(turn, w, h, weather_stats, top_tiles_xy)
                return {"status": "ok", "scores": scores,
                        "top_tiles": top, "kernel_ms": ms,
                        "disaster_active": d_active,
                        "disaster_epi_tile": d_epi}

            elif cmd == "terrain_event":
                turn = int(req.get("turn", 1))
                events, ms = self._terrain_event(w, h, turn)
                extra_name = EVENT_EXTRAS[turn % len(EVENT_EXTRAS)]
                # Store for storyteller banner (tile_score fires next, prints banner)
                self.storyteller.store_events(turn, len(events), extra_name)
                print(f"[ttlang-server] terrain_event {w}x{h} turn={turn}"
                      f" events={len(events)}: {ms:.3f}ms", flush=True)
                return {"status": "ok", "events": events, "kernel_ms": ms}

            elif cmd == "city_influence":
                turn    = int(req.get("turn", 1))
                cities  = req.get("cities", [])
                # cities is [{t: tile_idx, p: player_id}, ...] from C
                cities_norm = [{'tile_idx': c['t'], 'player_id': c['p']}
                               for c in cities]
                mods, stats = self.city_model.compute(cities_norm, w, h)
                n_rival = stats['n_rival']
                ms = stats['kernel_ms']
                print(f"  [TERRITORY] {n_rival} rival cities  "
                      f"128 Tensix cores  {ms:.2f}ms  "
                      f"influence blended into AI scores", flush=True)
                # Pass influence stats to storyteller via turn cache
                self.storyteller.store_influence(turn, stats)
                return {"status": "ok", "influence": mods, "kernel_ms": ms}

            elif cmd == "gen_civs":
                n_civs     = int(req.get("n_civs", 6))
                civ_seed   = int(req.get("seed", 42))
                out_dir    = req.get("output_dir", None)
                civs, tt_ms, out_path = self._gen_civs(n_civs, civ_seed, out_dir)
                print(f"[ttlang-server] gen_civs {n_civs} civs"
                      f" seed={civ_seed}: TT {tt_ms:.3f}ms", flush=True)
                return {
                    "status":     "ok",
                    "civs":       civs,
                    "output_dir": out_path,
                    "kernel_ms":  tt_ms,
                }

            elif cmd == "unit_eval":
                turn   = int(req.get("turn", 1))
                units  = req.get("units", [])
                # Reuse pathfield from the tile_score call that ran this same turn.
                # If tile_score hasn't run yet (unusual), fall back to empty field.
                pathfield = self._last_pathfield
                if not pathfield:
                    print("[ttlang-server] unit_eval: no pathfield cached "
                          "(tile_score must run first)", flush=True)
                    return {"status": "ok", "recommendations": [], "kernel_ms": 0.0}

                recs, ms = self.unit_eval_model.evaluate(units, pathfield, w, h)
                n_settlers = len([u for u in units if u.get('is_settler', 1)])
                print(f"  [UNIT-EVAL] {n_settlers} settlers  "
                      f"128 Tensix cores  {ms:.2f}ms  "
                      f">> {len(recs)} recommendations", flush=True)
                return {"status": "ok", "recommendations": recs, "kernel_ms": ms}

            elif cmd == "city_prod":
                # City production priority scoring.
                # Runs ThreatFieldModel + CityProdModel on TT hardware.
                # Also caches the threat field for combat_pos later this turn.
                turn    = int(req.get("turn", 1))
                cities  = req.get("cities", [])
                enemy   = req.get("enemy",  [])

                # Normalize compact keys sent from C: t→tile_idx, f→food, etc.
                cities_norm = [{
                    'tile_idx':         int(c.get('t',  c.get('tile_idx',  0))),
                    'food':             float(c.get('f', c.get('food',     2.0))),
                    'shields':          float(c.get('s', c.get('shields',  1.0))),
                    'trade':            float(c.get('r', c.get('trade',    1.0))),
                    'pop':              float(c.get('p', c.get('pop',      1.0))),
                    'mil_units_nearby': int(c.get('m',  c.get('mil_units_nearby', 0))),
                } for c in cities]
                enemy_norm = [
                    {'tile_idx': int(e.get('t',   e.get('tile_idx', 0))),
                     'strength': float(e.get('s', e.get('strength', 1.0)))}
                    for e in enemy
                ]

                # Run threat field diffusion; cache for combat_pos this turn
                threat, threat_stats = self.threat_model.compute(enemy_norm, w, h)
                self._last_threat_field = threat

                recs, ms = self.city_prod_model.score(cities_norm, threat, w, h)
                n_mil  = sum(1 for r in recs if r['prod_cat'] == 0 and r['urgency'] > 0.7)
                n_grw  = sum(1 for r in recs if r['prod_cat'] == 1 and r['urgency'] > 0.7)
                n_sci  = sum(1 for r in recs if r['prod_cat'] == 2 and r['urgency'] > 0.7)
                print(f"  [CITY-PROD] {len(cities)} cities  "
                      f"128 Tensix cores  {ms:.2f}ms  "
                      f">> mil={n_mil} grw={n_grw} sci={n_sci} urgent",
                      flush=True)
                return {"status": "ok", "recommendations": recs, "kernel_ms": ms}

            elif cmd == "combat_pos":
                # Combat positioning: advance / hold / retreat orders.
                # Reuses threat field cached from city_prod earlier this turn.
                # Must be called AFTER city_prod in the same turn.
                turn      = int(req.get("turn", 1))
                own_units = req.get("own_units", [])
                threat    = self._last_threat_field
                if not threat:
                    print("[ttlang-server] combat_pos: no threat field cached "
                          "(city_prod must run first)", flush=True)
                    return {"status": "ok", "orders": [], "kernel_ms": 0.0}

                orders, ms = self.combat_pos_model.position(own_units, threat, w, h)
                n_adv  = sum(1 for o in orders if o['action'] == 'advance')
                n_ret  = sum(1 for o in orders if o['action'] == 'retreat')
                n_hold = sum(1 for o in orders if o['action'] == 'hold')
                print(f"  [COMBAT-POS] {len(own_units)} units  "
                      f"128 Tensix cores  {ms:.2f}ms  "
                      f">> adv={n_adv} ret={n_ret} hold={n_hold}",
                      flush=True)
                return {"status": "ok", "orders": orders, "kernel_ms": ms}

            elif cmd == "ping":
                return {"status": "ok", "message": "pong"}

            else:
                return {"status": "error", "message": f"unknown cmd: {cmd}"}

        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}


# ── Main server loop ───────────────────────────────────────────────────────────

def main():
    srv = TTLangServer()
    srv.open_device()

    try:
        srv.warm_up()

        # Remove stale socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ssock:
            ssock.bind(SOCKET_PATH)
            ssock.listen(1)
            os.chmod(SOCKET_PATH, 0o666)

            print(f"\n[ttlang-server] Ready on {SOCKET_PATH}", flush=True)
            print("[ttlang-server] Waiting for FreeCiv to connect...\n", flush=True)

            while True:
                conn, _ = ssock.accept()
                print("[ttlang-server] FreeCiv connected.", flush=True)
                with conn:
                    while True:
                        req = recv_msg(conn)
                        if req is None:
                            print("[ttlang-server] FreeCiv disconnected.", flush=True)
                            break
                        resp = srv.handle(req)
                        send_msg(conn, resp)

    except KeyboardInterrupt:
        print("\n[ttlang-server] Shutting down.", flush=True)
    finally:
        srv.close_device()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    main()
