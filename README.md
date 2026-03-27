# TT-Lang FreeCiv Integration

A live demonstration of [TT-Lang](https://github.com/tenstorrent/tt-lang) running on
Tenstorrent P300C Blackhole hardware — using the open-source strategy game
[FreeCiv](https://www.freeciv.org/) as a visual playground.

Every turn, four P300C chips compute the game world in real time:
terrain diffusion, weather simulation, disaster spread, AI tile scoring,
city influence fields, and turn-event narrative — all running on Tensix cores
while you play.

---

## What the hardware does

| Kernel | Grid | Time | What it computes |
|---|---|---|---|
| `scale_height_map` | 8×8 = 64 cores | 0.07 ms | Initial terrain pass-through / scaling |
| `smooth_height_map` | 8×8 = 64 cores | 0.06 ms | 2D box-filter averaging (E-W + N-S passes) |
| `weather_kernel` | 8×8 = 64 cores | 0.09 ms | 512×512 temperature + precipitation field |
| Disaster diffusion | 64 cores × 2 | 0.12 ms | Plague/famine/locusts/eruption spread field |
| City influence | 64 cores × 16 | 9 ms | Territorial reach from 8-pass diffusion |
| Tile scorer | 64 cores × 2 | 0.10 ms | `2×food + 1.5×shield + 1×trade` per tile |

The `smooth_height_map` kernel runs at **204 GB/s effective bandwidth** (71% of P300C peak),
confirmed with the hardware profiler. Eltwise-add compute is only 8% of runtime — the kernel
is memory-bandwidth-bound, which is optimal for this workload.

---

## Architecture

```
FreeCiv Server (C)                  TT-Lang Server (Python)
─────────────────                   ───────────────────────
height_map.c ─────── terrain_gen ──► scale_height_map kernel
srv_main.c ────────── tile_score ──► weather + tile_scorer kernels
srv_main.c ─────── terrain_event ──► terrain_event kernel
srv_main.c ──────── city_influence ► 8-pass diffusion kernel
                                     │
                   Unix socket        └── P300C Blackhole (4 chips)
              /tmp/ttlang_freeciv.sock
```

**Protocol:** 4-byte little-endian length prefix + UTF-8 JSON body (both directions).
The C side gracefully falls back to static terrain if the socket is not present.

---

## Directory structure

```
tt-lang-freeciv/
├── kernels/
│   ├── height_map_simple.py      Procedural terrain generation (sine-wave + noise)
│   ├── height_map_smooth.py      2D box-filter smoothing kernel (used by all diffusion)
│   ├── weather_kernel.py         512×512 climate simulation (temperature + precipitation)
│   ├── terrain_event.py          Per-turn resource event kernel (Gold/Coal/Pheasant/Fish)
│   ├── tile_scorer.py            Weighted tile scoring: 2×food + 1.5×shields + 1×trade
│   ├── disaster_kernel.py        Stateful disaster spread (plague/famine/locusts/eruption)
│   ├── city_influence_kernel.py  8-pass territorial influence diffusion from city list
│   ├── terrain_classify.py       Terrain type classification from height values
│   ├── terrain_colorize.py       RGB colorisation of terrain for debug output
│   ├── civ_genome.py             Civilization personality kernel (colour, traits)
│   └── storyteller.py            Rule-based narrative engine for turn chronicles
│
├── bridge/
│   └── ttlang_server.py          Persistent Unix socket server — keeps device open,
│                                  dispatches terrain_gen / tile_score / terrain_event /
│                                  city_influence / gen_civs commands
│
├── freeciv_integration/
│   ├── run_freeciv_ttlang.sh     Launch script (server + FreeCiv together)
│   └── ttlang_game.serv          FreeCiv server script (map size, AI count, start)
│
├── hardware/
│   └── benchmark_scaling.py     Scaling benchmark: 128² → 2048² tensor sizes
│
├── output/                       Generated files (height maps, benchmark JSON)
├── run_demo.sh                   Full 3-pane tmux demo launcher
└── README.md                     This file
```

**FreeCiv C patches** (in `~/code/freeciv/`):
```
server/generator/height_map.c          USE_TTLANG_TERRAIN block — live terrain gen
server/generator/ttlang_client.c/.h    Unix socket client (connect/request/disconnect)
server/generator/tt_tile_cache.c/.h    Per-tile score cache + influence blending
server/srv_main.c                      Per-turn hooks: tile_score, terrain_event, city_influence
server/generator/meson.build           ttlang_client.c added to build
```

---

## Quick start

### 1. Prerequisites

```bash
# TT-Lang environment
source ~/code/tt-lang/build/env/activate

# FreeCiv built with TT-Lang patches
cd ~/code/freeciv
ninja -C build_ttlang freeciv-server
```

### 2. Full demo (3-pane tmux layout)

```bash
~/tt-lang-freeciv/run_demo.sh
```

This opens a tmux session with:
- **Top left** — TT-Lang server (hardware log, kernel timings)
- **Top right** — FreeCiv server (game log)
- **Bottom** — Live story chronicle (`tail -f /tmp/ttlang_story.log`)

After ~12 s warm-up, the FreeCiv GUI client launches automatically.

### 3. Manual start

```bash
# Terminal 1 — TT server
source ~/code/tt-lang/build/env/activate
python ~/tt-lang-freeciv/bridge/ttlang_server.py

# Terminal 2 — FreeCiv server
cd ~/code/freeciv
build_ttlang/freeciv-server \
  --read ~/tt-lang-freeciv/freeciv_integration/ttlang_game.serv \
  --port 5557 -d v -l /tmp/fc_ttlang.log

# Terminal 3 — Watch the story
tail -f /tmp/ttlang_story.log
```

In the FreeCiv client, connect to `localhost:5557`.
Set the map generator to **RANDOM** before starting the game — only `MAPGEN_RANDOM`
calls the `make_random_hmap` path where TT-Lang terrain is injected.

---

## Kernel walkthrough

### `smooth_height_map` — the core operator

All spatial diffusion (weather, disaster, city influence) is built from one kernel:

```python
# E-W pass: output = damping/2 * field  +  damping/2 * roll(field, 1, dims=1)
f_h  = (field                       * (d / 2.0)).to(torch.bfloat16)
fs_h = (torch.roll(field, 1, dims=1) * (d / 2.0)).to(torch.bfloat16)
smooth_height_map(a_t, b_t, ew_t)   # a_blk + b_blk on Tensix

# N-S pass: output = 0.5 * ew  +  0.5 * roll(ew, 1, dims=0)
smooth_height_map(ew_h_t, ews_h_t, ns_t)
```

Each call does one E-W and one N-S averaging pass on a 512×512 tensor using 64 Tensix cores.
Running N passes covers an ~N-tile diffusion radius.

### `disaster_kernel.py`

Stateful across turns. Seeds a 5×5 hotspot every 20 turns, then diffuses it:

```
Disaster types    damping    max_mod    character
plague            0.90       −40%       slow spread, medium hit
famine            0.85       −35%       wide area, milder
locusts           0.80       −50%       fast, heavy
eruption          0.70       −70%       intense, short-lived
```

### `city_influence_kernel.py`

Recomputed each turn from the city list. 8 diffusion passes = 16 `smooth_height_map`
calls = ~8-tile territorial radius per city. Rival cities: weight 1.0. Own cities: 0.5.
Output ∈ [−0.6, 0] blended into tile scores via `tt_tile_cache_blend_influence()`.

### `storyteller.py`

Rule-based narrative engine. Aggregates data from `terrain_event` and `tile_score`
handlers, then prints a formatted 64-char-wide banner each turn:

```
════════════════════════════════════════════════════════════════
  TURN  21  ·  32×32 World  ·  P300C Computing History
────────────────────────────────────────────────────────────────
  SEASON   High summer — the sun reaches its zenith.
  WEATHER  [DROUGHT ] A great drought grips the continent — 53% of the land bakes.
  CALAMITY [FAMINE  ] The rains have failed for the third year running. The famine takes hold.
  TERRITORY The world grows crowded. TT hardware maps 3 rival spheres of influence.
  HARVEST  Dark veins of coal surface across 12 tiles. Industry beckons.
  AI BIAS  The P300C calculates destiny: settlers march toward the highest-scored territories.
           Top-scored tiles: (14,22)  (18,9)  (7,31)
════════════════════════════════════════════════════════════════
```

Also appended to `/tmp/ttlang_story.log` for a running game chronicle.

---

## Hardware profiler results

Profiled with `TT_METAL_DEVICE_PROFILER=1 TTLANG_AUTO_PROFILE=1` on a 512×512 tensor:

```
Grid:         8×16 = 128 Tensix cores  (full P300C chip, after harvesting)
Duration:     10,363 cycles  ≈ 7.7 μs
Effective BW: 204.9 GB/s  (71% of P300C peak)
DRAM read:    24.6 MB  (2× 512×512 bfloat16 inputs)
DRAM write:   12.3 MB  (1× 512×512 bfloat16 output)
```

Thread hotspots (fraction of cycles in each thread):
- **NCRISC** (read DMA): 86.8% waiting on `tx_a.wait()` — DMA-limited as expected
- **BRISC** (write DMA): 85.2% waiting on `out_dfb.wait()`
- **TRISC_0** (compute): 87.7% waiting on `a_dfb.wait()`; actual `a_blk + b_blk` = **8% of compute cycles**

The kernel is memory-bandwidth-bound at 71% of peak — no structural inefficiency.
The `tx_b` DMA (126 cycle wait) completes while `tx_a` is in flight (2,176 cycles),
confirming correct double-buffered DMA overlap.

To reproduce:
```bash
source ~/code/tt-lang/build/env/activate
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TTLANG_AUTO_PROFILE=1 \
  python /tmp/profile_smooth_512.py
```

---

## Scaling benchmark

Results from `hardware/benchmark_scaling.py` (warm kernels, device kept open):

| Tensor size | Kernel time | Throughput |
|---|---|---|
| 128×128 | 0.064 ms | 0.3 M tiles/s |
| 256×256 | 0.077 ms | 0.8 M tiles/s |
| 512×512 | 0.091 ms | 2.9 M tiles/s |
| 1024×1024 | 0.270 ms | 3.9 M tiles/s |
| 2048×2048 | 0.305 ms | 13.4 M tiles/s |

Cold start (first call, JIT compilation) ≈ 400 ms. All subsequent calls use the
compiled kernel cache. The server pre-warms at startup, so FreeCiv never pays this cost.

---

## Extending the project

### Add a new kernel

1. Write `kernels/my_kernel.py` using `smooth_height_map` as a template
2. Add an import and handler to `bridge/ttlang_server.py`
3. Add a C request call to `ttlang_client.c`
4. Rebuild FreeCiv: `ninja -C ~/code/freeciv/build_ttlang freeciv-server`

### Protocol format

**Request → server:**
```json
{"cmd": "terrain_gen",    "w": 32, "h": 64, "seed": 12345}
{"cmd": "tile_score",     "w": 32, "h": 64, "food": [...], "shields": [...], "trade": [...], "turn": 5}
{"cmd": "terrain_event",  "w": 32, "h": 64, "turn": 42}
{"cmd": "city_influence", "w": 32, "h": 64, "turn": 5, "cities": [{"t": 123, "p": 1}, ...]}
```

**Response ← server:**
```json
{"status": "ok", "data": [505, 372, ...], "kernel_ms": 0.07}
{"status": "ok", "scores": [1.2, 3.4, ...], "top_tiles": [17, 42], "kernel_ms": 0.10}
{"status": "ok", "events": [{"tile": 12, "extra": "Gold"}, ...], "kernel_ms": 0.13}
{"status": "ok", "influence": [-0.3, 0.0, ...], "kernel_ms": 9.9}
```

---

## Requirements

- **Tenstorrent P300C Blackhole** (or compatible) with `tt-smi` and `tt-kmd` installed
- **TT-Lang** built at `~/code/tt-lang/` (`source build/env/activate`)
- **FreeCiv 3.3** built with TT-Lang patches at `~/code/freeciv/build_ttlang/`
- Python packages: `torch`, `ttnn` (from TT-Lang venv)

Device reset if needed: `tt-smi -r`

---

## License

SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
