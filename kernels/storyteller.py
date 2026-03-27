# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Storyteller — Rule-based narrative engine for TT-driven FreeCiv events.

Generates per-turn "game chronicle" banners from hardware-computed data:
  - Weather events (drought, frost, storm, ideal) from the 512x512 climate kernel
  - Terrain events (Gold, Coal, Pheasant, Fish) from the sine-wave intensity kernel
  - AI tile scoring from the weighted eltwise-add scoring kernel

All template selection is deterministic (seeded by turn + event type), so
narrative is reproducible but varies meaningfully across turns and conditions.

The storyteller is stateless between game sessions; each TTLangServer instance
creates one Storyteller that persists for the session lifetime.
"""

import os

# ── Narrative templates ────────────────────────────────────────────────────────

# Each list has 4 entries so we can index by (turn % 4) for variety.
# Placeholders: {n}=count, {pct}=percent, {lon}/{lat}=storm position.

WEATHER_LINES = {
    'DROUGHT': [
        "The sun blazes mercilessly. Parched earth cracks across {pct}% of the known world.",
        "A great drought grips the continent — {pct}% of the land bakes under relentless heat.",
        "Rivers run dry. Dust storms scour {pct}% of the land.",
        "The rains have not come. {pct}% of all tiles wither under a pitiless sky.",
    ],
    'FROST': [
        "A killing frost descends from the poles, seizing {pct}% of the world in ice.",
        "Winter's grip tightens — {pct}% of the land falls silent under frost.",
        "The cold north wind cuts deep. Settlers brace as {pct}% of the land freezes.",
        "Ice spreads across the lowlands. {pct}% of all tiles sleep beneath a white shroud.",
    ],
    'STORM': [
        "A violent tempest tears across the map. Storms rage over {pct}% of all territory.",
        "Lightning splits the sky! A great storm front covers {pct}% of the known world.",
        "Torrential rains and thunder — the storm at ({lon},{lat}) grows stronger.",
        "The heavens open. {pct}% of all tiles are lashed by wind and rain this turn.",
    ],
    'IDEAL': [
        "Clear skies and gentle rains bless the land. {pct}% of all tiles thrive.",
        "The season turns favorable — mild weather nurtures {pct}% of the continent.",
        "Sun and rain in perfect balance. Farmers rejoice as {pct}% of territory flourishes.",
        "A golden calm settles over the world. {pct}% of tiles enjoy ideal conditions.",
    ],
}

SEASON_DESCS = {
    'summer': "High summer — the sun reaches its zenith.",
    'late_summer': "Late summer — the harvest approaches.",
    'autumn': "The leaves turn gold. Harvest season is upon us.",
    'late_autumn': "The last warmth fades. Winter waits on the horizon.",
    'winter': "Deep winter — frost and short days grip the land.",
    'late_winter': "Winter loosens its grip. The world stirs.",
    'spring': "New growth stirs. Spring breathes life into the world.",
    'late_spring': "Spring advances. Blossoms spread across the hillsides.",
}

RESOURCE_LINES = {
    'Gold': [
        "{n} gold deposits shimmer in the earth, revealed by the shifting terrain.",
        "Prospectors' dreams come true — gold surfaces across {n} tiles this turn.",
        "The land gives up its secrets: {n} gold seams glitter in the light.",
        "Word spreads of gold. {n} tiles are staked by eager claimants.",
    ],
    'Coal': [
        "Dark veins of coal surface across {n} tiles. Industry beckons.",
        "The earth churns, exposing {n} coal deposits across the continent.",
        "Smoke and promise — coal surfaces in {n} locations, fueling future empires.",
        "{n} coal seams break through. The age of industry draws near.",
    ],
    'Pheasant': [
        "Game birds populate {n} tiles. Hunters sharpen their bows.",
        "Pheasants take flight across {n} territories — hunting is excellent this turn.",
        "{n} tiles fill with the calls of pheasant. The hunt begins at dawn.",
        "A great flock settles across {n} tiles. Feathers and arrows will fly.",
    ],
    'Fish': [
        "Shoals of fish rise to {n} coastal and river tiles. Nets await.",
        "The waters teem — {n} tiles grow rich with silver-scaled fish this turn.",
        "A great spawning run fills {n} tiles. Fisher-folk haul in record catches.",
        "The sea gives its bounty freely: {n} tiles blessed with abundant fish.",
    ],
}

AI_LINES = [
    "Hardware intelligence surveys the land — civilizations are drawn toward fertile ground.",
    "The P300C calculates destiny: settlers march toward the highest-scored territories.",
    "Silicon oracles weigh every tile. Leaders heed the hardware's counsel.",
    "Ancient instincts, computed anew: {cores} Tensix cores guide civilization's next steps.",
]

DISASTER_LINES = {
    'plague': [
        "The plague has reached {n} settlements. Healers are powerless against its march.",
        "Dark clouds gather — plague spreads from the east, afflicting {n} territories.",
        "A great pestilence sweeps the continent. {n} tiles lie silent and empty.",
        "The sickness spreads. {n} once-fertile lands are now cursed ground.",
    ],
    'famine': [
        "Crops wither and livestock starve. Famine grips {n} tiles across the world.",
        "The granaries are empty. {n} territories face desperate hunger.",
        "A great famine descends — {n} tiles can no longer sustain life.",
        "The harvest has failed. Refugees flee from {n} starving provinces.",
    ],
    'locusts': [
        "A black cloud of locusts devours everything in its path — {n} tiles stripped bare.",
        "The sky darkens with locusts. {n} territories lose their harvests overnight.",
        "Locusts descend in their millions. {n} tiles are left as dust and silence.",
        "The locust swarm moves west, consuming {n} tiles of crops and pasture.",
    ],
    'eruption': [
        "The volcano erupts! Ash and fire consume {n} tiles near the epicenter.",
        "A catastrophic eruption shakes the earth. {n} tiles are buried under ash.",
        "The mountain speaks with fire. {n} territories tremble under the eruption's shadow.",
        "Molten rock flows across the land. {n} tiles are scorched beyond recovery.",
    ],
}

TERRITORY_LINES = [
    "{n} rival cities cast their shadow across the land. Open territory shrinks.",
    "The world grows crowded. TT hardware maps {n} rival spheres of influence.",
    "Borders tighten as {n} rival powers expand. Settlers must venture further.",
    "The frontier narrows. {n} rival civilizations contest every fertile valley.",
]

TERRITORY_EARLY_LINES = [
    "The land lies open. Settlers may yet claim unclaimed wilderness.",
    "Vast wilderness awaits. No rival influence detected this turn.",
]

DISASTER_SEED_LINES = {
    'plague':   "A new sickness has been reported. The {type} has begun its terrible journey.",
    'famine':   "The rains have failed for the third year running. The {type} takes hold.",
    'locusts':  "The first swarms have been spotted on the horizon. The {type} descends.",
    'eruption': "The ground shakes. Smoke rises from the mountain. The {type} is beginning.",
}

BANNER_WIDTH = 64


# ── Storyteller class ──────────────────────────────────────────────────────────

class Storyteller:
    """
    Maintains per-turn narrative state and prints a game chronicle banner.

    Usage:
        teller = Storyteller()
        # Call when terrain_event result arrives (stored for later)
        teller.store_events(turn, n_events, extra_name)
        # Call when tile_score result arrives (triggers full banner print)
        teller.narrate_turn(turn, map_w, map_h, weather_stats, top_tiles_xy)
    """

    LOG_PATH = "/tmp/ttlang_story.log"

    def __init__(self):
        # Turn-keyed caches so earlier handlers' data is available when tile_score fires
        self._events_cache:    dict[int, tuple[int, str]] = {}
        self._influence_cache: dict[int, dict] = {}
        # Wipe log on startup
        with open(self.LOG_PATH, 'w') as f:
            f.write("TT-Lang FreeCiv Storyteller — Game Chronicle\n")
            f.write("Powered by P300C Blackhole hardware (4× chips)\n")
            f.write("=" * BANNER_WIDTH + "\n\n")

    # ── State management ───────────────────────────────────────────────────────

    def store_events(self, turn: int, n_events: int, extra_name: str) -> None:
        """Store terrain event data so narrate_turn can include it."""
        self._events_cache[turn] = (n_events, extra_name)

    def store_influence(self, turn: int, stats: dict) -> None:
        """Store city influence stats so narrate_turn can include them."""
        self._influence_cache[turn] = stats

    # ── Main narrative entry point ─────────────────────────────────────────────

    def narrate_turn(self, turn: int, map_w: int, map_h: int,
                     weather_stats: dict, top_tiles_xy: list) -> None:
        """
        Print a full turn banner to stdout and append to the story log.

        Called from tile_score handler (which fires after terrain_event).
        Reads stored terrain event data for this turn from the cache.
        """
        n_events, extra_name = self._events_cache.get(turn, (0, "Gold"))
        cores = weather_stats.get('cores_used', 64)

        season_desc    = self._season_desc(weather_stats['season_phase'])
        weather_line   = self._weather_line(turn, weather_stats)
        resource_line  = self._resource_line(turn, n_events, extra_name)
        ai_line        = self._ai_line(turn, cores)
        disaster_stats  = weather_stats.get('disaster', {})
        disaster_line   = self._disaster_line(turn, disaster_stats)
        influence_stats = self._influence_cache.get(turn, {})
        territory_line  = self._territory_line(turn, influence_stats)

        dom = weather_stats['dominant'].upper()

        border  = "═" * BANNER_WIDTH
        thin    = "─" * BANNER_WIDTH

        top3_str = "  ".join(f"({x},{y})" for x, y in top_tiles_xy[:3]) if top_tiles_xy else "—"

        lines = [
            "",
            border,
            f"  TURN {turn:>3}  ·  {map_w}×{map_h} World  ·  P300C Computing History",
            thin,
            f"  SEASON   {season_desc}",
            f"  WEATHER  [{dom:<8}] {weather_line}",
        ]
        if disaster_line:
            dtype = (disaster_stats.get('type') or '').upper()
            lines.append(f"  CALAMITY [{dtype:<8}] {disaster_line}")
        if territory_line:
            lines.append(f"  TERRITORY {territory_line}")
        if resource_line:
            lines.append(f"  HARVEST  {resource_line}")
        lines += [
            f"  AI BIAS  {ai_line}",
            f"           Top-scored tiles: {top3_str}",
            border,
            "",
        ]

        text = "\n".join(lines)
        print(text, flush=True)

        with open(self.LOG_PATH, 'a') as f:
            f.write(text + "\n")

    # ── Template selection helpers ─────────────────────────────────────────────

    @staticmethod
    def _pick(templates: list, seed: int) -> str:
        """Deterministically pick a template string by seed."""
        return templates[seed % len(templates)]

    def _season_desc(self, phase: float) -> str:
        """Map season_phase (sin value, -1..+1) to a descriptive label."""
        if   phase >  0.85: key = 'summer'
        elif phase >  0.35: key = 'late_summer'
        elif phase >  0.00: key = 'spring'
        elif phase > -0.35: key = 'late_spring'
        elif phase > -0.70: key = 'autumn'
        elif phase > -0.85: key = 'late_autumn'
        else:               key = 'winter'
        return SEASON_DESCS[key]

    def _weather_line(self, turn: int, stats: dict) -> str:
        dom = stats['dominant'].upper()
        n   = stats.get(f'n_{stats["dominant"]}', 0)
        total = sum(stats.get(f'n_{k}', 0)
                    for k in ('drought', 'frost', 'storm', 'ideal'))
        pct = round(n / max(total, 1) * 100)
        lon = f"{stats.get('storm_lon', 0.5):.2f}"
        lat = f"{stats.get('storm_lat', 0.5):.2f}"
        tmpl = self._pick(WEATHER_LINES[dom], turn)
        return tmpl.format(n=n, pct=pct, lon=lon, lat=lat)

    def _resource_line(self, turn: int, n_events: int, extra_name: str) -> str | None:
        if n_events == 0:
            return None
        templates = RESOURCE_LINES.get(extra_name, ["{n} resource deposits discovered."])
        return self._pick(templates, turn).format(n=n_events)

    def _ai_line(self, turn: int, cores: int) -> str:
        return self._pick(AI_LINES, turn).format(cores=cores)

    def _territory_line(self, turn: int, stats: dict) -> str | None:
        if not stats:
            return None
        n_rival = stats.get('n_rival', 0)
        if n_rival == 0:
            return self._pick(TERRITORY_EARLY_LINES, turn)
        return self._pick(TERRITORY_LINES, turn).format(n=n_rival)

    def _disaster_line(self, turn: int, stats: dict) -> str | None:
        if not stats.get('active'):
            return None
        dtype  = stats.get('type', 'plague')
        n      = stats.get('n_affected', 0)
        age    = stats.get('turn_age', 0)
        # On turn 1 of a disaster (seeding turn), use the intro line
        if age <= 1:
            return DISASTER_SEED_LINES.get(dtype, "A new calamity begins.").format(type=dtype)
        templates = DISASTER_LINES.get(dtype, ["{n} tiles are affected by disaster."])
        return self._pick(templates, turn).format(n=n)
