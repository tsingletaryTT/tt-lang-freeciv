# FreeCiv Map Generation Analysis

**Date:** 2026-03-25

## Overview

FreeCiv uses a height-map based approach for procedural terrain generation.

## Key Components

### Height Map System

**File:** `server/generator/height_map.c`

**Core data structure:**
```c
extern int *height_map;           // Array of heights for each tile
#define hmap_max_level 1000       // Maximum height value
extern int hmap_shore_level;      // Threshold: >= shore = land, < shore = ocean
extern int hmap_mountain_level;   // Threshold for mountains/hills
```

**Access macro:**
```c
#define hmap(_tile) (height_map[tile_index(_tile)])
```

### Generation Functions

1. **`make_random_hmap(int smooth)`**
   - Creates uncorrelated random height values
   - Applies smoothing passes to correlate neighbors
   - Basic procedural generation

2. **`make_pseudofractal1_hmap(int extra_div)`**
   - More sophisticated fractal-based generation
   - Creates realistic terrain features

3. **`normalize_hmap_poles()`**
   - Adjusts heights near map edges and poles
   - Prevents too much land at extremes

4. **`height_map_to_map()`**
   - Converts height values to actual terrain types
   - Applies shore_level and mountain_level thresholds

## Terrain Mapping

**Process flow:**
```
1. Generate height_map[] with values [0, 1000]
2. Apply normalization (poles, edges)
3. Map heights to terrain types:
   - height < shore_level → Ocean
   - height >= shore_level → Land
   - height > mountain_level → Mountains/Hills
4. Additional passes: rivers, resources, forests, etc.
```

## TT-Lang Integration Opportunity

### Perfect Target: Height Map Generation

**Why this works:**
- ✅ Simple 2D array output (height per tile)
- ✅ Embarrassingly parallel (each tile independent)
- ✅ Clear input/output contract
- ✅ Easy to validate
- ✅ Measurable performance improvement

**Approach:**
1. Implement parallel noise generation in TT-Lang
2. Output: 2D array of heights [0, 1000]
3. FreeCiv converts heights → terrain (existing code)

### Kernel Design

**Input:**
- Map dimensions (width, height)
- Seed (for reproducibility)
- Smoothing parameters

**Output:**
- 2D array of integers [0, 1000]

**Processing:**
- Generate base noise per tile
- Apply smoothing (neighbor averaging)
- Normalize to [0, 1000] range

**TT-Lang advantages:**
- Parallel tile processing across cores
- Fast smoothing passes with dataflow
- Can process large maps quickly

## Data Structures

**Map representation:**
```c
struct tile {
  struct terrain *terrain;  // Pointer to terrain type
  // ... other fields
};

struct terrain {
  char name[MAX_LEN_NAME];
  // Properties: food, production, etc.
};
```

**Map access:**
- Linear array indexed by tile_index()
- 2D coordinate conversion via map utilities

## Next Steps

1. **Design simple TT-Lang kernel:**
   - Start with basic Perlin/simplex noise
   - Output 2D height array
   - Match FreeCiv's [0, 1000] range

2. **Test in simulator:**
   - Generate small maps (64x64, 128x128)
   - Validate output format
   - Check performance

3. **Python bridge:**
   - Call TT-Lang kernel
   - Convert to NumPy array
   - Integrate with FreeCiv map structure

4. **Visual validation:**
   - Generate height map with kernel
   - Visualize with matplotlib
   - Compare to FreeCiv default

## Integration Points

**Option 1: Replace height map generation**
- Modify `make_random_hmap()` to call our kernel
- Requires C → Python → TT-Lang bridge
- Most invasive but complete integration

**Option 2: Standalone tool**
- Generate height map externally
- Save to file format FreeCiv can load
- Less integration, easier to demo

**Option 3: Python prototype**
- Pure Python bridge demonstrating concept
- Visual output only (no FreeCiv integration yet)
- Fastest path to demo

**Recommendation:** Start with Option 3, iterate to Option 2 if time permits.

## Key Insights

1. **Simple target:** Height map is just integers, no complex data structures
2. **Clear boundary:** Generate heights, let FreeCiv handle terrain conversion
3. **Parallel-friendly:** Each tile can be computed independently
4. **Measurable:** Easy to compare CPU vs TT-Lang performance
5. **Visual:** Height maps are easy to visualize for validation

## References

- Height map: `server/generator/height_map.{c,h}`
- Map utilities: `server/generator/mapgen_utils.{c,h}`
- Terrain types: `common/terrain.h`
- Main generator: `server/generator/mapgen.c`
