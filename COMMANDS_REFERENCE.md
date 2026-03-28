# Working Commands for Local Simulator Setup

**Current Setup:** TT-Lang simulator-only mode (no hardware)

## ❌ What Doesn't Work (Yet)

The slash commands (`/ttl-profile`, `/ttl-export`, etc.) are designed for:
- Remote hardware execution
- SSH/Docker setups
- Real Tenstorrent devices

They **won't work** in our local simulator-only setup without hardware.

## ✅ What DOES Work

### 1. Test Kernels in Simulator

```bash
cd ~/code/tt-lang
source build/env/activate
./bin/ttlang-sim ~/tt-lang-freeciv/kernels/height_map_simple.py
```

**Output:**
```
✓ Height map generated!
  Size: 256x256
  Min:  227.0
  Max:  672.0
  Mean: 500.0
  ✓ PASSED!
```

### 2. Run Complete Pipeline

```bash
cd ~/tt-lang-freeciv
python demo/full_pipeline.py
```

**Generates:**
- Visualizations (pipeline_complete.png)
- FreeCiv exports (C code, CSV, terrain maps)
- Statistics and metadata

### 3. Test Individual Components

**Terrain Classification:**
```bash
cd ~/tt-lang-freeciv
python kernels/terrain_classify.py
```

**FreeCiv Export:**
```bash
cd ~/tt-lang-freeciv
python bridge/freeciv_export.py
```

**Visualization:**
```bash
cd ~/tt-lang-freeciv
python demo/visualize_standalone.py
```

### 4. View Generated Files

```bash
# List all outputs
ls -lh ~/tt-lang-freeciv/output/

# View visualization
display ~/tt-lang-freeciv/output/pipeline_complete.png

# Inspect terrain map (text file)
head -30 ~/tt-lang-freeciv/output/pipeline_terrain.txt

# Check statistics
cat ~/tt-lang-freeciv/output/pipeline_metadata.json

# View C integration code
head -50 ~/tt-lang-freeciv/output/pipeline_height.c
```

### 5. Explore FreeCiv Integration

```bash
# View FreeCiv's height map implementation
cat ~/code/freeciv/server/generator/height_map.h
cat ~/code/freeciv/server/generator/height_map.c | grep -A 20 "make_random_hmap"

# Compare our format
head ~/tt-lang-freeciv/output/pipeline_height.c
```

## 🔧 Common Issues

### "ModuleNotFoundError: No module named 'ttl'"

**Cause:** Running Python directly instead of through `ttlang-sim`

**Solution:** Use the simulator:
```bash
cd ~/code/tt-lang
source build/env/activate
./bin/ttlang-sim /path/to/kernel.py
```

**Or:** Our code gracefully falls back to CPU (this is intentional!)

### "ModuleNotFoundError: No module named 'ttnn'"

**Cause:** Not in TT-Lang environment

**Solution:**
```bash
cd ~/code/tt-lang
source build/env/activate
# Now Python knows about ttnn
```

### Slash Commands Not Working

**Cause:** Designed for remote hardware, we're local simulator

**Solution:** Use manual commands above, or wait for hardware access

## 📊 Performance Testing

### Current (Simulator)

```bash
cd ~/code/tt-lang
source build/env/activate
time ./bin/ttlang-sim ~/tt-lang-freeciv/kernels/height_map_simple.py
```

### With Hardware (When Available)

```bash
cd ~/code/tt-lang
# Rebuild for hardware
cmake -G Ninja -B build  # Remove -DTTLANG_SIM_ONLY=ON
cmake --build build
source build/env/activate

# Test on hardware
tt-smi  # Check device
python ~/tt-lang-freeciv/kernels/height_map_simple.py
```

## 🎯 Quick Tests

### Test Everything Works

```bash
#!/bin/bash
cd ~/tt-lang-freeciv

echo "Testing kernel in simulator..."
cd ~/code/tt-lang
source build/env/activate
./bin/ttlang-sim ~/tt-lang-freeciv/kernels/height_map_simple.py

echo "Testing terrain classification..."
cd ~/tt-lang-freeciv
python kernels/terrain_classify.py

echo "Testing complete pipeline..."
python demo/full_pipeline.py

echo "✓ All tests passed!"
ls -lh output/
```

Save as `test_all.sh` and run:
```bash
chmod +x test_all.sh
./test_all.sh
```

## 📁 File Locations

**TT-Lang Build:**
- Source: `~/code/tt-lang/`
- Build: `~/code/tt-lang/build/`
- Simulator: `~/code/tt-lang/build/env/bin/ttlang-sim`
- Environment: `source ~/code/tt-lang/build/env/activate`

**Our Project:**
- Root: `~/tt-lang-freeciv/`
- Kernels: `~/tt-lang-freeciv/kernels/`
- Bridge: `~/tt-lang-freeciv/bridge/`
- Demos: `~/tt-lang-freeciv/demo/`
- Outputs: `~/tt-lang-freeciv/output/`

**FreeCiv:**
- Root: `~/code/freeciv/`
- Map Generation: `~/code/freeciv/server/generator/`
- Scenarios: `~/code/freeciv/data/scenarios/`

## 🚀 When You Get Hardware Access

Then the slash commands will work! Here's the setup:

1. **Configure remote.conf:**
   ```bash
   vim ~/.claude/commands/tools/remote.conf
   # Set REMOTE_SHELL to your SSH command
   ```

2. **Run smoke test:**
   ```bash
   ~/.claude/commands/tools/smoke-test.sh
   ```

3. **Use slash commands:**
   - `/ttl-profile kernels/height_map_simple.py`
   - `/ttl-export kernels/height_map_simple.py`
   - `/ttl-optimize kernels/height_map_simple.py`

But for now, our manual workflow is perfect for simulator development!

## 📝 Summary

**What Works Now:**
✅ Simulator testing (`ttlang-sim`)
✅ Complete pipeline (`demo/full_pipeline.py`)
✅ All visualizations
✅ FreeCiv exports (C code, CSV, terrain maps)
✅ Terrain classification
✅ Documentation and demos

**What Needs Hardware:**
⏳ Slash commands (`/ttl-profile`, `/ttl-export`)
⏳ Performance profiling
⏳ Large-scale testing (1024x1024+ maps)
⏳ Real speedup measurements

**But we have everything we need for the demo!** 🎉
