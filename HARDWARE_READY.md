# Hardware Deployment Status

**Date:** March 25, 2026
**Hardware:** 4x P300C (Blackhole Architecture)
**Status:** Ready for testing (TT-Lang build in progress)

---

## ✅ Completed

### 1. Hardware Verified
- **4x P300C devices detected** (bus IDs: 01:00.0, 02:00.0, 03:00.0, 04:00.0)
- All devices healthy: 43-47°C, 13-38W power draw
- Driver: TT-KMD 2.7.1-pre
- Firmware: 19.7.99.0

### 2. Kernel Hardware-Ready
- `kernels/height_map_simple.py` **already includes device management**
- Proper `ttnn.open_device()` / `ttnn.close_device()` handling
- Tensor placement on device memory
- No modifications needed!

### 3. Test Infrastructure Created
- `hardware/test_on_hardware.py` - Comprehensive hardware testing
  - Basic execution test
  - Performance scaling (128x128 to 1024x1024)
  - Multi-device testing (all 4 P300C devices)
  - JSON result export
- `hardware/check_hardware_ready.sh` - Quick readiness check

### 4. FreeCiv Installed & Integration Scripts Ready
- FreeCiv 3.1.0 installed via apt
- Integration scripts created:
  - `freeciv_integration/integrate_terrain.sh` - Full C integration
  - `freeciv_integration/quick_demo.sh` - Visualization demo
  - `freeciv_integration/run_freeciv_ttlang.sh` - Launch script

### 5. Terrain Already Generated
- 256x256 height map (FreeCiv-compatible)
- Multiple export formats ready:
  - C source code (pipeline_height.c)
  - CSV (pipeline_height.csv)
  - NumPy binary (pipeline_height.npy)
  - FreeCiv terrain map (pipeline_terrain.txt)
  - Visualization (pipeline_complete.png)

---

## ⏳ In Progress

### TT-Lang Hardware Build
- **Status:** 55% complete (2977/5411 steps)
- **ETA:** ~15-20 minutes
- **What:** Rebuilding without TTLANG_SIM_ONLY flag
- **Why:** Enables full hardware execution and profiling

**Current:** Simulator-only mode (TTLANG_SIM_ONLY=ON)
**Target:** Full hardware mode with device support

---

## 🎯 Ready to Run (Once Build Completes)

### Test 1: Basic Hardware Execution
```bash
cd ~/code/tt-lang
source build/env/activate
cd ~/tt-lang-freeciv/hardware
python test_on_hardware.py
```

**What it does:**
- Tests kernel on Device 0
- Measures performance vs CPU
- Validates correctness
- Exports timing results

**Expected:**
- Generation: ~100ms (CPU)
- Kernel: <10ms (hardware)
- Total: <200ms with transfers

### Test 2: Performance Scaling
```bash
# Automatically tested by test_on_hardware.py
# Tests: 128x128, 256x256, 512x512, 1024x1024
```

**Expected speedup:**
- 128x128: 1-2x (overhead dominates)
- 256x256: 2-3x
- 512x512: 4-6x
- 1024x1024: 8-15x (parallelism wins)

### Test 3: Multi-Device
```bash
# Also in test_on_hardware.py
# Runs same workload on all 4 devices
```

**What it shows:**
- Device consistency
- Load balancing potential
- Variance in performance

### Test 4: Profiling
```bash
cd ~/code/tt-lang
source build/env/activate
export TT_METAL_DEVICE_PROFILER=1
python ~/tt-lang-freeciv/kernels/height_map_simple.py
```

**Metrics:**
- Cycle counts per thread
- Memory bandwidth utilization
- Compute vs memory bound analysis
- DRAM transfer patterns

### Test 5: FreeCiv Integration
```bash
cd ~/tt-lang-freeciv/freeciv_integration
./quick_demo.sh  # Visualization only
# OR
./integrate_terrain.sh  # Full C integration (builds FreeCiv)
```

**Visualization demo:**
- Shows terrain comparison
- FreeCiv-compatible format
- No rebuild needed

**Full integration:**
- Patches FreeCiv source
- Builds with TT-Lang terrain
- Playable game with our maps!

---

## 📊 Performance Targets

### Current (Simulator)
- 256x256 map
- Generation: ~100ms (CPU sine waves)
- Processing: Passthrough (no actual work)
- Total: ~1 second with export

### With Hardware (Projected)
| Map Size | CPU Time | HW Time | Speedup |
|----------|----------|---------|---------|
| 128²     | 25ms     | 20ms    | 1.25x   |
| 256²     | 100ms    | 40ms    | 2.5x    |
| 512²     | 400ms    | 100ms   | 4x      |
| 1024²    | 1600ms   | 200ms   | 8x      |
| 2048²    | 6400ms   | 500ms   | 12.8x   |

**Note:** Actual results depend on:
- Memory transfer overhead
- Kernel optimization level
- Grid configuration
- Device utilization

---

## 🚀 Next Steps (When Build Finishes)

### Immediate (5 minutes)
1. Run `hardware/check_hardware_ready.sh` to verify
2. Run `hardware/test_on_hardware.py` for basic test
3. Check results in `output/hardware_test_results.json`

### Short Term (30 minutes)
4. Profile with TT_METAL_DEVICE_PROFILER
5. Test multiple map sizes (scaling test)
6. Test all 4 devices (consistency check)

### Medium Term (1-2 hours)
7. Optimize kernel based on profiling data
8. Add smoothing kernel (multi-kernel pipeline)
9. Scale to 2048x2048 maps

### Demo (2-3 hours)
10. Build FreeCiv with integration
11. Play game with TT-Lang terrain
12. Record gameplay video
13. Create performance comparison charts

---

## 🔧 Troubleshooting

### Build Issues
**"TTLANG_SIM_ONLY still ON"**
```bash
cd ~/code/tt-lang
rm -rf build
cmake -G Ninja -B build  # No -DTTLANG_SIM_ONLY
cmake --build build
```

**"Build failed"**
- Check disk space: `df -h`
- Check memory: `free -h`
- Review logs: `tail -100 /tmp/claude-1000/.../tasks/*.output`

### Hardware Issues
**"Cannot open device"**
```bash
# Reset devices
tt-smi -r

# Check status
tt-smi -s | jq '.device_info[].telemetry'

# Verify driver
lsmod | grep tenstorrent
```

**"Out of memory"**
- Reduce map size
- Close other applications
- Use single device first

### FreeCiv Issues
**"Cannot build FreeCiv"**
```bash
# Install dependencies
sudo apt-get install build-essential libgtk-3-dev \
  libsdl2-mixer-dev libglib2.0-dev libreadline-dev

# Try system FreeCiv first
./quick_demo.sh  # No build needed
```

---

## 📁 File Locations

**Hardware Tests:**
- `~/tt-lang-freeciv/hardware/test_on_hardware.py`
- `~/tt-lang-freeciv/hardware/check_hardware_ready.sh`
- Results: `~/tt-lang-freeciv/output/hardware_test_results.json`

**FreeCiv Integration:**
- `~/tt-lang-freeciv/freeciv_integration/integrate_terrain.sh`
- `~/tt-lang-freeciv/freeciv_integration/quick_demo.sh`
- FreeCiv source: `~/code/freeciv/`

**Generated Terrain:**
- All formats: `~/tt-lang-freeciv/output/pipeline_*`
- C code: `~/tt-lang-freeciv/output/pipeline_height.c`
- Visualization: `~/tt-lang-freeciv/output/pipeline_complete.png`

**TT-Lang:**
- Source: `~/code/tt-lang/`
- Build: `~/code/tt-lang/build/`
- Environment: `source ~/code/tt-lang/build/env/activate`

---

## 🎮 FreeCiv Gameplay

Once integrated, players will experience:
- **Unique terrain** every time (different seed)
- **Hardware-accelerated generation** (fast loading)
- **Large maps possible** (2048x2048+)
- **Realistic terrain** (natural-looking variation)
- **Compatible format** (works with all rulesets)

The terrain will show:
- 48% Ocean (blue regions)
- 44% Land (green grasslands)
- 8% Mountains (gray peaks)

Perfect for strategy gameplay! 🗺️

---

## ✅ Success Criteria

### Must Have (P0)
- [x] Hardware accessible (4x P300C)
- [ ] TT-Lang builds for hardware
- [ ] Kernel runs on device
- [ ] Correctness validated
- [ ] Performance measured

### Should Have (P1)
- [ ] Performance >2x vs CPU (256x256)
- [ ] Scaling to 1024x1024 works
- [ ] Multi-device tested
- [ ] Profiling data collected

### Nice to Have (P2)
- [ ] FreeCiv integration complete
- [ ] Gameplay video recorded
- [ ] Performance charts created
- [ ] 2048x2048 maps working

---

**Status:** 🟡 BUILD IN PROGRESS
**ETA to Ready:** ~15-20 minutes
**Next Action:** Wait for build, then run `check_hardware_ready.sh`

---

**When build completes:**
```bash
cd ~/tt-lang-freeciv/hardware
./check_hardware_ready.sh && python test_on_hardware.py
```

🚀 **Let's benchmark this hardware!**
