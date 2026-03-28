# Running on Tenstorrent Hardware

## Current Status: Simulator Mode

The project is currently set up for **simulator-only mode**. This means:
- ✅ Kernels are tested and working in simulation
- ✅ All code is hardware-ready (follows TT-Lang patterns)
- ⚠️ Not yet run on actual hardware

## Why Use Simulator Mode?

**Advantages:**
- Fast iteration and development
- No hardware required
- Easy debugging
- Portable (works anywhere)

**When to Move to Hardware:**
- Ready to measure real performance
- Need to scale to large maps (1024x1024+)
- Want to demonstrate hardware acceleration
- Profiling and optimization phase

## How to Deploy on Hardware

### Prerequisites

1. **Tenstorrent hardware available:**
   ```bash
   tt-smi
   # Should show devices: P150, P300C, QB2, etc.
   ```

2. **TT-Lang built for hardware:**
   ```bash
   cd ~/code/tt-lang
   # Remove TTLANG_SIM_ONLY flag
   cmake -G Ninja -B build
   cmake --build build
   source build/env/activate
   ```

### Step 1: Modify Kernel for Hardware

Current kernel (`kernels/height_map_simple.py`) needs device management:

```python
import ttnn
import torch
import ttl

# Add device management
device = ttnn.open_device(device_id=0)

try:
    # Your kernel code here
    # ...

finally:
    ttnn.close_device(device)
```

### Step 2: Run on Hardware

```bash
cd ~/code/tt-lang
source build/env/activate
python ~/tt-lang-freeciv/kernels/height_map_simple.py
```

### Step 3: Profile Performance

```bash
# Enable profiling
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_MID_RUN_DUMP=1

python ~/tt-lang-freeciv/kernels/height_map_simple.py
```

### Step 4: Compare CPU vs Hardware

Measure:
- Generation time (CPU sine waves)
- Kernel execution time (hardware)
- Total pipeline time
- Speedup ratio

Expected results:
- Small maps (256x256): ~2-3x speedup
- Large maps (1024x1024): ~5-10x speedup
- Very large (2048x2048): ~10-20x speedup

## Troubleshooting

### "No device found"
```bash
# Check device status
tt-smi

# Reset device if needed
tt-smi -r

# Check kernel module
lsmod | grep tenstorrent
```

### "Module ttl not found"
```bash
# Make sure you're in TT-Lang environment
cd ~/code/tt-lang
source build/env/activate

# Verify build
ls build/env/bin/
# Should see ttlang-sim and other tools
```

### "Out of memory"
- Reduce map size
- Adjust GRANULARITY parameter
- Use smaller buffer_factor

## Performance Profiling

Use TT-Lang slash commands:

```bash
# Profile kernel performance
/ttl-profile kernels/height_map_simple.py

# Optimize hotspots
/ttl-optimize kernels/height_map_simple.py

# Export to C++ for production
/ttl-export kernels/height_map_simple.py
```

## Scaling Tests

Once on hardware, test different map sizes:

```python
# Test cases
MAP_SIZES = [
    128,    # Tiny (16K tiles)
    256,    # Small (64K tiles) - Current
    512,    # Medium (256K tiles)
    1024,   # Large (1M tiles)
    2048,   # Huge (4M tiles)
]

for size in MAP_SIZES:
    print(f"Testing {size}x{size}...")
    t_start = time.time()
    terrain = generate_terrain(size)
    t_end = time.time()
    print(f"  Time: {t_end - t_start:.3f}s")
```

## Next Steps

1. **Verify hardware access**: `tt-smi`
2. **Rebuild for hardware**: Remove `TTLANG_SIM_ONLY`
3. **Add device management**: Update kernels
4. **Run on hardware**: Test and profile
5. **Scale up**: Try larger maps
6. **Document results**: Speedup measurements

## Current vs Hardware Comparison

| Aspect | Simulator Mode | Hardware Mode |
|--------|---------------|---------------|
| Speed | CPU speed | Accelerated |
| Devices | None needed | Requires TT hardware |
| Development | Fast iteration | Production testing |
| Profiling | Limited | Full metrics |
| Scale | Limited by RAM | Scales well |
| Cost | Free | Requires hardware |

---

**Status:** Ready for hardware deployment when device is available!
