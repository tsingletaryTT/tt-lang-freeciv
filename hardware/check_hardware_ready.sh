#!/bin/bash
# Quick check if hardware and TT-Lang are ready

echo "======================================================================"
echo "Tenstorrent Hardware Readiness Check"
echo "======================================================================"
echo ""

# Check 1: Devices visible
echo "[1/5] Checking devices with tt-smi..."
if tt-smi -s > /dev/null 2>&1; then
    NUM_DEVICES=$(tt-smi -s | jq '.device_info | length')
    echo "  ✓ Found $NUM_DEVICES P300C devices"
else
    echo "  ✗ tt-smi failed"
    exit 1
fi
echo ""

# Check 2: TT-Lang build
echo "[2/5] Checking TT-Lang build..."
if [ -f ~/code/tt-lang/build/env/activate ]; then
    echo "  ✓ TT-Lang build environment exists"
else
    echo "  ✗ TT-Lang not built"
    exit 1
fi
echo ""

# Check 3: Check if simulator-only
echo "[3/5] Checking build configuration..."
cd ~/code/tt-lang
source build/env/activate
SIM_ONLY=$(grep "TTLANG_SIM_ONLY" build/CMakeCache.txt | cut -d= -f2)
if [ "$SIM_ONLY" = "ON" ]; then
    echo "  ⚠  Build is SIMULATOR-ONLY (needs rebuild for hardware)"
    echo "     Run: cd ~/code/tt-lang && rm -rf build && cmake -G Ninja -B build && cmake --build build"
    exit 1
else
    echo "  ✓ Build configured for hardware"
fi
echo ""

# Check 4: ttnn module
echo "[4/5] Checking ttnn module..."
if python3 -c "import ttnn; print('ttnn version:', ttnn.__version__)" 2>/dev/null; then
    echo "  ✓ ttnn module importable"
else
    echo "  ✗ ttnn module not found"
    echo "     Make sure you're in the TT-Lang environment:"
    echo "     cd ~/code/tt-lang && source build/env/activate"
    exit 1
fi
echo ""

# Check 5: Quick device open test
echo "[5/5] Testing device access..."
python3 << 'EOF'
import ttnn
try:
    device = ttnn.open_device(device_id=0)
    print("  ✓ Successfully opened device 0")
    ttnn.close_device(device)
except Exception as e:
    print(f"  ✗ Failed to open device: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Hardware is READY for TT-Lang execution!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  cd ~/tt-lang-freeciv/hardware"
    echo "  python test_on_hardware.py"
    exit 0
else
    exit 1
fi
