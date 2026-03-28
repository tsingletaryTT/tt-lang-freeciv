#!/bin/bash
# Test all components of the TT-Lang + FreeCiv project
# Run this to verify everything works in simulator mode

set -e  # Exit on any error

echo "======================================================================"
echo "TT-Lang + FreeCiv - Full Test Suite"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Kernel in Simulator
echo -e "${BLUE}[1/4] Testing kernel in simulator...${NC}"
cd ~/code/tt-lang
source build/env/activate
./bin/ttlang-sim ~/tt-lang-freeciv/kernels/height_map_simple.py > /tmp/test_kernel.log 2>&1
if grep -q "PASSED" /tmp/test_kernel.log; then
    echo -e "${GREEN}✓ Kernel test passed!${NC}"
else
    echo "✗ Kernel test failed! Check /tmp/test_kernel.log"
    exit 1
fi
echo ""

# Test 2: Terrain Classification
echo -e "${BLUE}[2/4] Testing terrain classification...${NC}"
cd ~/tt-lang-freeciv
python kernels/terrain_classify.py > /tmp/test_classify.log 2>&1
if grep -q "Classification complete" /tmp/test_classify.log; then
    echo -e "${GREEN}✓ Terrain classification passed!${NC}"
else
    echo "✗ Classification failed! Check /tmp/test_classify.log"
    exit 1
fi
echo ""

# Test 3: FreeCiv Export
echo -e "${BLUE}[3/4] Testing FreeCiv export bridge...${NC}"
cd ~/tt-lang-freeciv
python bridge/freeciv_export.py > /tmp/test_export.log 2>&1
if grep -q "Export complete" /tmp/test_export.log; then
    echo -e "${GREEN}✓ FreeCiv export passed!${NC}"
else
    echo "✗ Export failed! Check /tmp/test_export.log"
    exit 1
fi
echo ""

# Test 4: Complete Pipeline
echo -e "${BLUE}[4/4] Testing complete pipeline...${NC}"
cd ~/tt-lang-freeciv
python demo/full_pipeline.py > /tmp/test_pipeline.log 2>&1
if grep -q "PIPELINE COMPLETE" /tmp/test_pipeline.log; then
    echo -e "${GREEN}✓ Complete pipeline passed!${NC}"
else
    echo "✗ Pipeline failed! Check /tmp/test_pipeline.log"
    exit 1
fi
echo ""

# Summary
echo "======================================================================"
echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
echo "======================================================================"
echo ""
echo "Generated outputs:"
ls -lh ~/tt-lang-freeciv/output/*.png ~/tt-lang-freeciv/output/*.c ~/tt-lang-freeciv/output/*.json 2>/dev/null | awk '{printf "  %-40s %10s\n", $9, $5}'
echo ""
echo "To view visualization:"
echo "  display ~/tt-lang-freeciv/output/pipeline_complete.png"
echo ""
echo "To inspect terrain map:"
echo "  head -30 ~/tt-lang-freeciv/output/pipeline_terrain.txt"
echo ""
echo "To see statistics:"
echo "  cat ~/tt-lang-freeciv/output/pipeline_metadata.json | jq ."
echo ""
echo "======================================================================"
echo "Project Status: READY FOR HARDWARE DEPLOYMENT! 🚀"
echo "======================================================================"
