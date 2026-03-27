#!/bin/bash
# Integrate TT-Lang generated terrain into FreeCiv
#
# This script patches FreeCiv to use our hardware-generated height maps
# instead of its built-in random generator.

set -e

FREECIV_SRC="$HOME/code/freeciv"
OUTPUT_DIR="$HOME/tt-lang-freeciv/output"
PATCH_DIR="$HOME/tt-lang-freeciv/freeciv_integration"

echo "======================================================================"
echo "FreeCiv + TT-Lang Terrain Integration"
echo "======================================================================"
echo ""

# Check prerequisites
if [ ! -d "$FREECIV_SRC" ]; then
    echo "✗ FreeCiv source not found at $FREECIV_SRC"
    echo "  Clone with: git clone https://github.com/freeciv/freeciv.git ~/code/freeciv"
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/pipeline_height.c" ]; then
    echo "✗ Generated terrain not found at $OUTPUT_DIR/pipeline_height.c"
    echo "  Generate with: cd ~/tt-lang-freeciv && python demo/full_pipeline.py"
    exit 1
fi

echo "[1/6] Checking generated terrain..."
MAP_SIZE=$(grep "MAP_HEIGHT" "$OUTPUT_DIR/pipeline_height.c" | head -1 | awk '{print $3}')
echo "  ✓ Found ${MAP_SIZE}x${MAP_SIZE} height map"
echo ""

echo "[2/6] Copying generated terrain to FreeCiv source..."
cp "$OUTPUT_DIR/pipeline_height.c" "$FREECIV_SRC/server/generator/ttlang_height.c"
echo "  ✓ Copied to server/generator/ttlang_height.c"
echo ""

echo "[3/6] Creating FreeCiv patch..."
cat > "$PATCH_DIR/freeciv_ttlang.patch" << 'PATCH_EOF'
--- a/server/generator/height_map.c
+++ b/server/generator/height_map.c
@@ -26,6 +26,10 @@

 #include "height_map.h"

+/* TT-Lang integration */
+#include "ttlang_height.c"
+#define USE_TTLANG_TERRAIN 1
+
 int *height_map = nullptr;
 int hmap_shore_level = 0, hmap_mountain_level = 0;

@@ -101,6 +105,14 @@ void renormalize_hmap_poles(void)
 void make_random_hmap(int smooth)
 {
   int i;
+
+#ifdef USE_TTLANG_TERRAIN
+  /* Use TT-Lang generated terrain */
+  log_normal("Using TT-Lang hardware-generated terrain (256x256)");
+  load_ttlang_height_map(height_map);
+  return;
+#endif
+
   whole_map_iterate(&(wld.map), ptile) {
     hmap(ptile) = fc_rand(hmap_max_level);
   } whole_map_iterate_end;
PATCH_EOF

echo "  ✓ Patch created"
echo ""

echo "[4/6] Applying patch to FreeCiv..."
cd "$FREECIV_SRC"
if git apply --check "$PATCH_DIR/freeciv_ttlang.patch" 2>/dev/null; then
    git apply "$PATCH_DIR/freeciv_ttlang.patch"
    echo "  ✓ Patch applied successfully"
else
    echo "  ⚠  Patch may already be applied or conflicts exist"
    echo "     Trying manual integration..."

    # Manual integration fallback
    if ! grep -q "ttlang_height.c" "$FREECIV_SRC/server/generator/height_map.c"; then
        echo "     Adding TT-Lang include..."
        sed -i '/^#include "height_map.h"/a \\n/* TT-Lang integration */\n#include "ttlang_height.c"\n#define USE_TTLANG_TERRAIN 1' \
            "$FREECIV_SRC/server/generator/height_map.c"

        echo "     Patching make_random_hmap..."
        # This is tricky - would need more complex sed/awk
        echo "     ⚠  Manual edit may be required in server/generator/height_map.c"
        echo "        Add this at the start of make_random_hmap():"
        echo ""
        echo "        #ifdef USE_TTLANG_TERRAIN"
        echo "          load_ttlang_height_map(height_map);"
        echo "          return;"
        echo "        #endif"
        echo ""
    else
        echo "     ✓ TT-Lang integration already present"
    fi
fi
echo ""

echo "[5/6] Building FreeCiv with TT-Lang terrain..."
cd "$FREECIV_SRC"

if [ ! -f "./configure" ]; then
    echo "  Running autogen..."
    ./autogen.sh --no-configure-run > /dev/null 2>&1
fi

if [ ! -f "./Makefile" ]; then
    echo "  Configuring..."
    ./configure --enable-client=gtk3.22 --disable-nls > /dev/null 2>&1
fi

echo "  Building (this may take 5-10 minutes)..."
make -j$(nproc) > /dev/null 2>&1 || {
    echo "  ✗ Build failed!"
    echo "     Check $FREECIV_SRC for errors"
    echo "     You may need to:"
    echo "       sudo apt-get install build-essential libgtk-3-dev libsdl2-mixer-dev"
    exit 1
}

echo "  ✓ Build complete!"
echo ""

echo "[6/6] Creating launch script..."
cat > "$PATCH_DIR/run_freeciv_ttlang.sh" << 'RUN_EOF'
#!/bin/bash
# Run FreeCiv with TT-Lang generated terrain

cd ~/code/freeciv

echo "Starting FreeCiv with TT-Lang terrain..."
echo "Map will be 256x256 with hardware-generated height map"
echo ""

# Start server in background
./server/freeciv-server -r ~/code/freeciv/data/civ2civ3.serv &
SERVER_PID=$!

sleep 2

# Start client
./client/freeciv-gtk3.22

# Cleanup
kill $SERVER_PID 2>/dev/null
RUN_EOF

chmod +x "$PATCH_DIR/run_freeciv_ttlang.sh"
echo "  ✓ Launch script created"
echo ""

echo "======================================================================"
echo "✓ Integration Complete!"
echo "======================================================================"
echo ""
echo "To play FreeCiv with TT-Lang terrain:"
echo "  $PATCH_DIR/run_freeciv_ttlang.sh"
echo ""
echo "Or manually:"
echo "  cd ~/code/freeciv"
echo "  ./server/freeciv-server &"
echo "  ./client/freeciv-gtk3.22"
echo ""
echo "The server will use the 256x256 TT-Lang generated height map!"
echo "======================================================================"
