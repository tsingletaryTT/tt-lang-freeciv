#!/bin/bash
# Launch FreeCiv with TT-Lang hardware-generated terrain
#
# This build uses a 256x256 height map produced by TT-Lang kernels running
# on 4x Tenstorrent P300C Blackhole devices.

set -e

BUILD="$HOME/code/freeciv/build_ttlang"
DATA="$HOME/code/freeciv/data"

if [ ! -f "$BUILD/freeciv-server" ]; then
    echo "✗ FreeCiv not built. Run build first:"
    echo "  cd ~/code/freeciv && ninja -C build_ttlang"
    exit 1
fi

echo "======================================================================"
echo "FreeCiv with TT-Lang Terrain (P300C Blackhole hardware)"
echo "======================================================================"
echo ""
echo "Map type:   Custom 256x256 height map"
echo "Generated:  TT-Lang kernels on 4x P300C Blackhole"
echo "Kernel:     0.54-0.64ms per device | 78M tiles/second"
echo ""

# Try to find a good ruleset
RULESET=""
if [ -d "$DATA/civ2civ3" ]; then
    RULESET="civ2civ3"
elif [ -d "$DATA/classic" ]; then
    RULESET="classic"
fi

# Start server
echo "Starting FreeCiv server..."
# Must run from source dir so the server finds data/ rulesets.
# generator=RANDOM routes through our patched make_random_hmap (TT terrain).
# --read loads ttlang_game.serv which sets aifill=5 (5 AI opponents) and
# configures map/timeout so AI turns are fast enough to watch.
SERV_SCRIPT="$HOME/tt-lang-freeciv/freeciv_integration/ttlang_game.serv"
cd ~/code/freeciv
"$BUILD/freeciv-server" \
    -l /tmp/freeciv-ttlang-server.log \
    --port 5556 \
    --read "$SERV_SCRIPT" &
SERVER_PID=$!

echo "  Server PID: $SERVER_PID"
echo "  Log: /tmp/freeciv-ttlang-server.log"
sleep 2

# Verify server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "✗ Server failed to start. Check /tmp/freeciv-ttlang-server.log"
    exit 1
fi

echo "  ✓ Server running"
echo ""
echo "Starting FreeCiv client (connect to localhost:5556)..."

"$BUILD/freeciv-gtk3.22" &
CLIENT_PID=$!

echo "  Client PID: $CLIENT_PID"
echo ""
echo "======================================================================"
echo "FreeCiv + TT-Lang AI is running!"
echo ""
echo "  In the client: Connect → New Game → Start"
echo "  5 AI opponents will play, driven each turn by P300C hardware scoring"
echo ""
echo "  Watch server log live:"
echo "    tail -f /tmp/freeciv-ttlang-server.log | grep TT-Lang"
echo ""
echo "  Per-turn you will see:"
echo "    TT-Lang: turn N AI scoring — top tile #1 at (X,Y) score=S.S"
echo "    TT-Lang: turn N terrain events: K tiles updated"
echo "======================================================================"
echo ""
echo "Press Ctrl-C to stop both server and client."
echo ""

# Wait for client to exit, then clean up server
wait $CLIENT_PID 2>/dev/null || true
echo ""
echo "Client exited. Stopping server..."
kill $SERVER_PID 2>/dev/null || true
echo "Done."
