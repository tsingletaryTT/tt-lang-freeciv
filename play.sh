#!/usr/bin/env bash
# play.sh — Launch TT-Lang FreeCiv demo (human player + 5 AIs)
#
# Usage:  ./play.sh
#   Starts the TT-Lang hardware server and FreeCiv server, then opens
#   the game client.  You will be prompted to pick a civilization and
#   click "Start Game" — that triggers live P300C terrain generation.

set -e
cd "$(dirname "$0")"

FREECIV_DIR="$HOME/code/freeciv"
TTLANG_ENV="$HOME/code/tt-lang/build/env/activate"
SERVER_LOG="/tmp/fc-ttlang.log"
TTLANG_LOG="/tmp/ttlang_server.log"
PORT=5556

# ── 1. Kill any leftover processes ──────────────────────────────────────────
echo "[play.sh] Cleaning up old processes..."
pkill -f ttlang_server.py  2>/dev/null || true
pkill -f freeciv-server    2>/dev/null || true
pkill -f freeciv-gtk       2>/dev/null || true
rm -f /tmp/ttlang_freeciv.sock
sleep 1

# ── 2. Start TT-Lang hardware server ────────────────────────────────────────
echo "[play.sh] Starting TT-Lang server (P300C hardware)..."
# shellcheck disable=SC1090
source "$TTLANG_ENV"

python "$(pwd)/bridge/ttlang_server.py" \
    > "$TTLANG_LOG" 2>&1 &
TTLANG_PID=$!

echo "[play.sh] Waiting for TT kernels to compile and warm up..."
for i in $(seq 1 60); do
    if grep -q "TT-Lang server ready" "$TTLANG_LOG" 2>/dev/null; then
        echo "[play.sh] TT-Lang server ready (${i}s)"
        break
    fi
    if ! kill -0 "$TTLANG_PID" 2>/dev/null; then
        echo "[play.sh] ERROR: TT-Lang server crashed. Log:"
        cat "$TTLANG_LOG"
        exit 1
    fi
    sleep 1
done

# ── 3. Write FreeCiv startup script ─────────────────────────────────────────
# Sets TT-Lang RANDOM generator, leaves 1 human slot (aifill 5),
# and does NOT call 'start' — you start the game from the client.
cat > /tmp/fc-ttlang-setup.serv <<'EOF'
set generator RANDOM
set aifill 0
set timeout 60
set minplayers 1
EOF
# aifill is 0 now so NO AI slots are pre-created.  The human connects as
# player 1 automatically (no "take a slot" step needed).  The server then
# auto-fills remaining slots with AIs when Start is clicked.

# ── 4. Start FreeCiv server ──────────────────────────────────────────────────
echo "[play.sh] Starting FreeCiv server on port $PORT..."
cd "$FREECIV_DIR"
build_ttlang/freeciv-server \
    -d n \
    -l "$SERVER_LOG" \
    --port "$PORT" \
    --read /tmp/fc-ttlang-setup \
    > /tmp/fc-server-stdout.log 2>&1 &
SERVER_PID=$!

# Wait for "accepting connections"
for i in $(seq 1 15); do
    if grep -q "accepting new client connections" /tmp/fc-server-stdout.log 2>/dev/null; then
        echo "[play.sh] FreeCiv server ready (${i}s)"
        break
    fi
    sleep 1
done

# ── 5. Open game client ───────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  TT-Lang FreeCiv Demo"
echo "  Server: localhost:$PORT"
echo ""
echo "  In the client:"
echo "    1. Connect → localhost:$PORT"
echo "    2. You are auto-assigned player 1 — pick your nation"
echo "    3. Click 'Start Game'  (AIs fill the other slots)"
echo ""
echo "  Watch the hardware fire:"
echo "    grep TT-Lang $SERVER_LOG"
echo "    tail -f $TTLANG_LOG"
echo "══════════════════════════════════════════════════════"
echo ""

cd "$FREECIV_DIR"
build_ttlang/freeciv-gtk3.22 &

# ── 6. Follow TT-Lang activity ───────────────────────────────────────────────
echo "[play.sh] TT-Lang hardware activity (Ctrl-C to exit log view):"
echo ""
tail -f "$SERVER_LOG" | grep --line-buffered "TT-Lang"
