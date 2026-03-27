#!/usr/bin/env bash
# ============================================================================
# TT-Lang FreeCiv Demo Launcher
# ============================================================================
#
# Starts everything in a single tmux session with four panes:
#
#   ┌──────────────────────────┬──────────────────────────┐
#   │  Pane 1 — TT-Lang server │  Pane 3 — FreeCiv server │
#   │  (Python + P300C device) │  (game logic + AI turns) │
#   ├──────────────────────────┴──────────────────────────┤
#   │  Pane 2 — Game Chronicle (full-width, live tail)    │
#   └─────────────────────────────────────────────────────┘
#
# The FreeCiv GUI client is launched as a separate window (not in tmux).
#
# Usage:
#   ./run_demo.sh            # start full demo
#   ./run_demo.sh --stop     # kill all demo processes
#   ./run_demo.sh --logs     # show all three logs in tmux (no new game)
#
# Requirements:
#   - tt-lang built at ~/code/tt-lang/build  (cmake -G Ninja -B build)
#   - freeciv built at ~/code/freeciv/build_ttlang
#   - tmux installed
# ============================================================================

set -euo pipefail

SESSION="tt-freeciv"
TT_SERVER_LOG="/tmp/ttlang_server.log"
FC_SERVER_LOG="/tmp/ttlang_fc_server.log"
STORY_LOG="/tmp/ttlang_story.log"

TTLANG_VENV="$HOME/code/tt-lang/build/env/activate"
TT_SERVER="$HOME/tt-lang-freeciv/bridge/ttlang_server.py"
FC_BUILD="$HOME/code/freeciv/build_ttlang"
FC_SERV_SCRIPT="$HOME/tt-lang-freeciv/freeciv_integration/ttlang_game.serv"
FC_CLIENT="$FC_BUILD/freeciv-gtk3.22"
FC_PORT=5556

# ── Helpers ──────────────────────────────────────────────────────────────────

die() { echo "✗ $*" >&2; exit 1; }

check_prereqs() {
    [ -f "$TTLANG_VENV" ]            || die "TT-Lang venv not found: $TTLANG_VENV"
    [ -f "$FC_BUILD/freeciv-server" ] || die "FreeCiv not built. Run: cd ~/code/freeciv && ninja -C build_ttlang"
    command -v tmux >/dev/null        || die "tmux not installed"
}

stop_demo() {
    echo "Stopping TT-Lang demo..."
    pkill -f ttlang_server.py 2>/dev/null || true
    pkill -f "freeciv-server.*5556"       2>/dev/null || true
    pkill -f "freeciv-gtk3.22"            2>/dev/null || true
    tmux kill-session -t "$SESSION"       2>/dev/null || true
    echo "Stopped."
    exit 0
}

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
    stop_demo
fi

check_prereqs

# Kill any stale demo processes before starting fresh
pkill -f ttlang_server.py 2>/dev/null || true
pkill -f "freeciv-server.*5556"       2>/dev/null || true
sleep 1

# ── Story log: create empty file so tail -f works immediately ────────────────
printf "TT-Lang FreeCiv — Game Chronicle\nStarting...\n" > "$STORY_LOG"

# ── Create tmux session ───────────────────────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Layout strategy: send-keys to the ACTIVE pane right after each split-window,
# since tmux always makes the new pane active.  This avoids pane-index
# renumbering confusion (tmux re-sorts indices by position after each split).

# Split 1: vertical → creates full-width bottom pane (active after split)
tmux split-window -t "$SESSION:1.1" -v -l 12
# Active = bottom pane → story log
tmux send-keys -t "$SESSION" \
    "echo '=== Game Chronicle (live) ===' && sleep 3 && tail -f $STORY_LOG" Enter

# Split 2: horizontal on top-left → creates top-right pane (active after split)
tmux split-window -t "$SESSION:1.1" -h
# Active = top-right pane → FreeCiv server
tmux send-keys -t "$SESSION" \
    "cd ~/code/freeciv && \
     echo '=== FreeCiv Server ===' && \
     $FC_BUILD/freeciv-server \
         --read $FC_SERV_SCRIPT \
         --port $FC_PORT \
         -d v \
         2>&1 | tee $FC_SERVER_LOG" Enter

# Top-left pane (1.1): TT-Lang server — started last so warm-up happens
# while all three panes are already visible
tmux select-pane -t "$SESSION:1.1"
tmux send-keys -t "$SESSION" \
    "echo '=== TT-Lang Python Server ===' && \
     source $TTLANG_VENV && \
     python $TT_SERVER 2>&1 | tee $TT_SERVER_LOG" Enter

# Wait for TT server to open device + warm up (~10s)
echo "Starting TT-Lang server (warm-up ~10s)..."
sleep 12

# ── Launch FreeCiv GUI client in background (not in tmux) ────────────────────
echo "Waiting for FreeCiv server to accept connections..."
sleep 5

if [ -f "$FC_CLIENT" ]; then
    cd ~/code/freeciv
    "$FC_CLIENT" --server localhost --port "$FC_PORT" &
    echo "FreeCiv GUI launched (PID $!)."
else
    echo "⚠  FreeCiv GTK client not found at $FC_CLIENT"
    echo "   Connect manually:  $FC_CLIENT --server localhost --port $FC_PORT"
fi

# ── Attach to tmux session ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TT-Lang FreeCiv Demo — P300C Blackhole Hardware"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Pane layout:"
echo "    Pane 1 top-left  : TT-Lang Python server (P300C compute output)"
echo "    Pane 2 top-right : FreeCiv server log (game state, TT-Lang events)"
echo "    Pane 3 bottom    : Game Chronicle (narrative log, updates each turn)"
echo ""
echo "  In the FreeCiv client:"
echo "    1. Click 'Connect to network game' → localhost:$FC_PORT"
echo "    2. OR just wait — game may auto-start via the .serv script"
echo ""
echo "  Key things to watch each turn:"
echo "    [WEATHER ] — 128 Tensix cores  climate field"
echo "    [DISASTER] — 128 Tensix cores  plague/famine/locusts spread"
echo "    [SCORING ] — tile scores from 4-pass TT kernel pipeline"
echo "    CALAMITY/HARVEST/AI BIAS in the Chronicle pane"
echo ""
echo "  Stop the demo:  ./run_demo.sh --stop"
echo "  Or Ctrl-C then: tmux kill-session -t $SESSION"
echo ""
echo "Attaching to tmux session '$SESSION'..."
echo ""

tmux attach-session -t "$SESSION"
