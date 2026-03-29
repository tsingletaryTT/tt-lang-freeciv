#!/usr/bin/env bash
# ============================================================================
# TT-Lang FreeCiv Demo Launcher
# ============================================================================
#
# Starts everything in a single tmux session with three panes:
#
#   ┌──────────────────────┬──────────────────────┐
#   │  Pane 1 — TT server  │  Pane 2 — FreeCiv    │
#   │  (P300C compute log) │  (game + AI turns)   │
#   ├──────────────────────┴──────────────────────┤
#   │  Pane 3 — Chronicle  (turn narrative)        │
#   └─────────────────────────────────────────────┘
#
# The FreeCiv GUI client is launched as a separate window (not in tmux).
#
# Usage:
#   ./run_demo.sh            # start full demo
#   ./run_demo.sh --stop     # kill all demo processes
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
    pkill -f ttlang_server.py          2>/dev/null || true
    pkill -f "freeciv-server.*$FC_PORT" 2>/dev/null || true
    pkill -f "freeciv-gtk3.22"         2>/dev/null || true
    tmux kill-session -t "$SESSION"    2>/dev/null || true
    sleep 1
    tt-smi -r 2>/dev/null || true
    echo "Stopped."
    exit 0
}

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
    stop_demo
fi

check_prereqs

# Kill stale processes and reset TT devices before starting fresh.
# Crashed processes leave device handles open without close_device() —
# tt-smi -r clears all chip locks so the new session starts clean.
pkill -f ttlang_server.py          2>/dev/null || true
pkill -f "freeciv-server.*$FC_PORT" 2>/dev/null || true
sleep 1
echo "Resetting TT devices..."
tt-smi -r 2>/dev/null || true
sleep 2

# ── Prep ─────────────────────────────────────────────────────────────────────
printf "TT-Lang FreeCiv — Game Chronicle\nStarting...\n" > "$STORY_LOG"

# ── Create tmux session (3-pane layout) ──────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 52

# Pane 1 (top-left): TT-Lang server — active pane at start
# Pane 2 (top-right): FreeCiv server
# Pane 3 (bottom): Chronicle tail

# Split bottom strip off → Pane 3
tmux split-window -t "$SESSION:1.1" -v -l 10
tmux send-keys -t "$SESSION" \
    "echo '=== Game Chronicle ===' && sleep 3 && tail -f $STORY_LOG" Enter

# Split top row right → Pane 2 (FreeCiv server)
tmux split-window -t "$SESSION:1.1" -h
tmux send-keys -t "$SESSION" \
    "echo '=== FreeCiv Server ===' && \
     cd ~/code/freeciv && \
     $FC_BUILD/freeciv-server \
         --read $FC_SERV_SCRIPT \
         --port $FC_PORT \
         -d v \
         2>&1 | tee $FC_SERVER_LOG" Enter

# Pane 1 (top-left): TT-Lang server
# Source venv as its own command so it runs in the foreground shell;
# chaining it before '&' would background it in a subshell and lose activation.
tmux select-pane -t "$SESSION:1.1"
tmux send-keys -t "$SESSION" "source $TTLANG_VENV" Enter
tmux send-keys -t "$SESSION" \
    "echo '=== TT-Lang Python Server (P300C Blackhole) ===' && \
     python $TT_SERVER 2>&1 | tee $TT_SERVER_LOG" Enter

# ── Wait for warm-up + launch client ─────────────────────────────────────────
echo "Starting TT-Lang server (warm-up ~14s)..."
sleep 17

echo "Launching FreeCiv GUI client..."
if [ -f "$FC_CLIENT" ]; then
    cd ~/code/freeciv
    "$FC_CLIENT" --server localhost --port "$FC_PORT" &
    echo "FreeCiv GUI launched (PID $!)."
else
    echo "⚠  FreeCiv GTK client not found at $FC_CLIENT"
    echo "   Connect manually:  $FC_CLIENT --server localhost --port $FC_PORT"
fi

# ── Attach ────────────────────────────────────────────────────────────────────
cat <<'BANNER'

════════════════════════════════════════════════════════════════
  TT-Lang FreeCiv Demo — P300C Blackhole Hardware
════════════════════════════════════════════════════════════════

  Pane layout (3 panes):
    Top-left    TT-Lang server  — P300C compute timing each turn
    Top-right   FreeCiv server  — game log, TT-Lang event messages
    Bottom      Game Chronicle  — AI-generated narrative per turn

  Visual effects on the map each turn:
    Roads       → top-5 TT-scored land tiles (prime expansion zones)
    Pollution   → disaster epicenter spreading outward
    Gold/Coal   → hills+mountains only   (ecological filter)
    Pheasant    → fertile grassland only
    Fish        → coastal/shallow water tiles only

  FreeCiv client tips:
    Zoom out    Ctrl+scroll  or  -  key
    Full map    F1 (overview map)
    Messages    View → Messagewin — TT hardware notifications appear here
    Press n     Cycle through active units to follow the action

  Stop:  ./run_demo.sh --stop

BANNER

tmux attach-session -t "$SESSION"
