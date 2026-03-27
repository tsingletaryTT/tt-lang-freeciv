#!/usr/bin/env bash
# ============================================================================
# TT-Lang FreeCiv Demo Launcher
# ============================================================================
#
# Starts everything in a single tmux session with four panes:
#
#   ┌──────────────────────┬──────────────────────┐
#   │  Pane 1 — TT server  │  Pane 2 — FreeCiv    │
#   │  (P300C compute log) │  (game + AI turns)   │
#   ├──────────────────────┼──────────────────────┤
#   │  Pane 3 — Chronicle  │  Pane 4 — Live Art   │
#   │  (turn narrative)    │  (Gray-Scott on TT)  │
#   └──────────────────────┴──────────────────────┘
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
#   - tmux + feh installed (sudo apt install feh)
# ============================================================================

set -euo pipefail

SESSION="tt-freeciv"
TT_SERVER_LOG="/tmp/ttlang_server.log"
FC_SERVER_LOG="/tmp/ttlang_fc_server.log"
STORY_LOG="/tmp/ttlang_story.log"
ART_DIR="/tmp/tt_art"

TTLANG_VENV="$HOME/code/tt-lang/build/env/activate"
TT_SERVER="$HOME/tt-lang-freeciv/bridge/ttlang_server.py"
REACT_KERNEL="$HOME/tt-lang-freeciv/kernels/react_diffuse.py"
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
    pkill -f ttlang_server.py   2>/dev/null || true
    pkill -f react_diffuse.py   2>/dev/null || true
    pkill -f "freeciv-server.*$FC_PORT"  2>/dev/null || true
    pkill -f "freeciv-gtk3.22"  2>/dev/null || true
    pkill -f "feh.*tt_art"      2>/dev/null || true
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "Stopped."
    exit 0
}

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
    stop_demo
fi

check_prereqs

# Kill stale processes before starting fresh
pkill -f ttlang_server.py 2>/dev/null || true
pkill -f react_diffuse.py 2>/dev/null || true
pkill -f "freeciv-server.*$FC_PORT" 2>/dev/null || true
sleep 1

# ── Prep output directories ───────────────────────────────────────────────────
mkdir -p "$ART_DIR"
printf "TT-Lang FreeCiv — Game Chronicle\nStarting...\n" > "$STORY_LOG"

# ── Create tmux session (4-pane layout) ──────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 52

# Build layout bottom-up so indices stabilise:
#
# Start: [1.1 full screen — becomes top-left]
#
# Step 1: split bottom row off (active = new bottom-full-width)
tmux split-window -t "$SESSION:1.1" -v -l 14
# Active = bottom-left → Pane 3 (chronicle)
tmux send-keys -t "$SESSION" \
    "echo '=== Game Chronicle ===' && sleep 3 && tail -f $STORY_LOG" Enter

# Step 2: split bottom row right → Pane 4 (art viewer)
tmux split-window -t "$SESSION" -h
# Active = bottom-right → art viewer
tmux send-keys -t "$SESSION" \
    "mkdir -p $ART_DIR && echo '=== Gray-Scott Reaction-Diffusion Art ===' && \
     echo 'Waiting for first frame...' && \
     while [ ! -f $ART_DIR/frame_00000.png ]; do sleep 1; done; \
     feh --slideshow-delay 0.08 --zoom fill --no-menus $ART_DIR/" Enter

# Step 3: split top row right → Pane 2 (FreeCiv server)
tmux split-window -t "$SESSION:1.1" -h
# Active = top-right → FreeCiv server
tmux send-keys -t "$SESSION" \
    "echo '=== FreeCiv Server ===' && \
     cd ~/code/freeciv && \
     $FC_BUILD/freeciv-server \
         --read $FC_SERV_SCRIPT \
         --port $FC_PORT \
         -d v \
         2>&1 | tee $FC_SERVER_LOG" Enter

# Step 4: top-left → TT-Lang server (Pane 1)
tmux select-pane -t "$SESSION:1.1"
tmux send-keys -t "$SESSION" \
    "echo '=== TT-Lang Python Server + Gray-Scott Art ===' && \
     source $TTLANG_VENV && \
     python $TT_SERVER 2>&1 | tee $TT_SERVER_LOG &
     sleep 14 && \
     echo '=== Starting Gray-Scott art kernel ===' && \
     python $REACT_KERNEL --preset coral --frames 9999 --steps 12 --out $ART_DIR" Enter

# ── Wait for warm-up + launch client ─────────────────────────────────────────
echo "Starting TT-Lang server (warm-up ~12s)..."
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

  Pane layout (4 panes):
    Top-left    TT-Lang server  — P300C compute timing each turn
    Top-right   FreeCiv server  — game log, TT-Lang event messages
    Bot-left    Game Chronicle  — AI-generated narrative per turn
    Bot-right   Gray-Scott Art  — reaction-diffusion on TT hardware

  Visual effects on the map each turn:
    Roads       → top-5 TT-scored land tiles (prime expansion zones)
    Pollution   → disaster epicenter spreading outward
    Gold/Coal   → hills+mountains only   (ecological filter)
    Pheasant    → fertile grassland only
    Fish        → water tiles only

  FreeCiv client tips:
    Zoom out   Ctrl+scroll  or  -  key
    Full map   F1 (overview map)
    Messages   open Events pane — TT hardware notifications appear here
    Follow AI  View → Show All Activities

  Stop:  ./run_demo.sh --stop

BANNER

tmux attach-session -t "$SESSION"
