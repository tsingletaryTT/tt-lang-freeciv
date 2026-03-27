#!/bin/bash
# capture_demo.sh — Record TT-Lang AI FreeCiv demo to video
#
# Launches FreeCiv as observer, waits for the window, captures with ffmpeg.
# Output: /tmp/ttlang_demo_<timestamp>.mp4
#
# Usage: ./capture_demo.sh [duration_seconds]
#   Default duration: 120 seconds (2 minutes)

set -e

BUILD="$HOME/code/freeciv/build_ttlang"
DURATION="${1:-120}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="/tmp/ttlang_demo_${TIMESTAMP}.mp4"
DISPLAY="${DISPLAY:-:0}"

echo "================================================================"
echo "TT-Lang AI Demo Capture"
echo "  Duration:  ${DURATION}s"
echo "  Output:    ${OUTFILE}"
echo "  Display:   ${DISPLAY}"
echo "================================================================"
echo ""

# ── Step 1: Connect FreeCiv as observer ──────────────────────────────────────
echo "[1/3] Launching FreeCiv observer..."
cd "$HOME/code/freeciv"
"$BUILD/freeciv-gtk3.22" \
    --server localhost --port 5556 \
    > /tmp/freeciv-observer.log 2>&1 &
OBSERVER_PID=$!
echo "  Observer PID: $OBSERVER_PID"

# Wait for the FreeCiv window to appear (poll xwininfo)
echo "[2/3] Waiting for FreeCiv window..."
WINDOW_ID=""
for i in $(seq 1 30); do
    sleep 1
    # xwininfo -root -tree prints all windows; grep for Freeciv title
    WINDOW_ID=$(xwininfo -root -tree 2>/dev/null \
        | grep -i '"freeciv\|Freeciv' \
        | grep -v "^\s*0x0 " \
        | head -1 \
        | awk '{print $1}')
    if [ -n "$WINDOW_ID" ]; then
        echo "  Found window: $WINDOW_ID"
        break
    fi
done

if [ -z "$WINDOW_ID" ]; then
    echo "  WARNING: Could not find FreeCiv window by name."
    echo "  Falling back to full-display capture."
    # Get full display resolution
    RESOLUTION=$(xdpyinfo | grep dimensions | awk '{print $2}')
    CAPTURE_SPEC="-video_size ${RESOLUTION} -i ${DISPLAY}.0"
else
    # Get window geometry (position + size)
    GEOM=$(xwininfo -id "$WINDOW_ID" 2>/dev/null \
        | awk '/Absolute upper-left X:/{x=$NF} /Absolute upper-left Y:/{y=$NF} /Width:/{w=$NF} /Height:/{h=$NF} END{print w"x"h"+"x"+"y}')
    echo "  Window geometry: ${GEOM}"
    # ffmpeg x11grab format: WxH+X+Y
    SIZE=$(echo "$GEOM" | cut -d+ -f1)
    OFFSET=$(echo "$GEOM" | cut -d+ -f2-3 | tr '+' ',')
    # Round size down to even numbers (ffmpeg requirement for h264)
    W=$(echo "$SIZE" | cut -dx -f1)
    H=$(echo "$SIZE" | cut -dx -f2)
    W=$(( (W / 2) * 2 ))
    H=$(( (H / 2) * 2 ))
    CAPTURE_SPEC="-video_size ${W}x${H} -i ${DISPLAY}.0+${OFFSET}"
fi

echo ""

# ── Step 3: Record with ffmpeg ────────────────────────────────────────────────
echo "[3/3] Recording ${DURATION}s → ${OUTFILE}"
echo "  Press Ctrl-C to stop early."
echo ""

ffmpeg -loglevel warning \
    -f x11grab \
    -framerate 15 \
    $CAPTURE_SPEC \
    -t "$DURATION" \
    -c:v libx264 \
    -preset fast \
    -crf 23 \
    -pix_fmt yuv420p \
    "$OUTFILE"

echo ""
echo "================================================================"
echo "Recording complete: ${OUTFILE}"
echo "  $(du -h "$OUTFILE" | cut -f1)  $(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$OUTFILE" 2>/dev/null | xargs printf '%.0fs')s"
echo "================================================================"

# Clean up observer
kill "$OBSERVER_PID" 2>/dev/null || true
