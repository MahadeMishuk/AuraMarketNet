#!/usr/bin/env bash
─
#stop.sh — SSH into RunPod and stop the AuraMarketNet app
#Run from your Mac: bash scripts/stop.sh
#
#UPDATE THESE TWO LINES after every pod restart:
─
RUNPOD_HOST="194.68.***.***" 
RUNPOD_PORT="******"
set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/AuraMarketNet"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  AuraMarketNet — Stopping app on RunPod"
echo "  Host: $RUNPOD_HOST   Port: $RUNPOD_PORT"
echo "═══════════════════════════════════════════════════"

if [[ "$RUNPOD_HOST" == "<FILL_IN>" || "$RUNPOD_PORT" == "<FILL_IN>" ]]; then
  echo "ERROR: Update RUNPOD_HOST and RUNPOD_PORT at the top of this script."
  exit 1
fi

ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "root@$RUNPOD_HOST" \
  "PID_FILE=$REMOTE_DIR/app.pid
   if [[ -f \"\$PID_FILE\" ]]; then
     PID=\$(cat \"\$PID_FILE\")
     if kill -0 \"\$PID\" 2>/dev/null; then
       kill \"\$PID\" && echo \"✓ App stopped (PID \$PID).\"
     else
       echo \"App was not running.\"
     fi
     rm -f \"\$PID_FILE\"
   else
     pkill -f 'python api/app.py' && echo '✓ App stopped.' || echo 'App was not running.'
   fi"

echo ""
echo "  NOTE: /workspace persists after stop — files and model are safe."
echo "  COST REMINDER: Stop the RunPod pod in the dashboard when done."
echo "═══════════════════════════════════════════════════"
