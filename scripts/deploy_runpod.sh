#!/usr/bin/env bash
─
#deploy_runpod.sh — Deploy AuraMarketNet to RunPod GPU instance
#
#UPDATE THESE TWO LINES after every pod restart:
#  RunPod dashboard → Connect tab → "SSH over exposed TCP"
─
RUNPOD_HOST="194.68.***.***" 
RUNPOD_PORT="******"

set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/AuraMarketNet"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  AuraMarketNet → RunPod Deployment"
echo "  Host: $RUNPOD_HOST   Port: $RUNPOD_PORT"
echo "  Local:  $LOCAL_DIR"
echo "  Remote: root@$RUNPOD_HOST:$REMOTE_DIR"
echo "═══════════════════════════════════════════════════"
echo ""

#1. Validate config
if [[ "$RUNPOD_HOST" == "<FILL_IN>" || "$RUNPOD_PORT" == "<FILL_IN>" ]]; then
  echo "ERROR: Update RUNPOD_HOST and RUNPOD_PORT at the top of this script."
  echo "       Check RunPod dashboard → Connect tab → SSH over exposed TCP."
  exit 1
fi

#2. Ensure rsync is installed on RunPod 
echo "► Ensuring rsync is installed on RunPod..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "root@$RUNPOD_HOST" \
  "which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)"
echo "✓ rsync ready."
echo ""

#3. Rsync project files──
echo "► Syncing project files to RunPod..."
rsync -rltz --progress --no-owner --no-group --no-perms \
  -e "ssh -i $SSH_KEY -p $RUNPOD_PORT -o StrictHostKeyChecking=no" \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude=".DS_Store" \
  --exclude="data/cache/" \
  --exclude="data/raw/" \
  --exclude="AuraMarketNet-v1_checkpoints/" \
  --exclude="logs/" \
  --exclude="notebooks/" \
  --exclude=".env" \
  "$LOCAL_DIR/" \
  "root@$RUNPOD_HOST:$REMOTE_DIR/"

echo "✓ Files synced."

#3. Install dependencies and start app on RunPod ─
echo ""
echo "► Installing dependencies and starting app on RunPod..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "root@$RUNPOD_HOST" \
  "cd $REMOTE_DIR && bash scripts/start_app.sh"

echo ""
echo "✓ App started."

#4. Print access info
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  To access the dashboard, open an SSH tunnel:"
echo "  ssh -L 8080:localhost:8080 -i $SSH_KEY -p $RUNPOD_PORT root@$RUNPOD_HOST -N"
echo ""
echo "  Then open: http://localhost:8080"
echo ""
echo "  RunPod proxy URL (if TCP port 8080 is exposed):"
echo "  https://mv69k9pagou3nl-8080.proxy.runpod.net"
echo ""
echo "  REMINDER: RUNPOD_HOST and RUNPOD_PORT change on every"
echo "            pod restart — update the top of this script."
echo "═══════════════════════════════════════════════════"
