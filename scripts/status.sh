#!/usr/bin/env bash
─
#status.sh — Show RunPod status: containers, GPU, disk, models
#Run from your Mac: bash scripts/status.sh
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
echo "  AuraMarketNet — RunPod Status"
echo "  Host: $RUNPOD_HOST   Port: $RUNPOD_PORT"
echo "═══════════════════════════════════════════════════"

if [[ "$RUNPOD_HOST" == "<FILL_IN>" || "$RUNPOD_PORT" == "<FILL_IN>" ]]; then
  echo "ERROR: Update RUNPOD_HOST and RUNPOD_PORT at the top of this script."
  exit 1
fi

ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "root@$RUNPOD_HOST" << 'REMOTE'

echo ""
echo "── Docker containers─"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "(no containers running)"

echo ""
echo "── GPU status──"
nvidia-smi 2>/dev/null || echo "(no GPU / nvidia-smi not found)"

echo ""
echo "── Disk usage──"
df -h /workspace 2>/dev/null || df -h /
echo ""
du -sh /workspace/AuraMarketNet 2>/dev/null || true

echo ""
echo "── Model checkpoint sizes ─────"
find /workspace/AuraMarketNet/checkpoints -name "*.pt" -o -name "*.pth" 2>/dev/null \
  | xargs -I{} du -sh {} 2>/dev/null \
  || echo "(no .pt files found in checkpoints/)"

echo ""
echo "── App health──"
curl -sf http://localhost:8080/api/model_status 2>/dev/null \
  | python3 -m json.tool 2>/dev/null \
  || echo "(app not responding on port 8080)"

REMOTE

echo ""
echo "═══════════════════════════════════════════════════"
echo "  App URL (SSH tunnel):  http://localhost:8080"
echo "  RunPod proxy URL:      https://mv69k9pagou3nl-8080.proxy.runpod.net"
echo "═══════════════════════════════════════════════════"
