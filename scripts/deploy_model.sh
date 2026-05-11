#!/usr/bin/env bash
─
#deploy_model.sh — Copy a new .pt checkpoint to RunPod and restart
#Usage: bash scripts/deploy_model.sh path/to/aura_market_net_best.pt
#
#UPDATE THESE TWO LINES after every pod restart:
─
RUNPOD_HOST="194.68.***.***" 
RUNPOD_PORT="******"

set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/AuraMarketNet"
REMOTE_CHECKPOINT_DIR="$REMOTE_DIR/checkpoints"

#Validate args─────
MODEL_FILE="${1:-}"
if [[ -z "$MODEL_FILE" ]]; then
  echo "Usage: bash scripts/deploy_model.sh <path-to-model.pt>"
  echo ""
  echo "Example:"
  echo "  bash scripts/deploy_model.sh AuraMarketNet-v1_checkpoints/aura_market_net_best.pt"
  exit 1
fi

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: File not found: $MODEL_FILE"
  exit 1
fi

if [[ "$RUNPOD_HOST" == "<FILL_IN>" || "$RUNPOD_PORT" == "<FILL_IN>" ]]; then
  echo "ERROR: Update RUNPOD_HOST and RUNPOD_PORT at the top of this script."
  exit 1
fi

BASENAME="$(basename "$MODEL_FILE")"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  AuraMarketNet — Deploy Model Checkpoint"
echo "  File:   $MODEL_FILE  ($(du -sh "$MODEL_FILE" | cut -f1))"
echo "  Host:   $RUNPOD_HOST   Port: $RUNPOD_PORT"
echo "  Remote: $REMOTE_CHECKPOINT_DIR/$BASENAME"
echo "═══════════════════════════════════════════════════"
echo ""

#1. Ensure remote checkpoints dir exists 
echo "► Creating remote checkpoints directory..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "root@$RUNPOD_HOST" \
  "mkdir -p $REMOTE_CHECKPOINT_DIR"

#2. SCP model file─
echo "► Copying model file to RunPod..."
scp -i "$SSH_KEY" -P "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "$MODEL_FILE" \
  "root@$RUNPOD_HOST:$REMOTE_CHECKPOINT_DIR/$BASENAME"

echo "✓ Model uploaded."

#3. If destination name differs from best, also link as best ─
if [[ "$BASENAME" != "aura_market_net_best.pt" ]]; then
  echo ""
  read -r -p "► Also set as 'aura_market_net_best.pt' (the active model)? [y/N] " ans
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
      -o StrictHostKeyChecking=no \
      "root@$RUNPOD_HOST" \
      "cp $REMOTE_CHECKPOINT_DIR/$BASENAME $REMOTE_CHECKPOINT_DIR/aura_market_net_best.pt"
    echo "✓ Copied as aura_market_net_best.pt"
  fi
fi

#4. Restart container to load new model ─
echo ""
echo "► Restarting container to load new model..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
  -o StrictHostKeyChecking=no \
  "root@$RUNPOD_HOST" \
  "cd $REMOTE_DIR && docker compose -f docker-compose.gpu.yml restart"

echo ""
echo "✓ Container restarted with new model."
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Model deployed. Check status:"
echo "  bash scripts/status.sh"
echo "═══════════════════════════════════════════════════"
