#!/usr/bin/env bash
─
#start_app.sh — Run ON RunPod to start AuraMarketNet directly
#Usage: cd /workspace/AuraMarketNet && bash scripts/start_app.sh
─
set -euo pipefail

PROJECT_DIR="/workspace/AuraMarketNet"
LOG_FILE="$PROJECT_DIR/logs/app.log"
PID_FILE="$PROJECT_DIR/app.pid"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  AuraMarketNet — Start App (no Docker)"
echo "═══════════════════════════════════════════════════"

cd "$PROJECT_DIR"
mkdir -p logs

#1. GPU check
echo ""
echo "► GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null \
  || echo "  (nvidia-smi not available)"

#2. Check if already running───
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo ""
    echo "✓ App is already running (PID $PID)."
    echo "  Use 'bash scripts/stop.sh' first to restart."
    exit 0
  else
    rm -f "$PID_FILE"
  fi
fi

#3. Install / sync dependencies ──
echo ""
echo "► Installing Python dependencies..."
#Force-reinstall distutils-managed packages that pip can't uninstall normally
pip install -q --ignore-installed blinker
pip install -q -r requirements.txt
echo "✓ Dependencies ready."

#4. Start the app in the background ────
echo ""
echo "► Starting AuraMarketNet on port 8080..."
nohup python api/app.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "✓ App started (PID $(cat $PID_FILE)). Logs: $LOG_FILE"

#5. Wait for app to respond───
echo ""
echo "► Waiting for app to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8080/api/model_status > /dev/null 2>&1; then
    echo "✓ App is responding on port 8080."
    break
  fi
  printf "."
  sleep 2
done
echo ""

#6. Show recent logs─
echo ""
echo "► Last 20 log lines:"
tail -20 "$LOG_FILE" 2>/dev/null || true

#7. Disk usage────
echo ""
echo "► Disk usage:"
df -h /workspace 2>/dev/null || df -h /

#8. Access info────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  App running at: http://localhost:8080"
echo ""
echo "  From your Mac, open an SSH tunnel:"
echo "  ssh -L 8080:localhost:8080 -i ~/.ssh/id_ed25519 -p 32146 root@213.173.108.102 -N"
echo "  Then open: http://localhost:8080"
echo "═══════════════════════════════════════════════════"
