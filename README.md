<p align="center">
  <img src="Images/logo.jpg" width="150" alt="Logo">
</p>
AuraMarketNet is a production-grade deep learning system that jointly encodes financial news sentiment and OHLCV time-series data to predict next-day market direction, expected return magnitude, and volatility. Built as a full-stack capstone: model training, fine-tuned sentiment engine, live REST/WebSocket API, professional backtesting, and a real-time quant terminal dashboard.

![Dashboard](Images/dashboard.png)

## Project Overview

Capabilities of AuraMarketNet

---

| Step | Description |
|------|-------------|
| **1 INGEST** | Real-time quotes, news & OHLCV from Yahoo Finance. |
| **2 ANALYZE** | Fine-tuned FinBERT reads every headline. 86.78% accuracy on Financial PhraseBank. |
| **3 PREDICT** | Multi-modal AI fusion (text + price). Direction, expected return, volatility - simultaneously. |
| **4 BACKTEST** | 6-strategy professional engine with Sharpe, Sortino, max drawdown. Commission + slippage modeled. |
| **5 VISUALIZE** | Production dashboard. WebSocket live updates. 4-panel synchronized Plotly charts. |


## API Reference

All endpoints return JSON. The dashboard runs on **port 8080** by default (`PORT` env var overrides).

### Market Data

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/price?ticker=AAPL` | Live quote: price, change, volume, market cap |
| GET | `/api/history?ticker=AAPL&range=1D` | OHLCV arrays for charting (`1D/5D/1W/1M/3M/6M/1Y/MAX`) |
| GET | `/api/ticker_tape` | Live quotes for all tracked tickers |
| GET | `/api/market_status` | US equity market open/closed + session times |
| GET | `/api/market_overview` | S&P 500, NASDAQ, DOW, VIX |
| GET | `/api/sparklines?tickers=AAPL,TSLA` | Last 30 hourly prices per ticker (watchlist) |
| GET | `/api/top_movers?n=6` | Top N gainers and losers |
| GET | `/api/company_info?ticker=AAPL` | Name, sector, description |

### Prediction & Sentiment

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/predict?ticker=AAPL&horizon=1D` | Full AuraMarketNet prediction (1H/1D/1W) |
| POST | `/api/analyze_text` | Body `{"text":"...", "ticker":"AAPL"}` — sentiment + market prediction |
| GET | `/api/sentiment_feed?ticker=AAPL` | FinBERT batch inference on live news |
| GET | `/api/news?ticker=AAPL&limit=20` | News with impact score, breaking flag, keyword highlights |
| POST | `/api/predict-sentiment` | Body `{"text":"..."}` or `{"texts":[...]}` — raw FinBERT inference |
| POST | `/api/predict-market-signal` | Body `{"texts":[...], "weights":[...]}` — aggregate bullish/neutral/bearish signal |
| GET | `/api/model_status` | Model health, checkpoint epoch, val accuracy, sentiment engine |

---

# AuraMarketNet — RunPod User Manual

---

## Every New Session

### Step 1 — Check RunPod Dashboard
Go to **RunPod → your pod → Connect tab** and copy the new **Host/Port** from *SSH over exposed TCP*.

### Step 2 — Update the 4 Scripts
If the pod was restarted, the host/port will have changed. Open each file and update `RUNPOD_HOST` and `RUNPOD_PORT` at the top:

- `scripts/deploy_runpod.sh`
- `scripts/stop.sh`
- `scripts/status.sh`
- `scripts/deploy_model.sh`

### Step 3 — Deploy & Start (from your Mac/Windows)
```bash
bash scripts/deploy_runpod.sh
```

### Step 4 — Open SSH Tunnel
Open a **new terminal tab** and keep it running:
```bash
ssh -L 8080:localhost:8080 -i ~/.ssh/id_ed25519 -p 32146 root@213.173.108.102 -N
```

### Step 5 — Open the Dashboard
http://localhost:8080

### Step 6 — On RunPod
```bash
cd /workspace/AuraMarketNet
```

---

## When You Update Your Code
Syncs changed files, installs new dependencies, and restarts the app:
```bash
bash scripts/deploy_runpod.sh
```

---

## When You Update Your Model Checkpoint
```bash
bash scripts/deploy_model.sh AuraMarketNet-v1_checkpoints/aura_market_net_best.pt
```

---

## Check Status / Troubleshoot
Shows GPU usage, app health, disk usage, and model file sizes:
```bash
bash scripts/status.sh
```

---

## Run the Application on RunPod
```bash
pip install -r /workspace/AuraMarketNet/requirements.txt && python /workspace/AuraMarketNet/api/app.py
```

---

## Training
```bash
python train.py \
  --resume AuraMarketNet-v1_checkpoints/aura_market_net_epoch_002.pt \
  --epochs 30 \
  --batch-size 64
```

---

## Restart App After Code Update (on RunPod)
```bash
pkill -f "python api/app.py" ; TRANSFORMERS_OFFLINE=1 python api/app.py
```

---

## When You're Done for the Day

**Step 1 — Stop the app:**
```bash
bash scripts/stop.sh
```

**Step 2 — Stop the pod:**
Go to the RunPod dashboard and click **Stop** on your pod — otherwise it keeps billing you.

>  **Stop ≠ Terminate**
> `/workspace` on RunPod **persists** when you Stop the pod but is **wiped permanently** if you Terminate it.

---

## Quick Reference

| What | Command |
|------|---------|
| Start / redeploy | `bash scripts/deploy_runpod.sh` |
| Check status | `bash scripts/status.sh` |
| Stop app | `bash scripts/stop.sh` |
| Deploy new model | `bash scripts/deploy_model.sh <path/to/model.pt>` |
| SSH tunnel | `ssh -L 8080:localhost:8080 -i ~/.ssh/id_ed25519 -p 32146 root@213.173.108.102 -N` |
| Dashboard | `http://localhost:8080` |

