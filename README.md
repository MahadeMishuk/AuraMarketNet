#AuraMarketNet
**Real-Time Multi-Modal Sentiment Analysis**

A production-grade deep learning system that jointly encodes financial news sentiment and OHLCV time-series data to predict next-day market direction, expected return magnitude, and volatility. Built as a full-stack capstone: model training, fine-tuned sentiment engine, live REST/WebSocket API, professional backtesting, and a real-time quant terminal dashboard.

---

##Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        AURAMARKETNET v1                          ║
║                                                                  ║
║   Financial Text              Numerical Time-Series              ║
║   (News Headlines)            (OHLCV + 20 TA Features)           ║
║         │                             │                          ║
║         ▼                             ▼                          ║
║  ┌─┐       ┌┐                ║
║  │  MultiText      │       │  Bi-LSTM (3 layers)│                ║
║  │  Encoder        │       │  + Self-Attention  │                ║
║  │  (FinBERT +     │       │  + Temporal Pooling│                ║
║  │   Attn Pooling) │       │  (30-day window)   │                ║
║  └──┬──┘       └───┬┘                ║
║           │  text_emb [256]          │  num_emb [256]            ║
║           └────┬─┘                           ║
║                          ▼                                       ║
║             ┌────┐                           ║
║             │  Cross-Attention Fusion│                           ║
║             │  num queries text  ←→  │                           ║
║             │  text queries num      │                           ║
║             │  + Learned Gate        │                           ║
║             └─┬──┘                           ║
║                         │  fused_emb [512]                       ║
║              ┌┼───┐                          ║
║              ▼          ▼             ▼                          ║
║        [UP / DOWN]  [Δ% return]  [volatility]                    ║
║         softmax      Huber         MSE                           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

##Results

###FinBERT Sentiment Model (fine-tuned on Financial PhraseBank)

| Metric | Value |
|---|---|
| Test Accuracy | **87.40%** |
| Test Macro F1 | **86.69** |
| F1 — Negative | 0.87 |
| F1 — Neutral | 0.89 |
| F1 — Positive | 0.84 |
| Dataset | lmassaron/FinancialPhraseBank (~4,840 sentences) |
| Training | AdamW, cosine schedule, FP16, label smoothing 0.1 |

---

##Features

###Model & Training
- **Dual-stream encoder**: FinBERT for text + Bi-LSTM + Self-Attention for 30-day OHLCV sequences
- **Cross-attention fusion**: bidirectional — numerical queries text and text queries numerical; learned sigmoid gate balances modality contributions
- **Multi-task learning**: jointly trains direction (CE), return (Huber), volatility (MSE), and auxiliary sentiment (CE)
- **FinBERT sentiment pre-training**: standalone `train_sentiment.py` fine-tunes FinBERT on Financial PhraseBank before full model training
- **Separate learning rates**: 1e-5 for BERT backbone, 2e-5 for all other parameters (prevents catastrophic forgetting)
- **FP16 mixed precision** with gradient accumulation (effective batch 64)
- **Resume from checkpoint**: restores model + optimizer + LR scheduler state

###Inference Pipeline
- **Singleton model loading**: both sentiment and market models load once at startup, thread-safe, CUDA-aware
- **Device-adaptive**: CUDA → MPS → CPU auto-detection; autocast on CUDA
- **FinBERT fusion**: FinBERT sentiment score adjusts AuraMarketNet direction probability by ±20% (weight 0.30)
- **Kernel warm-up**: dummy forward pass on startup eliminates first-request MPS/CUDA latency

###Real-Time API & Dashboard
- **15 REST endpoints + WebSocket** (Flask-SocketIO)
- **Live market data**: TTL-cached yfinance quotes (5s for prices, 5s for news)
- **Professional backtesting**: 5 strategies — RSI mean reversion, MA crossover, price momentum, Bollinger squeeze breakout, AI-driven (AuraMarketNet batch inference) — with commission, slippage, stop-loss, Sharpe, Sortino, Calmar, alpha/beta
- **News enrichment**: impact scoring (0–10), breaking news detection, keyword highlighting with XSS-safe HTML
- **VADER fallback**: FinBERT inference degrades gracefully to VADER if no checkpoint

###Data
- **20 technical indicators**: RSI-14, MACD (12/26/9), Bollinger Bands (20,2σ), ATR-14, EMA-12/26, SMA-20/50, OBV, Stochastic %K/%D, VWAP, log return, historical volatility (20-day annualised), Garman-Klass volatility
- **Multi-ticker training**: AAPL, TSLA, GOOGL, MSFT, AMZN, NVDA, META, NFLX, AMD, SPY etc
- **Local data cache**: yfinance OHLCV pickles + Hugging Face dataset parquet to avoid redundant downloads

---

##Project Structure

```
AuraMarketNet/
├── models/
│   ├── text_encoder.py        #FinBERTEncoder + MultiTextEncoder (attention pooling)
│   ├── numerical_encoder.py   #Bi-LSTM + Self-Attention + Temporal Pooling
│   ├── fusion.py              #CrossAttentionFusion
│   └── aura_market_net.py     #Full model assembly
├── utils/
│   ├── feature_engineering.py
│   ├── data_loader.py         #dataloaders
│   ├── text_preprocessing.py  #Financial text cleaning, WSB slang normalization
│   ├── metrics.py             #All evaluation metrics (classification + regression)
│   ├── realtime_data.py       #Thread-safe TTL-cached yfinance layer (VADER fallback)
│   ├── sentiment_inference.py #FinBERT singleton inference engine + VADER fallback
│   └── market_inference.py    #AuraMarketNet singleton inference + FinBERT fusion
├── training/
│   ├── trainer.py             
│   ├── losses.py             
│   └── callbacks.py         
├── evaluation/
│   ├── evaluator.py           #ModelEvaluator: full metrics + backtest report
│   └── explainability.py      #SHAP feature importance + attention visualization
├── api/
│   ├── app.py                 #Flask app
│   └── services/
│       ├── backtest_engine.py #5-strategy backtester with full financial metrics
│       ├── data_service.py    #Indicator endpoint computation
│       └── news_service.py    #Impact scoring, breaking detection, keyword highlight
├── dashboard/
│   ├── templates/index.html   #Real-time quant terminal UI
│   └── static/
│       ├── css/style.css      #theme
│       └── js/dashboard.js    #Plotly charts, live polling, WebSocket client
├── notebooks/
│   └── AuraMarketNet_Analysis.ipynb  #Full analysis walkthrough
├── scripts/
│   ├── start_app.sh           #Launch Flask dashboard
│   ├── deploy_runpod.sh       #Push to RunPod GPU pod
│   ├── deploy_model.sh        #Model deployment helper
│   ├── status.sh              #Check running processes
│   └── stop.sh                #Stop the dashboard
├── data/
├── logs/
│   ├── final_test_metrics.json       #AuraMarketNet test set results
│   └── sentiment_test_metrics.json   #FinBERT fine-tuning results
├── config.py                  #Centralized dataclass config (model/training/data/dashboard)
├── train.py                   #Full model training entry point
├── train_sentiment.py         #FinBERT sentiment fine-tuning
├── Dockerfile.gpu             #GPU Docker image (RunPod, CUDA 12.1)
├── docker-compose.gpu.yml     #GPU Docker Compose
├── .env.example               #Environment variable template
└── requirements.txt
```

---

##Quick Start

###1. Install Dependencies

```bash
git clone https://github.com/MahadeMishuk/AuraMarketNet.git
cd AuraMarketNet
python -m venv venv
source venv/bin/activate        #Windows: venv\Scripts\activate
pip install -r requirements.txt
```

###2. Configure Environment (Optional)

API keys for Reddit and NewsAPI are optional — the system works without them using Yahoo Finance data.

```bash
cp .env.example .env
#Edit .env and add your keys (Reddit, AlphaVantage, NewsAPI)
```

###3. Fine-Tune the Sentiment Model

Fine-tunes `finbert` on Financial PhraseBank for 3-class financial sentiment. The resulting checkpoint (`checkpoints/finbert_sentiment_best.pt`) is loaded automatically by the inference engine.

```bash
#Full 10-epoch fine-tuning (downloads ~400 MB FinBERT on first run)
python train_sentiment.py

#Quick smoke test: 2 epochs, 200 samples
python train_sentiment.py --dry-run

#Custom hyperparameters
python train_sentiment.py --epochs 10 --batch-size 32 --lr 2e-5
```

Results are saved to `logs/sentiment_test_metrics.json`.

###4. Train the Full AuraMarketNet Model

Fetches OHLCV for 10 tickers (2020–2024), engineers 20 TA features, builds 30-day sequences, trains the dual-stream model.

```bash
#Full 30-epoch training on all default tickers
python train.py

#Custom tickers, epochs, batch size
python train.py --tickers AAPL TSLA NVDA MSFT --epochs 30 --batch-size 32

#Resume from a checkpoint
python train.py --resume AuraMarketNet-v1_checkpoints/aura_market_net_best.pt

#Pipeline validation (2 epochs, 4 samples — fast sanity check)
python train.py --dry-run

#Numerical-only ablation (disable FinBERT text encoder)
python train.py --no-text
```

Checkpoints are saved every 2 epochs and on every best validation score to `AuraMarketNet-v1_checkpoints/`. Final weights are saved as `AuraMarketNet-v1_final.pth` after training completes.

Test metrics are saved to `logs/final_test_metrics.json`.

###5. Launch the Dashboard

```bash
python api/app.py
#Open http://localhost:8080
```

Both models load at startup. If a checkpoint is missing, the endpoint returns a 503 with a clear error — it does not crash.

###6. Run the Analysis Notebook

```bash
jupyter notebook notebooks/AuraMarketNet_Analysis.ipynb
```

---

##Model Details

###Text Encoder — `MultiTextEncoder`

- **Backbone**: `ProsusAI/finbert` (BERT-base fine-tuned on Bloomberg + Reuters + earnings calls)
- **Multi-text pooling**: encodes up to 5 headlines per sample in parallel, then aggregates via learned attention pooling (which headline matters most is learned)
- **Layer freezing**: first 8 of 12 transformer blocks frozen to preserve financial vocabulary
- **Pooling strategy**: weighted mean over the last 4 hidden layers (outperforms CLS-only on financial tasks)
- **Projection**: Linear(768→384) → GELU → LayerNorm → Dropout → Linear(384→256) → LayerNorm
- **Auxiliary head**: `Linear(256→3)` predicts sentiment (neg/neutral/pos) as a regularisation task
- **Output**: `text_emb [batch, 256]`

###Numerical Encoder — `NumericalEncoder`

- **Input**: 30-day rolling window × 20 features per timestep `[batch, 30, 20]`
- **Pipeline**: Linear(20→256) → Positional Encoding → Bi-LSTM(256, 3 layers, dropout=0.2) → Multi-Head Self-Attention(8 heads) → Temporal Attention Pooling
- **Output**: `num_emb [batch, 256]`

###Fusion — `CrossAttentionFusion`

- **Cross-attention A**: numerical queries text → weighted text context
- **Cross-attention B**: text queries numerical → weighted market context
- **Gating**: `sigmoid(Linear(512))` learns per-sample modality balance
- **MLP**: `[512, 256]` with LayerNorm and GELU activations
- **Output**: `fused_emb [batch, 512]`

###Output Heads

| Head | Architecture | Loss | Weight |
|---|---|---|---|
| Direction (UP/DOWN) | Linear(512→2) + Softmax | CrossEntropy | 1.0 |
| Return Magnitude | Linear(512→1) | Huber (δ=0.1) | 0.5 |
| Volatility | Linear(512→1) + Softplus | MSE | 0.3 |
| Sentiment (auxiliary) | Linear(256→3) on text_emb | CrossEntropy | 0.2 |

**Total loss**: `L = 1.0·L_dir + 0.5·L_ret + 0.3·L_vol + 0.2·L_sent`

---

##Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| BERT backbone LR | 1e-5 |
| All other layers LR | 2e-5 |
| Weight decay | 0.01 |
| LR scheduler | Cosine with warmup (10% of steps) |
| Batch size | 16 (effective 64 with 4× gradient accumulation) |
| Gradient clipping | max norm 1.0 |
| Max epochs | 30 |
| Early stopping | patience=7 on `val_directional_accuracy` |
| Mixed precision | FP16 (CUDA only) |
| Label smoothing | 0.1 |
| Sequence length | 30 days |
| Val / Test split | 15% / 15% |
| Seed | 42 |

---

##Data Sources

| Source | Purpose | Access |
|---|---|---|
| Yahoo Finance (`yfinance`) | OHLCV daily bars for all tickers | No key required |
| Financial PhraseBank (`lmassaron/FinancialPhraseBank`) | FinBERT fine-tuning (3-class sentiment) | Auto-downloaded via HuggingFace |
| PRAW (Reddit) | Social sentiment from r/wallstreetbets, r/stocks | Optional — `REDDIT_CLIENT_ID` in `.env` |
| NewsAPI | Financial headlines | Optional — `NEWS_API_KEY` in `.env` |

Tickers: `AAPL TSLA GOOGL MSFT AMZN NVDA META NFLX AMD SPY`  
Date range: `2020-01-01` → `2024-12-31`

---

##Technical Indicators (20 Features)

| Category | Indicators |
|---|---|
| Trend | SMA-20, SMA-50, EMA-12, EMA-26 |
| Momentum | RSI-14, MACD line, MACD signal, MACD histogram, Stochastic %K, Stochastic %D |
| Volatility | Bollinger upper/mid/lower, ATR-14, historical volatility (20-day), Garman-Klass volatility |
| Volume | OBV, VWAP, volume ratio |
| Returns | Log return, price ratios (high-low/close, close-open/open) |

---

##Backtesting Engine

Five strategies, all self-contained (no external backtesting library):

| Strategy | Signal Logic |
|---|---|
| `rsi` | Buy on RSI crossover below 30, sell on crossover above 70 |
| `ma_cross` | SMA-20/50 golden cross buy, death cross sell |
| `momentum` | 20-day price momentum + volume surge (≥1.2× SMA) confirmation |
| `volatility` | Bollinger Band squeeze (≤25th percentile bandwidth) followed by upper-band breakout |
| `ai` | AuraMarketNet batch inference — UP probability > 0.55 → BUY, < 0.45 → SELL (CUDA only; falls back to RSI+momentum composite on CPU/MPS) |

All strategies support **commission**, **slippage**, and **stop-loss** parameters. The simulator reports:

- Total return vs buy-and-hold benchmark
- Annualised return, Sharpe, Sortino, Calmar ratios
- Max drawdown, exposure %, alpha, beta
- Per-trade log (entry/exit price, PnL, reason)
- Statistical reliability warning (< 10 trades = not significant)

Use `strategy=all` to run all five side-by-side in a single response.

---

##API Reference

All endpoints return JSON. The dashboard runs on **port 8080** by default (`PORT` env var overrides).

###Market Data

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

###ML Prediction & Sentiment

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/predict?ticker=AAPL&horizon=1D` | Full AuraMarketNet prediction (1H/1D/1W) |
| POST | `/api/analyze_text` | Body `{"text":"...", "ticker":"AAPL"}` — sentiment + market prediction |
| GET | `/api/sentiment_feed?ticker=AAPL` | FinBERT batch inference on live news |
| GET | `/api/news?ticker=AAPL&limit=20` | News with impact score, breaking flag, keyword highlights |
| POST | `/api/predict-sentiment` | Body `{"text":"..."}` or `{"texts":[...]}` — raw FinBERT inference |
| POST | `/api/predict-market-signal` | Body `{"texts":[...], "weights":[...]}` — aggregate bullish/neutral/bearish signal |
| GET | `/api/model_status` | Model health, checkpoint epoch, val accuracy, sentiment engine |

###Backtesting

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/backtest?ticker=AAPL&strategy=rsi&days=252` | Run backtest (`strategy`: rsi/ma_cross/momentum/volatility/ai/all) |

Optional params: `commission` (default 0.001), `slippage` (0.0005), `stop_loss` (disabled by default), `days` (max 756).

###WebSocket (Flask-SocketIO)

Connect to `ws://localhost:8080/socket.io`. Server pushes `ticker_tape` and `market_status` every 5 seconds to all connected clients. Emit `subscribe_ticker` with `{"ticker": "AAPL"}` for an immediate quote snapshot.

---

##GPU Deployment (RunPod)

Quick summary:

```bash
#On the pod
git clone https://github.com/MahadeMishuk/AuraMarketNet.git
cd AuraMarketNet
pip install -r requirements.txt

#Sentiment fine-tuning
python train_sentiment.py --epochs 10 --num-workers 4

#Full model training
python train.py --epochs 30 --batch-size 32 --num-workers 4

#Launch dashboard
python api/app.py
```

**GPU behaviour**: FP16 autocast on CUDA, cuDNN benchmark enabled, 4 DataLoader workers with `pin_memory=True`, VRAM usage logged per epoch.

Expected throughput on RTX 4090 (24 GB): ~5–10 min/epoch with 10 tickers, batch size 32.

###Docker (alternative)

```bash
docker build -f Dockerfile.gpu -t auramarketnet-gpu .
docker-compose -f docker-compose.gpu.yml up
```

---

##Architecture Rationale

**FinBERT over general BERT** — Pre-trained on 1.8 M Bloomberg, Reuters, and earnings call documents. Understands domain-specific polarity ("bearish" = negative, "headwinds" = risk). Fine-tuning on Financial PhraseBank achieves 79.1% accuracy on financial sentiment, significantly above general BERT baselines.

**Bi-LSTM + Transformer over pure Transformer** — Bi-LSTM efficiently captures local momentum (day-over-day trends) while the Transformer self-attention layer captures global periodic patterns across the 30-day window (e.g., monthly options expiry cycles, earnings windows). The hybrid generalises better than either architecture alone on the short sequence lengths used in financial time-series.

**Cross-attention over concatenation** — Markets exhibit either news-driven or technically-driven moves. Cross-attention lets the model learn per-sample which modality to trust: on high-impact news days (earnings surprises, Fed decisions) it weights text heavily; on low-news technical breakout days it weights numerical signals. Simple concatenation cannot make this dynamic per-sample distinction.

**Multi-task learning over single-head** — Jointly predicting direction + return magnitude + volatility prevents mode collapse (predicting "always UP"). The model must also be right about *how much* and *how volatile*, which acts as a strong regulariser. The auxiliary FinBERT sentiment head keeps the text encoder grounded in financial language semantics throughout fine-tuning.

---

##Disclaimer

This system is for research and educational purposes only. It is **not financial advice**. Market prediction is inherently uncertain. Past model performance does not guarantee future results. Never make investment decisions solely on the basis of any automated system.

---

##Citation

```bibtex
@software{auramarketnet2025,
  title  = {AuraMarketNet: Multi-Modal Market Prediction via FinBERT + Bi-LSTM Fusion},
  author = {Mishuk, M. H.},
  year   = {2025},
  note   = {Dual-stream FinBERT text encoder + Bi-LSTM time-series encoder
            with cross-attention fusion for stock market direction prediction}
}
```
