"""
Routes:
  GET  /                             → Dashboard HTML
  GET  /api/price?ticker=X           → Live quote
  GET  /api/history?ticker=X&range=1D → OHLCV for charting (with indicators)
  GET  /api/news?ticker=X&limit=20   → News with sentiment + impact scores
  GET  /api/sentiment_feed?ticker=X  → Sentiment feed with aggregate
  GET  /api/ticker_tape              → All tape tickers with live prices
  GET  /api/market_status            → US market open/closed
  GET  /api/market_overview          → S&P500, NASDAQ, DOW, VIX
  GET  /api/company_info?ticker=X    → Company metadata
  GET  /api/sparklines?tickers=A,B   → Mini price series for watchlist
  GET  /api/top_movers               → Top gainers and losers
  GET  /api/predict?ticker=X&horizon=1D → ML prediction with time horizon
  GET  /api/backtest?ticker=X        → Multi-strategy professional backtest vs buy-and-hold
  POST /api/analyze_text             → Custom text sentiment + prediction
  GET  /api/model_status             → Model health
  WS   /socket.io                    → Live streaming
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  #project root
sys.path.insert(0, str(Path(__file__).parent))           #api/ for services

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

try:
    from flask_socketio import SocketIO, emit
    _SOCKETIO_AVAILABLE = True
except ImportError:
    _SOCKETIO_AVAILABLE = False

from config import CFG
from utils.feature_engineering import FeatureEngineer
from utils.realtime_data import (
    get_live_quote, get_price_history, get_news,
    get_ticker_tape, get_market_status, get_company_info,
    get_market_overview, get_sparkline_data, get_top_movers,
)
from services.data_service import compute_all_indicators
from services.news_service import enrich_articles, sentiment_distribution
from utils.sentiment_inference import (
    load_sentiment_model,
    predict_sentiment,
    predict_batch_sentiment,
    score_sentiment_probability,
    aggregate_market_sentiment,
    generate_market_signal,
    is_loaded as sentiment_model_loaded,
)
from utils.market_inference import (
    load_market_model,
    is_market_model_loaded,
    predict_market,
    get_checkpoint_meta,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Load both models at startup.  Each is non-blocking: a missing checkpoint
#logs a warning and the prediction endpoint returns a 503 instead of crashing.
load_sentiment_model()
load_market_model()



#Flask App Setup

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "dashboard" / "templates"),
    static_folder=str(Path(__file__).parent.parent / "dashboard" / "static"),
)
app.config["SECRET_KEY"] = CFG.dashboard.secret_key
CORS(app)

if _SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    logger.info("Flask-SocketIO enabled")
else:
    socketio = None
    logger.info("Flask-SocketIO not available — falling back to polling")



#Feature Engineering (used by indicator endpoints)

_feature_engineer = FeatureEngineer()



#Real-Time API Endpoints


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/price")
def price():
    """
    GET /api/price?ticker=AAPL
    Returns live quote: price, change, change_pct, volume, market_cap, etc.
    """
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    if not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    return jsonify(get_live_quote(ticker))


@app.route("/api/history")
def history():
    """
    GET /api/history?ticker=AAPL&range=1D
    range: 1D | 5D | 1W | 1M | 3M | 6M | 1Y | MAX
    Returns OHLCV arrays for Plotly charting.
    """
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    range_key = request.args.get("range", "1D").upper().strip()

    if not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400

    valid_ranges = {"1D", "5D", "1W", "1M", "3M", "6M", "1Y", "MAX"}
    if range_key not in valid_ranges:
        range_key = "1D"

    return jsonify(get_price_history(ticker, range_key))


@app.route("/api/news")
def news():
    """
    GET /api/news?ticker=AAPL&limit=20
    Returns live news enriched with impact score, breaking flag, keyword highlights.
    """
    ticker = request.args.get("ticker", "").upper().strip() or None
    limit  = min(int(request.args.get("limit", 25)), 50)

    if ticker and not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400

    articles  = get_news(ticker, limit)
    enriched  = enrich_articles(articles)
    dist      = sentiment_distribution(enriched)
    return jsonify({
        "ticker":       ticker or "MARKET",
        "articles":     enriched,
        "count":        len(enriched),
        "distribution": dist,
    })


@app.route("/api/ticker_tape")
def ticker_tape():
    """
    GET /api/ticker_tape
    Returns live quotes for all tape tickers (batched fetch).
    """
    return jsonify({"tickers": get_ticker_tape(), "timestamp": datetime.now().isoformat()})


@app.route("/api/market_status")
def market_status():
    """GET /api/market_status — US equity market open/closed status."""
    return jsonify(get_market_status())


@app.route("/api/market_overview")
def market_overview():
    """GET /api/market_overview — S&P 500, NASDAQ, DOW, VIX. Cached 1 min."""
    return jsonify(get_market_overview())


@app.route("/api/sparklines")
def sparklines():
    """
    GET /api/sparklines?tickers=AAPL,TSLA,NVDA
    Returns last 30 hourly close prices per ticker for sparkline rendering.
    """
    raw     = request.args.get("tickers", "")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()][:15]
    tickers = [t for t in tickers if _valid_ticker(t)]
    if not tickers:
        return jsonify({"error": "No valid tickers"}), 400
    return jsonify(get_sparkline_data(tickers))


@app.route("/api/top_movers")
def top_movers():
    """GET /api/top_movers — Top N gainers and losers from tape tickers."""
    n = min(int(request.args.get("n", 6)), 10)
    return jsonify(get_top_movers(n))


@app.route("/api/backtest")
def backtest():
    """
    GET /api/backtest?ticker=AAPL&strategy=rsi&days=252&commission=0.001&slippage=0.0005&stop_loss=
    strategy: rsi | ma_cross | momentum | volatility | ai | all
    days: trading days to backtest (default 252, max 756)
    """
    ticker         = request.args.get("ticker",     "AAPL").upper().strip()
    strategy       = request.args.get("strategy",   "rsi").lower().strip()
    days           = min(int(request.args.get("days",       252)),   756)
    commission     = float(request.args.get("commission",  0.001))
    slippage       = float(request.args.get("slippage",    0.0005))
    stop_loss_raw  = request.args.get("stop_loss", "").strip()
    stop_loss      = float(stop_loss_raw) if stop_loss_raw else None

    if not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400

    try:
        from services.backtest_engine import run_backtest

        #Pick range that gives daily bars for the requested period
        range_key = "1Y" if days <= 252 else "5Y"
        hist = get_price_history(ticker, range_key)
        if not hist.get("close"):
            return jsonify({"error": "No price data available"}), 404

        n = min(days, len(hist["close"]))
        data_dict = {
            "timestamps": hist["timestamps"][-n:],
            "open":       hist["open"][-n:],
            "high":       hist["high"][-n:],
            "low":        hist["low"][-n:],
            "close":      hist["close"][-n:],
            "volume":     hist["volume"][-n:],
        }

        result = run_backtest(data_dict, strategy, commission, slippage, stop_loss)
        result["ticker"] = ticker
        return jsonify(result)

    except Exception as e:
        logger.warning(f"Backtest failed for {ticker}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/company_info")
def company_info():
    """GET /api/company_info?ticker=AAPL — Company name, sector, description."""
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    if not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    return jsonify(get_company_info(ticker))



#ML Prediction Endpoints


@app.route("/api/predict")
def predict():
    """
    GET /api/predict?ticker=AAPL&horizon=1D
    horizon: 1H | 1D | 1W

    Runs the full AuraMarketNet v1 pipeline:
      OHLCV → TA features → Bi-LSTM+Attention → CrossFusion w/ FinBERT text
      → direction (UP/DOWN) + price_change_pct + volatility
    Returns 503 if the model checkpoint is not loaded.
    """
    ticker  = request.args.get("ticker",  "AAPL").upper().strip()
    horizon = request.args.get("horizon", "1D").upper().strip()

    if not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    if horizon not in ("1H", "1D", "1W"):
        horizon = "1D"

    if not is_market_model_loaded():
        return jsonify({
            "error":   "Prediction model not available",
            "detail":  "AuraMarketNet checkpoint not loaded. Check server logs.",
            "mode":    "unavailable",
        }), 503

    try:
        news_items = get_news(ticker, limit=5)
        news_texts = [a["title"] for a in news_items if a.get("title")]

        result = run_prediction(ticker, news_texts)
        result["horizon"] = horizon

        #Sentiment / technical contribution estimates for the explainability panel
        news_score = float(news_items[0].get("sentiment_score", 0)) if news_items else 0.0
        result["sentiment_contrib"] = round(min(abs(news_score) * 80, 75), 1)
        result["technical_contrib"] = round(result["confidence"] * 0.55, 1)

        #Top 3 news driving the prediction
        enriched_news = enrich_articles(news_items) if news_items else []
        result["top_news"] = [
            {
                "title":             a.get("title", ""),
                "highlighted_title": a.get("highlighted_title", a.get("title", "")),
                "sentiment":         a.get("sentiment", "neutral"),
                "score":             a.get("sentiment_score", 0),
                "publisher":         a.get("publisher", ""),
            }
            for a in enriched_news[:3]
        ]
        return jsonify(result)

    except Exception as exc:
        logger.error(f"Prediction error for {ticker}: {exc}", exc_info=True)
        return jsonify({"error": str(exc), "mode": "error"}), 500


@app.route("/api/analyze_text", methods=["POST"])
def analyze_text():
    """
    POST /api/analyze_text
    Body: {"text": "...", "ticker": "AAPL"}
    Analyze custom text and run model prediction.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"][:2000]
    ticker = data.get("ticker", "AAPL").upper()

    if len(text.strip()) < 5:
        return jsonify({"error": "Text too short"}), 400

    result = run_prediction(ticker, [text])

    #FinBERT sentiment (fine-tuned on Financial PhraseBank)
    finbert_result = predict_sentiment(text)
    result["finbert_sentiment"]  = finbert_result["label"]
    result["finbert_score"]      = finbert_result["score"]
    result["finbert_probs"]      = finbert_result["probabilities"]
    result["sentiment_engine"]   = finbert_result["engine"]

    #Keep VADER as a secondary signal for comparison
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        scores = vader.polarity_scores(text)
        result["vader_compound"]  = scores["compound"]
        result["vader_sentiment"] = (
            "positive" if scores["compound"] > 0.05
            else "negative" if scores["compound"] < -0.05
            else "neutral"
        )
    except Exception:
        pass

    result["analyzed_text"] = text
    return jsonify(result)


@app.route("/api/sentiment_feed")
def sentiment_feed():
    """
    GET /api/sentiment_feed?ticker=AAPL&limit=20
    Returns live news headlines with VADER sentiment scores.
    Aggregates overall sentiment signal (bullish / bearish / neutral).
    """
    ticker = request.args.get("ticker", "").upper().strip() or None
    limit  = min(int(request.args.get("limit", 20)), 50)

    if ticker and not _valid_ticker(ticker):
        return jsonify({"error": "Invalid ticker"}), 400

    articles = get_news(ticker, limit)
    titles   = [a.get("title", "") for a in articles if a.get("title")]

    #FinBERT batch inference over all headlines in one forward pass
    if titles:
        predictions = predict_batch_sentiment(titles)
        for a, pred in zip(articles, predictions):
            a["sentiment"]        = pred["label"]
            a["sentiment_score"]  = round(
                pred["probabilities"]["positive"] - pred["probabilities"]["negative"], 3
            )
            a["sentiment_probs"]  = pred["probabilities"]
            a["sentiment_engine"] = pred["engine"]
    else:
        predictions = []

    agg    = aggregate_market_sentiment(predictions)
    signal = agg["overall_label"]
    color_map = {"bullish": "#00e676", "bearish": "#ff1744", "neutral": "#6b82a8"}

    return jsonify({
        "ticker":   ticker or "MARKET",
        "articles": articles,
        "count":    len(articles),
        "aggregate": {
            "overall":         signal,
            "overall_color":   color_map[signal],
            "overall_score":   agg["overall_score"],
            "signal_strength": agg["signal_strength"],
            "positive":        agg["bullish_count"],
            "negative":        agg["bearish_count"],
            "neutral":         agg["neutral_count"],
            "avg_positive":    agg["avg_positive"],
            "avg_negative":    agg["avg_negative"],
            "engine":          predictions[0]["engine"] if predictions else "unknown",
        },
    })


@app.route("/api/model_status")
def model_status():
    """GET /api/model_status — Model health and architecture info."""
    mkt_loaded  = is_market_model_loaded()
    meta        = get_checkpoint_meta()
    return jsonify({
        "model_loaded":       mkt_loaded,
        "model_name":         "AuraMarketNet v1",
        "architecture":       "FinBERT + Bi-LSTM + Cross-Attention Fusion",
        "tasks":              ["Direction (UP/DOWN)", "% Price Change", "Volatility"],
        "status":             "ready" if mkt_loaded else "unavailable",
        "checkpoint_epoch":   meta.get("epoch", "?"),
        "val_dir_accuracy":   meta.get("metrics", {}).get("val_directional_accuracy", 0),
        "sentiment_engine":   "finbert" if sentiment_model_loaded() else "vader_fallback",
        "prediction_engine":  "auramarketnet_v1" if mkt_loaded else "unavailable",
        "realtime_data":      True,
        "timestamp":          datetime.now().isoformat(),
    })


@app.route("/api/predict-sentiment", methods=["POST"])
def predict_sentiment_endpoint():
    """
    POST /api/predict-sentiment
    Body (JSON):
        {"text": "Apple beats earnings estimates by 12%"}
        or
        {"texts": ["headline 1", "headline 2", ...]}

    Returns structured FinBERT (or VADER fallback) sentiment predictions.
    Single text → one SentimentResult dict.
    Multiple texts → list of SentimentResult dicts + aggregate.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    if "texts" in data:
        texts = [str(t)[:2000] for t in data["texts"][:50] if t]
        if not texts:
            return jsonify({"error": "Empty texts list"}), 400

        predictions = predict_batch_sentiment(texts)
        aggregate   = aggregate_market_sentiment(predictions)
        signal      = aggregate["overall_label"]
        return jsonify({
            "predictions": predictions,
            "aggregate":   aggregate,
            "signal":      signal,
            "engine":      predictions[0]["engine"] if predictions else "unknown",
            "count":       len(predictions),
        })

    elif "text" in data:
        text = str(data["text"])[:2000].strip()
        if len(text) < 3:
            return jsonify({"error": "Text too short (min 3 characters)"}), 400
        result = predict_sentiment(text)
        return jsonify({**result, "text": text})

    else:
        return jsonify({"error": "Provide 'text' (str) or 'texts' (list)"}), 400


@app.route("/api/predict-market-signal", methods=["POST"])
def predict_market_signal_endpoint():
    """
    POST /api/predict-market-signal
    Body (JSON):
        {
            "texts":   ["AAPL beats Q3 earnings", "Fed raises rates", ...],
            "weights": [0.9, 0.7, ...]   #optional impact weights per headline
        }

    Aggregates FinBERT predictions across all headlines and returns a
    single market signal: "bullish" | "neutral" | "bearish".
    """
    data = request.get_json(silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "JSON body with 'texts' list required"}), 400

    texts   = [str(t)[:2000] for t in data["texts"][:50] if t]
    weights = data.get("weights")

    if not texts:
        return jsonify({"error": "Empty texts list"}), 400
    if weights is not None:
        if len(weights) != len(texts):
            return jsonify({"error": "'weights' length must match 'texts' length"}), 400
        weights = [float(w) for w in weights]

    predictions = predict_batch_sentiment(texts)
    aggregate   = aggregate_market_sentiment(predictions, weights)

    return jsonify({
        "signal":         aggregate["overall_label"],
        "overall_score":  aggregate["overall_score"],
        "signal_strength": aggregate["signal_strength"],
        "aggregate":      aggregate,
        "predictions":    predictions,
        "engine":         predictions[0]["engine"] if predictions else "unknown",
        "timestamp":      datetime.now().isoformat(),
    })



#WebSocket — Live Streaming


if _SOCKETIO_AVAILABLE:
    _streaming_clients: set = set()
    _stream_lock = threading.Lock()
    _stream_thread = None

    @socketio.on("connect")
    def on_connect():
        sid = request.sid
        with _stream_lock:
            _streaming_clients.add(sid)
        logger.info(f"Client connected: {sid} (total: {len(_streaming_clients)})")
        _ensure_stream_thread()
        #Send immediate snapshot
        emit("market_status", get_market_status())
        emit("ticker_tape", {"tickers": get_ticker_tape()})

    @socketio.on("disconnect")
    def on_disconnect():
        sid = request.sid
        with _stream_lock:
            _streaming_clients.discard(sid)
        logger.info(f"Client disconnected: {sid} (total: {len(_streaming_clients)})")

    @socketio.on("subscribe_ticker")
    def on_subscribe(data):
        """Client subscribes to a specific ticker for live updates."""
        ticker = str(data.get("ticker", "AAPL")).upper()[:10]
        if _valid_ticker(ticker):
            quote = get_live_quote(ticker)
            emit("quote_update", quote)

    def _stream_worker():
        """Background thread: push tape + market status every 5 seconds."""
        while True:
            with _stream_lock:
                n = len(_streaming_clients)
            if n > 0:
                try:
                    tape = get_ticker_tape()
                    status = get_market_status()
                    socketio.emit("ticker_tape", {"tickers": tape})
                    socketio.emit("market_status", status)
                except Exception as e:
                    logger.warning(f"Stream error: {e}")
            time.sleep(5)

    def _ensure_stream_thread():
        global _stream_thread
        if _stream_thread is None or not _stream_thread.is_alive():
            _stream_thread = threading.Thread(target=_stream_worker, daemon=True)
            _stream_thread.start()



#Prediction Logic


def run_prediction(ticker: str, recent_news: list = None) -> dict:
    """
    Run the full AuraMarketNet v1 prediction pipeline.

    Uses utils.market_inference which handles:
      - OHLCV fetch + 20-feature TA engineering
      - 30-day sequence tensor
      - MultiTextEncoder tokenisation of headlines
      - AuraMarketNet forward pass (direction / price_change / volatility)
      - FinBERT sentiment fusion (+30% weight adjustment)

    Returns a dict ready for jsonify().  Raises ValueError/RuntimeError
    on data or model errors (caller handles with 503).
    """
    headlines = recent_news or [f"{ticker} market outlook"]
    result    = predict_market(ticker, headlines=headlines)

    quote = get_live_quote(ticker)
    result.update({
        "ticker":               ticker,
        "current_price":        quote.get("price", 0),
        "price_change":         quote.get("change", 0),
        "price_change_day_pct": quote.get("change_pct", 0),
        "price_change_pct":     result.pop("expected_return", 0),
        "timestamp":            datetime.now().isoformat(),
        #direction_probs as percentages for the UI progress bars
        "direction_probs": {
            k: round(v * 100, 2)
            for k, v in result["direction_probs"].items()
        },
        "confidence":           round(result["confidence"] * 100, 2),
    })
    return result



#Validation Helper


def _valid_ticker(ticker: str) -> bool:
    return bool(ticker) and len(ticker) <= 10 and ticker.replace(".", "").replace("-", "").isalnum()



#Error Handlers


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Internal server error"}), 500



#Entry Point


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║      AuraMarketNet — Real-Time Market Intelligence Platform   ║
║                                                              ║
║  Dashboard :  http://localhost:8080                          ║
║  Live Quote:  http://localhost:8080/api/price?ticker=AAPL    ║
║  History   :  http://localhost:8080/api/history?ticker=AAPL  ║
║  News      :  http://localhost:8080/api/news?ticker=AAPL     ║
║  Tape      :  http://localhost:8080/api/ticker_tape          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    port = int(os.getenv("PORT", CFG.dashboard.port))

    if _SOCKETIO_AVAILABLE and socketio:
        socketio.run(app, host=CFG.dashboard.host, port=port,
                     debug=CFG.dashboard.debug, allow_unsafe_werkzeug=True)
    else:
        app.run(host=CFG.dashboard.host, port=port,
                debug=CFG.dashboard.debug, threaded=True)
