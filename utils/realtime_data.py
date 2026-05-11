import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_vader = None
_vader_lock = threading.Lock()

def _get_vader():
    global _vader
    if _vader is None:
        with _vader_lock:
            if _vader is None:
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    _vader = SentimentIntensityAnalyzer()
                except Exception:
                    _vader = None
    return _vader

def _score_sentiment(text: str):
    """Return (label, compound_score) using VADER."""
    vader = _get_vader()
    if vader is None or not text:
        return "neutral", 0.0
    try:
        compound = vader.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return label, round(compound, 3)
    except Exception:
        return "neutral", 0.0




class TTLCache:
    """Simple in-memory cache with per-key TTL. Thread-safe."""

    def __init__(self):
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_cache = TTLCache()





TTL_QUOTE      = 5
TTL_HISTORY_1D = 30
TTL_HISTORY_LG = 300  
TTL_NEWS       = 300   
TTL_TAPE       = 10
TTL_INFO       = 600 

INTERVAL_MAP = {
    "1D":  ("1d",  "1m"),
    "5D":  ("5d",  "5m"),
    "1W":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "5Y":  ("5y",  "1d"),
    "MAX": ("max", "1wk"),
}

TAPE_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL",
    "AMZN", "META", "AMD",  "SPY",  "QQQ",
    "NFLX", "UBER", "CRM",  "ORCL", "INTC",
    "JPM",  "BAC",  "GS",   "V",    "PYPL",
]


def _get_yf_ticker(symbol: str):
    """Return a cached yfinance Ticker object."""
    import yfinance as yf
    return yf.Ticker(symbol)



#Live quote


def get_live_quote(symbol: str) -> Dict[str, Any]:
    """
    Fetch real-time quote for a symbol.
    Returns price, change, change_pct, volume, market_cap, etc.
    Cache TTL: 5 seconds.
    """
    cache_key = f"quote:{symbol}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = _get_yf_ticker(symbol)

        #Use .info for reliable cross-version data
        info = ticker.info or {}

        current_price  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        prev_close     = float(info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price or 1)
        market_cap     = info.get("marketCap")
        volume         = info.get("volume") or info.get("regularMarketVolume")
        day_high       = info.get("dayHigh") or info.get("regularMarketDayHigh")
        day_low        = info.get("dayLow") or info.get("regularMarketDayLow")
        fifty_two_high = info.get("fiftyTwoWeekHigh")
        fifty_two_low  = info.get("fiftyTwoWeekLow")
        beta           = info.get("beta")
        pe_ratio       = info.get("trailingPE") or info.get("forwardPE")

        change     = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        result = {
            "symbol":       symbol,
            "price":        round(current_price, 2),
            "prev_close":   round(prev_close, 2),
            "change":       round(change, 2),
            "change_pct":   round(change_pct, 3),
            "volume":       int(volume) if volume else None,
            "market_cap":   int(market_cap) if market_cap else None,
            "day_high":     round(float(day_high), 2) if day_high else None,
            "day_low":      round(float(day_low), 2) if day_low else None,
            "52w_high":     round(float(fifty_two_high), 2) if fifty_two_high else None,
            "52w_low":      round(float(fifty_two_low), 2) if fifty_two_low else None,
            "beta":         round(float(beta), 2) if beta else None,
            "pe_ratio":     round(float(pe_ratio), 1) if pe_ratio else None,
            "is_up":        change >= 0,
            "timestamp":    datetime.now().isoformat(),
        }

        _cache.set(cache_key, result, TTL_QUOTE)
        return result

    except Exception as e:
        logger.warning(f"Quote fetch failed for {symbol}: {e}")
        return _fallback_quote(symbol)


def _fallback_quote(symbol: str) -> Dict[str, Any]:
    """Return a minimal error-safe quote dict."""
    return {
        "symbol": symbol, "price": 0.0, "prev_close": 0.0,
        "change": 0.0, "change_pct": 0.0, "volume": None,
        "market_cap": None, "day_high": None, "day_low": None,
        "52w_high": None, "52w_low": None, "beta": None, "pe_ratio": None,
        "is_up": True, "timestamp": datetime.now().isoformat(),
        "error": "data_unavailable",
    }



#Price history (for charts)


def get_price_history(symbol: str, range_key: str = "1D") -> Dict[str, Any]:
    """
    Fetch OHLCV history for the specified range.

    range_key: "1D" | "5D" | "1W" | "1M" | "3M" | "6M" | "1Y" | "MAX"
    Returns dict with dates, open, high, low, close, volume arrays.
    Cache TTL: 30s for 1D, 5min for longer ranges.
    """
    period, interval = INTERVAL_MAP.get(range_key.upper(), ("1d", "1m"))
    cache_key = f"history:{symbol}:{range_key}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = _get_yf_ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            raise ValueError("Empty dataframe")

        df.index = pd.to_datetime(df.index)
        #Strip timezone for JSON serialization
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        #Format timestamps
        if interval in ("1m", "5m", "15m", "1h"):
            timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in df.index]
        else:
            timestamps = [t.strftime("%Y-%m-%d") for t in df.index]

        result = {
            "symbol":    symbol,
            "range":     range_key,
            "interval":  interval,
            "timestamps": timestamps,
            "open":   [round(float(v), 4) for v in df["Open"]],
            "high":   [round(float(v), 4) for v in df["High"]],
            "low":    [round(float(v), 4) for v in df["Low"]],
            "close":  [round(float(v), 4) for v in df["Close"]],
            "volume": [int(v) for v in df["Volume"]],
            "fetched_at": datetime.now().isoformat(),
        }

        ttl = TTL_HISTORY_1D if range_key == "1D" else TTL_HISTORY_LG
        _cache.set(cache_key, result, ttl)
        return result

    except Exception as e:
        logger.warning(f"History fetch failed for {symbol}/{range_key}: {e}")
        return {"symbol": symbol, "range": range_key, "error": str(e),
                "timestamps": [], "open": [], "high": [], "low": [], "close": [], "volume": []}



#News feed


def get_news(symbol: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """
    Fetch financial news. Uses yfinance's built-in news (no API key needed).
    For general market news, fetches from SPY.
    Cache TTL: 5 minutes.
    """
    cache_key = f"news:{symbol or 'market'}:{limit}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    results = []
    try:
        fetch_symbol = symbol or "SPY"
        ticker = _get_yf_ticker(fetch_symbol)
        raw_news = ticker.news or []

        for item in raw_news[:limit]:
            #yfinance news dict structure (varies by version)
            content = item.get("content", item)  #Newer yfinance wraps in "content"
            if isinstance(content, dict):
                title     = content.get("title", item.get("title", ""))
                link      = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else content.get("clickThroughUrl", {}).get("url", item.get("link", ""))
                publisher = content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else content.get("publisher", item.get("publisher", ""))
                pub_time  = content.get("pubDate", "") or item.get("providerPublishTime", "")
            else:
                title     = item.get("title", "")
                link      = item.get("link", "")
                publisher = item.get("publisher", "")
                pub_time  = item.get("providerPublishTime", "")

            if not title:
                continue

            #Parse timestamp
            try:
                if isinstance(pub_time, (int, float)):
                    ts = datetime.fromtimestamp(pub_time)
                elif isinstance(pub_time, str) and pub_time:
                    ts = datetime.fromisoformat(pub_time.replace("Z", "+00:00").replace("+00:00", ""))
                else:
                    ts = datetime.now()
                time_str = ts.strftime("%b %d, %H:%M")
                age_mins = int((datetime.now() - ts).total_seconds() / 60)
            except Exception:
                time_str = "Recent"
                age_mins = 0

            sentiment_label, sentiment_score = _score_sentiment(title)

            results.append({
                "title":           title,
                "link":            link,
                "publisher":       publisher or "Yahoo Finance",
                "time_str":        time_str,
                "age_mins":        age_mins,
                "ticker":          symbol or "MARKET",
                "sentiment":       sentiment_label,
                "sentiment_score": sentiment_score,
            })

        #Sort by recency
        results.sort(key=lambda x: x["age_mins"])

        if not results:
            results = _static_news_fallback(symbol)

    except Exception as e:
        logger.warning(f"News fetch failed for {symbol}: {e}")
        results = _static_news_fallback(symbol)

    _cache.set(cache_key, results, TTL_NEWS)
    return results


def _static_news_fallback(symbol: Optional[str]) -> List[Dict]:
    """Minimal fallback news when API fails."""
    sym = symbol or "market"
    now = datetime.now()
    return [
        {
            "title": f"Markets open: {sym.upper()} trading in focus amid macro developments",
            "link": "", "publisher": "AuraMarketNet",
            "time_str": now.strftime("%b %d, %H:%M"), "age_mins": 0, "ticker": sym,
        }
    ]



#Ticker tape (top movers)


def get_ticker_tape() -> List[Dict]:
    """
    Fetch live quotes for the tape tickers.
    Batched fetch is faster than individual calls.
    Cache TTL: 10 seconds.
    """
    cache_key = "ticker_tape"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    results = []
    try:
        import yfinance as yf
        #Download all tickers at once — much faster than individual calls
        data = yf.download(
            " ".join(TAPE_TICKERS),
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            raise ValueError("Empty download")

        closes = data["Close"]
        if len(closes) >= 2:
            prev  = closes.iloc[-2]
            today = closes.iloc[-1]
        else:
            prev  = closes.iloc[0]
            today = closes.iloc[0]

        for ticker in TAPE_TICKERS:
            try:
                price  = float(today[ticker]) if ticker in today else 0.0
                p_prev = float(prev[ticker])  if ticker in prev  else price
                change_pct = ((price - p_prev) / p_prev * 100) if p_prev else 0.0
                if price > 0:
                    results.append({
                        "symbol":     ticker,
                        "price":      round(price, 2),
                        "change_pct": round(change_pct, 2),
                        "is_up":      change_pct >= 0,
                    })
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"Ticker tape fetch failed: {e}")
        #Fallback: fetch individually for a subset
        for ticker in TAPE_TICKERS[:8]:
            q = get_live_quote(ticker)
            if "error" not in q:
                results.append({
                    "symbol":     ticker,
                    "price":      q["price"],
                    "change_pct": q["change_pct"],
                    "is_up":      q["is_up"],
                })

    _cache.set(cache_key, results, TTL_TAPE)
    return results



#Market status


def get_market_status() -> Dict[str, Any]:
    """Check if US equity markets are currently open."""
    now_et = datetime.now()  #Server should be in ET or adjust accordingly
    weekday = now_et.weekday()
    hour    = now_et.hour
    minute  = now_et.minute

    is_weekday = weekday < 5
    is_hours   = (hour == 9 and minute >= 30) or (10 <= hour <= 15) or (hour == 16 and minute == 0)
    is_open    = is_weekday and is_hours

    next_open = "Mon 9:30 AM ET" if weekday >= 4 else "9:30 AM ET"

    return {
        "is_open":   is_open,
        "status":    "OPEN" if is_open else "CLOSED",
        "color":     "#00ff88" if is_open else "#ff3366",
        "next_open": next_open,
        "timestamp": now_et.isoformat(),
    }



#Company info


def get_company_info(symbol: str) -> Dict[str, Any]:
    """Fetch basic company info (name, sector, description). Cached 10 min."""
    cache_key = f"info:{symbol}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = _get_yf_ticker(symbol)
        info   = ticker.info or {}

        result = {
            "symbol":      symbol,
            "name":        info.get("longName") or info.get("shortName") or symbol,
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            "description": (info.get("longBusinessSummary") or "")[:300],
            "website":     info.get("website", ""),
            "country":     info.get("country", ""),
            "pe_ratio":    info.get("trailingPE"),
            "eps":         info.get("trailingEps"),
            "dividend":    info.get("dividendYield"),
            "beta":        info.get("beta"),
        }
        _cache.set(cache_key, result, TTL_INFO)
        return result

    except Exception as e:
        logger.warning(f"Company info failed for {symbol}: {e}")
        return {"symbol": symbol, "name": symbol, "sector": "", "industry": "",
                "description": "", "website": "", "country": ""}



#Market overview (major indices + VIX)


_INDEX_SYMBOLS = {
    "sp500":  "^GSPC",
    "nasdaq": "^IXIC",
    "dow":    "^DJI",
    "vix":    "^VIX",
    "russell":"^RUT",
}
_INDEX_LABELS = {
    "sp500": "S&P 500", "nasdaq": "NASDAQ", "dow": "DOW",
    "vix": "VIX", "russell": "Russell 2000",
}

TTL_INDEX = 60  #1 min for indices


def get_market_overview() -> Dict[str, Any]:
    """Fetch S&P 500, NASDAQ, DOW, VIX. Cached 1 minute."""
    cache_key = "market_overview"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Any] = {}
    for name, symbol in _INDEX_SYMBOLS.items():
        try:
            ticker = _get_yf_ticker(symbol)
            info   = ticker.info or {}
            price  = float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
            prev   = float(info.get("regularMarketPreviousClose") or info.get("previousClose") or price or 1)
            change     = price - prev
            change_pct = (change / prev * 100) if prev else 0.0
            result[name] = {
                "symbol":     symbol,
                "label":      _INDEX_LABELS.get(name, name.upper()),
                "price":      round(price, 2),
                "change":     round(change, 2),
                "change_pct": round(change_pct, 2),
                "is_up":      change >= 0,
            }
        except Exception as e:
            logger.warning(f"Index fetch failed for {symbol}: {e}")
            result[name] = {
                "symbol": symbol, "label": _INDEX_LABELS.get(name, name),
                "price": 0, "change": 0, "change_pct": 0, "is_up": True, "error": True,
            }

    _cache.set(cache_key, result, TTL_INDEX)
    return result



#Sparklines (mini price history per ticker)


TTL_SPARKLINE = 120  #2 min


def get_sparkline_data(symbols: List[str], n_points: int = 30) -> Dict[str, List[float]]:
    """
    Fetch last n_points hourly close prices for each symbol.
    Used to render sparkline charts in the watchlist.
    Returns {symbol: [price, ...]} — empty list on failure.
    """
    cache_key = f"sparkline:{','.join(sorted(symbols))}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, List[float]] = {}
    for sym in symbols:
        try:
            ticker = _get_yf_ticker(sym)
            hist   = ticker.history(period="5d", interval="1h", auto_adjust=True)
            if not hist.empty:
                prices = hist["Close"].dropna().tolist()[-n_points:]
                result[sym] = [round(float(p), 2) for p in prices]
            else:
                result[sym] = []
        except Exception:
            result[sym] = []

    _cache.set(cache_key, result, TTL_SPARKLINE)
    return result



#Top movers (gainers + losers from tape)


def get_top_movers(n: int = 6) -> Dict[str, List[Dict]]:
    """Return top N gainers and losers from TAPE_TICKERS."""
    tape = get_ticker_tape()
    if not tape:
        return {"gainers": [], "losers": []}
    sorted_tape = sorted(tape, key=lambda x: x.get("change_pct", 0), reverse=True)
    return {
        "gainers": sorted_tape[:n],
        "losers":  list(reversed(sorted_tape[-n:])),
    }
