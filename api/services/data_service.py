from typing import List, Optional, Tuple, Dict, Any


def compute_sma(prices: List[float], period: int) -> List[Optional[float]]:
    """Simple moving average. Returns None for indices without enough data."""
    result: List[Optional[float]] = [None] * len(prices)
    for i in range(period - 1, len(prices)):
        result[i] = round(sum(prices[i - period + 1: i + 1]) / period, 4)
    return result


def compute_ema(prices: List[float], period: int) -> List[Optional[float]]:
    """Exponential moving average. Initial value = SMA of first `period` values."""
    result: List[Optional[float]] = [None] * len(prices)
    if len(prices) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = round(sum(prices[:period]) / period, 4)
    for i in range(period, len(prices)):
        prev = result[i - 1]
        result[i] = round(prices[i] * k + prev * (1 - k), 4)  #type: ignore[operator]
    return result


def compute_rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
    """
    Wilder's RSI. Returns overbought (>70) / oversold (<30) levels.
    Returns None for indices that don't have enough history.
    """
    result: List[Optional[float]] = [None] * len(prices)
    if len(prices) < period + 2:
        return result

    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period

    def _rsi(g: float, l: float) -> float:
        return round(100 - 100 / (1 + g / l), 2) if l else 100.0

    result[period] = _rsi(avg_g, avg_l)

    for i in range(period + 1, len(prices)):
        avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        result[i] = _rsi(avg_g, avg_l)

    return result


def compute_macd(
    prices: List[float],
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    MACD = EMA(fast) − EMA(slow).
    Signal = EMA(MACD, signal_period).
    Histogram = MACD − Signal.
    """
    ema_f = compute_ema(prices, fast)
    ema_s = compute_ema(prices, slow)
    n     = len(prices)

    macd: List[Optional[float]] = [None] * n
    sig:  List[Optional[float]] = [None] * n
    hist: List[Optional[float]] = [None] * n

    for i in range(n):
        if ema_f[i] is not None and ema_s[i] is not None:
            macd[i] = round(ema_f[i] - ema_s[i], 4)  #type: ignore[operator]

    #EMA of MACD values
    valid_vals = [v for v in macd if v is not None]
    ema_sig    = compute_ema(valid_vals, signal)
    j = 0
    for i in range(n):
        if macd[i] is not None:
            if ema_sig[j] is not None:
                sig[i] = ema_sig[j]
            j += 1

    for i in range(n):
        if macd[i] is not None and sig[i] is not None:
            hist[i] = round(macd[i] - sig[i], 4)  #type: ignore[operator]

    return macd, sig, hist


def compute_bollinger(
    prices: List[float], period: int = 20, std_mult: float = 2.0,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Bollinger Bands: upper, mid (SMA), lower."""
    n     = len(prices)
    upper = [None] * n
    mid   = [None] * n
    lower = [None] * n
    for i in range(period - 1, n):
        window = prices[i - period + 1: i + 1]
        m   = sum(window) / period
        std = (sum((x - m) ** 2 for x in window) / period) ** 0.5
        mid[i]   = round(m, 4)
        upper[i] = round(m + std_mult * std, 4)
        lower[i] = round(m - std_mult * std, 4)
    return upper, mid, lower  #type: ignore[return-value]


def compute_all_indicators(history: Dict[str, Any]) -> Dict[str, Any]:
    close = history.get("close", [])
    if not close:
        history["indicators"] = {}
        return history

    macd_line, macd_sig, macd_hist = compute_macd(close)
    bb_upper, bb_mid, bb_lower     = compute_bollinger(close)

    history["indicators"] = {
        "sma_20":    compute_sma(close, 20),
        "sma_50":    compute_sma(close, 50),
        "ema_12":    compute_ema(close, 12),
        "ema_26":    compute_ema(close, 26),
        "rsi_14":    compute_rsi(close, 14),
        "macd":      macd_line,
        "macd_sig":  macd_sig,
        "macd_hist": macd_hist,
        "bb_upper":  bb_upper,
        "bb_mid":    bb_mid,
        "bb_lower":  bb_lower,
    }
    return history


def simple_backtest(
    timestamps: List[str],
    close: List[float],
    rsi: List[Optional[float]],
    initial: float = 10_000.0,
    buy_threshold: float = 40.0,
    sell_threshold: float = 60.0,
) -> Dict[str, Any]:

    n      = len(close)
    port   = initial
    shares = 0.0
    in_pos = False

    port_values  = []
    bh_values    = []
    bh_shares    = initial / close[0] if close[0] else 0

    trades_won = 0
    trades_lost = 0
    entry_price = 0.0

    for i in range(n):
        r = rsi[i] if rsi[i] is not None else 50.0

        #RSI buy signal (was above threshold, now below)
        if not in_pos and r < buy_threshold:
            shares    = port / close[i]
            in_pos    = True
            entry_price = close[i]
            port      = 0.0

        #RSI sell signal (was below threshold, now above)
        elif in_pos and r > sell_threshold:
            port   = shares * close[i]
            in_pos = False
            if close[i] >= entry_price:
                trades_won += 1
            else:
                trades_lost += 1
            shares = 0.0

        current_val = port + shares * close[i]
        port_values.append(round(current_val, 2))
        bh_values.append(round(bh_shares * close[i], 2))

    #Close open position at end
    if in_pos:
        final_port = shares * close[-1]
    else:
        final_port = port

    strat_return = (final_port - initial) / initial * 100
    bh_return    = (bh_values[-1] - initial) / initial * 100 if bh_values else 0.0

    #Sharpe (approximate, daily returns)
    daily = [
        (port_values[i] - port_values[i - 1]) / port_values[i - 1]
        for i in range(1, n) if port_values[i - 1] > 0
    ]
    sharpe = 0.0
    if len(daily) > 1:
        import math
        mu  = sum(daily) / len(daily)
        std = math.sqrt(sum((x - mu) ** 2 for x in daily) / len(daily))
        sharpe = round((mu / std * (252 ** 0.5)) if std else 0.0, 2)

    return {
        "timestamps":     timestamps,
        "strategy":       port_values,
        "buy_hold":       bh_values,
        "strat_return":   round(strat_return, 2),
        "bh_return":      round(bh_return, 2),
        "trades_won":     trades_won,
        "trades_lost":    trades_lost,
        "sharpe":         sharpe,
        "initial":        initial,
    }
