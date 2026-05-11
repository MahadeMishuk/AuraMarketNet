import abc
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RISK_FREE_DAILY     = 0.05 / 252   #5% annual
MIN_TRADES_RELIABLE = 10
MIN_TRADES_ROBUST   = 30


#Data containers

@dataclass
class Signal:
    action: str       #"BUY" | "SELL" | "HOLD"
    reason: str = ""  #human-readable explanation


@dataclass
class ClosedTrade:
    entry_date:  str
    exit_date:   str
    entry_price: float
    exit_price:  float
    return_pct:  float
    net_pnl:     float
    won:         bool
    reason_entry: str = ""
    reason_exit:  str = ""


#Abstract strategy────

class Strategy(abc.ABC):
    @property
    @abc.abstractmethod
    def label(self) -> str:
        pass

    @abc.abstractmethod
    def generate_signals(self, data: Dict) -> List[Signal]:
        """
        data keys: timestamps, open, high, low, close, volume (all lists).
        Returns a Signal for every bar.
        """


#Indicator helpers (self-contained, no external deps) ─

def _to_arr(lst) -> np.ndarray:
    """Convert list (may contain None) to float64 array with NaN for None."""
    return np.array([float(v) if v is not None else np.nan for v in lst], dtype=np.float64)


def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = np.nanmean(arr[i - n + 1:i + 1])
    return out


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    k = 2.0 / (n + 1)
    seed = np.nanmean(arr[:n])
    if np.isnan(seed):
        return out
    out[n - 1] = seed
    for i in range(n, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(arr: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    if len(arr) <= n:
        return out
    diff = np.diff(arr)
    gains  = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    out[n] = 100.0 - (100.0 / (1 + avg_g / avg_l)) if avg_l else 100.0
    for i in range(n + 1, len(arr)):
        avg_g = (avg_g * (n - 1) + gains[i - 1]) / n
        avg_l = (avg_l * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 - (100.0 / (1 + avg_g / avg_l)) if avg_l else 100.0
    return out


def _bollinger(arr: np.ndarray, n: int = 20, mult: float = 2.0):
    mid   = _sma(arr, n)
    upper = np.full_like(arr, np.nan)
    lower = np.full_like(arr, np.nan)
    for i in range(n - 1, len(arr)):
        std = np.std(arr[i - n + 1:i + 1], ddof=0)
        upper[i] = mid[i] + mult * std
        lower[i] = mid[i] - mult * std
    return upper, mid, lower


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.full_like(close, np.nan)
    atr[n - 1] = np.mean(tr[:n])
    for i in range(n, len(close)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def _macd(arr: np.ndarray):
    e12  = _ema(arr, 12)
    e26  = _ema(arr, 26)
    line = e12 - e26
    sig  = _ema(np.where(np.isnan(line), 0, line), 9)
    hist = line - sig
    return line, sig, hist


def _compute_all_indicators(data: Dict) -> Dict[str, np.ndarray]:
    close  = _to_arr(data["close"])
    high   = _to_arr(data.get("high",  data["close"]))
    low    = _to_arr(data.get("low",   data["close"]))
    open_  = _to_arr(data.get("open",  data["close"]))
    volume = _to_arr(data.get("volume", np.ones(len(close)) * 1_000_000))
    n      = len(close)

    rsi_arr              = _rsi(close, 14)
    sma20                = _sma(close, 20)
    sma50                = _sma(close, 50)
    ema12                = _ema(close, 12)
    ema26                = _ema(close, 26)
    macd_line, macd_sig, macd_hist = _macd(close)
    bb_upper, bb_mid, bb_lower     = _bollinger(close, 20)
    atr_arr              = _atr(high, low, close, 14)
    vol_sma20            = _sma(volume, 20)

    #Log return
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-9) / np.maximum(close[:-1], 1e-9))

    #Historical volatility (20-day annualised)
    hist_vol = np.full(n, np.nan)
    for i in range(20, n):
        hist_vol[i] = np.std(log_ret[i - 19:i + 1], ddof=1) * math.sqrt(252)

    #Garman-Klass volatility
    with np.errstate(divide="ignore", invalid="ignore"):
        gk = np.where(
            (high > 0) & (low > 0) & (close > 0) & (open_ > 0),
            0.5 * np.log(np.maximum(high / low, 1e-9)) ** 2
            - (2 * math.log(2) - 1) * np.log(np.maximum(close / open_, 1e-9)) ** 2,
            0.0
        )

    #OBV change (normalised)
    obv = np.zeros(n)
    for i in range(1, n):
        obv[i] = obv[i - 1] + (volume[i] if close[i] > close[i - 1] else
                                -volume[i] if close[i] < close[i - 1] else 0)
    obv_chg = np.diff(obv, prepend=obv[0])
    vol_norm = max(float(volume.max()), 1.0)

    #Volume-price trend (normalised)
    vpt = np.zeros(n)
    for i in range(1, n):
        if close[i - 1] > 0:
            vpt[i] = vpt[i - 1] + volume[i] * (close[i] - close[i - 1]) / close[i - 1]
    vpt_max = max(float(np.abs(vpt).max()), 1.0)

    #Stochastic %K and %D
    stoch_k = np.full(n, np.nan)
    for i in range(14, n):
        ph = np.max(high[i - 13:i + 1])
        pl = np.min(low[i - 13:i + 1])
        stoch_k[i] = (close[i] - pl) / (ph - pl) * 100 if ph != pl else 50.0
    stoch_d = _sma(np.where(np.isnan(stoch_k), 50.0, stoch_k), 3)

    safe = np.where(close > 0, close, 1.0)
    safe_vol = np.where(vol_sma20 > 0, vol_sma20, 1.0)
    safe_sma20 = np.where(np.isnan(sma20) | (sma20 == 0), 1.0, sma20)
    safe_sma50 = np.where(np.isnan(sma50) | (sma50 == 0), 1.0, sma50)
    safe_ema12 = np.where(np.isnan(ema12) | (ema12 == 0), 1.0, ema12)
    safe_ema26 = np.where(np.isnan(ema26) | (ema26 == 0), 1.0, ema26)

    return {
        "rsi_14":            rsi_arr,
        "sma_20":            sma20,
        "sma_50":            sma50,
        "ema_12":            ema12,
        "ema_26":            ema26,
        "macd_line":         macd_line,
        "macd_signal":       macd_sig,
        "macd_hist":         macd_hist,
        "bb_upper":          bb_upper,
        "bb_mid":            bb_mid,
        "bb_lower":          bb_lower,
        "atr":               atr_arr,
        "vol_sma20":         vol_sma20,
        "log_return":        log_ret,
        "hist_volatility":   hist_vol,
        "garman_klass_vol":  gk,
        "stoch_k":           stoch_k,
        "stoch_d":           stoch_d,
        #Feature-matrix columns (normalised, for AI strategy)
        "high_low_ratio":    (high - low) / safe,
        "close_open_ratio":  (close - open_) / np.where(open_ > 0, open_, 1.0),
        "volume_ratio":      volume / safe_vol,
        "log_volume":        np.log(np.maximum(volume, 1.0)),
        "sma_20_ratio":      close / safe_sma20,
        "sma_50_ratio":      close / safe_sma50,
        "ema_12_ratio":      close / safe_ema12,
        "ema_26_ratio":      close / safe_ema26,
        "macd_norm":         macd_line / safe,
        "macd_signal_norm":  macd_sig  / safe,
        "macd_hist_norm":    macd_hist / safe,
        "atr_ratio":         atr_arr   / safe,
        "obv_change":        obv_chg   / vol_norm,
        "volume_price_trend": vpt      / vpt_max,
    }


#Concrete strategies───

class RSIMeanReversionStrategy(Strategy):
    """Buy on RSI crossover below 30, sell on crossover above 70."""

    def __init__(self, buy_th=30, sell_th=70):
        self.buy_th  = buy_th
        self.sell_th = sell_th

    @property
    def label(self): return "RSI Mean Reversion"

    def generate_signals(self, data: Dict) -> List[Signal]:
        ind   = _compute_all_indicators(data)
        rsi   = ind["rsi_14"]
        n     = len(data["close"])
        sigs  = [Signal("HOLD")] * n
        in_pos = False

        for i in range(1, n):
            r, r_prev = rsi[i], rsi[i - 1]
            if np.isnan(r) or np.isnan(r_prev):
                continue
            if not in_pos and r < self.buy_th and r_prev >= self.buy_th:
                sigs[i] = Signal("BUY", f"RSI crossed below {self.buy_th} ({r:.1f})")
                in_pos   = True
            elif in_pos and r > self.sell_th and r_prev <= self.sell_th:
                sigs[i] = Signal("SELL", f"RSI crossed above {self.sell_th} ({r:.1f})")
                in_pos   = False
        return sigs


class MACrossoverStrategy(Strategy):
    """20/50 SMA golden-cross buy, death-cross sell."""

    def __init__(self, fast=20, slow=50):
        self.fast = fast
        self.slow = slow

    @property
    def label(self): return f"MA Crossover {self.fast}/{self.slow}"

    def generate_signals(self, data: Dict) -> List[Signal]:
        close  = _to_arr(data["close"])
        n      = len(close)
        fast_m = _sma(close, self.fast)
        slow_m = _sma(close, self.slow)
        sigs   = [Signal("HOLD")] * n
        in_pos = False

        for i in range(1, n):
            f, f_p = fast_m[i], fast_m[i - 1]
            s, s_p = slow_m[i], slow_m[i - 1]
            if np.isnan(f) or np.isnan(s) or np.isnan(f_p) or np.isnan(s_p):
                continue
            golden = f > s and f_p <= s_p
            death  = f < s and f_p >= s_p
            if not in_pos and golden:
                sigs[i] = Signal("BUY",  f"Golden cross SMA{self.fast}/SMA{self.slow}")
                in_pos   = True
            elif in_pos and death:
                sigs[i] = Signal("SELL", f"Death cross SMA{self.fast}/SMA{self.slow}")
                in_pos   = False
        return sigs


class MomentumStrategy(Strategy):
    """20-day price momentum with volume surge confirmation."""

    def __init__(self, lookback=20, threshold=0.04, vol_mult=1.2):
        self.lookback   = lookback
        self.threshold  = threshold
        self.vol_mult   = vol_mult

    @property
    def label(self): return "Momentum"

    def generate_signals(self, data: Dict) -> List[Signal]:
        close   = _to_arr(data["close"])
        volume  = _to_arr(data.get("volume", np.ones(len(close)) * 1e6))
        n       = len(close)
        lb      = self.lookback
        vol_sma = _sma(volume, lb)
        sigs    = [Signal("HOLD")] * n
        in_pos  = False

        for i in range(lb + 1, n):
            if close[i - lb] <= 0:
                continue
            mom     = (close[i] - close[i - lb]) / close[i - lb]
            mom_p   = (close[i - 1] - close[i - lb - 1]) / max(close[i - lb - 1], 1e-9)
            vol_ok  = (not np.isnan(vol_sma[i])) and (volume[i] > self.vol_mult * vol_sma[i])

            if not in_pos and mom > self.threshold and vol_ok:
                sigs[i] = Signal("BUY",  f"Momentum {mom * 100:.1f}% + volume surge")
                in_pos   = True
            elif in_pos and mom < 0:
                sigs[i] = Signal("SELL", f"Momentum turned negative ({mom * 100:.1f}%)")
                in_pos   = False
        return sigs


class VolatilityBreakoutStrategy(Strategy):
    """Bollinger Band squeeze followed by breakout entry."""

    def __init__(self, bb_period=20, mult=2.0, squeeze_pctile=25):
        self.bb_period      = bb_period
        self.mult           = mult
        self.squeeze_pctile = squeeze_pctile

    @property
    def label(self): return "Volatility Breakout (BB)"

    def generate_signals(self, data: Dict) -> List[Signal]:
        close  = _to_arr(data["close"])
        n      = len(close)
        upper, mid, lower = _bollinger(close, self.bb_period, self.mult)
        bandwidth = np.where(mid > 0, (upper - lower) / mid, np.nan)
        sigs   = [Signal("HOLD")] * n
        in_pos = False

        for i in range(self.bb_period + 20, n):
            if np.isnan(bandwidth[i]) or np.isnan(upper[i]):
                continue
            bw_hist   = bandwidth[i - 20:i]
            bw_hist   = bw_hist[~np.isnan(bw_hist)]
            if len(bw_hist) < 10:
                continue
            sq_thresh = np.percentile(bw_hist, self.squeeze_pctile)
            in_squeeze = bandwidth[i - 1] < sq_thresh

            if not in_pos and in_squeeze and close[i] > upper[i] and close[i - 1] <= upper[i - 1]:
                sigs[i] = Signal("BUY",  "BB squeeze breakout above upper band")
                in_pos   = True
            elif in_pos and close[i] < mid[i]:
                sigs[i] = Signal("SELL", "Price closed below BB midline")
                in_pos   = False
        return sigs


class AIDrivenStrategy(Strategy):

    FEATURE_COLS = [
        "log_return", "high_low_ratio", "close_open_ratio", "volume_ratio",
        "log_volume", "sma_20_ratio", "sma_50_ratio", "ema_12_ratio",
        "ema_26_ratio", "rsi_14", "macd_norm", "macd_signal_norm",
        "macd_hist_norm", "stoch_k", "stoch_d", "atr_ratio",
        "hist_volatility", "garman_klass_vol", "obv_change", "volume_price_trend",
    ]

    def __init__(self, confidence_thresh=0.58, up_thresh=0.55, down_thresh=0.45):
        self.conf_thresh = confidence_thresh
        self.up_thresh   = up_thresh
        self.down_thresh = down_thresh

    @property
    def label(self): return "AI-Driven (AuraMarketNet)"

    def generate_signals(self, data: Dict) -> List[Signal]:
        try:
            return self._model_signals(data)
        except Exception as exc:
            logger.warning(f"AI strategy fell back to composite: {exc}")
            return self._composite_fallback(data)

    def _model_signals(self, data: Dict) -> List[Signal]:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from utils.market_inference import (
            is_market_model_loaded, _model, _device, _SEQ_LEN,
        )
        import torch
        import torch.nn.functional as F

        if not is_market_model_loaded() or _model is None:
            return self._composite_fallback(data)

        #MPS/CPU inference is unreliable in Flask worker threads — fall back
        if _device.type != "cuda":
            logger.info("AI backtest: non-CUDA device (%s), using composite fallback", _device.type)
            return self._composite_fallback(data)

        ind    = _compute_all_indicators(data)
        close  = _to_arr(data["close"])
        n      = len(close)
        sigs   = [Signal("HOLD")] * n

        #Build feature matrix [n, 20]
        feat = np.zeros((n, len(self.FEATURE_COLS)), dtype=np.float32)
        for col_idx, col in enumerate(self.FEATURE_COLS):
            arr = ind.get(col)
            if arr is None:
                continue
            col_arr = np.asarray(arr, dtype=np.float32)
            col_arr = np.where(np.isnan(col_arr), 0.0, col_arr)
            feat[:, col_idx] = col_arr[:n]

        #Forward-fill zeros
        for col_idx in range(feat.shape[1]):
            last = 0.0
            for i in range(n):
                if feat[i, col_idx] != 0.0:
                    last = feat[i, col_idx]
                elif i > 0:
                    feat[i, col_idx] = last

        if n <= _SEQ_LEN:
            return sigs

        MAX_TEXTS, MAX_TOKEN_LEN = 5, 128
        windows  = np.stack([feat[i - _SEQ_LEN:i] for i in range(_SEQ_LEN, n)])
        n_win    = len(windows)
        up_probs = np.zeros(n_win, dtype=np.float32)
        BATCH    = 64  #safe for CPU/MPS/CUDA

        for b in range(0, n_win, BATCH):
            batch_np = windows[b:b + BATCH]
            bs       = batch_np.shape[0]
            num_t    = torch.tensor(batch_np, dtype=torch.float32, device=_device)
            ids      = torch.zeros(bs, MAX_TEXTS, MAX_TOKEN_LEN, dtype=torch.long,  device=_device)
            mask     = torch.zeros(bs, MAX_TEXTS, MAX_TOKEN_LEN, dtype=torch.long,  device=_device)
            tmask    = torch.zeros(bs, MAX_TEXTS,                dtype=torch.bool,   device=_device)
            tmask[:, 0] = True  #mark first slot as valid so model doesn't NaN

            with torch.no_grad():
                out = _model(
                    input_ids=ids, attention_mask=mask,
                    text_mask=tmask, numerical_features=num_t,
                )
            probs = F.softmax(out["direction_logits"], dim=-1)  #[bs, 2]
            up_probs[b:b + bs] = probs[:, 1].cpu().numpy()

        in_pos = False
        for idx, up_p in enumerate(up_probs):
            i     = idx + _SEQ_LEN
            conf  = max(up_p, 1 - up_p)
            if conf < self.conf_thresh:
                continue
            if not in_pos and up_p > self.up_thresh:
                sigs[i] = Signal("BUY",  f"Model UP {up_p:.2%} conf {conf:.2%}")
                in_pos   = True
            elif in_pos and up_p < self.down_thresh:
                sigs[i] = Signal("SELL", f"Model DOWN {1-up_p:.2%} conf {conf:.2%}")
                in_pos   = False
        return sigs

    def _composite_fallback(self, data: Dict) -> List[Signal]:
        """RSI + momentum composite when model not available."""
        ind    = _compute_all_indicators(data)
        rsi    = ind["rsi_14"]
        close  = _to_arr(data["close"])
        n      = len(close)
        sigs   = [Signal("HOLD")] * n
        in_pos = False

        for i in range(21, n):
            r = rsi[i] if not np.isnan(rsi[i]) else 50.0
            if close[i - 20] > 0:
                mom = (close[i] - close[i - 20]) / close[i - 20]
            else:
                mom = 0.0
            score = (50 - r) / 50 * 0.6 + mom * 5 * 0.4
            if not in_pos and score > 0.25:
                sigs[i] = Signal("BUY",  f"RSI+momentum composite score {score:.2f}")
                in_pos   = True
            elif in_pos and score < -0.15:
                sigs[i] = Signal("SELL", f"Composite score {score:.2f}")
                in_pos   = False
        return sigs


#Simulator──

class BacktestSimulator:
    def __init__(
        self,
        initial:    float = 10_000.0,
        commission: float = 0.001,
        slippage:   float = 0.0005,
        stop_loss:  Optional[float] = None,
    ):
        self.initial    = initial
        self.commission = commission
        self.slippage   = slippage
        self.stop_loss  = stop_loss

    def run(self, data: Dict, signals: List[Signal]) -> Dict:
        close  = data["close"]
        high   = data.get("high",  close)
        low    = data.get("low",   close)
        ts     = data.get("timestamps", [str(i) for i in range(len(close))])
        n      = len(close)

        cash      = self.initial
        shares    = 0.0
        in_pos    = False
        entry_px  = 0.0
        entry_idx = 0
        entry_reason = ""

        equity      : List[float] = []
        bh_shares   = self.initial / close[0] if close[0] else 0.0
        bh_curve    : List[float] = []
        in_pos_arr  : List[bool]  = []
        trades      : List[ClosedTrade] = []

        for i in range(n):
            px = close[i]
            if px <= 0:
                equity.append(equity[-1] if equity else self.initial)
                bh_curve.append(round(bh_shares * px, 2))
                in_pos_arr.append(in_pos)
                continue

            #Stop-loss check───
            if in_pos and self.stop_loss:
                sl_px = entry_px * (1 - self.stop_loss)
                if low[i] <= sl_px:
                    exit_px = sl_px * (1 - self.slippage)
                    proceeds = shares * exit_px * (1 - self.commission)
                    net_pnl  = proceeds - shares * entry_px
                    trades.append(ClosedTrade(
                        entry_date=ts[entry_idx], exit_date=ts[i],
                        entry_price=round(entry_px, 4), exit_price=round(exit_px, 4),
                        return_pct=round((exit_px / entry_px - 1) * 100, 3),
                        net_pnl=round(net_pnl, 2), won=False,
                        reason_entry=entry_reason, reason_exit=f"Stop-loss at {sl_px:.2f}",
                    ))
                    cash    = proceeds
                    shares  = 0.0
                    in_pos  = False

            sig = signals[i]

            #Buy────
            if sig.action == "BUY" and not in_pos and cash > 0:
                buy_px   = px * (1 + self.slippage)
                cost     = cash * self.commission
                shares   = (cash - cost) / buy_px
                entry_px = buy_px
                entry_idx = i
                entry_reason = sig.reason
                cash     = 0.0
                in_pos   = True

            #Sell───
            elif sig.action == "SELL" and in_pos:
                sell_px  = px * (1 - self.slippage)
                proceeds = shares * sell_px * (1 - self.commission)
                net_pnl  = proceeds - shares * entry_px
                trades.append(ClosedTrade(
                    entry_date=ts[entry_idx], exit_date=ts[i],
                    entry_price=round(entry_px, 4), exit_price=round(sell_px, 4),
                    return_pct=round((sell_px / entry_px - 1) * 100, 3),
                    net_pnl=round(net_pnl, 2), won=net_pnl > 0,
                    reason_entry=entry_reason, reason_exit=sig.reason,
                ))
                cash    = proceeds
                shares  = 0.0
                in_pos  = False

            current = cash + shares * px
            equity.append(round(current, 2))
            bh_curve.append(round(bh_shares * px, 2))
            in_pos_arr.append(shares > 0)

        #Close open position at final bar (mark-to-market, no commission)
        if in_pos and shares > 0:
            final_px = close[-1] * (1 - self.slippage)
            net_pnl  = shares * final_px - shares * entry_px
            trades.append(ClosedTrade(
                entry_date=ts[entry_idx], exit_date=ts[-1],
                entry_price=round(entry_px, 4), exit_price=round(final_px, 4),
                return_pct=round((final_px / entry_px - 1) * 100, 3),
                net_pnl=round(net_pnl, 2), won=net_pnl > 0,
                reason_entry=entry_reason, reason_exit="End of backtest (MTM)",
            ))

        #Drawdown series
        peak = self.initial
        drawdown: List[float] = []
        for v in equity:
            if v > peak:
                peak = v
            drawdown.append(round((v - peak) / peak * 100, 3))

        return {
            "equity":    equity,
            "bh_curve":  bh_curve,
            "drawdown":  drawdown,
            "trades":    trades,
            "in_pos":    in_pos_arr,
        }


#Metrics────

class MetricsComputer:
    @staticmethod
    def compute(
        equity:   List[float],
        bh_curve: List[float],
        trades:   List[ClosedTrade],
        in_pos:   List[bool],
        initial:  float,
        trading_days: int = 252,
    ) -> Dict[str, Any]:
        n = len(equity)
        if n < 2:
            return {}

        final    = equity[-1]
        bh_final = bh_curve[-1] if bh_curve else initial
        years    = n / trading_days

        total_ret    = (final - initial) / initial * 100
        bh_ret       = (bh_final - initial) / initial * 100
        ann_ret      = ((final / initial) ** (1 / max(years, 1e-6)) - 1) * 100

        #Active-only daily returns (avoids idle-cash Sharpe inflation) ──
        eq_arr = np.array(equity)
        ip_arr = np.array(in_pos)
        #Days where we were IN position
        active_idx = np.where(ip_arr)[0]
        if len(active_idx) > 1:
            active_eq  = eq_arr[active_idx]
            active_ret = np.diff(active_eq) / active_eq[:-1]
        else:
            active_ret = np.array([0.0])

        exposure_pct = ip_arr.sum() / n * 100

        excess = active_ret - RISK_FREE_DAILY
        std_e  = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
        sharpe = float(np.mean(excess) / std_e * math.sqrt(trading_days)) if std_e > 0 else 0.0

        down = excess[excess < 0]
        std_d = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
        sortino = float(np.mean(excess) / std_d * math.sqrt(trading_days)) if std_d > 0 else 0.0

        #Max drawdown─
        peak   = initial
        max_dd = 0.0
        for v in equity:
            peak   = max(peak, v)
            max_dd = min(max_dd, (v - peak) / peak * 100)

        calmar = abs(ann_ret / max_dd) if max_dd < -0.01 else 0.0

        #Trade stats
        num_trades  = len(trades)
        won         = [t for t in trades if t.won]
        lost        = [t for t in trades if not t.won]
        win_rate    = len(won) / num_trades * 100 if num_trades else 0.0
        avg_trade   = float(np.mean([t.return_pct for t in trades])) if trades else 0.0
        avg_win     = float(np.mean([t.return_pct for t in won]))    if won    else 0.0
        avg_loss    = float(np.mean([t.return_pct for t in lost]))   if lost   else 0.0
        gross_won   = sum(t.net_pnl for t in won)
        gross_lost  = abs(sum(t.net_pnl for t in lost))
        profit_fac  = gross_won / gross_lost if gross_lost > 0 else (99.9 if gross_won > 0 else 0.0)

        #Alpha / Beta─
        all_ret = np.diff(eq_arr) / eq_arr[:-1]
        bh_ret_arr = np.diff(np.array(bh_curve)) / np.array(bh_curve)[:-1] if bh_curve else all_ret
        min_len = min(len(all_ret), len(bh_ret_arr))
        beta = alpha = 0.0
        if min_len > 5:
            cov = np.cov(all_ret[:min_len], bh_ret_arr[:min_len])
            var_bh = cov[1, 1]
            beta  = float(cov[0, 1] / var_bh) if var_bh > 0 else 0.0
            alpha = float((np.mean(all_ret[:min_len]) - beta * np.mean(bh_ret_arr[:min_len]))
                          * trading_days * 100)

        #Statistical warning───
        if num_trades == 0:
            warning = "No completed trades — strategy generated no signals in this period."
        elif num_trades < MIN_TRADES_RELIABLE:
            warning = (f"Only {num_trades} trade(s) — results are NOT statistically significant "
                       f"(minimum {MIN_TRADES_RELIABLE} recommended).")
        elif num_trades < MIN_TRADES_ROBUST:
            warning = (f"{num_trades} trades — treat with caution "
                       f"(minimum {MIN_TRADES_ROBUST} recommended for robustness).")
        else:
            warning = None

        return {
            "total_return":       round(total_ret,   2),
            "bh_return":          round(bh_ret,       2),
            "annualized_return":  round(ann_ret,      2),
            "sharpe":             round(sharpe,        2),
            "sortino":            round(sortino,       2),
            "calmar":             round(calmar,        2),
            "max_drawdown":       round(max_dd,        2),
            "win_rate":           round(win_rate,      1),
            "num_trades":         num_trades,
            "trades_won":         len(won),
            "trades_lost":        len(lost),
            "avg_trade_return":   round(avg_trade,    2),
            "avg_win":            round(avg_win,       2),
            "avg_loss":           round(avg_loss,      2),
            "profit_factor":      round(min(profit_fac, 99.9), 2),
            "exposure_pct":       round(exposure_pct, 1),
            "beta":               round(beta,          3),
            "alpha":              round(alpha,          2),
            "statistical_warning": warning,
        }


#Registry & orchestrator

STRATEGY_REGISTRY: Dict[str, type] = {
    "rsi":        RSIMeanReversionStrategy,
    "ma_cross":   MACrossoverStrategy,
    "momentum":   MomentumStrategy,
    "volatility": VolatilityBreakoutStrategy,
    "ai":         AIDrivenStrategy,
}


def _run_single(data: Dict, strategy: Strategy, sim: BacktestSimulator) -> Dict:
    sigs      = strategy.generate_signals(data)
    sim_res   = sim.run(data, sigs)
    metrics   = MetricsComputer.compute(
        sim_res["equity"], sim_res["bh_curve"],
        sim_res["trades"], sim_res["in_pos"],
        sim.initial,
    )
    trade_dicts = [
        {
            "entry_date":   t.entry_date,
            "exit_date":    t.exit_date,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "return_pct":   t.return_pct,
            "net_pnl":      t.net_pnl,
            "won":          t.won,
            "reason_entry": t.reason_entry,
            "reason_exit":  t.reason_exit,
        }
        for t in sim_res["trades"]
    ]
    return {
        "equity":    sim_res["equity"],
        "bh_curve":  sim_res["bh_curve"],
        "drawdown":  sim_res["drawdown"],
        "trades":    trade_dicts,
        "metrics":   metrics,
    }


def run_backtest(
    data_dict:   Dict,
    strategy_key: str   = "rsi",
    commission:   float = 0.001,
    slippage:     float = 0.0005,
    stop_loss:    Optional[float] = None,
    initial:      float = 10_000.0,
) -> Dict[str, Any]:
    """
    Main entry point.

    data_dict must contain: timestamps, open, high, low, close, volume (lists).
    Returns a JSON-serialisable dict.
    """
    close = data_dict.get("close", [])
    if len(close) < 30:
        return {"error": "Insufficient data (need ≥30 bars)"}

    sim = BacktestSimulator(
        initial=initial, commission=commission,
        slippage=slippage, stop_loss=stop_loss,
    )
    ts = data_dict.get("timestamps", [])

    #Comparison mode
    if strategy_key == "all":
        strategies_out = {}
        bh_curve       = None
        for key, cls in STRATEGY_REGISTRY.items():
            try:
                res = _run_single(data_dict, cls(), sim)
                strategies_out[key] = {
                    "label":    cls().label,
                    "equity":   res["equity"],
                    "drawdown": res["drawdown"],
                    "metrics":  res["metrics"],
                }
                if bh_curve is None:
                    bh_curve = res["bh_curve"]
            except Exception as exc:
                logger.warning(f"Strategy {key} failed: {exc}")

        #Legacy fields from best-sharpe strategy
        best = max(strategies_out.values(),
                   key=lambda v: v["metrics"].get("sharpe", -99),
                   default={})
        bm = best.get("metrics", {})
        return {
            "mode":         "comparison",
            "timestamps":   ts,
            "buy_hold":     bh_curve or [],
            "strategies":   strategies_out,
            #legacy
            "strat_return": bm.get("total_return",  0),
            "bh_return":    bm.get("bh_return",      0),
            "sharpe":       bm.get("sharpe",         0),
            "trades_won":   bm.get("trades_won",     0),
            "trades_lost":  bm.get("trades_lost",    0),
        }

    #Single strategy
    key = strategy_key if strategy_key in STRATEGY_REGISTRY else "rsi"
    strategy = STRATEGY_REGISTRY[key]()
    res      = _run_single(data_dict, strategy, sim)
    m        = res["metrics"]

    return {
        "mode":           "single",
        "strategy":       key,
        "strategy_label": strategy.label,
        "timestamps":     ts,
        "equity":         res["equity"],
        "buy_hold":       res["bh_curve"],
        "drawdown":       res["drawdown"],
        "trades":         res["trades"],
        "metrics":        m,
        #legacy fields
        "strat_return":   m.get("total_return",  0),
        "bh_return":      m.get("bh_return",      0),
        "sharpe":         m.get("sharpe",         0),
        "trades_won":     m.get("trades_won",     0),
        "trades_lost":    m.get("trades_lost",    0),
    }
