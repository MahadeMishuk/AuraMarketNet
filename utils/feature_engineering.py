import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    FEATURE_COLUMNS = [
        "log_return",
        "high_low_ratio",
        "close_open_ratio",
        "volume_ratio",     #volume / 20-day avg volume
        "log_volume",
        "sma_20_ratio",     #close / SMA20 - 1
        "sma_50_ratio",     #close / SMA50 - 1
        "ema_12_ratio",
        "ema_26_ratio",
        "rsi_14",           #Normalized 0-1
        "macd_norm",        #MACD / ATR
        "macd_signal_norm",
        "macd_hist_norm",
        "stoch_k",          #0-100 → normalized 0-1
        "stoch_d",
        "atr_ratio",        #ATR / close
        "hist_volatility",  #20-day historical volatility (annualized)
        "garman_klass_vol", #Garman-Klass volatility estimator
        "obv_change",       #OBV % change
        "volume_price_trend",
    ]

    def __init__(self, normalize: bool = True):
        self.normalize = normalize


    #TREND INDICATORS


    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Moving Average Convergence/Divergence."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        })

    @staticmethod
    def bollinger_bands(
        close: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands: mean ± k*std over rolling window."""
        sma = close.rolling(window=window, min_periods=1).mean()
        std = close.rolling(window=window, min_periods=1).std()
        return pd.DataFrame({
            "bb_upper": sma + num_std * std,
            "bb_mid": sma,
            "bb_lower": sma - num_std * std,
            "bb_width": (2 * num_std * std) / sma.replace(0, np.nan),
            "bb_pct_b": (close - (sma - num_std * std)) / (2 * num_std * std).replace(0, np.nan),
        })


    #MOMENTUM INDICATORS


    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k_window: int = 14, d_window: int = 3
    ) -> pd.DataFrame:
        """Stochastic Oscillator %K and %D."""
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(window=d_window, min_periods=1).mean()
        return pd.DataFrame({"stoch_k": k, "stoch_d": d})

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """Williams %R momentum indicator."""
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


    #VOLATILITY INDICATORS


    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range — measures market volatility."""
        prev_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return true_range.ewm(alpha=1 / window, adjust=False).mean()

    @staticmethod
    def historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
        """Rolling historical volatility (annualized standard deviation of log returns)."""
        log_returns = np.log(close / close.shift(1))
        return log_returns.rolling(window=window, min_periods=2).std() * np.sqrt(252)

    @staticmethod
    def garman_klass_volatility(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        Uses OHLC data for a more efficient volatility estimate than close-to-close.
        """
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return gk.rolling(window=window, min_periods=2).mean().apply(np.sqrt) * np.sqrt(252)


    #VOLUME INDICATORS


    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume — cumulative volume reflecting price direction."""
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend — variant of OBV using % price change."""
        pct_change = close.pct_change()
        return (pct_change * volume).cumsum()


    #MAIN COMPUTATION


    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators from OHLCV data.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
                and DatetimeIndex

        Returns:
            DataFrame with all computed indicators appended
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        #Log returns
        df["log_return"] = np.log(close / close.shift(1)).fillna(0)
        df["return_1d"] = close.pct_change().fillna(0)
        df["return_5d"] = close.pct_change(5).fillna(0)
        df["return_20d"] = close.pct_change(20).fillna(0)

        #Price ratios──
        df["high_low_ratio"] = (high / low - 1).fillna(0)
        df["close_open_ratio"] = (close / open_ - 1).fillna(0)

        #Moving averages─────
        for window in [10, 20, 50, 200]:
            df[f"sma_{window}"] = self.sma(close, window)
            df[f"sma_{window}_ratio"] = (close / df[f"sma_{window}"] - 1).fillna(0)

        for span in [12, 26, 50]:
            df[f"ema_{span}"] = self.ema(close, span)
            df[f"ema_{span}_ratio"] = (close / df[f"ema_{span}"] - 1).fillna(0)

        #MACD─────
        macd_df = self.macd(close)
        atr_series = self.atr(high, low, close)
        #Normalize MACD by ATR to make it scale-invariant across different stocks
        df["macd_norm"] = (macd_df["macd"] / atr_series.replace(0, np.nan)).fillna(0)
        df["macd_signal_norm"] = (macd_df["macd_signal"] / atr_series.replace(0, np.nan)).fillna(0)
        df["macd_hist_norm"] = (macd_df["macd_hist"] / atr_series.replace(0, np.nan)).fillna(0)

        #Bollinger Bands
        bb = self.bollinger_bands(close)
        df["bb_width"] = bb["bb_width"].fillna(0)
        df["bb_pct_b"] = bb["bb_pct_b"].fillna(0.5)

        #RSI─
        df["rsi_14"] = self.rsi(close) / 100.0  #Normalize to [0, 1]
        df["rsi_7"] = self.rsi(close, window=7) / 100.0

        #Stochastic
        stoch = self.stochastic(high, low, close)
        df["stoch_k"] = stoch["stoch_k"] / 100.0
        df["stoch_d"] = stoch["stoch_d"] / 100.0

        #Volatility
        df["atr"] = atr_series
        df["atr_ratio"] = (atr_series / close.replace(0, np.nan)).fillna(0)
        df["hist_volatility"] = self.historical_volatility(close).fillna(0)
        df["garman_klass_vol"] = self.garman_klass_volatility(
            open_, high, low, close
        ).fillna(0)

        #Volume────
        vol_sma_20 = volume.rolling(20, min_periods=1).mean()
        df["volume_ratio"] = (volume / vol_sma_20.replace(0, np.nan)).fillna(1.0)
        df["log_volume"] = np.log1p(volume)

        obv = self.on_balance_volume(close, volume)
        df["obv_change"] = obv.pct_change().fillna(0).clip(-1, 1)  #Clip extreme values
        df["volume_price_trend"] = self.volume_price_trend(close, volume)
        #Normalize VPT
        vpt_std = df["volume_price_trend"].std()
        if vpt_std > 0:
            df["volume_price_trend"] = (df["volume_price_trend"] - df["volume_price_trend"].mean()) / vpt_std

        #Forward return (prediction target) ──
        df["target_return_1d"] = close.pct_change().shift(-1)  #Next day return
        df["target_direction"] = (df["target_return_1d"] > 0).astype(int)
        df["target_volatility"] = df["hist_volatility"].shift(-1)

        logger.info(f"Computed {len(df.columns)} features for {len(df)} rows")
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the model input feature matrix from a processed DataFrame.

        Returns:
            numpy array of shape [n_timesteps, n_features]
        """
        available = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        missing = [col for col in self.FEATURE_COLUMNS if col not in df.columns]

        if missing:
            logger.warning(f"Missing feature columns: {missing}")

        feature_matrix = df[available].values.astype(np.float32)

        #Replace NaN/inf with 0 (should be minimal after proper processing)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix

    def get_target_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract target labels for multi-task learning.

        Returns:
            - direction labels: [n] binary (0=DOWN, 1=UP)
            - return labels:    [n] float (% change)
            - volatility labels:[n] float
        """
        direction = df["target_direction"].values.astype(np.int64)
        returns = df["target_return_1d"].values.astype(np.float32)
        volatility = df["target_volatility"].ffill().values.astype(np.float32)

        return direction, returns, volatility

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for time-series input.

        Each sample is: X[t-seq_len+1 : t] → predict at t+1

        Returns:
            features:    [n_samples, seq_len, n_features]
            directions:  [n_samples]
            returns:     [n_samples]
            volatilities:[n_samples]
            dates:       [n_samples] — the prediction date
        """
        feature_matrix = self.get_feature_matrix(df)
        direction, returns, volatility = self.get_target_labels(df)
        dates = df.index.values

        n = len(df)
        samples_features = []
        samples_direction = []
        samples_returns = []
        samples_vol = []
        samples_dates = []

        for i in range(sequence_length, n - 1):
            #Features: past sequence_length days
            x = feature_matrix[i - sequence_length: i]

            #Target: next day
            d = direction[i]
            r = returns[i]
            v = volatility[i]

            #Skip samples with NaN targets
            if np.isnan(r) or np.isnan(v):
                continue

            samples_features.append(x)
            samples_direction.append(d)
            samples_returns.append(r)
            samples_vol.append(v)
            samples_dates.append(dates[i])

        return (
            np.array(samples_features, dtype=np.float32),
            np.array(samples_direction, dtype=np.int64),
            np.array(samples_returns, dtype=np.float32),
            np.array(samples_vol, dtype=np.float32),
            np.array(samples_dates),
        )
