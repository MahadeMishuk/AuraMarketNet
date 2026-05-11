import logging
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#Checkpoint & model constants─────
#Prefer the 30-epoch final weights; fall back to the epoch-1 best checkpoint
_CHECKPOINT_CANDIDATES = [
    Path("AuraMarketNet-v1_checkpoints/AuraMarketNet-v1_final.pth"),
    Path("AuraMarketNet-v1_checkpoints/aura_market_net_best.pt"),
    Path("checkpoints/aura_market_net_best.pt"),
]
_CHECKPOINT_PATH = next((p for p in _CHECKPOINT_CANDIDATES if p.exists()),
                        _CHECKPOINT_CANDIDATES[-1])
_TOKENIZER_MODEL = "ProsusAI/finbert"
_SEQ_LEN         = 30    #30-day rolling window expected by numerical encoder
_MAX_TEXTS       = 5     #max headlines fed to MultiTextEncoder per prediction
_MAX_TOKEN_LEN   = 128

#Fusion weights (tunable once a better checkpoint is available)
_SENTIMENT_WEIGHT = 0.30   #FinBERT contribution to direction probability

#Horizon → (sqrt-scale factor for volatility)
_HORIZON_VOL_SCALE = {"1H": (1 / 6.5) ** 0.5, "1D": 1.0, "1W": 5.0 ** 0.5}

#Singleton state─
_lock            = threading.Lock()
_model           = None
_tokenizer       = None
_device: Optional[torch.device] = None
_feature_eng     = None
_load_attempted  = False
_checkpoint_meta: Dict[str, Any] = {}



#DEVICE SELECTION


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



#MODEL LOADING


def load_market_model(
    checkpoint_path: str = str(_CHECKPOINT_PATH),
) -> bool:
    """
    Thread-safe singleton load of the AuraMarketNet checkpoint.

    Accepts two on-disk formats:
        • Full checkpoint dict  — saved by ModelCheckpoint; key "model_state_dict"
        • Plain state dict      — saved by torch.save(model.state_dict(), ...)

    Old checkpoints store numpy scalars in the metrics dict (saved before
    torch 2.6).  We allowlist numpy._core.multiarray.scalar for weights_only
    safety; fallback to weights_only=False if that still fails.

    Returns:
        True  → model loaded and ready for inference.
        False → checkpoint missing or load error.
    """
    global _model, _tokenizer, _device, _feature_eng, _load_attempted, _checkpoint_meta

    with _lock:
        if _load_attempted:
            return _model is not None

        _load_attempted = True
        ckpt_path = Path(checkpoint_path)

        if not ckpt_path.exists():
            logger.warning(
                f"AuraMarketNet checkpoint not found: {ckpt_path}. "
                "Predictions will be unavailable."
            )
            return False

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from models import create_model
            from utils.feature_engineering import FeatureEngineer
            from transformers import AutoTokenizer

            _device = _resolve_device()
            logger.info(f"Loading AuraMarketNet v1 on {_device} ← {ckpt_path}")

            #Load checkpoint — allowlist numpy scalar for weights_only safety
            try:
                import numpy._core.multiarray as _np_core
                torch.serialization.add_safe_globals([_np_core.scalar])
                ckpt = torch.load(
                    str(ckpt_path), map_location=_device, weights_only=True
                )
            except Exception:
                ckpt = torch.load(
                    str(ckpt_path), map_location=_device, weights_only=False
                )

            #Handle both full-checkpoint dict and plain state-dict formats
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                _checkpoint_meta = {
                    "epoch":   ckpt.get("epoch", "?"),
                    "metrics": ckpt.get("metrics", {}),
                }
            else:
                #Plain state dict — no epoch / metrics metadata available
                state_dict = ckpt
                _checkpoint_meta = {
                    "epoch":   "30",
                    "metrics": {"note": "plain state-dict, no per-epoch metrics"},
                }

            model = create_model()
            model.load_state_dict(state_dict)
            model = model.to(_device).eval()
            _model = model

            #Offline-first tokenizer (falls back to network on first run)
            try:
                _tokenizer = AutoTokenizer.from_pretrained(
                    _TOKENIZER_MODEL, local_files_only=True
                )
            except Exception:
                _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)

            _feature_eng = FeatureEngineer()
            _warmup_model()


            val_acc = _checkpoint_meta["metrics"].get("val_directional_accuracy", 0)
            val_vol_r = _checkpoint_meta["metrics"].get("val_vol_correlation", 0)
            logger.info(
                f"AuraMarketNet v1 ready | "
                f"epoch={_checkpoint_meta['epoch']} | "
                f"val_dir_acc={val_acc:.4f} | "
                f"val_vol_corr={val_vol_r:.4f} | "
                f"device={_device}"
            )
            return True

        except Exception as exc:
            logger.error(f"Failed to load AuraMarketNet: {exc}", exc_info=True)
            _model = None
            _tokenizer = None
            return False


@torch.no_grad()
def _warmup_model() -> None:
    """Single dummy forward pass to pre-compile the device kernel."""
    try:
        dummy_ids   = torch.zeros(1, _MAX_TEXTS, _MAX_TOKEN_LEN, dtype=torch.long, device=_device)
        dummy_mask  = torch.zeros(1, _MAX_TEXTS, _MAX_TOKEN_LEN, dtype=torch.long, device=_device)
        dummy_tmask = torch.zeros(1, _MAX_TEXTS, dtype=torch.bool, device=_device)
        dummy_tmask[0, 0] = True
        dummy_num   = torch.zeros(1, _SEQ_LEN, 20, dtype=torch.float32, device=_device)
        _model(
            input_ids=dummy_ids, attention_mask=dummy_mask,
            text_mask=dummy_tmask, numerical_features=dummy_num,
        )
        logger.info("AuraMarketNet kernel warm-up complete")
    except Exception as exc:
        logger.debug(f"Warm-up skipped ({exc})")


def is_market_model_loaded() -> bool:
    """Return True if the AuraMarketNet model is loaded and ready."""
    return _model is not None and _tokenizer is not None


def get_checkpoint_meta() -> Dict[str, Any]:
    """Return epoch and metric metadata from the loaded checkpoint."""
    return _checkpoint_meta.copy()



#INTERNAL HELPERS


def _fetch_ohlcv(ticker: str, days: int = 120) -> Optional[pd.DataFrame]:
    """Fetch raw OHLCV from yfinance.  Returns None on failure."""
    try:
        import yfinance as yf
        end   = datetime.utcnow()
        start = end - timedelta(days=days)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index).tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        logger.warning(f"OHLCV fetch failed for {ticker}: {exc}")
        return None


def _build_feature_tensor(df_raw: pd.DataFrame) -> torch.Tensor:
    """
    Engineer TA indicators and return [1, SEQ_LEN, 20] float32 tensor.
    Raises ValueError if insufficient data.
    """
    df_feat = _feature_eng.compute_all_indicators(df_raw)
    feat    = _feature_eng.get_feature_matrix(df_feat).astype(np.float32)

    if len(feat) < _SEQ_LEN:
        #Zero-pad from the left to reach SEQ_LEN
        pad  = np.zeros((_SEQ_LEN - len(feat), feat.shape[1]), dtype=np.float32)
        feat = np.vstack([pad, feat])

    seq = feat[-_SEQ_LEN:]                                    #[30, 20]
    return torch.tensor(seq).unsqueeze(0).to(_device)         #[1, 30, 20]


def _tokenize_headlines(headlines: List[str]) -> Dict[str, torch.Tensor]:
    """
    Tokenize up to _MAX_TEXTS headlines → padded tensors for MultiTextEncoder.

    Returns dict:
        input_ids:      [1, MAX_TEXTS, MAX_LEN]
        attention_mask: [1, MAX_TEXTS, MAX_LEN]
        text_mask:      [1, MAX_TEXTS]  — True for real slots, False for padding
    """
    texts = [(h or "")[:500] for h in headlines[:_MAX_TEXTS]] or ["market analysis"]
    n     = len(texts)

    enc = _tokenizer(
        texts,
        max_length=_MAX_TOKEN_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    #Pad empty slots to reach _MAX_TEXTS
    if n < _MAX_TEXTS:
        z = torch.zeros(_MAX_TEXTS - n, _MAX_TOKEN_LEN, dtype=torch.long)
        enc["input_ids"]      = torch.cat([enc["input_ids"],      z], dim=0)
        enc["attention_mask"] = torch.cat([enc["attention_mask"], z], dim=0)

    text_mask        = torch.zeros(_MAX_TEXTS, dtype=torch.bool)
    text_mask[:n]    = True

    return {
        "input_ids":      enc["input_ids"].unsqueeze(0).to(_device),       #[1, 5, 128]
        "attention_mask": enc["attention_mask"].unsqueeze(0).to(_device),   #[1, 5, 128]
        "text_mask":      text_mask.unsqueeze(0).to(_device),               #[1, 5]
    }



#CORE INFERENCE


@torch.no_grad()
def predict_market(
    ticker: str,
    headlines: Optional[List[str]] = None,
    horizon: str = "1D",
) -> Dict[str, Any]:

    if not is_market_model_loaded():
        raise RuntimeError("AuraMarketNet model not loaded — call load_market_model() first")

    if not headlines:
        headlines = [f"{ticker} market outlook"]


    df_raw = _fetch_ohlcv(ticker)
    if df_raw is None or len(df_raw) < max(_SEQ_LEN, 52):
        raise ValueError(f"Insufficient OHLCV data for {ticker} (got {len(df_raw) if df_raw is not None else 0} rows, need {_SEQ_LEN})")

    num_tensor   = _build_feature_tensor(df_raw)                 

    txt          = _tokenize_headlines(headlines)

    amp_ctx = torch.amp.autocast("cuda") if _device.type == "cuda" else nullcontext()
    with amp_ctx:
        raw = _model(
            input_ids          = txt["input_ids"],
            attention_mask     = txt["attention_mask"],
            text_mask          = txt["text_mask"],
            numerical_features = num_tensor,
        )

    dir_probs   = F.softmax(raw["direction_logits"], dim=-1)[0] 
    model_down  = dir_probs[0].item()
    model_up    = dir_probs[1].item()

    #Regression head: predicted % price change (model outputs fractional)
    price_chg   = raw["price_change"][0, 0].item() * 100           #→ %

    #Volatility head (Softplus output → always positive)
    raw_vol     = raw["volatility"][0, 0].item()

    #Auxiliary sentiment from text encoder head
    aux_sent_label = "neutral"
    if "sentiment_logits" in raw:
        sent_p      = F.softmax(raw["sentiment_logits"][0], dim=-1)
        #Label ordering inside model: 0=neutral, 1=positive, 2=negative
        aux_sent_label = ["neutral", "positive", "negative"][sent_p.argmax().item()]


    finbert_score = 0.0
    finbert_probs = {"positive": 0.333, "neutral": 0.333, "negative": 0.333}
    try:
        from utils.sentiment_inference import predict_batch_sentiment, is_loaded as fb_loaded
        if fb_loaded():
            fb_preds      = predict_batch_sentiment(headlines[:5])
            #Weighted mean: weight by position (lead headline has highest weight)
            n_fb          = len(fb_preds)
            pos_weights   = np.array([1.0 / (i + 1) for i in range(n_fb)])
            pos_weights  /= pos_weights.sum()
            finbert_score = float(sum(
                w * (p["probabilities"]["positive"] - p["probabilities"]["negative"])
                for w, p in zip(pos_weights, fb_preds)
            ))
            #Aggregate probabilities for the response payload
            finbert_probs = {
                "positive": round(float(sum(w * p["probabilities"]["positive"] for w, p in zip(pos_weights, fb_preds))), 4),
                "neutral":  round(float(sum(w * p["probabilities"]["neutral"]  for w, p in zip(pos_weights, fb_preds))), 4),
                "negative": round(float(sum(w * p["probabilities"]["negative"] for w, p in zip(pos_weights, fb_preds))), 4),
            }
    except Exception as exc:
        logger.debug(f"FinBERT fusion skipped ({exc})")

    #Adjust UP probability by sentiment (bounded ±0.20)
    sentiment_adj = float(np.clip(finbert_score * _SENTIMENT_WEIGHT, -0.20, 0.20))
    fused_up      = float(np.clip(model_up + sentiment_adj, 0.05, 0.95))
    fused_down    = 1.0 - fused_up

    direction   = "UP" if fused_up >= 0.50 else "DOWN"
    confidence  = fused_up if direction == "UP" else fused_down


    vol_scale  = _HORIZON_VOL_SCALE.get(horizon, 1.0)
    volatility = float(raw_vol * vol_scale)

    #Price change sign should match direction
    if (direction == "UP" and price_chg < 0) or (direction == "DOWN" and price_chg > 0):
        price_chg = -price_chg

    return {
        "direction":        direction,
        "confidence":       round(confidence, 4),
        "direction_probs":  {"UP": round(fused_up, 4), "DOWN": round(fused_down, 4)},
        "expected_return":  round(price_chg, 3),
        "volatility":       round(volatility, 4),
        "sentiment":        aux_sent_label,
        "sentiment_score":  round(finbert_score, 4),
        "finbert_probs":    finbert_probs,
        "engine":           "auramarketnet_v1",
        "model_loaded":     True,
        "ticker":           ticker,
        "horizon":          horizon,
        "mode":             "model",
    }
