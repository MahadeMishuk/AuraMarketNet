"""
Public API
----------
    load_sentiment_model()           → bool
    predict_sentiment(text)          → SentimentResult dict
    predict_batch_sentiment(texts)   → List[SentimentResult]
    score_sentiment_probability(text)→ SentimentResult dict  (alias)
    aggregate_market_sentiment(preds, weights) → AggregateResult dict
    generate_market_signal(preds)    → "bullish" | "neutral" | "bearish"
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#Label constants consistent with training─────
LABELS    = ["negative", "neutral", "positive"]
LABEL_IDX = {"negative": 0, "neutral": 1, "positive": 2}

#Signal thresholds
#avg_positive >= BULLISH_THRESHOLD  → bullish
#avg_negative >= BEARISH_THRESHOLD  → bearish
#otherwise                          → neutral
BULLISH_THRESHOLD = 0.50
BEARISH_THRESHOLD = 0.35

#Singleton state (protected by _lock)──
_lock:      threading.Lock     = threading.Lock()
_model:     Optional[object]   = None
_tokenizer: Optional[object]   = None
_device:    Optional[torch.device] = None
_load_attempted: bool          = False



#DEVICE SELECTION


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



#MODEL LOADING


def load_sentiment_model(
    checkpoint_path: str = "checkpoints/finbert_sentiment_best.pt",
    model_name: str = "ProsusAI/finbert",
) -> bool:
    """
    Load the fine-tuned FinBERT sentiment classifier from checkpoint.

    Thread-safe and idempotent — safe to call multiple times.
    On the first call it loads; subsequent calls return immediately.

    Args:
        checkpoint_path: Path to the .pt checkpoint saved by train_sentiment.py.
        model_name:      HuggingFace model identifier used by the tokenizer.

    Returns:
        True  → model loaded and ready for inference.
        False → checkpoint not found; predict_* functions fall back to VADER.
    """
    global _model, _tokenizer, _device, _load_attempted

    with _lock:
        if _load_attempted:
            return _model is not None

        _load_attempted = True
        ckpt_path = Path(checkpoint_path)

        if not ckpt_path.exists():
            logger.warning(
                f"FinBERT checkpoint not found: {ckpt_path}. "
                "Sentiment endpoints will fall back to VADER."
            )
            return False

        try:
            from transformers import AutoTokenizer
            #Import the classifier class from the training script
            import sys, importlib
            sys.path.insert(0, str(Path(__file__).parent.parent))
            ts = importlib.import_module("train_sentiment")
            FinBERTSentimentClassifier = ts.FinBERTSentimentClassifier

            from config import CFG

            _device = _resolve_device()
            logger.info(f"Loading FinBERT sentiment model on {_device} ← {ckpt_path}")

            #Prefer cached files; fall back to network on first run
            try:
                _tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception:
                _tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = FinBERTSentimentClassifier(CFG.model.text_encoder)
            ckpt  = torch.load(str(ckpt_path), map_location=_device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            model = model.to(_device).eval()

            _model = model
            logger.info(
                f"FinBERT sentiment ready | "
                f"val_f1={ckpt.get('val_f1', 0):.4f} | "
                f"epoch={ckpt.get('epoch', '?')} | "
                f"device={_device}"
            )
            return True

        except Exception as exc:
            logger.error(f"Failed to load FinBERT sentiment model: {exc}")
            _model     = None
            _tokenizer = None
            return False


def is_loaded() -> bool:
    """Return True if FinBERT is loaded and ready."""
    return _model is not None and _tokenizer is not None



#VADER FALLBACK


def _vader_fallback(text: str) -> Dict:
    """VADER-based sentiment when FinBERT is unavailable."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        c = vader.polarity_scores(text.strip() or "")["compound"]

        if c >= 0.05:
            label = "positive"
            pos   = min(0.97, 0.55 + abs(c) * 0.40)
            neg   = 0.03
            neu   = max(0.0, 1.0 - pos - neg)
        elif c <= -0.05:
            label = "negative"
            neg   = min(0.97, 0.55 + abs(c) * 0.40)
            pos   = 0.03
            neu   = max(0.0, 1.0 - pos - neg)
        else:
            label = "neutral"
            neu, pos, neg = 0.80, 0.10, 0.10

        total = pos + neu + neg
        return {
            "label": label,
            "score": round({"positive": pos, "neutral": neu, "negative": neg}[label] / total, 4),
            "probabilities": {
                "negative": round(neg / total, 4),
                "neutral":  round(neu / total, 4),
                "positive": round(pos / total, 4),
            },
            "engine": "vader_fallback",
        }
    except Exception:
        return {
            "label": "neutral",
            "score": 1.0,
            "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            "engine": "default_neutral",
        }



#CORE INFERENCE


@torch.no_grad()
def predict_batch_sentiment(
    texts: List[str],
    max_length: int = 128,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Batch FinBERT sentiment inference over a list of financial texts.

    Falls back to VADER per-text if the model is not loaded.

    Args:
        texts:      Financial headlines or sentences.
        max_length: Maximum tokenized length (FinBERT supports up to 512).
        batch_size: Internal sub-batch size for GPU memory management.

    Returns:
        List[SentimentResult] — one dict per input text, same order.
    """
    if not texts:
        return []

    if not is_loaded():
        return [_vader_fallback(t) for t in texts]

    results: List[Dict] = []
    use_amp = _device.type == "cuda"

    for i in range(0, len(texts), batch_size):
        batch = [str(t).strip() or "no text" for t in texts[i : i + batch_size]]

        enc = _tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(_device)
        attention_mask = enc["attention_mask"].to(_device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = _model(input_ids, attention_mask)   #[B, 3]

        probs = F.softmax(logits, dim=-1).cpu()          #[B, 3]

        for row in probs:
            neg_p, neu_p, pos_p = row[0].item(), row[1].item(), row[2].item()
            pred_idx = int(row.argmax())
            label    = LABELS[pred_idx]
            results.append({
                "label": label,
                "score": round(row[pred_idx].item(), 4),
                "probabilities": {
                    "negative": round(neg_p, 4),
                    "neutral":  round(neu_p, 4),
                    "positive": round(pos_p, 4),
                },
                "engine": "finbert",
            })

    return results


def predict_sentiment(text: str) -> Dict:
    """
    Single-text FinBERT (or VADER fallback) sentiment prediction.

    Args:
        text: A financial headline, sentence, or short paragraph.

    Returns:
        SentimentResult dict.
    """
    if not text or not str(text).strip():
        return _vader_fallback("")
    return predict_batch_sentiment([text.strip()])[0]


#Explicit alias used in API endpoints for clarity
score_sentiment_probability = predict_sentiment



#AGGREGATION & SIGNAL GENERATION


def aggregate_market_sentiment(
    predictions: List[Dict],
    weights: Optional[List[float]] = None,
) -> Dict:
    """
    Collapse multiple FinBERT predictions into one market-level sentiment score.

    Uses impact-weighted mean of class probabilities. The signed overall_score
    is positive for bullish conditions and negative for bearish conditions.

    Args:
        predictions: Output of predict_batch_sentiment().
        weights:     Per-prediction importance weights (e.g., impact scores).
                     Defaults to equal weighting.

    Returns:
        {
            "overall_label":   "bullish" | "neutral" | "bearish",
            "overall_score":   float,     #positive = bullish, negative = bearish
            "avg_positive":    float,
            "avg_neutral":     float,
            "avg_negative":    float,
            "signal_strength": float,     #0–1, conviction level
            "bullish_count":   int,
            "bearish_count":   int,
            "neutral_count":   int,
            "headline_count":  int,
        }
    """
    if not predictions:
        return {
            "overall_label":   "neutral",
            "overall_score":   0.0,
            "avg_positive":    0.0,
            "avg_neutral":     1.0,
            "avg_negative":    0.0,
            "signal_strength": 0.0,
            "bullish_count":   0,
            "bearish_count":   0,
            "neutral_count":   0,
            "headline_count":  0,
        }

    n = len(predictions)
    if weights is None:
        weights = [1.0] * n

    total_w = sum(weights) or 1.0
    w_norm  = [w / total_w for w in weights]

    avg_pos = sum(p["probabilities"]["positive"] * w for p, w in zip(predictions, w_norm))
    avg_neu = sum(p["probabilities"]["neutral"]  * w for p, w in zip(predictions, w_norm))
    avg_neg = sum(p["probabilities"]["negative"] * w for p, w in zip(predictions, w_norm))

    overall_score = avg_pos - avg_neg   #signed: positive=bullish, negative=bearish

    if avg_pos >= BULLISH_THRESHOLD:
        label = "bullish"
    elif avg_neg >= BEARISH_THRESHOLD:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "overall_label":   label,
        "overall_score":   round(overall_score, 4),
        "avg_positive":    round(avg_pos, 4),
        "avg_neutral":     round(avg_neu, 4),
        "avg_negative":    round(avg_neg, 4),
        "signal_strength": round(abs(overall_score), 4),
        "bullish_count":   sum(1 for p in predictions if p["label"] == "positive"),
        "bearish_count":   sum(1 for p in predictions if p["label"] == "negative"),
        "neutral_count":   sum(1 for p in predictions if p["label"] == "neutral"),
        "headline_count":  n,
    }


def generate_market_signal(
    predictions: List[Dict],
    weights: Optional[List[float]] = None,
) -> str:
    """
    Return a one-word market signal from a list of FinBERT predictions.

    Returns: "bullish" | "neutral" | "bearish"
    """
    return aggregate_market_sentiment(predictions, weights)["overall_label"]
