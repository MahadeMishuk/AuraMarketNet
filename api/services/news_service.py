import re
import html as html_lib
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

#Financial keyword weights (impact on market reaction likelihood) 
POSITIVE_KW: Dict[str, float] = {
    "beats": 0.9, "beat": 0.9, "exceeds": 0.8, "record": 0.7, "surge": 0.8,
    "rally": 0.7, "upgrade": 0.8, "buyout": 0.9, "acquisition": 0.7,
    "dividend": 0.6, "buyback": 0.7, "partnership": 0.5, "growth": 0.5,
    "breakthrough": 0.9, "raises": 0.7, "raised": 0.7, "profit": 0.6,
    "outperform": 0.8, "strong": 0.5, "soars": 0.8, "jumps": 0.7,
    "gains": 0.6, "rebounds": 0.6, "recovery": 0.6, "expansion": 0.5,
}
NEGATIVE_KW: Dict[str, float] = {
    "miss": 0.9, "misses": 0.9, "falls": 0.7, "drop": 0.7, "crash": 1.0,
    "downgrade": 0.8, "lawsuit": 0.8, "loss": 0.7, "recall": 0.8,
    "investigation": 0.8, "fine": 0.7, "layoffs": 0.7, "bankruptcy": 1.0,
    "warning": 0.7, "cut": 0.6, "plunges": 0.9, "tumbles": 0.8,
    "slump": 0.7, "decline": 0.6, "concern": 0.5, "risk": 0.5,
    "disappoints": 0.8, "struggles": 0.6, "weak": 0.6, "shortfall": 0.7,
}

#Breaking threshold────
BREAKING_THRESHOLD_MINS = 30


def compute_impact_score(title: str, sentiment_score: float) -> float:
    """
    0–10 score representing expected market impact.
    Base = abs(sentiment) × 4, boosted by high-weight keywords.
    """
    text   = title.lower()
    impact = abs(float(sentiment_score)) * 4.0

    for kw, wt in POSITIVE_KW.items():
        if kw in text:
            impact += wt * 1.5
    for kw, wt in NEGATIVE_KW.items():
        if kw in text:
            impact += wt * 1.5

    return round(min(impact, 10.0), 2)


def is_breaking(age_mins: int) -> bool:
    return int(age_mins) <= BREAKING_THRESHOLD_MINS


def highlight_keywords(title: str) -> str:
    """
    Returns HTML-safe string with positive/negative keywords wrapped in
    <mark class='kw-pos'> / <mark class='kw-neg'> spans.
    The source text is HTML-escaped before wrapping so no XSS is possible.
    """
    escaped = html_lib.escape(title)
    words   = escaped.split()
    result  = []
    for word in words:
        clean = re.sub(r"[^a-z]", "", word.lower())
        if clean in POSITIVE_KW:
            result.append(f'<mark class="kw-pos">{word}</mark>')
        elif clean in NEGATIVE_KW:
            result.append(f'<mark class="kw-neg">{word}</mark>')
        else:
            result.append(word)
    return " ".join(result)


def enrich_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not articles:
        return articles

    #FinBERT batch inference──
    titles = [a.get("title", "") for a in articles]
    try:
        from utils.sentiment_inference import predict_batch_sentiment, is_loaded
        if is_loaded():
            finbert_preds = predict_batch_sentiment(titles)
            for a, pred in zip(articles, finbert_preds):
                a["sentiment"]        = pred["label"]
                a["sentiment_score"]  = round(
                    pred["probabilities"]["positive"] - pred["probabilities"]["negative"], 3
                )
                a["sentiment_probs"]  = pred["probabilities"]
                a["sentiment_engine"] = pred["engine"]
    except Exception as exc:
        logger.debug(f"FinBERT enrichment skipped ({exc}); using existing VADER scores")

    #Keyword scoring and metadata──
    for a in articles:
        score = a.get("sentiment_score", 0.0)
        title = a.get("title", "")
        a["impact_score"]      = compute_impact_score(title, score)
        a["is_breaking"]       = is_breaking(a.get("age_mins", 9999))
        a["highlighted_title"] = highlight_keywords(title)

    articles.sort(key=lambda x: (-x["impact_score"], x.get("age_mins", 9999)))
    return articles


def sentiment_distribution(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns % positive / neutral / negative for a list of articles.
    """
    total = len(articles)
    if total == 0:
        return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}

    pos = sum(1 for a in articles if a.get("sentiment") == "positive")
    neg = sum(1 for a in articles if a.get("sentiment") == "negative")
    neu = total - pos - neg

    return {
        "positive": round(pos / total * 100, 1),
        "neutral":  round(neu / total * 100, 1),
        "negative": round(neg / total * 100, 1),
        "total":    total,
    }
