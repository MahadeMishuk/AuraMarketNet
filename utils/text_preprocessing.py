"""
Text Preprocessing for Financial NLP
Cleans, normalizes, and tokenizes financial text from multiple sources (news, Reddit, Twitter) for FinBERT sentiment analysis.
"""

import re
import string
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import logging

try:
    import torch
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


WSB_SLANG_MAP = {
    r"\bdd\b": "due diligence",
    r"\byolo\b": "high risk",
    r"\bhodl\b": "hold",
    r"\bfomo\b": "fear of missing out",
    r"\bmooning\b": "rising sharply",
    r"\bto the moon\b": "extremely bullish",
    r"\bbagholders?\b": "investors holding losses",
    r"\bapes?\b": "retail investors",
    r"\bgme\b": "GameStop",
    r"\bamcgang\b": "AMC investors",
    r"\bshort squeeze\b": "short squeeze",
    r"\bgamma squeeze\b": "gamma squeeze",
    r"\bretail investors?\b": "retail investors",
    r"\bhedge funds?\b": "hedge fund",
    r"\bwallstreetbets?\b": "Reddit investors",
    r"\bstonks?\b": "stocks",
    r"\btendies\b": "profits",
    r"\bgains?\b": "profits",
    r"\b10-?bagger\b": "ten times return",
    r"\b100x\b": "hundred times return",
}


class TextPreprocessor:
    """
    Preprocesses financial text for FinBERT tokenization.

    Features:
    - Ticker symbol handling ($AAPL → AAPL)
    - Number normalization (1.5B → 1.5 billion)
    - WSB slang expansion
    - Sentiment-preserving cleaning (doesn't strip negations!)
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 512,
        tokenizer_cache_dir: Optional[str] = None,
    ):
        self.max_length = max_length

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=tokenizer_cache_dir,
        )




    @staticmethod
    def normalize_ticker(text: str) -> str:
        """
        Handle ticker symbols: $AAPL → AAPL (token)
        FinBERT has been trained with tickers as regular words.
        """

        text = re.sub(r'\$([A-Z]{1,5})\b', r'\1', text)
        return text

    @staticmethod
    def normalize_numbers(text: str) -> str:
        """
        Normalize number formats to help the model understand magnitudes.
        $1.5B → 1.5 billion dollars
        """
        #Billions
        text = re.sub(r'\$?([\d.]+)\s*[Bb](?:illion)?\b', r'\1 billion dollars', text)
        #Millions
        text = re.sub(r'\$?([\d.]+)\s*[Mm](?:illion)?\b', r'\1 million dollars', text)
        #Percentages
        text = re.sub(r'([\d.]+)\s*%', r'\1 percent', text)
        #Prices
        text = re.sub(r'\$([\d,.]+)', r'\1 dollars', text)
        return text

    @staticmethod
    def expand_wsb_slang(text: str) -> str:
        """Replace WSB-specific slang with more formal equivalents."""
        text = text.lower()
        for pattern, replacement in WSB_SLANG_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def clean_url(text: str) -> str:
        """Remove URLs."""
        return re.sub(r'http[s]?://\S+', '', text)

    @staticmethod
    def clean_html(text: str) -> str:
        """Strip HTML tags."""
        return re.sub(r'<[^>]+>', ' ', text)

    @staticmethod
    def clean_special_chars(text: str) -> str:
        """
        Remove special characters but PRESERVE negations and financial symbols.
        DO NOT remove: %, -1, not, no, down, loss, loss
        """
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_text(self, text: str, source: str = "news") -> str:
        """
        Full text cleaning pipeline.

        Args:
            text: Raw input text
            source: "news", "reddit", "twitter" — different cleaning logic

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or len(text.strip()) < 3:
            return ""

        text = self.clean_html(text)
        text = self.clean_url(text)
        text = self.normalize_ticker(text)
        text = self.normalize_numbers(text)


        if source == "reddit":
            text = self.expand_wsb_slang(text)
        elif source == "twitter":
            text = re.sub(r'#(\w+)', r'\1', text)
            text = re.sub(r'@\w+', '', text)

        text = self.clean_special_chars(text)
        return text.strip()



    def tokenize(
        self,
        texts: Union[str, List[str]],
        source: str = "news",
    ) -> Dict[str, torch.Tensor]:
        """
        Clean and tokenize text(s) for FinBERT.

        Args:
            texts:  Single string or list of strings
            source: Data source for cleaning strategy

        Returns:
            Dict with "input_ids", "attention_mask", "token_type_ids"
            All tensors of shape [batch, max_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        #Clean all texts
        cleaned = [self.clean_text(t, source) for t in texts]

        #Replace empty strings with a neutral financial statement
        cleaned = [t if len(t) > 0 else "no significant news today" for t in cleaned]

        #Tokenize
        encoding = self.tokenizer(
            cleaned,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return encoding

    def tokenize_batch_of_texts(
        self,
        batch_texts: List[List[str]], 
        source: str = "news",
        max_texts: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize multiple texts per sample (e.g., N headlines per day).

        Args:
            batch_texts: List of lists — each inner list is the texts for one sample
            source:      Data source
            max_texts:   Maximum texts per sample (pad/truncate to this)

        Returns:
            Dict with tensors of shape [batch, max_texts, seq_len]
            Plus "text_mask" [batch, max_texts] indicating valid texts
        """
        batch_size = len(batch_texts)

        all_input_ids = []
        all_attention_masks = []
        all_text_masks = []

        pad_encoding = self.tokenizer(
            "no significant news today",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for texts in batch_texts:
            #Truncate to max_texts
            texts = texts[:max_texts]
            n_valid = len(texts)

            #Tokenize valid texts
            if n_valid > 0:
                enc = self.tokenize(texts, source)
                ids = enc["input_ids"]     
                masks = enc["attention_mask"]    
            else:
                ids = torch.zeros(0, self.max_length, dtype=torch.long)
                masks = torch.zeros(0, self.max_length, dtype=torch.long)

            #Pad to max_texts
            n_pad = max_texts - n_valid
            if n_pad > 0:
                pad_ids = pad_encoding["input_ids"].repeat(n_pad, 1)
                pad_masks = pad_encoding["attention_mask"].repeat(n_pad, 1)
                ids = torch.cat([ids, pad_ids], dim=0)
                masks = torch.cat([masks, pad_masks], dim=0)

            #Text validity mask
            text_mask = torch.zeros(max_texts, dtype=torch.bool)
            text_mask[:n_valid] = True

            all_input_ids.append(ids)
            all_attention_masks.append(masks)
            all_text_masks.append(text_mask)

        return {
            "input_ids": torch.stack(all_input_ids),          
            "attention_mask": torch.stack(all_attention_masks), 
            "text_mask": torch.stack(all_text_masks),     
        }

    def get_token_words(self, input_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs back to words for attention visualization.

        Args:
            input_ids: [seq_len] tensor

        Returns:
            List of token strings
        """
        return self.tokenizer.convert_ids_to_tokens(input_ids.tolist())


class SentimentAggregator:
    """
    Aggregates multiple sentiment scores (from multiple news articles)
    into a single representation for a time period.
    """

    @staticmethod
    def aggregate_sentiments(
        sentiments: List[Dict[str, float]],
        method: str = "weighted_mean",
    ) -> Dict[str, float]:
        """
        Aggregate sentiment scores from multiple articles.

        Args:
            sentiments: List of dicts with "positive", "negative", "neutral" keys
            method: "mean", "weighted_mean", "max_impact"

        Returns:
            Aggregated sentiment dict
        """
        if not sentiments:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}

        pos = [s.get("positive", 0) for s in sentiments]
        neg = [s.get("negative", 0) for s in sentiments]
        neu = [s.get("neutral", 0) for s in sentiments]

        if method == "mean":
            return {
                "positive": np.mean(pos),
                "negative": np.mean(neg),
                "neutral": np.mean(neu),
                "compound": np.mean(pos) - np.mean(neg),
            }

        elif method == "weighted_mean":
   
            weights = [abs(s.get("positive", 0) - s.get("negative", 0)) + 1e-6
                      for s in sentiments]
            w = np.array(weights) / sum(weights)
            return {
                "positive": float(np.dot(w, pos)),
                "negative": float(np.dot(w, neg)),
                "neutral": float(np.dot(w, neu)),
                "compound": float(np.dot(w, pos) - np.dot(w, neg)),
            }

        elif method == "max_impact":

            compounds = [s.get("positive", 0) - s.get("negative", 0) for s in sentiments]
            max_idx = np.argmax(np.abs(compounds))
            return {
                "positive": pos[max_idx],
                "negative": neg[max_idx],
                "neutral": neu[max_idx],
                "compound": compounds[max_idx],
            }

        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}
