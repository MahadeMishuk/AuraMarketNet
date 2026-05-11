"""
Handles data ingestion from multiple sources:
1. Market data via yfinance
2. Financial PhraseBank (labeled sentiment)
3. Reddit (PRAW)
4. News API / web scraping

Implements PyTorch Dataset classes for efficient batching.
"""

import os
import re
import json
import time
import random
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)



#MARKET DATA INGESTION


class MarketDataFetcher:
    """
    Fetches OHLCV data from Yahoo Finance with caching.

    We use yfinance as primary source — it's reliable, free, and covers
    most major stocks/ETFs. AlphaVantage is configured as a fallback.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        return self.cache_dir / f"{ticker}_{start}_{end}.pkl"

    def fetch(
        self,
        ticker: str,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        
        cache_path = self._cache_path(ticker, start_date, end_date)

        if use_cache and cache_path.exists():
            logger.info(f"Loading {ticker} from cache")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        try:
            import yfinance as yf
            logger.info(f"Fetching {ticker} from Yahoo Finance ({start_date} to {end_date})")

            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,   #Adjust for splits and dividends
                back_adjust=False,
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            #Standardize column names
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            #Cache to disk
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        data = {}
        for ticker in tickers:
            df = self.fetch(ticker, start_date, end_date)
            if not df.empty:
                data[ticker] = df
            time.sleep(0.5)  #Rate limiting
        return data



#FINANCIAL PHRASEBANK — HUGGING FACE SOURCE


#Sentiment label mapping used throughout the project
LABEL_MAP     = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_MAP_INV = {0: "negative", 1: "neutral", 2: "positive"}

HF_DATASET_ID = "lmassaron/FinancialPhraseBank"

#Column name variants across HuggingFace uploads of the same underlying dataset
_TEXT_COLS  = ["sentence", "Sentence", "text", "Text", "headline", "news", "content"]
_LABEL_COLS = ["label", "Label", "sentiment", "Sentiment", "label_int", "target"]


def _detect_columns(features: dict) -> Tuple[str, str]:
    """Return (text_col, label_col) by scanning a datasets.Features dict."""
    text_col  = next((c for c in _TEXT_COLS  if c in features), None)
    label_col = next((c for c in _LABEL_COLS if c in features), None)
    if text_col is None:
        raise ValueError(
            f"Cannot detect text column in dataset. Available: {list(features.keys())}"
        )
    if label_col is None:
        raise ValueError(
            f"Cannot detect label column in dataset. Available: {list(features.keys())}"
        )
    return text_col, label_col


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:

    raw = df["raw_label"]

    if pd.api.types.is_integer_dtype(raw) and raw.between(0, 2).all():
        df["label_int"] = raw.astype(int)
        df["label"]     = df["label_int"].map(LABEL_MAP_INV)
        return df.drop(columns=["raw_label"])

    if pd.api.types.is_string_dtype(raw) or pd.api.types.is_object_dtype(raw):
        lowered = raw.str.lower().str.strip()
        df["label"]     = lowered
        df["label_int"] = lowered.map(LABEL_MAP)
        bad = df["label_int"].isna()
        if bad.any():
            raise ValueError(
                f"Unknown label strings: {set(lowered[bad].unique())}. "
                f"Expected one of: {set(LABEL_MAP)}"
            )
        df["label_int"] = df["label_int"].astype(int)
        return df.drop(columns=["raw_label"])

    raise ValueError(
        f"Unsupported label dtype: {raw.dtype}. Sample values: {raw.unique()[:5]}"
    )


def _log_dataset_stats(df: pd.DataFrame) -> None:
    sep = "─" * 56
    logger.info(sep)
    logger.info("Financial PhraseBank — Dataset Sanity Check")
    logger.info(f"  Total rows    : {len(df):,}")
    logger.info(f"  Missing text  : {int(df['text'].isna().sum())}")
    logger.info(f"  Missing labels: {int(df['label_int'].isna().sum())}")
    logger.info("  Class distribution:")
    for lbl, cnt in df["label"].value_counts().items():
        bar = "█" * int(cnt / len(df) * 30)
        logger.info(f"    {lbl:10s}: {cnt:5,}  ({cnt / len(df):.1%})  {bar}")
    logger.info("  Sample sentences:")
    for _, row in df.sample(min(3, len(df)), random_state=42).iterrows():
        logger.info(f"    [{row['label']:8s}] {str(row['text'])[:80]}")
    logger.info(sep)


def load_financial_phrasebank(
    hf_dataset_id: str = HF_DATASET_ID,
    cache_dir: str = "data/hf_cache",
    max_retries: int = 3,
) -> pd.DataFrame:
    
    cache_file = Path(cache_dir) / f"{hf_dataset_id.replace('/', '_')}.parquet"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        logger.info(f"Loading Financial PhraseBank from cache: {cache_file}")
        df = pd.read_parquet(cache_file)
        _log_dataset_stats(df)
        return df

    last_err: Exception = RuntimeError("No download attempted")
    for attempt in range(1, max_retries + 1):
        try:
            from datasets import load_dataset as hf_load_dataset
            logger.info(
                f"Downloading {hf_dataset_id} from Hugging Face "
                f"(attempt {attempt}/{max_retries}) ..."
            )
            raw_dataset = hf_load_dataset(hf_dataset_id, trust_remote_code=True)

            #Combine all splits — Financial PhraseBank typically ships as 'train' only
            frames: List[pd.DataFrame] = []
            for split_name, split_data in raw_dataset.items():
                text_col, label_col = _detect_columns(split_data.features)
                split_df = split_data.to_pandas()[[text_col, label_col]].copy()
                split_df.rename(
                    columns={text_col: "text", label_col: "raw_label"}, inplace=True
                )
                logger.info(f"  Split '{split_name}': {len(split_df):,} rows")
                frames.append(split_df)

            df = pd.concat(frames, ignore_index=True)
            df = _normalize_labels(df)
            df["source"] = hf_dataset_id

            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached dataset → {cache_file}")
            _log_dataset_stats(df)
            return df

        except Exception as exc:
            last_err = exc
            logger.warning(f"Download attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(
        f"Could not load '{hf_dataset_id}' after {max_retries} attempts. "
        f"Last error: {last_err}\n"
        "Ensure `datasets` is installed (`pip install datasets`) "
        "and that you have internet access on the first run."
    ) from last_err


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    before = len(df)
    df.dropna(subset=["text", "label_int"], inplace=True)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with null text/labels")

    before = len(df)
    df.drop_duplicates(subset=["text"], inplace=True)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} duplicate sentences")

    #Vectorised cleaning pipeline — preserves financial terminology
    df["text"] = (
        df["text"]
        .astype(str)
        .str.lower()
        .apply(lambda t: re.sub(r"[\x00-\x1f\x7f]", "", t))   #control chars
        .apply(lambda t: re.sub(r"\s+", " ", t).strip())        #whitespace
    )

    before = len(df)
    df = df[df["text"].str.len() >= 10]
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} sentences shorter than 10 chars")

    df.reset_index(drop=True, inplace=True)
    logger.info(f"Preprocessing complete: {len(df):,} clean samples")
    return df


class SentimentDataset(Dataset):
   

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_seq_length: int = 128,
    ):
        assert len(texts) == len(labels), "texts/labels length mismatch"
        self.texts          = texts
        self.labels         = torch.tensor(labels, dtype=torch.long)
        self.tokenizer      = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),      #[seq_len]
            "attention_mask": enc["attention_mask"].squeeze(0),  #[seq_len]
            "label":          self.labels[idx],                  #scalar
        }

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the 3 sentiment classes."""
        counts  = torch.bincount(self.labels, minlength=3).float()
        weights = 1.0 / counts.clamp(min=1.0)
        return (weights / weights.sum()) * 3.0  #normalise so mean weight = 1.0


def create_sentiment_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    train_ratio: float  = 0.80,
    val_ratio: float    = 0.10,
    test_ratio: float   = 0.10,
    batch_size: int     = 16,
    num_workers: int    = 2,
    seed: int           = 42,
    max_seq_length: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
   
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
    )

    texts  = df["text"].tolist()
    labels = df["label_int"].tolist()

    #First split: train vs (val + test)
    temp_size = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=temp_size, stratify=labels, random_state=seed
    )

    #Second split: val vs test within the held-out portion
    relative_test = test_ratio / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=seed
    )

    logger.info(
        f"Sentiment dataset splits — "
        f"train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}"
    )

    train_ds = SentimentDataset(X_train, y_train, tokenizer, max_seq_length)
    val_ds   = SentimentDataset(X_val,   y_val,   tokenizer, max_seq_length)
    test_ds  = SentimentDataset(X_test,  y_test,  tokenizer, max_seq_length)

    #Weighted sampler balances classes during each training epoch
    sample_weights = train_ds.get_class_weights()[train_ds.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


class RedditFetcher:
    """
    Fetches posts from financial subreddits using PRAW.
    Falls back to a pre-cached dataset if API credentials aren't set.
    """

    def __init__(self, client_id: str = "", client_secret: str = "", user_agent: str = "AuraMarketNet"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._reddit = None

    def _get_client(self):
        if not self.client_id:
            return None
        try:
            import praw
            return praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Reddit client: {e}")
            return None

    def fetch_subreddit_posts(
        self,
        subreddit: str = "wallstreetbets",
        limit: int = 100,
        time_filter: str = "day",
    ) -> List[Dict]:
        """Fetch top posts from a subreddit."""
        reddit = self._get_client()
        if reddit is None:
            return self._generate_mock_posts(subreddit, limit)

        try:
            posts = []
            sub = reddit.subreddit(subreddit)
            for post in sub.top(time_filter=time_filter, limit=limit):
                posts.append({
                    "title": post.title,
                    "text": post.selftext[:1000] if post.selftext else post.title,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.fromtimestamp(post.created_utc),
                    "subreddit": subreddit,
                    "source": "reddit",
                })
            return posts
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
            return self._generate_mock_posts(subreddit, min(limit, 20))

    def _generate_mock_posts(self, subreddit: str, n: int) -> List[Dict]:
        """Generate mock Reddit posts for demo purposes."""
        wsb_posts = [
            {"title": "AAPL to the moon! 🚀 Earnings crushed it!", "score": 5000},
            {"title": "DD: Why TSLA is massively undervalued right now", "score": 3200},
            {"title": "Bears getting destroyed this week - SPY only goes up", "score": 2800},
            {"title": "Just YOLO'd my life savings into NVDA calls, wish me luck", "score": 7500},
            {"title": "GME short interest at 3 year high - squeeze incoming?", "score": 4100},
            {"title": "Market crash incoming - recession indicators flashing red", "score": 1200},
            {"title": "AMD vs NVDA - which do you like for the next 6 months?", "score": 890},
            {"title": "Fed pivot: what it means for your portfolio", "score": 2100},
            {"title": "Lost 50% on options this week. AMA", "score": 9800},
            {"title": "Why I'm all in on tech in 2024", "score": 3400},
        ]
        now = datetime.now()
        posts = []
        for i in range(min(n, len(wsb_posts))):
            p = wsb_posts[i % len(wsb_posts)].copy()
            p.update({
                "text": p["title"],
                "num_comments": random.randint(50, 500),
                "created_utc": now - timedelta(hours=random.randint(1, 24)),
                "subreddit": subreddit,
                "source": "reddit_mock",
            })
            posts.append(p)
        return posts



#PYTORCH DATASETS


class AuraMarketDataset(Dataset):
    """
    PyTorch Dataset for AuraMarketNet training.

    Each sample contains:
    - Tokenized text (multiple headlines for the prediction window)
    - Numerical feature sequence [seq_len, n_features]
    - Labels: direction (UP/DOWN), return (%), volatility

    The dataset aligns text and numerical data by date, ensuring
    no future information leaks into the model inputs.
    """

    def __init__(
        self,
        numerical_features: np.ndarray,   #[n_samples, seq_len, n_features]
        direction_labels: np.ndarray,      #[n_samples] — 0 or 1
        return_labels: np.ndarray,         #[n_samples] — float
        volatility_labels: np.ndarray,     #[n_samples] — float
        text_data: List[List[str]] = None, #[n_samples, n_texts]
        tokenizer=None,
        max_seq_length: int = 128,
        max_texts_per_sample: int = 5,
        augment: bool = False,
    ):
        assert len(numerical_features) == len(direction_labels), "Feature/label length mismatch"

        self.numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        self.direction_labels = torch.tensor(direction_labels, dtype=torch.long)
        self.return_labels = torch.tensor(return_labels, dtype=torch.float32)
        self.volatility_labels = torch.tensor(volatility_labels, dtype=torch.float32)
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_texts = max_texts_per_sample
        self.augment = augment

        logger.info(f"Dataset initialized: {len(self)} samples")
        logger.info(f"  Direction distribution: {self._get_class_dist()}")

    def _get_class_dist(self) -> str:
        counts = torch.bincount(self.direction_labels)
        total = len(self.direction_labels)
        return " | ".join([f"Class {i}: {c/total:.1%}" for i, c in enumerate(counts)])

    def __len__(self) -> int:
        return len(self.numerical_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "numerical_features": self.numerical_features[idx],   #[seq_len, n_features]
            "direction": self.direction_labels[idx],               #scalar
            "price_change": self.return_labels[idx],               #scalar
            "volatility": self.volatility_labels[idx],             #scalar
        }

        #Text tokenization────
        if self.tokenizer is not None and self.text_data is not None:
            texts = self.text_data[idx] if idx < len(self.text_data) else []
            texts = [t for t in texts if isinstance(t, str) and len(t) > 0]
            texts = texts[:self.max_texts]

            if not texts:
                texts = ["no significant financial news today"]

            #Optional augmentation: randomly drop some texts
            if self.augment and len(texts) > 1 and random.random() < 0.3:
                n_drop = random.randint(1, len(texts) - 1)
                texts = random.sample(texts, len(texts) - n_drop)

            encoding = self.tokenizer(
                texts,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            n_texts = len(texts)
            n_pad = self.max_texts - n_texts

            if n_pad > 0:
                pad_ids = torch.zeros(n_pad, self.max_seq_length, dtype=torch.long)
                pad_mask = torch.zeros(n_pad, self.max_seq_length, dtype=torch.long)
                encoding["input_ids"] = torch.cat([encoding["input_ids"], pad_ids], dim=0)
                encoding["attention_mask"] = torch.cat([encoding["attention_mask"], pad_mask], dim=0)

            text_valid_mask = torch.zeros(self.max_texts, dtype=torch.bool)
            text_valid_mask[:n_texts] = True

            sample["input_ids"] = encoding["input_ids"][:self.max_texts]         #[max_texts, seq_len]
            sample["attention_mask"] = encoding["attention_mask"][:self.max_texts]
            sample["text_mask"] = text_valid_mask

        else:
            #Dummy text tokens when no text data is provided
            sample["input_ids"] = torch.zeros(self.max_texts, self.max_seq_length, dtype=torch.long)
            sample["attention_mask"] = torch.zeros(self.max_texts, self.max_seq_length, dtype=torch.long)
            sample["text_mask"] = torch.zeros(self.max_texts, dtype=torch.bool)

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse class frequency weights for handling class imbalance.
        Markets tend to have slightly more UP days than DOWN days.
        """
        counts = torch.bincount(self.direction_labels, minlength=2).float()
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum()

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return class_weights[self.direction_labels]


def create_dataloaders(
    dataset: AuraMarketDataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 2,
    balance_classes: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create time-aware train/val/test splits without data leakage.

    CRITICAL: We split chronologically, NOT randomly.
    Random splitting would leak future data into training — the model
    would "see" future market patterns during training.

    Split: [train] [val] [test] sorted by time
    """
    n = len(dataset)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    #Chronological split
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    #Balanced sampling for training (optional)
    if balance_classes:
        sample_weights = dataset.get_sample_weights()[train_indices]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"DataLoader splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader
