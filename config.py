import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict



#PROJECT PATHS
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

#Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)



#MODEL CONFIGURATION
@dataclass
class TextEncoderConfig:
    """Configuration for the FinBERT-based text encoder."""
    model_name: str = "ProsusAI/finbert"            #Pre-trained FinBERT
    max_seq_length: int = 512                       #Max tokens per input
    hidden_size: int = 768                          #BERT hidden dimension
    output_dim: int = 256                           #Projected embedding size
    dropout: float = 0.1
    freeze_layers: int = 8                          #Freeze first N transformer layers
    use_pooler: bool = False                        #Use CLS token (not pooler)
    gradient_checkpointing: bool = True             #Save memory during training


@dataclass
class NumericalEncoderConfig:
    """Configuration for the time-series numerical encoder."""
    input_dim: int = 20                             #Number of input features per timestep
    hidden_dim: int = 256                           #LSTM hidden dimension
    num_layers: int = 3                             #LSTM depth
    output_dim: int = 256                           #Final embedding dimension
    dropout: float = 0.2
    bidirectional: bool = True                      #Bi-LSTM for richer context
    use_attention: bool = True                      #Self-attention over time steps
    num_attention_heads: int = 8                    #Multi-head attention
    sequence_length: int = 30                       #Days of historical data per sample


@dataclass
class FusionConfig:
    """Configuration for the multi-modal fusion layer."""
    text_dim: int = 256
    numerical_dim: int = 256
    fusion_dim: int = 512                           #Post-fusion dimension
    num_heads: int = 8                              #Cross-attention heads
    dropout: float = 0.1
    fusion_type: str = "cross_attention"            #Options: "concat", "cross_attention", "bilinear"
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256])


@dataclass
class OutputHeadsConfig:
    """Configuration for prediction heads."""
    input_dim: int = 512
    num_classes: int = 2                           #UP / DOWN
    regression_output: int = 1                     #% price change
    volatility_output: int = 1                     #Volatility prediction
    dropout: float = 0.15


@dataclass
class ModelConfig:
    """Full model configuration."""
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    numerical_encoder: NumericalEncoderConfig = field(default_factory=NumericalEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output_heads: OutputHeadsConfig = field(default_factory=OutputHeadsConfig)
    model_name: str = "AuraMarketNet-v1"



#TRAINING CONFIGURATION

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5                    
    weight_decay: float = 0.01
    bert_lr: float = 1e-5                          
    scheduler: str = "cosine_with_warmup"           
    warmup_ratio: float = 0.1                     
    batch_size: int = 16
    gradient_accumulation_steps: int = 4            
    max_grad_norm: float = 1.0                  
    num_epochs: int = 30
    early_stopping_patience: int = 7
    early_stopping_metric: str = "val_directional_accuracy"
    classification_weight: float = 1.0
    regression_weight: float = 0.5
    volatility_weight: float = 0.3
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    save_every_n_epochs: int = 2
    keep_n_checkpoints: int = 3
    seed: int = 42
    device: str = "auto"                           
    mixed_precision: bool = True           




@dataclass
class DataConfig:
    """Data pipeline configuration."""
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
        "NVDA", "META", "NFLX", "AMD", "SPY"
    ])


    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"


    prediction_horizon: int = 1                     

    #Technical indicators to compute
    indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "macd_signal", "macd_hist",
        "sma_20", "sma_50", "ema_12", "ema_26",
        "bb_upper", "bb_lower", "bb_mid",
        "atr", "obv", "vwap", "stoch_k", "stoch_d"
    ])


    sentiment_window_days: int = 3


    max_news_per_day: int = 10
    min_text_length: int = 20


    use_financial_phrasebank: bool = True
    use_reddit: bool = True
    use_news_scraper: bool = True


    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "wallstreetbets", "stocks", "investing", "StockMarket"
    ])
    reddit_limit: int = 100          


    normalize_prices: bool = True
    normalization: str = "zscore"              




@dataclass
class DashboardConfig:
    """Flask dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    secret_key: str = os.getenv("FLASK_SECRET_KEY", "aura-market-net-secret-2024")
    refresh_interval: int = 30
    prediction_cache_ttl: int = 60
    sentiment_cache_ttl: int = 300





class Config:
    """Master configuration object — import this everywhere."""
    model = ModelConfig()
    training = TrainingConfig()
    data = DataConfig()
    dashboard = DashboardConfig()

    #API keys (from environment or .env file)
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "AuraMarketNet/1.0")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")


CFG = Config()
