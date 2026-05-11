import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

#Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger("train")

from config import CFG, CACHE_DIR
from models import create_model
from utils.data_loader import MarketDataFetcher, load_financial_phrasebank, AuraMarketDataset, create_dataloaders
from utils.feature_engineering import FeatureEngineer
from utils.text_preprocessing import TextPreprocessor
from training.trainer import AuraTrainer
from evaluation.evaluator import ModelEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train AuraMarketNet")
    parser.add_argument("--tickers", nargs="+", default=None, help="Stock tickers to use")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes (default: auto)")
    parser.add_argument("--no-text", action="store_true", help="Disable text encoder")
    parser.add_argument("--dry-run", action="store_true", help="Quick validation pass")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def log_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {props.total_memory / 1e9:.1f} GB | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    else:
        logger.info("No CUDA GPU detected — training on CPU")


def build_dataset(args):
    """
    Full data pipeline:
    1. Fetch OHLCV data for all configured tickers
    2. Compute technical indicators
    3. Create sliding window sequences
    4. Associate with text sentiment data

    Returns PyTorch Dataset ready for training.
    """
    tickers = args.tickers or CFG.data.tickers
    if args.dry_run:
        tickers = ["AAPL", "TSLA"]

    logger.info(f"Building dataset for {len(tickers)} tickers: {tickers}")

    feature_engineer = FeatureEngineer()
    fetcher = MarketDataFetcher(cache_dir=str(CACHE_DIR))

    all_features = []
    all_directions = []
    all_returns = []
    all_volatilities = []
    all_texts = []

    for ticker in tickers:
        logger.info(f"Processing {ticker}...")

        #Fetch market data
        df = fetcher.fetch(
            ticker=ticker,
            start_date=CFG.data.start_date,
            end_date=CFG.data.end_date,
        )

        if df.empty or len(df) < CFG.model.numerical_encoder.sequence_length + 30:
            logger.warning(f"Insufficient data for {ticker}, skipping")
            continue

        #Compute indicators
        df_features = feature_engineer.compute_all_indicators(df)

        #Create sequences
        features, directions, returns, volatilities, dates = feature_engineer.create_sequences(
            df=df_features,
            sequence_length=CFG.model.numerical_encoder.sequence_length,
        )

        if len(features) == 0:
            logger.warning(f"No valid sequences for {ticker}")
            continue

        #Generate placeholder text data for each sample
        #In production: fetch real news headlines aligned to dates
        n_samples = len(features)
        ticker_texts = []
        for d in dates:
            date_str = str(d)[:10]
            ticker_texts.append([
                f"Latest {ticker} stock market analysis and financial news for {date_str}",
                f"{ticker} trading volume and price movement analysis",
            ])

        all_features.append(features)
        all_directions.append(directions)
        all_returns.append(returns)
        all_volatilities.append(volatilities)
        all_texts.extend(ticker_texts)

        logger.info(f"  {ticker}: {n_samples} sequences | "
                    f"UP={directions.mean():.2%} | "
                    f"Return: μ={returns.mean():.4f} σ={returns.std():.4f}")

    if not all_features:
        raise RuntimeError("No valid data found. Check your configuration and network connection.")

    #Concatenate across all tickers
    features_combined = np.concatenate(all_features, axis=0)
    directions_combined = np.concatenate(all_directions, axis=0)
    returns_combined = np.concatenate(all_returns, axis=0)
    volatilities_combined = np.concatenate(all_volatilities, axis=0)

    logger.info(f"\nTotal dataset: {len(features_combined):,} samples")
    logger.info(f"Feature shape: {features_combined.shape}")
    logger.info(f"Direction balance: UP={directions_combined.mean():.2%}")

    #Load tokenizer for text encoding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.text_encoder.model_name)

    #Build dataset
    dataset = AuraMarketDataset(
        numerical_features=features_combined,
        direction_labels=directions_combined,
        return_labels=returns_combined,
        volatility_labels=volatilities_combined,
        text_data=all_texts,
        tokenizer=tokenizer,
        max_seq_length=128,
        max_texts_per_sample=5,
        augment=True,
    )

    return dataset


def train(args):
    logger.info("\n" + "="*70)
    logger.info("  AuraMarketNet Training Pipeline")
    logger.info("="*70 + "\n")

    log_gpu_info()
    total_start = time.time()

    #Override config from args
    if args.epochs:    CFG.training.num_epochs = args.epochs
    if args.batch_size: CFG.training.batch_size = args.batch_size
    if args.lr:        CFG.training.learning_rate = args.lr
    if args.dry_run:
        CFG.training.num_epochs = 2
        CFG.training.batch_size = 4

    #Build dataset
    start_time = time.time()
    dataset = build_dataset(args)
    logger.info(f"Dataset built in {time.time() - start_time:.1f}s")

    #Create data loaders
    if args.dry_run:
        nw = 0
    elif args.num_workers is not None:
        nw = args.num_workers
    else:
        nw = 4 if torch.cuda.is_available() else 2

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        val_ratio=CFG.training.val_ratio,
        test_ratio=CFG.training.test_ratio,
        batch_size=CFG.training.batch_size,
        num_workers=nw,
        balance_classes=True,
    )

    #Build model
    logger.info("\nBuilding model...")
    model = create_model(CFG)
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    #Setup trainer
    trainer = AuraTrainer(model=model, config=CFG)

    #Resume from checkpoint
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)
        from training.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint.load_checkpoint(
            str(resume_path),
            model,
            trainer.optimizer,
            device=str(trainer.device),   #load directly onto training device
        )
        resumed_epoch = checkpoint.get("epoch", 0)
        start_epoch = resumed_epoch + 1
        trainer.current_epoch = resumed_epoch
        logger.info(f"Resumed from epoch {resumed_epoch} — continuing from epoch {start_epoch}")

    #Train
    logger.info("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CFG.training.num_epochs,
        start_epoch=start_epoch,
    )

    logger.info("\nEvaluating on test set...")
    evaluator = ModelEvaluator(model=model, device=str(trainer.device))
    test_metrics = evaluator.evaluate(test_loader)

    import json
    results_path = Path("logs") / "final_test_metrics.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"\nTest metrics saved to {results_path}")

    pth_dir = Path(f"{CFG.model.model_name}_checkpoints")
    pth_dir.mkdir(exist_ok=True)
    pth_path = pth_dir / f"{CFG.model.model_name}_final.pth"
    torch.save(model.state_dict(), str(pth_path))
    logger.info(f"Final model weights saved to {pth_path}")

    logger.info(f"Total training time: {(time.time() - total_start) / 60:.1f} min")
    return model, test_metrics


def main():
    args = parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    try:
        model, metrics = train(args)
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f}")
        logger.info(f"  F1 Score:             {metrics.get('clf_f1', 0):.4f}")
        logger.info(f"  Return MAE:           {metrics.get('ret_mae', 0):.6f}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
