"""
FinBERT Sentiment Fine-Tuning — Financial PhraseBank
=====================================================
Downloads the official lmassaron/FinancialPhraseBank dataset from Hugging Face
and fine-tunes the FinBERT text encoder for 3-class financial sentiment:

  0 = negative | 1 = neutral | 2 = positive
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score

#Logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/sentiment_training.log", mode="a"),
    ],
)
logger = logging.getLogger("train_sentiment")

from config import CFG, LOGS_DIR, CHECKPOINTS_DIR
from models.text_encoder import FinBERTEncoder
from utils.data_loader import (
    HF_DATASET_ID,
    LABEL_MAP,
    LABEL_MAP_INV,
    load_financial_phrasebank,
    preprocess_dataset,
    create_sentiment_dataloaders,
)



#ARGUMENT PARSING


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune FinBERT on Financial PhraseBank (lmassaron/FinancialPhraseBank)"
    )
    p.add_argument("--epochs",      type=int,   default=10,   help="Training epochs (default: 10)")
    p.add_argument("--batch-size",  type=int,   default=16,   help="Batch size per device (default: 16)")
    p.add_argument("--lr",          type=float, default=2e-5, help="Top-layer learning rate (default: 2e-5)")
    p.add_argument("--bert-lr",     type=float, default=1e-5, help="BERT backbone LR (default: 1e-5)")
    p.add_argument("--max-len",     type=int,   default=128,  help="Max token length (default: 128)")
    p.add_argument("--num-workers", type=int,   default=None, help="DataLoader workers (auto if omitted)")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed (default: 42)")
    p.add_argument("--dry-run",     action="store_true",      help="2 epochs on 200 samples — smoke test")
    p.add_argument("--no-hf-cache", action="store_true",      help="Delete local cache and re-download")
    return p.parse_args()



#DEVICE SELECTION


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {props.total_memory / 1e9:.1f} GB  |  CUDA {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU (no GPU detected — training will be slow)")
    return device



#SENTIMENT CLASSIFIER


class FinBERTSentimentClassifier(nn.Module):
    """
    Thin wrapper around FinBERTEncoder that exposes the built-in sentiment_head.

    Reuses models/text_encoder.py so weights transfer directly to AuraMarketNet
    without any shape mismatches.

    Output: logits of shape [batch, 3]  (neg | neutral | pos)
    """

    def __init__(self, encoder_cfg):
        super().__init__()
        self.encoder = FinBERTEncoder(
            model_name=encoder_cfg.model_name,
            output_dim=encoder_cfg.output_dim,
            dropout=encoder_cfg.dropout,
            freeze_layers=encoder_cfg.freeze_layers,
            use_pooler=encoder_cfg.use_pooler,
            gradient_checkpointing=encoder_cfg.gradient_checkpointing,
        )

    def forward(
        self,
        input_ids: torch.Tensor,      
        attention_mask: torch.Tensor, 
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out["sentiment_logits"] 

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



#TRAINING


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    bert_lr: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_clip: float,
    use_amp: bool,
    device: torch.device,
    checkpoint_dir: Path,
    class_weights: torch.Tensor,
) -> dict:

    bert_params  = [p for n, p in model.named_parameters() if "bert"     in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad]

    optimizer = AdamW(
        [
            {"params": bert_params,  "lr": bert_lr},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,  #mild smoothing to reduce overconfidence
    )
    #GradScaler is a pure no-op when enabled=False (CPU / MPS paths)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = 0.0
    best_ckpt   = checkpoint_dir / "finbert_sentiment_best.pt"
    history: dict = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        #Train────
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        #Validate─
        val_loss, val_acc, val_f1 = _run_eval(
            model, val_loader, criterion, device, use_amp
        )

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))

        logger.info(
            f"Epoch {epoch:02d}/{num_epochs} [{time.time()-t0:.0f}s] | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1":          val_f1,
                    "val_acc":         val_acc,
                    "val_loss":        val_loss,
                },
                best_ckpt,
            )
            logger.info(f"  Saved best checkpoint (val_f1={val_f1:.4f}) → {best_ckpt}")

    logger.info(f"Training complete.  Best val macro-F1: {best_val_f1:.4f}")
    return history



#EVALUATION


@torch.no_grad()
def _run_eval(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple:
    """Return (avg_loss, accuracy, macro_f1) on a DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds:  list = []
    all_labels: list = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = float((np.array(all_preds) == np.array(all_labels)).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """
    Full test-set evaluation with per-class metrics and confusion matrix.

    Returns a dict with keys:
      accuracy, macro_f1, per_class_f1, confusion_matrix,
      classification_report (string)
    """
    model.eval()
    all_preds:  list = []
    all_labels: list = []

    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids, attention_mask)

        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    label_names = [LABEL_MAP_INV[i] for i in range(3)]
    report      = classification_report(all_labels, all_preds, target_names=label_names)
    cm          = confusion_matrix(all_labels, all_preds).tolist()
    acc         = float((np.array(all_preds) == np.array(all_labels)).mean())
    macro_f1    = float(f1_score(all_labels, all_preds, average="macro",  zero_division=0))
    per_class   = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

    return {
        "accuracy":               round(acc,      4),
        "macro_f1":               round(macro_f1, 4),
        "per_class_f1":           {LABEL_MAP_INV[i]: round(v, 4) for i, v in enumerate(per_class)},
        "confusion_matrix":       cm,
        "classification_report":  report,
    }
















def main() -> None:
    args = parse_args()

    #Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 68)
    logger.info("  FinBERT Sentiment Fine-Tuning  —  Financial PhraseBank")
    logger.info(f"  Dataset : {HF_DATASET_ID}")
    logger.info("=" * 68 + "\n")

    device  = resolve_device()
    use_amp = device.type == "cuda"

    #Optionally clear HuggingFace cache
    cache_dir  = "data/hf_cache"
    cache_file = Path(cache_dir) / f"{HF_DATASET_ID.replace('/', '_')}.parquet"
    if args.no_hf_cache and cache_file.exists():
        cache_file.unlink()
        logger.info(f"Deleted local cache ({cache_file}) — will re-download from HuggingFace")

    #Load & preprocess dataset
    df = load_financial_phrasebank(cache_dir=cache_dir)
    df = preprocess_dataset(df)

    if args.dry_run:
        df         = df.sample(n=min(200, len(df)), random_state=args.seed).reset_index(drop=True)
        args.epochs     = 2
        args.batch_size = 8
        logger.info("DRY RUN: 200 samples, 2 epochs, batch_size=8")

    #Tokenizer
    model_name = CFG.model.text_encoder.model_name
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #DataLoaders─
    num_workers = (
        args.num_workers if args.num_workers is not None
        else (4 if device.type == "cuda" else 0)
    )


    if device.type != "cuda" and num_workers > 0:
        logger.warning(
            f"Overriding --num-workers {num_workers} → 0.  "
            "HuggingFace fast tokenizers deadlock inside forked DataLoader "
            "workers on CPU (Linux fork + Rust threads).  "
            "Workers are re-enabled automatically when a CUDA GPU is detected."
        )
        num_workers = 0

    train_loader, val_loader, test_loader = create_sentiment_dataloaders(
        df=df,
        tokenizer=tokenizer,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed,
        max_seq_length=args.max_len,
    )

    #Build model
    logger.info("Building FinBERTSentimentClassifier ...")
    model = FinBERTSentimentClassifier(CFG.model.text_encoder).to(device)
    logger.info(f"Trainable parameters: {model.count_trainable():,}")

    class_weights = train_loader.dataset.get_class_weights()
    logger.info(f"Class weights: neg={class_weights[0]:.3f}  neu={class_weights[1]:.3f}  pos={class_weights[2]:.3f}")

    #Train
    logger.info(f"\nStarting training — {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    t_start = time.time()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        bert_lr=args.bert_lr,
        weight_decay=CFG.training.weight_decay,
        warmup_ratio=CFG.training.warmup_ratio,
        grad_clip=CFG.training.max_grad_norm,
        use_amp=use_amp,
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR,
        class_weights=class_weights,
    )

    logger.info(f"Total training time: {(time.time() - t_start) / 60:.1f} min")

    #Load best checkpoint and evaluate on test set
    best_ckpt = CHECKPOINTS_DIR / "finbert_sentiment_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val_f1={ckpt['val_f1']:.4f})")

    logger.info("\nEvaluating on held-out test set ...")
    test_metrics = evaluate_model(model, test_loader, device, use_amp)

    logger.info("\n" + "=" * 68)
    logger.info("  Test Results")
    logger.info("=" * 68)
    logger.info(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1  : {test_metrics['macro_f1']:.4f}")
    for cls_name, f1_val in test_metrics["per_class_f1"].items():
        logger.info(f"  F1 [{cls_name:8s}]: {f1_val:.4f}")
    logger.info("\nClassification Report:\n" + test_metrics["classification_report"])

    #Persist results──
    results_path = LOGS_DIR / "sentiment_test_metrics.json"
    save_payload = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    save_payload["training_history"] = history
    save_payload["config"] = {
        "dataset":        HF_DATASET_ID,
        "model":          model_name,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "bert_lr":        args.bert_lr,
        "max_seq_length": args.max_len,
        "seed":           args.seed,
    }

    with open(results_path, "w") as f:
        json.dump(save_payload, f, indent=2)

    logger.info(f"\nResults saved  → {results_path}")
    logger.info(f"Best checkpoint → {best_ckpt}")
    logger.info("\n✓  Sentiment fine-tuning complete!")


if __name__ == "__main__":
    main()
