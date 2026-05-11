import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
try:
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
except ImportError:
    #Fallback scheduler implementations when transformers not installed
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **kwargs):
        import math
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda)

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **kwargs):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda)

from .losses import MultiTaskLoss
from .callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger
from utils.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class AuraTrainer:
    """
    Manages the full training and evaluation lifecycle.

    Key features:
    - Differential learning rates: lower LR for BERT (fine-tuning),
      higher for fusion/heads (training from scratch)
    - FP16 mixed precision for 2x speedup on modern GPUs
    - Gradient accumulation to simulate large batch sizes on limited VRAM
    - Time-aware data splits to prevent look-ahead bias
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.tc = config.training

        #Device setup──
        if device is None:
            if self.tc.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.tc.device

        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            props = torch.cuda.get_device_properties(self.device)
            logger.info(f"GPU: {torch.cuda.get_device_name(self.device)} | VRAM: {props.total_memory / 1e9:.1f} GB")

        #Optimizer with differential LRs─
        self.optimizer = self._build_optimizer()

        #Loss function──
        self.criterion = MultiTaskLoss(
            classification_weight=self.tc.classification_weight,
            regression_weight=self.tc.regression_weight,
            volatility_weight=self.tc.volatility_weight,
            label_smoothing=0.1,
        ).to(self.device)

        #Mixed precision
        self.use_amp = self.tc.mixed_precision and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        #Callbacks─
        self.early_stopping = EarlyStopping(
            patience=self.tc.early_stopping_patience,
            mode="max",
            metric_name=self.tc.early_stopping_metric,
        )

        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=str(Path(config.model.model_name + "_checkpoints").resolve()),
            metric_name=self.tc.early_stopping_metric,
            mode="max",
            save_every_n_epochs=self.tc.save_every_n_epochs,
            keep_n_checkpoints=self.tc.keep_n_checkpoints,
        )

        self.logger = TrainingLogger(
            log_dir="logs",
            experiment_name=config.model.model_name,
        )

        #LR scheduler (set in train())───
        self.scheduler = None

        #Training state─
        self.current_epoch = 0
        self.global_step = 0

        #Set seeds for reproducibility
        self._set_seed(self.tc.seed)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with differential learning rates.

        BERT layers: lower LR (fine-tuning pre-trained weights)
        Other layers: higher LR (training from scratch)
        """
        bert_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bert" in name or "text_encoder" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {
                "params": bert_params,
                "lr": self.tc.bert_lr,
                "weight_decay": self.tc.weight_decay,
            },
            {
                "params": other_params,
                "lr": self.tc.learning_rate,
                "weight_decay": self.tc.weight_decay,
            },
        ])

        logger.info(f"Optimizer: AdamW | BERT LR={self.tc.bert_lr} | Other LR={self.tc.learning_rate}")
        logger.info(f"  BERT params: {sum(p.numel() for p in bert_params):,}")
        logger.info(f"  Other params: {sum(p.numel() for p in other_params):,}")

        return optimizer

    def _build_scheduler(self, total_steps: int):
        """Build cosine LR scheduler with linear warmup."""
        warmup_steps = int(total_steps * self.tc.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Scheduler: Cosine with {warmup_steps} warmup steps / {total_steps} total")
        return scheduler

    def _forward_batch(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Move batch to device and run forward pass."""
        #Move tensors to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        text_mask = batch.get("text_mask")
        if text_mask is not None:
            text_mask = text_mask.to(self.device)

        numerical = batch["numerical_features"].to(self.device)

        targets = {
            "direction": batch["direction"].to(self.device),
            "price_change": batch["price_change"].to(self.device),
            "volatility": batch["volatility"].to(self.device),
        }

        #Forward pass (with optional AMP) ──
        amp_ctx = autocast("cuda") if self.use_amp else nullcontext()
        with amp_ctx:
            predictions = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_mask=text_mask,
                numerical_features=numerical,
            )

        return predictions, targets

    def _train_step(self, batch: Dict, is_accumulation_step: bool) -> Dict[str, float]:
        """Execute one optimization step."""
        predictions, targets = self._forward_batch(batch)

        #Compute loss──
        amp_ctx = autocast("cuda") if self.use_amp else nullcontext()
        with amp_ctx:
            losses = self.criterion(predictions, targets)
            loss = losses["total_loss"] / self.tc.gradient_accumulation_steps

        #Backward pass──
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        #Optimizer step (after gradient accumulation) ──
        if not is_accumulation_step:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tc.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tc.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataloader. Returns dict of metrics."""
        self.model.eval()

        all_direction_true = []
        all_direction_pred = []
        all_direction_prob = []
        all_return_true = []
        all_return_pred = []
        all_vol_true = []
        all_vol_pred = []
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            predictions, targets = self._forward_batch(batch)
            losses = self.criterion(predictions, targets)
            total_loss += losses["total_loss"].item()
            n_batches += 1

            #Collect predictions
            dir_probs = torch.softmax(predictions["direction_logits"], dim=-1)
            dir_pred = dir_probs.argmax(dim=-1)

            all_direction_true.extend(targets["direction"].cpu().numpy())
            all_direction_pred.extend(dir_pred.cpu().numpy())
            all_direction_prob.extend(dir_probs.cpu().numpy())
            all_return_true.extend(targets["price_change"].cpu().numpy())
            all_return_pred.extend(predictions["price_change"].squeeze(-1).cpu().numpy())
            all_vol_true.extend(targets["volatility"].cpu().numpy())
            all_vol_pred.extend(predictions["volatility"].squeeze(-1).cpu().numpy())

        #Compute metrics
        metrics = compute_all_metrics(
            direction_true=np.array(all_direction_true),
            direction_pred=np.array(all_direction_pred),
            direction_probs=np.array(all_direction_prob),
            return_true=np.array(all_return_true),
            return_pred=np.array(all_return_pred),
            volatility_true=np.array(all_vol_true),
            volatility_pred=np.array(all_vol_pred),
        )

        metrics["loss"] = total_loss / n_batches
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        start_epoch: int = 1,
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            train_loader: Training DataLoader
            val_loader:   Validation DataLoader
            num_epochs:   Total epochs to train (inclusive upper bound)
            start_epoch:  Epoch to start from (1 = fresh, >1 = resumed)

        Returns:
            Training history dict
        """
        if num_epochs is None:
            num_epochs = self.tc.num_epochs

        remaining_epochs = num_epochs - start_epoch + 1
        if remaining_epochs <= 0:
            logger.info(f"Nothing to train: start_epoch={start_epoch} >= num_epochs={num_epochs}")
            return self.logger.get_history()

        #Scheduler covers only the epochs still to run so LR curve is correct on resume
        total_steps = (
            len(train_loader) //self.tc.gradient_accumulation_steps
        ) * remaining_epochs
        self.scheduler = self._build_scheduler(total_steps)

        logger.info(f"\n{'='*60}")
        logger.info(f"Training epochs {start_epoch}–{num_epochs} ({remaining_epochs} remaining)")
        logger.info(f"Scheduler steps: {total_steps}")
        logger.info(f"{'='*60}\n")

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            #Training phase───
            self.model.train()
            self.optimizer.zero_grad()

            train_losses = []
            for step, batch in enumerate(train_loader):
                is_accumulation = (step + 1) % self.tc.gradient_accumulation_steps != 0

                step_losses = self._train_step(batch, is_accumulation)
                train_losses.append(step_losses)

                if step % 20 == 0:
                    avg_loss = np.mean([l["total_loss"] for l in train_losses[-20:]])
                    lr = self.optimizer.param_groups[1]["lr"]
                    logger.info(
                        f"  Epoch {epoch}/{num_epochs} | Step {step}/{len(train_loader)}"
                        f" | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

            #Aggregate training losses
            train_metrics = {
                f"train_{k}": np.mean([l[k] for l in train_losses])
                for k in train_losses[0]
            }

            #Validation phase─
            val_metrics = self._eval_epoch(val_loader)
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            #Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["lr"] = self.optimizer.param_groups[1]["lr"]

            epoch_time = time.time() - epoch_start

            #Logging─
            self.logger.log(epoch, epoch_metrics)
            self._print_epoch_summary(epoch, epoch_metrics, epoch_time)

            #Callbacks──
            self.checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                metrics=epoch_metrics,
                epoch=epoch,
            )

            if self.early_stopping(self.model, epoch_metrics, epoch):
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break

        logger.info(f"\nTraining complete!")
        best = self.logger.get_best("val_directional_accuracy", mode="max")
        logger.info(f"Best epoch {best.get('epoch')}: "
                    f"val_directional_accuracy={best.get('val_directional_accuracy', 0):.4f}")

        return self.logger.get_history()

    def _print_epoch_summary(
        self, epoch: int, metrics: Dict, epoch_time: float
    ) -> None:
        """Print a clean epoch summary table."""
        print(f"\n{'─'*65}")
        gpu_info = ""
        if self.device.type == "cuda":
            used = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            gpu_info = f" | GPU Mem: {used:.1f}/{total:.1f} GB"
        print(f" Epoch {epoch:3d} | Time: {epoch_time:.1f}s{gpu_info}")
        print(f"{'─'*65}")
        print(f"  {'Metric':<35} {'Train':>10} {'Val':>10}")
        print(f"{'─'*65}")

        key_metrics = [
            ("Loss", "train_total_loss", "val_loss"),
            ("Directional Accuracy", "N/A", "val_directional_accuracy"),
            ("Classification F1", "N/A", "val_clf_f1"),
            ("Return MAE", "N/A", "val_ret_mae"),
            ("Return Correlation", "N/A", "val_ret_correlation"),
        ]

        for name, train_key, val_key in key_metrics:
            train_val = f"{metrics.get(train_key, 0):.4f}" if train_key in metrics else "   —"
            val_val = f"{metrics.get(val_key, 0):.4f}" if val_key in metrics else "   —"
            print(f"  {name:<35} {train_val:>10} {val_val:>10}")

        print(f"{'─'*65}\n")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set and return all metrics."""
        logger.info("Evaluating on test set...")
        metrics = self._eval_epoch(test_loader)
        logger.info("\nTest Set Results:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                logger.info(f"  {k:<35}: {v:.4f}")
        return metrics
