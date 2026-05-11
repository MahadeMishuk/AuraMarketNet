import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

logger = logging.getLogger(__name__)


class EarlyStopping:
    

    def __init__(
        self,
        patience: int = 7,
        mode: str = "max",       #"max" for accuracy/F1, "min" for loss
        min_delta: float = 1e-4, #Minimum improvement to count as progress
        metric_name: str = "val_directional_accuracy",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.restore_best_weights = restore_best_weights

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_weights = None
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.stopped_epoch = 0

    def __call__(self, model: torch.nn.Module, metrics: Dict[str, float], epoch: int) -> bool:
      
        current_value = metrics.get(self.metric_name, 0.0)

        improved = (
            current_value > self.best_value + self.min_delta
            if self.mode == "max"
            else current_value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.clone().cpu()
                    for k, v in model.state_dict().items()
                }
            logger.info(f"  [EarlyStopping] New best {self.metric_name}: {current_value:.4f}")
        else:
            self.epochs_without_improvement += 1
            logger.info(
                f"  [EarlyStopping] No improvement for {self.epochs_without_improvement}/{self.patience} epochs"
                f" (best={self.best_value:.4f} at epoch {self.best_epoch})"
            )

        if self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"  [EarlyStopping] Stopping training at epoch {epoch}")

            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(f"  [EarlyStopping] Restored weights from epoch {self.best_epoch}")

            return True

        return False

    @property
    def is_best(self) -> bool:
        return self.epochs_without_improvement == 0


class ModelCheckpoint:
    """
    Saves model checkpoints during training.

    Saves:
    - Latest checkpoint (every N epochs)
    - Best checkpoint (when validation metric improves)
    - Keeps only the last K checkpoints to save disk space
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        metric_name: str = "val_directional_accuracy",
        mode: str = "max",
        save_every_n_epochs: int = 2,
        keep_n_checkpoints: int = 3,
        model_name: str = "aura_market_net",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.save_every_n = save_every_n_epochs
        self.keep_n = keep_n_checkpoints
        self.model_name = model_name

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.saved_checkpoints = []

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        epoch: int,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Conditionally save checkpoint.

        Returns path to saved checkpoint, or None if not saved.
        """
        saved_path = None
        current_value = metrics.get(self.metric_name, 0.0)

        #Save periodic checkpoint──
        if epoch % self.save_every_n == 0:
            path = self._save_checkpoint(model, optimizer, metrics, epoch, tag=f"epoch_{epoch:03d}")
            self.saved_checkpoints.append(path)
            saved_path = path

            #Prune old checkpoints
            self._prune_old_checkpoints()

        #Save best checkpoint─
        is_best = (
            current_value > self.best_value
            if self.mode == "max"
            else current_value < self.best_value
        )

        if is_best:
            self.best_value = current_value
            best_path = self._save_checkpoint(
                model, optimizer, metrics, epoch, tag="best"
            )
            logger.info(f"  [Checkpoint] New best model saved: {self.metric_name}={current_value:.4f}")
            saved_path = best_path

        return saved_path

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        epoch: int,
        tag: str = "",
    ) -> Path:
        """Save a checkpoint to disk."""
        filename = f"{self.model_name}_{tag}.pt"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model.get_model_config() if hasattr(model, "get_model_config") else {},
        }

        torch.save(checkpoint, path)

        #Also save metrics as JSON for easy inspection
        metrics_path = path.with_suffix(".json")
        with open(metrics_path, "w") as f:
            json.dump({"epoch": epoch, "metrics": metrics, "tag": tag}, f, indent=2)

        logger.info(f"  [Checkpoint] Saved: {path}")
        return path

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent N."""
        #Exclude "best" checkpoint from pruning
        periodic = [p for p in self.saved_checkpoints if "best" not in str(p)]
        while len(periodic) > self.keep_n:
            old = periodic.pop(0)
            if old.exists():
                old.unlink()
                json_path = old.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
                logger.debug(f"  [Checkpoint] Pruned: {old}")

    @classmethod
    def load_checkpoint(
        cls, path: str, model: torch.nn.Module, optimizer=None, device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load a saved checkpoint."""
        try:
            import numpy._core.multiarray as _np_core
            torch.serialization.add_safe_globals([_np_core.scalar])
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        return checkpoint


class TrainingLogger:


    def __init__(self, log_dir: str = "logs", experiment_name: str = "aura_market_net"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history = []

        #Log file
        self.log_file = self.log_dir / f"{experiment_name}_training.jsonl"

    def log(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for one epoch."""
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_history(self) -> list:
        return self.history

    def get_best(self, metric: str, mode: str = "max") -> Dict:
        if not self.history:
            return {}
        return max(self.history, key=lambda x: x.get(metric, float("-inf"))) \
            if mode == "max" else min(self.history, key=lambda x: x.get(metric, float("inf")))
