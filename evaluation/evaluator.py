import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path

from utils.metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_directional_accuracy,
    BacktestSimulator,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        backtest_capital: float = 100_000.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.backtester = BacktestSimulator(initial_capital=backtest_capital)

    @torch.no_grad()
    def predict_dataloader(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Run inference on all samples in a DataLoader.

        Returns dict with all predictions and targets.
        """
        self.model.eval()

        all_preds = {
            "direction_true": [],
            "direction_pred": [],
            "direction_prob_up": [],
            "return_true": [],
            "return_pred": [],
            "vol_true": [],
            "vol_pred": [],
            "confidence": [],
        }

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            text_mask = batch.get("text_mask")
            if text_mask is not None:
                text_mask = text_mask.to(self.device)
            numerical = batch["numerical_features"].to(self.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_mask=text_mask,
                numerical_features=numerical,
            )

            #Direction
            dir_probs = torch.softmax(out["direction_logits"], dim=-1)
            dir_pred = dir_probs.argmax(dim=-1)
            confidence = dir_probs.max(dim=-1).values

            all_preds["direction_true"].extend(batch["direction"].numpy())
            all_preds["direction_pred"].extend(dir_pred.cpu().numpy())
            all_preds["direction_prob_up"].extend(dir_probs[:, 1].cpu().numpy())
            all_preds["confidence"].extend(confidence.cpu().numpy())

            #Return
            all_preds["return_true"].extend(batch["price_change"].numpy())
            all_preds["return_pred"].extend(out["price_change"].squeeze(-1).cpu().numpy())

            #Volatility
            all_preds["vol_true"].extend(batch["volatility"].numpy())
            all_preds["vol_pred"].extend(out["volatility"].squeeze(-1).cpu().numpy())

        return {k: np.array(v) for k, v in all_preds.items()}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:

        logger.info("Running full evaluation...")
        preds = self.predict_dataloader(dataloader)

        #Core metrics
        metrics = compute_all_metrics(
            direction_true=preds["direction_true"],
            direction_pred=preds["direction_pred"],
            direction_probs=np.column_stack([
                1 - preds["direction_prob_up"],
                preds["direction_prob_up"]
            ]),
            return_true=preds["return_true"],
            return_pred=preds["return_pred"],
            volatility_true=preds["vol_true"],
            volatility_pred=preds["vol_pred"],
        )

        #Backtest simulation (long-only strategy)
        bt_results = self.backtester.run(
            returns=preds["return_true"],
            predictions=preds["direction_pred"],
            long_only=True,
        )
        metrics.update({f"backtest_{k}": v for k, v in bt_results.items()})

        #Confidence-stratified analysis
        conf_metrics = self._analyze_by_confidence(preds)
        metrics.update(conf_metrics)

        self._print_evaluation_report(metrics, preds)
        return metrics

    def _analyze_by_confidence(
        self, preds: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
       
        confidence = preds["confidence"]
        results = {}

        for threshold in [0.6, 0.7, 0.8, 0.9]:
            mask = confidence >= threshold
            n_filtered = mask.sum()

            if n_filtered < 10:
                continue

            filtered_dir_acc = np.mean(
                preds["direction_true"][mask] == preds["direction_pred"][mask]
            )
            coverage = n_filtered / len(confidence)

            results[f"dir_acc_conf{int(threshold*100)}"] = float(filtered_dir_acc)
            results[f"coverage_conf{int(threshold*100)}"] = float(coverage)

        return results

    def _print_evaluation_report(
        self, metrics: Dict[str, float], preds: Dict[str, np.ndarray]
    ) -> None:
        """Print a formatted evaluation report."""
        print("\n" + "="*70)
        print("  AURAMARKETNET EVALUATION REPORT")
        print("="*70)

        print("\n  DIRECTION CLASSIFICATION")
        print(f"  {'Accuracy':<35} {metrics.get('clf_accuracy', 0):.4f}")
        print(f"  {'F1 Score':<35} {metrics.get('clf_f1', 0):.4f}")
        print(f"  {'Precision':<35} {metrics.get('clf_precision', 0):.4f}")
        print(f"  {'Recall':<35} {metrics.get('clf_recall', 0):.4f}")
        print(f"  {'ROC-AUC':<35} {metrics.get('clf_auc_roc', 0):.4f}")

        print("\n  RETURN REGRESSION")
        print(f"  {'MAE':<35} {metrics.get('ret_mae', 0):.6f}")
        print(f"  {'RMSE':<35} {metrics.get('ret_rmse', 0):.6f}")
        print(f"  {'R²':<35} {metrics.get('ret_r2', 0):.4f}")
        print(f"  {'Correlation':<35} {metrics.get('ret_correlation', 0):.4f}")

        print("\n  FINANCIAL METRICS")
        print(f"  {'Directional Accuracy':<35} {metrics.get('directional_accuracy', 0):.4f}")
        print(f"  {'UP Day Accuracy':<35} {metrics.get('up_accuracy', 0):.4f}")
        print(f"  {'DOWN Day Accuracy':<35} {metrics.get('down_accuracy', 0):.4f}")

        print("\n  BACKTEST (Long-Only Strategy)")
        print(f"  {'Total Return':<35} {metrics.get('backtest_total_return_pct', 0):.2f}%")
        print(f"  {'Benchmark (Buy & Hold)':<35} {metrics.get('backtest_benchmark_return', 0)*100:.2f}%")
        print(f"  {'Sharpe Ratio':<35} {metrics.get('backtest_sharpe_ratio', 0):.4f}")
        print(f"  {'Max Drawdown':<35} {metrics.get('backtest_max_drawdown_pct', 0):.2f}%")
        print(f"  {'Win Rate':<35} {metrics.get('backtest_win_rate', 0):.4f}")

        print("\n  CONFIDENCE STRATIFICATION")
        for thresh in [60, 70, 80, 90]:
            da = metrics.get(f'dir_acc_conf{thresh}', None)
            cov = metrics.get(f'coverage_conf{thresh}', None)
            if da is not None:
                print(f"  {'Conf ≥ ' + str(thresh) + '%':<35} DA={da:.4f} | Coverage={cov:.2%}")

        print("="*70 + "\n")

    def save_predictions(self, preds: Dict[str, np.ndarray], output_path: str) -> None:
        """Save predictions to CSV for further analysis."""
        df = pd.DataFrame({
            "true_direction": preds["direction_true"],
            "pred_direction": preds["direction_pred"],
            "prob_up": preds["direction_prob_up"],
            "confidence": preds["confidence"],
            "true_return": preds["return_true"],
            "pred_return": preds["return_pred"],
            "true_volatility": preds["vol_true"],
            "pred_volatility": preds["vol_pred"],
        })
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
