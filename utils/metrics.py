import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: List[str] = None,
) -> Dict[str, float]:

    if labels is None:
        labels = ["DOWN", "UP"]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None and len(y_prob.shape) > 1:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["auc_roc"] = 0.5

    #Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics for price change prediction.

    Args:
        y_true: Ground truth returns [n]
        y_pred: Predicted returns [n]

    Returns:
        Dict with MAE, RMSE, MAPE, R², and correlation
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    #MAPE — handle zero returns gracefully
    nonzero_mask = np.abs(y_true) > 1e-6
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    #R² score
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    #Pearson correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "correlation": correlation,
    }


def compute_directional_accuracy(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray = None,
    y_pred_direction: np.ndarray = None,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute directional accuracy — the most important financial metric.

    "Did the model correctly predict whether the price went UP or DOWN?"

    This is more practically useful than absolute return error because
    directional accuracy directly maps to trading strategy profitability.

    Args:
        y_true_returns:  Actual returns [n]
        y_pred_returns:  Predicted returns [n] (use OR provide direction)
        y_pred_direction: Predicted direction [n] (0=DOWN, 1=UP)
        threshold:       Min absolute return to include in calculation

    Returns:
        Dict with directional_accuracy, up_accuracy, down_accuracy
    """
    y_true_returns = np.asarray(y_true_returns, dtype=np.float64)

    true_direction = (y_true_returns > threshold).astype(int)

    if y_pred_direction is not None:
        pred_direction = np.asarray(y_pred_direction, dtype=int)
    elif y_pred_returns is not None:
        pred_direction = (np.asarray(y_pred_returns, dtype=np.float64) > threshold).astype(int)
    else:
        raise ValueError("Must provide either y_pred_returns or y_pred_direction")

    #Overall directional accuracy
    dir_acc = float(np.mean(true_direction == pred_direction))

    #Accuracy on UP days (precision for bullish signal)
    up_mask = true_direction == 1
    up_acc = float(np.mean(pred_direction[up_mask] == 1)) if up_mask.sum() > 0 else 0.0

    #Accuracy on DOWN days (precision for bearish signal)
    down_mask = true_direction == 0
    down_acc = float(np.mean(pred_direction[down_mask] == 0)) if down_mask.sum() > 0 else 0.0

    return {
        "directional_accuracy": dir_acc,
        "up_accuracy": up_acc,
        "down_accuracy": down_acc,
        "n_up_days": int(up_mask.sum()),
        "n_down_days": int(down_mask.sum()),
    }


def compute_all_metrics(
    direction_true: np.ndarray,
    direction_pred: np.ndarray,
    direction_probs: Optional[np.ndarray],
    return_true: np.ndarray,
    return_pred: np.ndarray,
    volatility_true: Optional[np.ndarray] = None,
    volatility_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all metrics for the full multi-task model evaluation.

    Returns a flat dict suitable for logging to W&B, MLflow, etc.
    """
    metrics = {}

    #Classification metrics
    clf_metrics = compute_classification_metrics(
        y_true=direction_true,
        y_pred=direction_pred,
        y_prob=direction_probs,
    )
    metrics.update({f"clf_{k}": v for k, v in clf_metrics.items()})

    #Regression metrics (returns)
    reg_metrics = compute_regression_metrics(
        y_true=return_true,
        y_pred=return_pred,
    )
    metrics.update({f"ret_{k}": v for k, v in reg_metrics.items()})

    #Directional accuracy (from return predictions)
    dir_metrics = compute_directional_accuracy(
        y_true_returns=return_true,
        y_pred_direction=direction_pred,
    )
    metrics.update(dir_metrics)

    #Volatility metrics (optional)
    if volatility_true is not None and volatility_pred is not None:
        vol_metrics = compute_regression_metrics(
            y_true=volatility_true,
            y_pred=volatility_pred,
        )
        metrics.update({f"vol_{k}": v for k, v in vol_metrics.items()})

    return metrics


class BacktestSimulator:
    """
    Simple backtest simulation to estimate trading strategy performance.

    Strategy: Long if model predicts UP, Short (or cash) if model predicts DOWN.
    This is a simplified backtest for demonstration — real backtesting
    requires transaction costs, slippage, position sizing, etc.
    """

    def __init__(self, initial_capital: float = 100_000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run(
        self,
        returns: np.ndarray,        #Actual daily returns [n]
        predictions: np.ndarray,    #Model predictions: 1=UP, 0=DOWN [n]
        long_only: bool = True,
    ) -> Dict[str, float]:
        """
        Run backtest and return performance metrics.

        Args:
            returns: Actual daily returns as decimals (0.01 = 1%)
            predictions: Model direction predictions
            long_only: If True, go to cash on DOWN prediction; if False, go short

        Returns:
            Dict with total return, Sharpe ratio, max drawdown, win rate
        """
        n = len(returns)
        capital = self.initial_capital
        portfolio_values = [capital]
        trade_returns = []
        position = 0  #0=cash, 1=long, -1=short

        for i in range(n):
            pred = predictions[i]
            ret = returns[i]

            #Determine position
            if pred == 1:
                new_position = 1  #Long
            else:
                new_position = -1 if not long_only else 0  #Short or cash

            #Transaction cost on position change
            if new_position != position:
                capital *= (1 - self.transaction_cost)
                position = new_position

            #Apply return
            if position == 1:
                daily_return = ret
            elif position == -1:
                daily_return = -ret
            else:
                daily_return = 0.0

            capital *= (1 + daily_return)
            portfolio_values.append(capital)

            if position != 0:
                trade_returns.append(daily_return)

        portfolio_values = np.array(portfolio_values)
        total_return = (capital - self.initial_capital) / self.initial_capital

        #Benchmark (buy-and-hold)
        bh_return = np.prod(1 + returns) - 1

        #Sharpe ratio (annualized)
        if len(trade_returns) > 1:
            daily_sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8)
            annualized_sharpe = daily_sharpe * np.sqrt(252)
        else:
            annualized_sharpe = 0.0

        #Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = float(drawdown.min())

        #Win rate
        win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0

        return {
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "benchmark_return": float(bh_return),
            "excess_return": float(total_return - bh_return),
            "sharpe_ratio": float(annualized_sharpe),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown * 100),
            "win_rate": float(win_rate),
            "n_trades": len(trade_returns),
            "final_capital": float(capital),
        }
