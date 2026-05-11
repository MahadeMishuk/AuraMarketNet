import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Loss = w1 * L_direction + w2 * L_return + w3 * L_volatility + w4 * L_sentiment
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 0.5,
        volatility_weight: float = 0.3,
        sentiment_weight: float = 0.2,
        use_uncertainty_weighting: bool = False,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.w_clf = classification_weight
        self.w_reg = regression_weight
        self.w_vol = volatility_weight
        self.w_sent = sentiment_weight
        self.use_uncertainty = use_uncertainty_weighting

        if use_uncertainty_weighting:
            #Learnable log-variance parameters (Kendall et al. 2018)
            #L_i = (1 / 2*sigma_i^2) * l_i + log(sigma_i)
            #Using log_sigma for numerical stability
            self.log_sigma_clf = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_vol = nn.Parameter(torch.tensor(0.0))
            logger.info("Using learned uncertainty weighting for multi-task loss")

        #Task-specific losses
        #Label smoothing for classification: reduces overconfidence
        #especially important for financial prediction where uncertainty is high
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        #Huber loss for return regression: robust to outlier returns
        #(earnings surprises, black swan events) compared to MSE
        self.regression_loss = nn.HuberLoss(delta=0.1)

        #MSE for volatility: volatility prediction benefits from mean-squared
        #error as we care about the absolute scale
        self.volatility_loss = nn.MSELoss()

        #Auxiliary sentiment loss
        self.sentiment_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}

        #Direction classification loss───
        l_clf = self.classification_loss(
            predictions["direction_logits"],
            targets["direction"],
        )
        losses["direction_loss"] = l_clf

        #Return regression loss─
        l_reg = self.regression_loss(
            predictions["price_change"].squeeze(-1),
            targets["price_change"],
        )
        losses["regression_loss"] = l_reg

        #Volatility loss (optional)─
        if "volatility" in predictions and "volatility" in targets:
            l_vol = self.volatility_loss(
                predictions["volatility"].squeeze(-1),
                targets["volatility"],
            )
            losses["volatility_loss"] = l_vol
        else:
            l_vol = torch.tensor(0.0, device=l_clf.device)
            losses["volatility_loss"] = l_vol

        #Auxiliary sentiment loss───
        if "sentiment_logits" in predictions and "sentiment" in targets:
            l_sent = self.sentiment_loss(
                predictions["sentiment_logits"],
                targets["sentiment"],
            )
            losses["sentiment_loss"] = l_sent
        else:
            l_sent = torch.tensor(0.0, device=l_clf.device)
            losses["sentiment_loss"] = l_sent

        #Combine with weighting strategy─
        if self.use_uncertainty:
            #Learned uncertainty weighting (Kendall et al. 2018)
            #Minimizing this automatically adjusts task weights
            total_loss = (
                torch.exp(-self.log_sigma_clf) * l_clf + self.log_sigma_clf +
                torch.exp(-self.log_sigma_reg) * l_reg + self.log_sigma_reg +
                torch.exp(-self.log_sigma_vol) * l_vol + self.log_sigma_vol
            ) + self.w_sent * l_sent
        else:
            total_loss = (
                self.w_clf * l_clf +
                self.w_reg * l_reg +
                self.w_vol * l_vol +
                self.w_sent * l_sent
            )

        losses["total_loss"] = total_loss
        return losses


class FocalLoss(nn.Module):
    

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [batch, num_classes]
            targets: [batch] — class indices

        Returns:
            Focal loss scalar
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        #Probability of the true class
        p_t = (probs * targets_one_hot).sum(dim=-1)  #[batch]

        #Focal weight: down-weight easy examples
        focal_weight = (1 - p_t) ** self.gamma

        #Alpha weighting
        alpha_t = self.alpha * targets_one_hot.sum(dim=-1) + (1 - self.alpha) * (1 - targets_one_hot.sum(dim=-1))
        alpha_t = alpha_t.clamp(min=self.alpha, max=1 - self.alpha)

        #Cross-entropy component
        ce = -torch.log(p_t.clamp(min=1e-8))

        loss = alpha_t * focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DirectionAwareLoss(nn.Module):
   

    def __init__(self, direction_penalty: float = 2.0):
        super().__init__()
        self.direction_penalty = direction_penalty
        self.regression_loss = nn.HuberLoss(delta=0.1)

    def forward(
        self,
        pred_returns: torch.Tensor,   #[batch]
        true_returns: torch.Tensor,   #[batch]
    ) -> torch.Tensor:
        """
        Huber loss + extra penalty for wrong directional predictions.
        """
        base_loss = self.regression_loss(pred_returns, true_returns)

        #Direction indicator: 1 if same direction, -1 if opposite
        pred_dir = torch.sign(pred_returns)
        true_dir = torch.sign(true_returns)
        direction_match = (pred_dir == true_dir).float()

        #Extra penalty when direction is wrong
        direction_penalty = (1 - direction_match) * self.direction_penalty * torch.abs(true_returns)

        return base_loss + direction_penalty.mean()
