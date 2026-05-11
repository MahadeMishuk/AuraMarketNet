import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

from .text_encoder import FinBERTEncoder, MultiTextEncoder
from .numerical_encoder import NumericalEncoder
from .fusion import CrossAttentionFusion, ConcatFusion

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):

    def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()

        #Classification head: UP / DOWN (binary)
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, input_dim //2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim //2, num_classes),
        )

        #Regression head: percentage price change (continuous)
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, input_dim //2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim //2, 1),
        )

        #Volatility head: next-day volatility estimate (continuous, positive)
        self.volatility_head = nn.Sequential(
            nn.Linear(input_dim, input_dim //4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim //4, 1),
            nn.Softplus(),  #Ensure positive output (volatility ≥ 0)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "direction_logits": self.classification_head(x),  #[batch, 2]
            "price_change": self.regression_head(x),           #[batch, 1]
            "volatility": self.volatility_head(x),             #[batch, 1]
        }


class AuraMarketNet(nn.Module):

    def __init__(
        self,
        #Text encoder params
        bert_model: str = "ProsusAI/finbert",
        text_output_dim: int = 256,
        text_dropout: float = 0.1,
        freeze_bert_layers: int = 8,
        multi_text: bool = True,          #Use multiple texts per sample

        #Numerical encoder params
        numerical_input_dim: int = 20,
        numerical_hidden_dim: int = 256,
        numerical_output_dim: int = 256,
        numerical_num_layers: int = 3,
        numerical_dropout: float = 0.2,
        sequence_length: int = 30,

        #Fusion params
        fusion_type: str = "cross_attention",  #"cross_attention", "concat", "bilinear"
        fusion_dim: int = 512,
        num_fusion_heads: int = 8,
        fusion_dropout: float = 0.1,

        #Output params
        num_classes: int = 2,
        output_dropout: float = 0.15,
    ):
        super().__init__()

        self.multi_text = multi_text
        self.fusion_type = fusion_type

        #Text Encoder──
        if multi_text:
            self.text_encoder = MultiTextEncoder(
                model_name=bert_model,
                output_dim=text_output_dim,
                dropout=text_dropout,
                freeze_layers=freeze_bert_layers,
            )
        else:
            self.text_encoder = FinBERTEncoder(
                model_name=bert_model,
                output_dim=text_output_dim,
                dropout=text_dropout,
                freeze_layers=freeze_bert_layers,
            )

        #Numerical Encoder───
        self.numerical_encoder = NumericalEncoder(
            input_dim=numerical_input_dim,
            hidden_dim=numerical_hidden_dim,
            num_layers=numerical_num_layers,
            output_dim=numerical_output_dim,
            dropout=numerical_dropout,
            bidirectional=True,
            num_attention_heads=8,
            use_attention=True,
        )

        #Fusion Layer──
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                text_dim=text_output_dim,
                numerical_dim=numerical_output_dim,
                fusion_dim=fusion_dim,
                num_heads=num_fusion_heads,
                dropout=fusion_dropout,
                hidden_layers=[fusion_dim, fusion_dim //2],
            )
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(
                text_dim=text_output_dim,
                numerical_dim=numerical_output_dim,
                hidden_layers=[fusion_dim, fusion_dim //2],
                dropout=fusion_dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        #Output Heads──
        self.output_heads = MultiTaskHead(
            input_dim=self.fusion.fusion_output_dim,
            num_classes=num_classes,
            dropout=output_dropout,
        )

        self._log_model_info()

    def _log_model_info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"AuraMarketNet initialized")
        logger.info(f"  Total parameters:     {total:,}")
        logger.info(f"  Trainable parameters: {trainable:,}")
        logger.info(f"  Frozen parameters:    {total - trainable:,}")

    def forward(
        self,
        #Text inputs
        input_ids: torch.Tensor,           #[batch, n_texts, seq_len] or [batch, seq_len]
        attention_mask: torch.Tensor,      #same shape as input_ids
        token_type_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,  #[batch, n_texts] — valid text mask

        #Numerical inputs
        numerical_features: torch.Tensor = None,  #[batch, seq_len, feature_dim]
        sequence_lengths: Optional[torch.Tensor] = None,

        #Control
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the dual-stream architecture.

        Args:
            input_ids:          Tokenized text input IDs
            attention_mask:     Text attention masks
            token_type_ids:     Token type IDs (optional)
            text_mask:          Which text slots have valid content
            numerical_features: Time-series market features
            sequence_lengths:   Actual lengths of numerical sequences
            return_intermediates: Return intermediate embeddings for analysis

        Returns:
            Dict containing all predictions and optionally intermediate states
        """
        #Text Encoding──
        if self.multi_text:
            text_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_mask=text_mask,
            )
        else:
            text_out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        text_embedding = text_out["embedding"]  #[batch, text_output_dim]

        #Numerical Encoding───
        num_out = self.numerical_encoder(
            x=numerical_features,
            lengths=sequence_lengths,
        )
        num_embedding = num_out["embedding"]    #[batch, numerical_output_dim]

        #Fusion────
        fusion_out = self.fusion(
            text_emb=text_embedding,
            num_emb=num_embedding,
        )
        fused = fusion_out["fused_embedding"]   #[batch, fusion_output_dim]

        #Multi-task Prediction
        outputs = self.output_heads(fused)

        #Add auxiliary sentiment prediction (from text encoder)
        if "sentiment_logits" in text_out:
            outputs["sentiment_logits"] = text_out["sentiment_logits"]

        #Optionally return intermediate states for visualization
        if return_intermediates:
            outputs.update({
                "text_embedding": text_embedding,
                "numerical_embedding": num_embedding,
                "fused_embedding": fused,
                "temporal_weights": num_out.get("temporal_weights"),
                "gate_weights": fusion_out.get("gate_weights"),
                "text_attention": text_out.get("text_attention_weights"),
            })

        return outputs

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference-mode prediction with human-readable output.

        Returns dict with:
          - direction: "UP" or "DOWN"
          - confidence: probability of predicted direction (0-1)
          - price_change_pct: predicted % change
          - volatility: predicted volatility
          - sentiment: "positive" / "negative" / "neutral"
        """
        self.eval()
        raw = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical_features=numerical_features,
            **kwargs,
        )

        #Classification
        direction_probs = torch.softmax(raw["direction_logits"], dim=-1)
        direction_pred = direction_probs.argmax(dim=-1)
        confidence = direction_probs.max(dim=-1).values

        #Sentiment
        sentiment_probs = torch.softmax(raw["sentiment_logits"], dim=-1)
        sentiment_pred = sentiment_probs.argmax(dim=-1)
        sentiment_labels = ["neutral", "positive", "negative"]

        return {
            "direction": ["DOWN", "UP"][direction_pred.item()],
            "confidence": confidence.item(),
            "direction_probs": {
                "DOWN": direction_probs[0, 0].item(),
                "UP": direction_probs[0, 1].item(),
            },
            "price_change_pct": raw["price_change"].item() * 100,
            "volatility": raw["volatility"].item(),
            "sentiment": sentiment_labels[sentiment_pred.item()],
            "sentiment_probs": {
                "neutral": sentiment_probs[0, 0].item(),
                "positive": sentiment_probs[0, 1].item(),
                "negative": sentiment_probs[0, 2].item(),
            },
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Return model configuration for saving/loading."""
        return {
            "architecture": "AuraMarketNet",
            "version": "1.0",
            "text_encoder": str(type(self.text_encoder).__name__),
            "fusion_type": self.fusion_type,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_model(config=None) -> AuraMarketNet:
    """
    Factory function to create AuraMarketNet from config.
    Use this rather than calling the constructor directly.
    """
    if config is None:
        from config import CFG
        config = CFG

    model = AuraMarketNet(
        bert_model=config.model.text_encoder.model_name,
        text_output_dim=config.model.text_encoder.output_dim,
        text_dropout=config.model.text_encoder.dropout,
        freeze_bert_layers=config.model.text_encoder.freeze_layers,
        multi_text=True,

        numerical_input_dim=config.model.numerical_encoder.input_dim,
        numerical_hidden_dim=config.model.numerical_encoder.hidden_dim,
        numerical_output_dim=config.model.numerical_encoder.output_dim,
        numerical_num_layers=config.model.numerical_encoder.num_layers,
        numerical_dropout=config.model.numerical_encoder.dropout,

        fusion_type=config.model.fusion.fusion_type,
        fusion_dim=config.model.fusion.fusion_dim,
        num_fusion_heads=config.model.fusion.num_heads,
        fusion_dropout=config.model.fusion.dropout,

        num_classes=config.model.output_heads.num_classes,
        output_dropout=config.model.output_heads.dropout,
    )

    return model
