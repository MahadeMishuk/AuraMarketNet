import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FinBERTEncoder(nn.Module):
    """
    Transformer-based text encoder fine-tuned for financial text.

    Input:  tokenized text (input_ids, attention_mask, token_type_ids)
    Output: embedding vector of shape [batch, output_dim]
            + optional attention weights for visualization
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_layers: int = 8,
        use_pooler: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_pooler = use_pooler
        logger.info(f"Loading pretrained model: {model_name}")
        try:
            self.bert = AutoModel.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=True,
            )
        except Exception:
            self.bert = AutoModel.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True,
            )

        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()


        self._freeze_bert_layers(freeze_layers)

        hidden_size = self.bert.config.hidden_size  #768 for BERT-base


        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size //2),
            nn.GELU(),
            nn.LayerNorm(hidden_size //2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size //2, output_dim),
            nn.LayerNorm(output_dim),
        )


        self.sentiment_head = nn.Linear(output_dim, 3)  #pos / neg / neutral

    def _freeze_bert_layers(self, num_frozen_layers: int) -> None:
        """Freeze the first num_frozen_layers transformer blocks."""
        #Always freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        #Freeze specified encoder layers
        if hasattr(self.bert, "encoder"):
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < num_frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        frozen = sum(1 for p in self.bert.parameters() if not p.requires_grad)
        total = sum(1 for p in self.bert.parameters())
        logger.info(f"Frozen {frozen}/{total} BERT parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
      
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.hidden_states  #tuple of [batch, seq, 768]
        last_4 = torch.stack(hidden_states[-4:], dim=0)  #[4, batch, seq, 768]
        pooled = last_4.mean(dim=0)                       #[batch, seq, 768]

        mask = attention_mask.unsqueeze(-1).float()       #[batch, seq, 1]
        embedding = (pooled * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embedding = self.projection(embedding)            #[batch, output_dim]
        sentiment_logits = self.sentiment_head(embedding) #[batch, 3]

        result = {
            "embedding": embedding,
            "sentiment_logits": sentiment_logits,
        }

        if return_attentions:
            result["attentions"] = outputs.attentions    #One per layer

        return result

    @torch.no_grad()
    def encode(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        """Inference-only forward pass. Returns embedding only."""
        out = self.forward(input_ids, attention_mask, token_type_ids)
        return out["embedding"]

    def get_tokenizer(self):
        """Convenience method to load the matching tokenizer."""
        return AutoTokenizer.from_pretrained(self.model_name)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTextEncoder(nn.Module):
    """
    Encodes MULTIPLE text inputs per sample (e.g., 3 days of news headlines)
    and aggregates them into a single embedding via attention pooling.

    This is more realistic than encoding just one headline per day — markets
    react to the cumulative sentiment of multiple news sources.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        output_dim: int = 256,
        max_texts_per_sample: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.max_texts = max_texts_per_sample

        #Shared encoder for all text inputs
        self.encoder = FinBERTEncoder(
            model_name=model_name,
            output_dim=output_dim,
            **kwargs,
        )

        #Attention pooling: learn which headlines matter most
        self.attention_pooler = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,       #[batch, n_texts, seq_len]
        attention_mask: torch.Tensor,  #[batch, n_texts, seq_len]
        text_mask: Optional[torch.Tensor] = None,  #[batch, n_texts] which texts are valid
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      [batch, n_texts, seq_len]
            attention_mask: [batch, n_texts, seq_len]
            text_mask:      [batch, n_texts] — 1 for valid text, 0 for padding

        Returns:
            dict with "embedding" [batch, output_dim]
        """
        batch_size, n_texts, seq_len = input_ids.shape

        #Encode all texts in parallel by flattening batch×n_texts
        flat_ids = input_ids.view(batch_size * n_texts, seq_len)
        flat_mask = attention_mask.view(batch_size * n_texts, seq_len)

        out = self.encoder(flat_ids, flat_mask)
        embeddings = out["embedding"]                         #[B*n, output_dim]
        embeddings = embeddings.view(batch_size, n_texts, -1) #[B, n, output_dim]

        #Attention pooling over text dimension
        attn_scores = self.attention_pooler(embeddings)       #[B, n, 1]

        if text_mask is not None:
            #Mask out padding texts before softmax
            attn_scores = attn_scores.squeeze(-1)             #[B, n]
            attn_scores = attn_scores.masked_fill(~text_mask.bool(), float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1) #[B, n]
            pooled = (embeddings * attn_weights.unsqueeze(-1)).sum(dim=1)  #[B, output_dim]
        else:
            attn_weights = torch.softmax(attn_scores, dim=1)  #[B, n, 1]
            pooled = (embeddings * attn_weights).sum(dim=1)   #[B, output_dim]

        return {
            "embedding": pooled,
            "text_attention_weights": attn_weights,
        }
