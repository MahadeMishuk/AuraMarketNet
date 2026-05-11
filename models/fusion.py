import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 256,
        numerical_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden_layers: list = None,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256]

        assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"

        #Project both modalities to same dimension for attention compatibility
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.num_proj = nn.Linear(numerical_dim, fusion_dim)

        #Cross-attention: numerical queries text ──
        #Question: "Given this market state, which sentiment features matter?"
        self.num_to_text_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        #Cross-attention: text queries numerical ──
        #Question: "Given this sentiment, which market features confirm/deny it?"
        self.text_to_num_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        #Layer norms after each attention ──
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        #Learned gating: decide modality importance per sample ──
        #Different samples may be more text-driven (surprise earnings news)
        #vs. numerics-driven (technical breakout with no news).
        #The gate is a function of BOTH modalities.
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )

        #Final fusion MLP────
        #Combines cross-attended features into prediction-ready embedding
        layers = []
        in_dim = fusion_dim * 2  #Concatenate both attention outputs
        for out_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.fusion_mlp = nn.Sequential(*layers)
        self.fusion_output_dim = hidden_layers[-1] if hidden_layers else fusion_dim * 2

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,    #[batch, text_dim]
        num_emb: torch.Tensor,     #[batch, numerical_dim]
    ) -> dict:
        """
        Args:
            text_emb: Text embedding from FinBERT encoder
            num_emb:  Numerical embedding from LSTM encoder

        Returns:
            dict with:
              "fused_embedding"     : [batch, fusion_output_dim]
              "gate_weights"        : [batch, fusion_dim] — modality importance
              "cross_attn_num2text" : [batch, 1, 1] attention weights
              "cross_attn_text2num" : [batch, 1, 1] attention weights
        """
        #Project to common fusion dimension
        t = self.text_proj(text_emb).unsqueeze(1)   #[batch, 1, fusion_dim]
        n = self.num_proj(num_emb).unsqueeze(1)     #[batch, 1, fusion_dim]

        #Cross-attention: numerical queries text ──
        n2t, attn_n2t = self.num_to_text_attention(
            query=n,  #numerical "asks" about text
            key=t,
            value=t,
            need_weights=True,
        )
        n2t = self.norm1(n + self.dropout(n2t)).squeeze(1)  #[batch, fusion_dim]

        #Cross-attention: text queries numerical ──
        t2n, attn_t2n = self.text_to_num_attention(
            query=t,  #text "asks" about numerical
            key=n,
            value=n,
            need_weights=True,
        )
        t2n = self.norm2(t + self.dropout(t2n)).squeeze(1)  #[batch, fusion_dim]

        #Concatenate both cross-attention outputs ─
        combined = torch.cat([n2t, t2n], dim=-1)  #[batch, fusion_dim * 2]

        #Learnable gating────
        #The gate decides how much to weight each modality direction
        gate = self.gate(combined)  #[batch, fusion_dim]

        #Final MLP
        fused = self.fusion_mlp(combined)  #[batch, fusion_output_dim]

        return {
            "fused_embedding": fused,
            "gate_weights": gate,
            "cross_attn_num2text": attn_n2t,
            "cross_attn_text2num": attn_t2n,
        }


class ConcatFusion(nn.Module):
    def __init__(self, text_dim: int, numerical_dim: int, hidden_layers: list = None, dropout: float = 0.1):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256]

        layers = []
        in_dim = text_dim + numerical_dim
        for out_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.fusion_output_dim = hidden_layers[-1] if hidden_layers else text_dim + numerical_dim

    def forward(self, text_emb: torch.Tensor, num_emb: torch.Tensor) -> dict:
        combined = torch.cat([text_emb, num_emb], dim=-1)
        return {"fused_embedding": self.mlp(combined)}


class BilinearFusion(nn.Module):
    def __init__(self, text_dim: int, numerical_dim: int, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        #Bilinear interaction
        self.bilinear = nn.Bilinear(text_dim, numerical_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_output_dim = output_dim

        #Residual from concat
        self.residual = nn.Linear(text_dim + numerical_dim, output_dim)

    def forward(self, text_emb: torch.Tensor, num_emb: torch.Tensor) -> dict:
        bilinear_out = self.dropout(F.gelu(self.bilinear(text_emb, num_emb)))
        residual_out = self.residual(torch.cat([text_emb, num_emb], dim=-1))
        fused = self.norm(bilinear_out + residual_out)
        return {"fused_embedding": fused}
