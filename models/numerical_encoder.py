import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
   

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        #Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  #[1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalSelfAttention(nn.Module):
   

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        #Self-attention with residual
        attn_out, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = self.norm(x + self.dropout(attn_out))

        #Feed-forward with residual
        x = self.ff_norm(x + self.dropout(self.ff(x)))

        return x, attn_weights


class NumericalEncoder(nn.Module):
    

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 256,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_attention_heads: int = 8,
        use_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.input_norm = nn.BatchNorm1d(input_dim)

        #Feature projection──
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        #Positional encoding─
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        #Bi-LSTM──
        #Use LSTM over GRU for better long-range memory (LSTM cell state
        #acts as an explicit "memory tape" unlike GRU's hidden state)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        #Transformer self-attention
        if use_attention:
            #Project LSTM output back to consistent dim for attention
            self.lstm_projection = nn.Linear(lstm_output_dim, hidden_dim)
            self.temporal_attention = TemporalSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            pool_dim = hidden_dim
        else:
            pool_dim = lstm_output_dim

        #Attention pooling over time─────
        #Learn which timesteps are most predictive
        #(e.g., recent days before earnings release)
        self.time_attention = nn.Sequential(
            nn.Linear(pool_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        #Output projection───
        self.output_projection = nn.Sequential(
            nn.Linear(pool_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LSTM weights with orthogonal initialization for stability."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                #Set forget gate bias to 1 — helps with learning long dependencies
                n = param.size(0)
                param.data[n //4: n //2].fill_(1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            x:        [batch, seq_len, input_dim] — raw market features
            lengths:  [batch] — actual sequence lengths (for masking)

        Returns:
            dict with:
              "embedding"       : [batch, output_dim]
              "temporal_weights": [batch, seq_len] — which timesteps mattered
              "lstm_output"     : [batch, seq_len, hidden_dim] — for explainability
        """
        batch_size, seq_len, _ = x.shape

        #Normalize input features──
        #BatchNorm expects [batch, features, seq] so we permute
        x = self.input_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        #Project to hidden dim─
        x = self.input_projection(x)  #[batch, seq_len, hidden_dim]

        #Add positional encoding───
        x = self.pos_encoding(x)

        #Bi-LSTM──
        if lengths is not None:
            #Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, _ = self.lstm(x)
        #lstm_out: [batch, seq_len, hidden_dim * 2 if bidirectional]

        #Transformer self-attention
        if self.use_attention:
            #Project to consistent dimension
            h = self.lstm_projection(lstm_out)  #[batch, seq_len, hidden_dim]

            #Create padding mask
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            else:
                mask = None

            h, attn_weights = self.temporal_attention(h, key_padding_mask=mask)
        else:
            h = lstm_out
            attn_weights = None

        #Attention-weighted pooling over time ─────
        time_scores = self.time_attention(h).squeeze(-1)  #[batch, seq_len]

        if lengths is not None:
            #Mask padding positions
            time_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            time_scores = time_scores.masked_fill(time_mask, float("-inf"))

        time_weights = torch.softmax(time_scores, dim=-1)  #[batch, seq_len]
        pooled = (h * time_weights.unsqueeze(-1)).sum(dim=1)  #[batch, hidden_dim]

        #Final projection────
        embedding = self.output_projection(pooled)  #[batch, output_dim]

        return {
            "embedding": embedding,
            "temporal_weights": time_weights,
            "lstm_output": lstm_out,
        }


class ConvTemporalEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        #Dilated causal convolutions with exponentially growing receptive field
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=2 ** i,
                    padding=2 ** i,  #Causal padding
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for i in range(4)  #4 layers → receptive field = 1+2+4+8 = 15 days
        ])

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_dim] → [batch, output_dim]"""
        x = x.permute(0, 2, 1)  #[batch, input_dim, seq_len]
        for conv in self.convs:
            x = conv(x)
        #Global average pooling over time
        x = x.mean(dim=-1)  #[batch, hidden_dim]
        return self.output_projection(x)
