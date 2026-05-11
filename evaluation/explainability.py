import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:

    def __init__(self, tokenizer, output_dir: str = "evaluation/plots"):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def extract_attention_weights(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
     
        model.eval()

        with torch.no_grad():
            if hasattr(model, "text_encoder"):
                encoder = model.text_encoder
                if hasattr(encoder, "encoder"):
                    encoder = encoder.encoder
            else:
                encoder = model

            out = encoder.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        #Take last layer attention: [batch, heads, seq, seq]
        last_attention = out.attentions[-1][0].cpu().numpy()  #[heads, seq, seq]
        return last_attention

    def visualize_token_attention(
        self,
        text: str,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prediction: str = "",
        save_path: Optional[str] = None,
    ) -> None:
    
        try:
            attention = self.extract_attention_weights(model, input_ids, attention_mask)
        except Exception as e:
            logger.warning(f"Could not extract attention: {e}")
            return

        #Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist()
        )

        #Average over attention heads, then take CLS token's attention to others
        #CLS token aggregates the full sequence for classification
        avg_attention = attention.mean(axis=0)       #[seq, seq]
        cls_attention = avg_attention[0, :]           #attention FROM [CLS] TO all tokens

        #Filter out special tokens and padding
        valid_tokens = []
        valid_attention = []
        for token, attn in zip(tokens, cls_attention):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            valid_tokens.append(token.replace("##", ""))
            valid_attention.append(attn)

        valid_attention = np.array(valid_attention)
        valid_attention = valid_attention / (valid_attention.max() + 1e-8)

        #Plot
        fig_width = max(12, len(valid_tokens) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, 3))

        colors = plt.cm.RdYlGn(valid_attention)  #Red=low, Green=high attention
        bars = ax.bar(range(len(valid_tokens)), valid_attention, color=colors, edgecolor="gray", linewidth=0.5)

        ax.set_xticks(range(len(valid_tokens)))
        ax.set_xticklabels(valid_tokens, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Attention Weight (normalized)", fontsize=10)
        ax.set_title(
            f"Token Attention — Prediction: {prediction}\n"
            f"Text: {text[:80]}{'...' if len(text) > 80 else ''}",
            fontsize=11, pad=10,
        )
        ax.set_ylim(0, 1.1)

        #Add value labels on bars
        for bar, val in zip(bars, valid_attention):
            if val > 0.3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/attention_viz.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Attention visualization saved: {path}")

    def highlight_important_phrases(
        self,
        text: str,
        attention_weights: np.ndarray,
        tokens: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Identify the top-K most attended phrases in the text.

        Returns list of (phrase, attention_score) tuples.
        """
        #Aggregate subword tokens into words
        words = []
        word_scores = []
        current_word = ""
        current_scores = []

        for token, score in zip(tokens, attention_weights):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if token.startswith("##"):
                current_word += token[2:]
                current_scores.append(score)
            else:
                if current_word:
                    words.append(current_word)
                    word_scores.append(np.mean(current_scores))
                current_word = token
                current_scores = [score]

        if current_word:
            words.append(current_word)
            word_scores.append(np.mean(current_scores))

        #Sort by attention score
        ranked = sorted(zip(words, word_scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class SHAPExplainer:
    """
    SHAP-based feature importance for the numerical (market data) inputs.

    Uses DeepExplainer for neural networks when available,
    falls back to KernelExplainer (model-agnostic) otherwise.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: str = "cpu",
        output_dir: str = "evaluation/plots",
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = torch.device(device)
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def compute_shap_values(
        self,
        background_numerical: torch.Tensor,   #[n_background, seq_len, n_features]
        test_numerical: torch.Tensor,          #[n_test, seq_len, n_features]
        background_text: Dict,
        test_text: Dict,
        n_samples: int = 50,
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for numerical features using DeepSHAP.

        Returns shap values of shape [n_test, seq_len, n_features]
        """
        try:
            import shap

            #Wrapper that takes only numerical features as input
            #(holds text inputs constant at background values)
            background_text_device = {
                k: v.to(self.device) for k, v in background_text.items()
            }

            class NumericOnlyModel(nn.Module):
                def __init__(self, full_model, fixed_text):
                    super().__init__()
                    self.model = full_model
                    self.fixed_text = fixed_text

                def forward(self, x):
                    with torch.no_grad():
                        out = self.model(
                            input_ids=self.fixed_text["input_ids"][:x.shape[0]],
                            attention_mask=self.fixed_text["attention_mask"][:x.shape[0]],
                            text_mask=self.fixed_text.get("text_mask"),
                            numerical_features=x,
                        )
                    return out["direction_logits"]

            numeric_model = NumericOnlyModel(self.model, background_text_device)
            numeric_model.eval()

            #Use mean over time as SHAP input (aggregate temporal dimension)
            background_agg = background_numerical.mean(dim=1).to(self.device)
            test_agg = test_numerical.mean(dim=1).to(self.device)

            #GradientExplainer works with PyTorch models
            explainer = shap.GradientExplainer(
                numeric_model, background_agg[:n_samples]
            )
            shap_values = explainer.shap_values(test_agg)

            return np.array(shap_values)

        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            return None
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return None

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
       
        if shap_values is None:
            logger.warning("No SHAP values to plot")
            return

        #For binary classification, use class 1 (UP) SHAP values
        if len(shap_values.shape) == 3:
            vals = shap_values[1]   #Class 1 = UP
        else:
            vals = shap_values

        mean_shap = np.abs(vals).mean(axis=0)  #[n_features]

        #Align with feature names
        n_features = min(len(mean_shap), len(self.feature_names))
        names = self.feature_names[:n_features]
        values = mean_shap[:n_features]

        #Sort by importance
        sorted_idx = np.argsort(values)[::-1]
        names = [names[i] for i in sorted_idx]
        values = values[sorted_idx]

        #Plot
        fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.4)))

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_features))
        bars = ax.barh(range(n_features), values, color=colors)

        ax.set_yticks(range(n_features))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Mean |SHAP Value| — Feature Importance for UP prediction", fontsize=11)
        ax.set_title("Numerical Feature Importance (SHAP)", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        #Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=8,
            )

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/shap_feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP importance plot saved: {path}")

    def plot_temporal_importance(
        self,
        temporal_weights: np.ndarray,  #[n_samples, seq_len] from model
        sequence_length: int = 30,
        save_path: Optional[str] = None,
    ) -> None:
        
        mean_weights = temporal_weights.mean(axis=0)  #[seq_len]
        days_ago = list(range(-sequence_length + 1, 1))[::-1]

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(days_ago, mean_weights, alpha=0.3, color="royalblue")
        ax.plot(days_ago, mean_weights, "o-", color="royalblue", linewidth=2, markersize=4)

        ax.set_xlabel("Days Before Prediction Date", fontsize=11)
        ax.set_ylabel("Mean Temporal Attention Weight", fontsize=11)
        ax.set_title(
            "Temporal Attention: Which Historical Days Matter Most for Prediction",
            fontsize=13, fontweight="bold",
        )

        #Mark key time points
        for day, label in [(-1, "Yesterday"), (-5, "1 Week"), (-22, "1 Month")]:
            if day in days_ago:
                idx = days_ago.index(day)
                ax.axvline(x=day, color="red", linestyle="--", alpha=0.5)
                ax.text(day, mean_weights.max() * 0.95, label,
                       ha="center", color="red", fontsize=9)

        ax.set_xlim(days_ago[0], days_ago[-1])
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = save_path or f"{self.output_dir}/temporal_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Temporal importance plot saved: {path}")
