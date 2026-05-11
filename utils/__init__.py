from .feature_engineering import FeatureEngineer
from .metrics import compute_classification_metrics, compute_regression_metrics, compute_directional_accuracy

#TextPreprocessor requires transformers — import lazily
def _import_text_preprocessor():
    from .text_preprocessing import TextPreprocessor
    return TextPreprocessor

__all__ = [
    "FeatureEngineer",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_directional_accuracy",
]
