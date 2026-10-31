"""Four supervised models (linear regression, KNN, random forest, SVM) over the nine acoustic features."""

from .registry import CONFIGS, MODEL_NAMES, TASKS, build_model
from .targets import genre_target, popularity_target

__all__ = ["CONFIGS", "MODEL_NAMES", "TASKS", "build_model", "genre_target", "popularity_target"]
