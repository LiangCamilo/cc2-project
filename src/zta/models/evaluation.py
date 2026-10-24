"""Metrics. Genre: ranking quality of the score matrix. Popularity: regression errors."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, label_ranking_average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# metric used to choose each model's configuration, and whether higher is better
PRIMARY = {"genre": ("lrap", True), "popularity": ("rmse", False)}


def genre_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    rows = np.arange(len(y_true))[:, None]
    order = np.argsort(-scores, axis=1)
    has_label = y_true.sum(axis=1) > 0
    top1 = y_true[rows[:, 0], order[:, 0]]
    top5 = y_true[rows, order[:, :5]].max(axis=1)
    positives = y_true.sum(axis=0) > 0
    return {
        "lrap": float(label_ranking_average_precision_score(y_true[has_label], scores[has_label])),
        "macro_ap": float(average_precision_score(y_true[:, positives], scores[:, positives], average="macro")),
        "top1_hit": float(top1[has_label].mean()),
        "top5_hit": float(top5[has_label].mean()),
    }


def popularity_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def evaluate(task: str, y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return genre_metrics(y_true, scores) if task == "genre" else popularity_metrics(y_true, scores)
