"""The four models, the baseline, and the small hyperparameter grids.

Every model exposes fit(X, y) and score(X) -> ndarray.
  genre      : score is an (n x n_genres) matrix; higher = more likely that genre.
  popularity : score is an (n,) vector with the predicted popularity.

KNN and Random Forest use their *regressor* versions on the 0/1 genre matrix:
the average of neighbours/leaves is exactly the class-probability estimate, it
handles 114 genres at once and avoids the list-of-arrays output of predict_proba.
SVM for genres is a one-vs-rest LinearSVC (an RBF SVC on 61k rows x 114 genres
is not feasible); for popularity it is an RBF SVR trained on a fixed subsample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVC

TASKS = ("genre", "popularity")
MODEL_NAMES = ("linear", "knn", "rf", "svm")

CONFIGS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "genre": {
        "baseline": [{}],
        "linear": [{}],
        "knn": [{"n_neighbors": 15}, {"n_neighbors": 50}, {"n_neighbors": 100}],
        "rf": [
            {"n_estimators": 100, "min_samples_leaf": 5, "max_samples": 0.5},
            {"n_estimators": 100, "min_samples_leaf": 20, "max_samples": 0.5},
        ],
        "svm": [{"C": 0.1}, {"C": 1.0}],
    },
    "popularity": {
        "baseline": [{}],
        "linear": [{}],
        "knn": [{"n_neighbors": 15}, {"n_neighbors": 50}, {"n_neighbors": 100}],
        "rf": [
            {"n_estimators": 200, "min_samples_leaf": 5},
            {"n_estimators": 200, "min_samples_leaf": 20},
        ],
        "svm": [{"C": 1.0, "epsilon": 1.0, "max_rows": 15000}, {"C": 10.0, "epsilon": 1.0, "max_rows": 15000}],
    },
}


class SubsampledSVR(BaseEstimator, RegressorMixin):
    """RBF SVR fitted on a fixed random subsample (SVR cost grows ~quadratically with rows)."""

    def __init__(self, C: float = 1.0, epsilon: float = 1.0, max_rows: int = 15000, random_state: int = 42):
        self.C, self.epsilon, self.max_rows, self.random_state = C, epsilon, max_rows, random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        rows = rng.choice(len(X), size=min(self.max_rows, len(X)), replace=False)
        self.model_ = SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma="scale").fit(X[rows], y[rows])
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


@dataclass
class Model:
    name: str
    task: str
    params: dict
    estimator: Any
    scorer: str  # "predict" or "decision_function"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        y = np.asarray(y)
        y = y.astype(int) if self.scorer == "decision_function" else y.astype("float32" if y.ndim == 2 else float)
        self.estimator.fit(np.asarray(X, dtype=float), y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return getattr(self.estimator, self.scorer)(np.asarray(X, dtype=float))


def build_model(name: str, task: str, params: dict | None = None, random_state: int = 42) -> Model:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    p = dict(params or {})
    scorer = "predict"
    if name == "baseline":
        est = DummyRegressor(strategy="mean")
    elif name == "linear":
        est = LinearRegression()
    elif name == "knn":
        est = KNeighborsRegressor(weights="distance", n_jobs=-1, **p)
    elif name == "rf":
        est = RandomForestRegressor(n_jobs=-1, random_state=random_state, **p)
    elif name == "svm" and task == "genre":
        est = OneVsRestClassifier(LinearSVC(C=p.get("C", 1.0), dual=False), n_jobs=-1)
        scorer = "decision_function"
    elif name == "svm":
        est = SubsampledSVR(random_state=random_state, **p)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Model(name, task, p, est, scorer)
