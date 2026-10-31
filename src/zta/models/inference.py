"""Load a saved model and score new songs (this is what the API will call)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .artifacts import validate_artifact


def load_artifact(path: str | Path) -> dict:
    artifact = joblib.load(path)
    validate_artifact(artifact)
    return artifact


def predict_scores(artifact: dict, raw_features: pd.DataFrame) -> pd.DataFrame:
    """`raw_features`: unscaled values of the nine acoustic columns, one row per song.

    Returns genre scores (one column per genre) or a single `popularity` column.
    """
    normalizer = validate_artifact(artifact)
    raw = raw_features[list(artifact["features"])].copy()
    raw.index = pd.Index([str(i) for i in raw.index], name=artifact["id_column"])
    x = normalizer.transform(raw).matrix
    scores = artifact["model"].score(x.to_numpy())
    columns = artifact["labels"] if artifact["task"] == "genre" else ["popularity"]
    return pd.DataFrame(scores.reshape(len(x), -1), index=x.index, columns=columns)


def top_genres(scores: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return pd.DataFrame({idx: row.nlargest(k).index.tolist() for idx, row in scores.iterrows()}).T
