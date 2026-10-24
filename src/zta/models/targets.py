"""Supervised targets built only from columns that already exist in the cleaned catalog.

No labels are invented here:
  * genre      -> multi-hot matrix (songs x genres) from track_genres.csv
  * popularity -> the original 0-100 `popularity` column
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from zta.config import Settings


def _track_index(catalog: pd.DataFrame, s: Settings) -> pd.Index:
    return pd.Index(catalog[s.id_col], name=s.id_col)


def genre_target(catalog: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Multi-hot genre matrix indexed by track_id (one column per genre, 0/1)."""
    s = settings or Settings()
    binarizer = MultiLabelBinarizer()
    values = binarizer.fit_transform(catalog[s.genres_col])
    return pd.DataFrame(values.astype("uint8"), index=_track_index(catalog, s), columns=binarizer.classes_)


def popularity_target(catalog: pd.DataFrame, settings: Settings | None = None) -> pd.Series:
    """Popularity (0-100) indexed by track_id."""
    s = settings or Settings()
    return pd.Series(catalog[s.popularity_col].to_numpy(dtype=float), index=_track_index(catalog, s), name=s.popularity_col)


def build_target(task: str, catalog: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame | pd.Series:
    if task == "genre":
        return genre_target(catalog, settings)
    if task == "popularity":
        return popularity_target(catalog, settings)
    raise ValueError(f"Unknown task: {task}")
