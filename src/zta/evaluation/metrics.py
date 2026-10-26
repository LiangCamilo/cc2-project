"""Project-defined list metrics; relevance is used only after recommendation."""

from __future__ import annotations

from collections.abc import Collection
import math

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data import CatalogContractError

from .profiles import positive_integer

METRICS = (
    "precision_at_k", "recall_at_k", "hit_at_k", "ild", "inverse_popularity",
    "genre_coverage", "completeness", "complete", "empty_zone",
)


def evaluate_list(
    table: pd.DataFrame,
    relevant_ids: Collection[str],
    k: int,
    normalized_features: pd.DataFrame,
    candidate_genres: Collection[str],
    settings: Settings | None = None,
) -> dict[str, float | int]:
    """Keep K as the precision denominator, even for incomplete lists.

    ILD always uses normalized Euclidean distance / sqrt(feature count),
    regardless of the distance used to generate a list. Undefined quantities
    remain NaN and are accompanied by aggregation counts in the reports.
    """
    s = settings or Settings()
    positive_integer(k, "k")
    if (
        not isinstance(table, pd.DataFrame) or table.columns.has_duplicates
        or not {s.id_col, s.genres_col, s.popularity_col}.issubset(table.columns)
        or len(table) > k or table[s.id_col].duplicated().any()
    ):
        raise CatalogContractError("Metrics require up to K distinct recommendations with metadata.")
    if (
        not isinstance(normalized_features, pd.DataFrame)
        or normalized_features.index.has_duplicates
        or normalized_features.index.name != s.id_col
        or list(normalized_features.columns) != list(s.features)
        # Only up to K IDs need membership checks. Passing a large candidate
        # index to Arrow Series.isin rebuilds its lookup on every short list.
        or any(track_id not in normalized_features.index for track_id in table[s.id_col])
    ):
        raise CatalogContractError("Recommended songs must belong to the fitted normalized feature matrix.")
    if isinstance(relevant_ids, (str, bytes)) or isinstance(candidate_genres, (str, bytes)):
        raise CatalogContractError("Provide relevance and genre collections, not strings.")
    relevant, universe = set(relevant_ids), set(candidate_genres)
    if not relevant.issubset(normalized_features.index):
        raise CatalogContractError("Relevant IDs must belong to the evaluation feature matrix.")
    ids = table[s.id_col].tolist()
    values = normalized_features.loc[ids].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise CatalogContractError("ILD requires finite normalized acoustic features in [0, 1].")
    labels = table[s.genres_col]
    if any(not isinstance(tags, (tuple, list)) or not tags for tags in labels):
        raise CatalogContractError("Recommended genres must be nonempty label collections.")
    covered = {genre for tags in labels for genre in tags}
    if not covered.issubset(universe):
        raise CatalogContractError("Recommended genres must belong to the candidate genre universe.")
    popularity = table[s.popularity_col].to_numpy(dtype=float)
    if not np.isfinite(popularity).all() or ((popularity < 0) | (popularity > 100)).any():
        raise CatalogContractError("Popularity must be finite and within [0, 100].")
    hits, count = len(set(ids) & relevant), len(ids)
    if count >= 2:
        left, right = np.triu_indices(count, k=1)
        ild = float(np.linalg.norm(values[left] - values[right], axis=1).mean() / math.sqrt(len(s.features)))
    else:
        ild = float("nan")
    return {
        "hits": hits, "relevant_count": len(relevant), "returned_k": count,
        "relevance_evaluable": int(bool(relevant)),
        "precision_at_k": hits / k if relevant else float("nan"),
        "recall_at_k": hits / len(relevant) if relevant else float("nan"),
        "hit_at_k": float(hits > 0) if relevant else float("nan"),
        "ild": ild,
        "inverse_popularity": float((-np.log2((popularity + 1) / 101)).mean()) if count else float("nan"),
        "genre_coverage": len(covered) / len(universe) if universe else float("nan"),
        "completeness": count / k, "complete": int(count == k),
    }
