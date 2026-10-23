"""Rank only the candidates inside one acoustic tolerance band."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data.loader import CatalogContractError
from zta.domain.models import ListeningMode
from zta.domain.zones import ZoneThresholds, validate_distance_series


@dataclass(frozen=True)
class RecommendationResult:
    table: pd.DataFrame
    thresholds: ZoneThresholds
    mode: ListeningMode
    requested_k: int
    available_in_zone: int
    candidate_rows: int
    seed_ids: tuple[str, ...]
    popularity_lambda: float

    def summary(self) -> dict:
        count = len(self.table)
        status = "complete" if count == self.requested_k else (
            "empty_zone" if not self.available_in_zone else "insufficient_candidates"
        )
        band = self.thresholds.band(self.mode)
        return {
            "metric": self.thresholds.metric, "mode": self.mode.value,
            "requested_k": self.requested_k, "returned_k": count,
            "candidate_rows": self.candidate_rows, "available_in_zone": self.available_in_zone,
            "status": status, "complete": count == self.requested_k,
            "popularity_lambda": self.popularity_lambda, "seed_ids": list(self.seed_ids),
            "lower_bound": band.lower, "upper_bound": band.upper,
            "lower_inclusive": band.include_lower,
        }


def _metadata(catalog: pd.DataFrame, ids: pd.Index, settings: Settings) -> pd.DataFrame:
    s = settings
    columns = [s.id_col, "track_name", "artists", s.genres_col, s.popularity_col]
    if not isinstance(catalog, pd.DataFrame) or catalog.columns.has_duplicates:
        raise CatalogContractError("Recommendation metadata must be a catalog with unique column names.")
    missing = [column for column in columns if column not in catalog.columns]
    if missing:
        raise CatalogContractError(f"Recommendation metadata is missing columns: {missing}.")
    all_ids = catalog[s.id_col]
    if all_ids.duplicated().any() or any(
        not isinstance(value, str) or not value or value != value.strip() for value in all_ids
    ):
        raise CatalogContractError("Recommendation metadata requires distinct, nonblank track IDs.")
    if not ids.isin(all_ids).all():
        raise CatalogContractError("Some candidate IDs have no matching catalog metadata.")
    selected = catalog.loc[:, columns].set_index(s.id_col).loc[ids].copy()
    for column in ("track_name", "artists"):
        if any(not isinstance(value, str) or not value.strip() for value in selected[column]):
            raise CatalogContractError(f"Recommendation metadata requires nonblank {column} values.")
    if any(
        not isinstance(tags, (tuple, list)) or not tags
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        for tags in selected[s.genres_col]
    ):
        raise CatalogContractError("Candidate genres must be nonempty collections of labels.")
    popularity = selected[s.popularity_col]
    if (
        not pd.api.types.is_numeric_dtype(popularity.dtype)
        or pd.api.types.is_bool_dtype(popularity.dtype) or pd.api.types.is_complex_dtype(popularity.dtype)
    ):
        raise CatalogContractError("Popularity must contain real numeric values.")
    values = popularity.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise CatalogContractError("Popularity must be finite and within [0, 100].")
    return selected


def rank_zone(
    candidate_distances: pd.Series,
    catalog: pd.DataFrame,
    thresholds: ZoneThresholds,
    mode: ListeningMode | str,
    k: int = 10,
    *,
    seed_ids: Sequence[str] = (),
    popularity_lambda: float | None = None,
    settings: Settings | None = None,
) -> RecommendationResult:
    """Filter by band, then sort by -distance - lambda * popularity / 100.

    Exact score ties are broken by track ID ascending. No outside-band fallback
    is used for empty/short lists. Input metadata and distances remain unchanged.
    """
    s = settings or Settings()
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise CatalogContractError("The requested recommendation count must be a positive integer.")
    weight = s.popularity_lambda if popularity_lambda is None else popularity_lambda
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise CatalogContractError("The popularity penalty must be finite and nonnegative.")
    if isinstance(seed_ids, (str, bytes)) or not isinstance(seed_ids, Sequence):
        raise CatalogContractError("Seed IDs must be a sequence of distinct strings.")
    seeds = tuple(seed_ids)
    if any(not isinstance(value, str) or not value or value != value.strip() for value in seeds) or len(set(seeds)) != len(seeds):
        raise CatalogContractError("Seed IDs must be distinct, nonblank strings.")
    band = thresholds.band(mode)
    mode = ListeningMode(mode)
    distances = validate_distance_series(candidate_distances, allow_empty=True)
    if distances.name != thresholds.metric or distances.index.name != s.id_col:
        raise CatalogContractError("Candidate metric or ID column does not match the calibrated zones.")
    distances = distances.loc[~distances.index.isin(seeds)]
    metadata = _metadata(catalog, distances.index, s)
    eligible = distances.loc[band.contains(distances.to_numpy())]
    table = metadata.loc[eligible.index].copy()
    table["distance"] = eligible
    with np.errstate(over="ignore", invalid="ignore"):
        table["score"] = -table["distance"] - weight * (table[s.popularity_col].astype(float) / 100.0)
    if not np.isfinite(table["score"].to_numpy()).all():
        raise CatalogContractError("The popularity penalty produced non-finite ranking scores.")
    available = len(table)
    table = table.reset_index().sort_values(["score", s.id_col], ascending=[False, True], kind="stable").head(k).copy()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["mode"] = mode.value
    table["metric"] = thresholds.metric
    table["lower_bound"] = band.lower
    table["upper_bound"] = band.upper
    table["lower_inclusive"] = band.include_lower
    return RecommendationResult(
        table=table.reset_index(drop=True), thresholds=thresholds, mode=mode,
        requested_k=k, available_in_zone=available, candidate_rows=len(distances),
        seed_ids=seeds, popularity_lambda=float(weight),
    )
