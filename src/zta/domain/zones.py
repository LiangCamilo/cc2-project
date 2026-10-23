"""Calibrate disjoint tolerance bands from training-reference distances."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data.loader import CatalogContractError
from zta.domain.models import ListeningMode, ZoneBand


def validate_distance_series(distances: pd.Series, *, allow_empty: bool = False) -> pd.Series:
    """Validate metric, song identities and nonnegative finite distances."""
    if not isinstance(distances, pd.Series) or distances.name not in ("euclidean", "mahalanobis"):
        raise CatalogContractError("Provide a named Euclidean or Mahalanobis distance Series.")
    if isinstance(distances.index, pd.MultiIndex) or distances.index.has_duplicates or any(
        not isinstance(value, str) or not value or value != value.strip() for value in distances.index
    ):
        raise CatalogContractError("Distances require distinct, nonblank string track IDs.")
    dtype = distances.dtype
    if (
        not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
    ):
        raise CatalogContractError("Distances must contain real numeric values.")
    values = distances.to_numpy(dtype=float, na_value=np.nan)
    if (not allow_empty and not len(values)) or not np.isfinite(values).all() or (values < 0).any():
        raise CatalogContractError("Distances must be finite and nonnegative; reference cannot be empty.")
    return pd.Series(values, index=distances.index.copy(), name=distances.name)


@dataclass(frozen=True)
class ZoneThresholds:
    metric: str
    percentiles: tuple[float, float, float]
    cutoffs: tuple[float, float, float]
    reference_rows: int

    def __post_init__(self) -> None:
        if self.metric not in ("euclidean", "mahalanobis"):
            raise CatalogContractError("Unknown distance metric for tolerance zones.")
        for values in (self.percentiles, self.cutoffs):
            if not isinstance(values, tuple) or len(values) != 3 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values
            ):
                raise CatalogContractError("Zones require three finite percentiles and three finite cutoffs.")
        if not 0 <= self.percentiles[0] < self.percentiles[1] < self.percentiles[2] <= 100:
            raise CatalogContractError("Zone percentiles must increase strictly between 0 and 100.")
        if not 0 <= self.cutoffs[0] <= self.cutoffs[1] <= self.cutoffs[2]:
            raise CatalogContractError("Zone cutoffs must be nonnegative and nondecreasing.")
        if isinstance(self.reference_rows, bool) or not isinstance(self.reference_rows, int) or self.reference_rows < 1:
            raise CatalogContractError("Zone calibration requires a nonempty training reference.")

    def band(self, mode: ListeningMode | str) -> ZoneBand:
        try:
            mode = ListeningMode(mode)
        except ValueError as error:
            raise CatalogContractError(f"Unknown listening mode: {mode}.") from error
        if mode is ListeningMode.FAMILIAR:
            return ZoneBand(0.0, self.cutoffs[0], include_lower=True)
        if mode is ListeningMode.DISCOVERY:
            return ZoneBand(self.cutoffs[0], self.cutoffs[1])
        return ZoneBand(self.cutoffs[1], self.cutoffs[2])

    def to_dict(self) -> dict:
        return {
            "schema_version": 1, "metric": self.metric, "quantile_method": "linear",
            "percentiles": dict(zip((mode.value for mode in ListeningMode), self.percentiles)),
            "cutoffs": dict(zip((mode.value for mode in ListeningMode), self.cutoffs)),
            "reference_rows": self.reference_rows,
        }


def calibrate_zones(reference_distances: pd.Series, settings: Settings | None = None) -> ZoneThresholds:
    """Use the training reference, never candidate distances, to estimate cutoffs.

    In the complete pipeline reference rows come from the distance model's
    training matrix, excluding any seeds by ID. Tied quantiles remain tied.
    """
    s = settings or Settings()
    names = [mode.value for mode in ListeningMode]
    if not isinstance(s.zone_percentiles, dict) or set(s.zone_percentiles) != set(names):
        raise CatalogContractError("Configure one percentile for each listening mode.")
    percentiles = tuple(s.zone_percentiles[name] for name in names)
    # Validate configuration before asking NumPy to compute its percentiles.
    ZoneThresholds("euclidean", percentiles, (0.0, 0.0, 0.0), 1)
    reference = validate_distance_series(reference_distances)
    cutoffs = tuple(float(value) for value in np.percentile(reference.to_numpy(), percentiles, method="linear"))
    return ZoneThresholds(reference.name, percentiles, cutoffs, len(reference))
