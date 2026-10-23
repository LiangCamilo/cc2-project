"""Represent seed songs by a centroid in a specific fitted acoustic space."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Sequence

import pandas as pd

from zta.data.loader import CatalogContractError
from zta.preprocessing import MinMaxNormalizer


def normalization_fingerprint(normalizer: MinMaxNormalizer) -> str:
    encoded = json.dumps(normalizer.to_dict(), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AcousticProfile:
    """Equal-weight centroid, seed identities and the fitted space it belongs to."""

    features: tuple[str, ...]
    centroid: tuple[float, ...]
    seed_ids: tuple[str, ...]
    normalization_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.features, tuple) or not self.features
            or any(not isinstance(name, str) or not name.strip() for name in self.features)
            or len(set(self.features)) != len(self.features)
            or not isinstance(self.centroid, tuple) or len(self.centroid) != len(self.features)
        ):
            raise CatalogContractError("A profile requires unique features and one centroid value per feature.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1 for value in self.centroid
        ):
            raise CatalogContractError("Profile centroid values must be finite numbers in [0, 1].")
        if (
            not isinstance(self.seed_ids, tuple) or not self.seed_ids
            or any(not isinstance(value, str) or not value or value != value.strip() for value in self.seed_ids)
            or len(set(self.seed_ids)) != len(self.seed_ids)
        ):
            raise CatalogContractError("A profile requires distinct, nonblank seed IDs.")
        if (
            not isinstance(self.normalization_sha256, str) or len(self.normalization_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.normalization_sha256)
        ):
            raise CatalogContractError("A profile must identify its fitted normalization parameters.")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "features": list(self.features),
            "centroid": list(self.centroid),
            "seed_ids": list(self.seed_ids),
            "normalization_sha256": self.normalization_sha256,
        }

    @classmethod
    def from_dict(cls, state: dict) -> AcousticProfile:
        try:
            if not isinstance(state, dict) or state.get("schema_version") != 1:
                raise CatalogContractError("Unsupported acoustic profile format.")
            if any(not isinstance(state[key], list) for key in ("features", "centroid", "seed_ids")):
                raise CatalogContractError("Saved profile features, centroid and seeds must be lists.")
            return cls(
                tuple(state["features"]), tuple(state["centroid"]), tuple(state["seed_ids"]),
                state["normalization_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CatalogContractError):
                raise
            raise CatalogContractError(f"Invalid saved acoustic profile: {error}") from error


def build_acoustic_profile(
    matrix: pd.DataFrame, seed_ids: Sequence[str], normalizer: MinMaxNormalizer
) -> AcousticProfile:
    """Transform only the requested seeds with fixed parameters and average them."""
    if isinstance(seed_ids, (str, bytes)) or not isinstance(seed_ids, Sequence):
        raise CatalogContractError("Provide seed IDs as a sequence of distinct strings.")
    seeds = tuple(seed_ids)
    if (
        not seeds or any(not isinstance(value, str) or not value or value != value.strip() for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise CatalogContractError("Choose at least one distinct, nonblank seed ID.")
    if (
        not isinstance(matrix, pd.DataFrame) or isinstance(matrix.index, pd.MultiIndex)
        or matrix.index.name != normalizer.id_column or matrix.index.has_duplicates
    ):
        raise CatalogContractError("Seed features must have a unique track ID index.")
    missing = [value for value in seeds if value not in matrix.index]
    if missing:
        raise CatalogContractError(f"Unknown seed IDs: {missing}.")
    normalized = normalizer.transform(matrix.loc[list(seeds)]).matrix
    return AcousticProfile(
        features=normalizer.features,
        centroid=tuple(float(value) for value in normalized.mean()),
        seed_ids=seeds,
        normalization_sha256=normalization_fingerprint(normalizer),
    )
