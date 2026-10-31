"""Euclidean and regularized Mahalanobis distances in a fitted acoustic space."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from zta.data.loader import CatalogContractError
from zta.domain.profiles import AcousticProfile, normalization_fingerprint
from zta.preprocessing import MinMaxNormalizer


@dataclass(frozen=True)
class AcousticDistanceModel:
    """Sample covariance from training only; no fitting occurs during a query."""

    normalizer: MinMaxNormalizer
    covariance: tuple[tuple[float, ...], ...]
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.normalizer, MinMaxNormalizer) or self.normalizer.training_rows < 2:
            raise CatalogContractError("Distance fitting requires at least two training songs.")
        if (
            isinstance(self.regularization, bool) or not isinstance(self.regularization, (int, float))
            or not math.isfinite(self.regularization) or self.regularization <= 0
        ):
            raise CatalogContractError("Mahalanobis regularization must be finite and strictly positive.")
        dimension = len(self.normalizer.features)
        if (
            not isinstance(self.covariance, tuple) or len(self.covariance) != dimension
            or any(not isinstance(row, tuple) or len(row) != dimension for row in self.covariance)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for row in self.covariance for value in row
            )
        ):
            raise CatalogContractError("Covariance must be a finite square matrix matching the features.")
        covariance = np.array(self.covariance, dtype=float)
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
            raise CatalogContractError("Covariance must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise CatalogContractError("Covariance must be positive semidefinite.")
        self._cholesky()

    def _cholesky(self) -> np.ndarray:
        regularized = np.array(self.covariance) + self.regularization * np.eye(len(self.normalizer.features))
        try:
            return np.linalg.cholesky(regularized)
        except np.linalg.LinAlgError as error:
            raise CatalogContractError("Regularized covariance is not numerically positive definite.") from error

    @classmethod
    def fit(
        cls, training: pd.DataFrame, normalizer: MinMaxNormalizer, regularization: float = 1e-6
    ) -> AcousticDistanceModel:
        """Verify the normalizer's training source and estimate sample covariance."""
        normalizer.validate_training(training)
        if len(training) < 2:
            raise CatalogContractError("Distance fitting requires at least two training songs.")
        normalized = normalizer.transform(training).matrix
        covariance = np.atleast_2d(np.cov(normalized.to_numpy(), rowvar=False, ddof=1))
        return cls(
            normalizer=normalizer,
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            regularization=regularization,
        )

    def distances(self, profile: AcousticProfile, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return both distances by ID, excluding seeds and preserving input order.

        A different song with identical acoustic values remains eligible at
        distance zero. Empty candidate sets yield an empty table with its schema.
        Mahalanobis is computed with a Cholesky solve, without an explicit inverse.
        """
        if (
            not isinstance(profile, AcousticProfile) or profile.features != self.normalizer.features
            or profile.normalization_sha256 != normalization_fingerprint(self.normalizer)
        ):
            raise CatalogContractError("Profile and distance model use different fitted acoustic spaces.")
        # Transform validates the complete input, including rows excluded as seeds.
        normalized = self.normalizer.transform(candidates).matrix
        eligible = normalized.loc[~normalized.index.isin(profile.seed_ids)]
        delta = eligible.to_numpy() - np.array(profile.centroid)
        euclidean = np.linalg.norm(delta, axis=1)
        whitened = np.linalg.solve(self._cholesky(), delta.T)
        mahalanobis = np.linalg.norm(whitened, axis=0)
        if not np.isfinite(euclidean).all() or not np.isfinite(mahalanobis).all():
            raise CatalogContractError("Distance calculation produced non-finite values.")
        return pd.DataFrame(
            {"euclidean": euclidean, "mahalanobis": mahalanobis}, index=eligible.index.copy()
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "covariance_ddof": 1,
            "normalizer": self.normalizer.to_dict(),
            "covariance": [list(row) for row in self.covariance],
            "regularization": self.regularization,
        }

    @classmethod
    def from_dict(cls, state: dict) -> AcousticDistanceModel:
        """Restore fitted parameters without observing seed or candidate values."""
        try:
            if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("covariance_ddof") != 1:
                raise CatalogContractError("Unsupported acoustic distance model format.")
            if not isinstance(state["covariance"], list) or any(not isinstance(row, list) for row in state["covariance"]):
                raise CatalogContractError("Saved covariance must be a list of rows.")
            return cls(
                normalizer=MinMaxNormalizer.from_dict(state["normalizer"]),
                covariance=tuple(tuple(row) for row in state["covariance"]),
                regularization=state["regularization"],
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CatalogContractError):
                raise
            raise CatalogContractError(f"Invalid saved acoustic distance model: {error}") from error
