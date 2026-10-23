"""Fit immutable Min-Max parameters and report held-out range violations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data.loader import CatalogContractError
from zta.data.partitions import CatalogPartitions


def _validated_matrix(
    matrix: pd.DataFrame, features: tuple[str, ...], id_column: str, *, allow_empty: bool = False
) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise CatalogContractError("Normalization requires a feature DataFrame indexed by track ID.")
    if isinstance(matrix.index, pd.MultiIndex) or matrix.index.name != id_column:
        raise CatalogContractError(f"Feature rows must be indexed by {id_column}.")
    if matrix.columns.has_duplicates or list(matrix.columns) != list(features):
        raise CatalogContractError("Normalization requires the configured features in their fixed order.")
    if not allow_empty and matrix.empty:
        raise CatalogContractError("Cannot fit normalization on an empty training matrix.")
    ids = matrix.index
    if ids.has_duplicates or any(
        not isinstance(value, str) or not value or value != value.strip() for value in ids
    ):
        raise CatalogContractError("Normalization requires unique, nonblank string track IDs.")
    for column in features:
        dtype = matrix[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise CatalogContractError(f"Normalization feature {column} must contain real numeric values.")
    values = matrix.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise CatalogContractError("Normalization features contain missing or non-finite values.")
    return pd.DataFrame(values, index=matrix.index.copy(), columns=list(features))


def _matrix_hash(matrix: pd.DataFrame) -> str:
    canonical = matrix.sort_index().to_csv(lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NormalizedFeatures:
    """Transformed rows and range counts; source values are left untouched.

    The report counts original values strictly below/above the fitted range.
    Constant features are reported too, but always map to zero, including when
    an incoming value differs from their training constant.
    """

    matrix: pd.DataFrame
    range_report: pd.DataFrame
    rows_outside_range: int

    @property
    def values_outside_range(self) -> int:
        return int(self.range_report["outside_range"].sum())


@dataclass(frozen=True)
class MinMaxNormalizer:
    """Fitted parameters; transform never learns from its input.

    Create with fit(training). For the final evaluation, fit a new instance on
    development after model selection and transform test with that instance.
    This is an explicit project rule for constant columns, which map to zero.
    """

    features: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    id_column: str
    training_rows: int
    training_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.features, tuple) or not self.features
            or any(not isinstance(name, str) or not name.strip() for name in self.features)
            or len(set(self.features)) != len(self.features)
            or not isinstance(self.id_column, str) or not self.id_column.strip()
            or self.id_column in self.features
        ):
            raise CatalogContractError("Invalid feature names or ID column in normalization parameters.")
        if not isinstance(self.minimum, tuple) or not isinstance(self.maximum, tuple):
            raise CatalogContractError("Fitted bounds must be immutable tuples.")
        if len(self.minimum) != len(self.features) or len(self.maximum) != len(self.features):
            raise CatalogContractError("Each normalization feature must have two fitted bounds.")
        for lower, upper in zip(self.minimum, self.maximum):
            if (
                isinstance(lower, bool) or isinstance(upper, bool)
                or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float))
                or not math.isfinite(lower) or not math.isfinite(upper)
                or lower > upper or not math.isfinite(upper - lower)
            ):
                raise CatalogContractError("Normalization bounds must define finite, nonnegative ranges.")
        if isinstance(self.training_rows, bool) or not isinstance(self.training_rows, int) or self.training_rows < 1:
            raise CatalogContractError("Normalization parameters require a positive training row count.")
        if (
            not isinstance(self.training_sha256, str) or len(self.training_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.training_sha256)
        ):
            raise CatalogContractError("Normalization parameters require the training matrix SHA-256.")

    @classmethod
    def fit(cls, training: pd.DataFrame, settings: Settings | None = None) -> MinMaxNormalizer:
        """Estimate bounds only from the supplied training rows; do not mutate them."""
        s = settings or Settings()
        features = tuple(s.features)
        if not features or len(set(features)) != len(features) or s.id_col in features:
            raise CatalogContractError("Normalization requires unique configured features excluding the ID.")
        matrix = _validated_matrix(training, features, s.id_col)
        return cls(
            features=features,
            minimum=tuple(float(value) for value in matrix.min()),
            maximum=tuple(float(value) for value in matrix.max()),
            id_column=s.id_col,
            training_rows=len(matrix),
            training_sha256=_matrix_hash(matrix),
        )

    def transform(self, matrix: pd.DataFrame) -> NormalizedFeatures:
        """Reuse fitted bounds, clip to [0, 1] and preserve row IDs/order."""
        source = _validated_matrix(matrix, self.features, self.id_column, allow_empty=True)
        values = source.to_numpy()
        lower, upper = np.array(self.minimum), np.array(self.maximum)
        below, above = values < lower, values > upper
        constant = upper == lower
        # Clip before subtracting to avoid overflow for extreme finite inputs.
        bounded = np.clip(values, lower, upper)
        scaled = (bounded - lower) / np.where(constant, 1.0, upper - lower)
        scaled[:, constant] = 0.0
        scaled = np.clip(scaled, 0.0, 1.0)
        report = pd.DataFrame({
            "feature": self.features,
            "rows": len(source),
            "training_min": lower,
            "training_max": upper,
            "constant_feature": constant,
            "below_min": below.sum(axis=0),
            "above_max": above.sum(axis=0),
            "outside_range": (below | above).sum(axis=0),
        })
        return NormalizedFeatures(
            matrix=pd.DataFrame(scaled, index=source.index.copy(), columns=list(self.features)),
            range_report=report,
            rows_outside_range=int((below | above).any(axis=1).sum()),
        )

    def to_dict(self) -> dict:
        """Return JSON-safe parameters sufficient to transform without refitting."""
        return {
            "schema_version": 1,
            "method": "min-max-clip-constant-zero-v1",
            "features": list(self.features),
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "id_column": self.id_column,
            "training_rows": self.training_rows,
            "training_sha256": self.training_sha256,
        }

    @classmethod
    def from_dict(cls, parameters: dict) -> MinMaxNormalizer:
        """Restore and validate the saved parameters without observing new data."""
        try:
            if (
                not isinstance(parameters, dict) or parameters.get("schema_version") != 1
                or parameters.get("method") != "min-max-clip-constant-zero-v1"
            ):
                raise CatalogContractError("Unsupported normalization parameter format.")
            if any(not isinstance(parameters[key], list) for key in ("features", "minimum", "maximum")):
                raise CatalogContractError("Saved normalization features and bounds must be lists.")
            return cls(
                features=tuple(parameters["features"]),
                minimum=tuple(parameters["minimum"]),
                maximum=tuple(parameters["maximum"]),
                id_column=parameters["id_column"],
                training_rows=parameters["training_rows"],
                training_sha256=parameters["training_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CatalogContractError):
                raise
            raise CatalogContractError(f"Invalid saved normalization parameters: {error}") from error


@dataclass(frozen=True)
class NormalizedFold:
    fold: int
    normalizer: MinMaxNormalizer
    training: NormalizedFeatures
    validation: NormalizedFeatures


def normalize_fold(
    matrix: pd.DataFrame,
    partitions: CatalogPartitions,
    fold: int,
    settings: Settings | None = None,
) -> NormalizedFold:
    """Fit on four development folds and transform the remaining fold.

    Test rows never enter the fitting or transformation performed here.
    The complete input matrix must match the saved partition IDs.
    """
    training = partitions.select_features(matrix, "train", fold)
    validation = partitions.select_features(matrix, "validation", fold)
    normalizer = MinMaxNormalizer.fit(training, settings)
    return NormalizedFold(
        fold=fold,
        normalizer=normalizer,
        training=normalizer.transform(training),
        validation=normalizer.transform(validation),
    )
