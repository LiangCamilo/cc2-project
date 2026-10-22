"""Select unscaled acoustic features while preserving each track's identity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data.loader import CatalogContractError


def select_acoustic_features(
    catalog: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    """Return a numeric feature matrix indexed by track ID.

    Columns follow Settings.features and rows retain the catalog's order.
    Original values and dtypes are preserved; no scaling, filtering or changes
    to the source catalog are performed. Metadata remain in the input catalog.
    """
    s = settings or Settings()
    features = list(s.features)
    if not features or len(features) != len(set(features)) or s.id_col in features:
        raise CatalogContractError("Feature names must be nonempty, unique and exclude the track ID.")
    if catalog.columns.has_duplicates:
        raise CatalogContractError("The catalog contains duplicate column names.")
    missing = [column for column in [s.id_col, *features] if column not in catalog.columns]
    if missing:
        raise CatalogContractError(f"Cannot select acoustic features: missing columns {missing}.")
    if catalog.empty:
        raise CatalogContractError("Cannot select acoustic features from an empty catalog.")

    ids = catalog[s.id_col]
    text_ids = ids.astype("string")
    if (text_ids.isna() | text_ids.str.strip().eq("")).any():
        raise CatalogContractError("Cannot associate features with missing or blank track IDs.")
    if text_ids.ne(text_ids.str.strip()).any():
        raise CatalogContractError("Cannot associate features with track IDs containing surrounding whitespace.")
    if ids.duplicated().any():
        raise CatalogContractError("Cannot associate features with duplicate track IDs.")

    for column in features:
        values = catalog[column]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
        ):
            raise CatalogContractError(f"Acoustic feature {column} must contain real numeric values.")
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            raise CatalogContractError(f"Acoustic feature {column} contains missing or non-finite values.")

    matrix = catalog.loc[:, features].copy()
    matrix.index = pd.Index(ids, name=s.id_col, copy=True)
    return matrix
