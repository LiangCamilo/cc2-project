from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from zta.config import Settings

class CatalogContractError(ValueError):
    """The catalog delivered by the ETL does not comply with the agreed-upon contract"""

class CatalogSource(Protocol):
    def load(self) -> pd.DataFrame: ...

class CsvCatalogSource:

    def __init__(self, path: Path, settings: Settings | None = None) -> None:
        self._path = Path(path)
        self._s = settings or Settings()

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        s = self._s
        required = [s.id_col, s.popularity_col, s.genres_col, s.primary_genre_col, *s.features]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise CatalogContractError(f"Columns are missing: {missing}")
        if df[s.id_col].duplicated().any():
            raise CatalogContractError(f"There are duplicate track_ids")
        if df[list(s.features)].isna().any().any():
            raise CatalogContractError(f"There are null values in the acoustic features")