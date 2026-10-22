from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from zta.config import Settings


class CatalogContractError(ValueError):
    """The cleaned catalog does not satisfy the model's input contract."""


class CatalogSource(Protocol):
    def load(self) -> pd.DataFrame: ...


class CsvCatalogSource:
    """Read the cleaned ETL tables and attach genre tuples to unique tracks.

    All original track columns and their row order are preserved. This loader
    validates its inputs without cleaning, scaling or writing either CSV.
    """

    def __init__(
        self,
        tracks_path: Path | None = None,
        track_genres_path: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._s = settings or Settings()
        self._tracks_path = Path(tracks_path) if tracks_path is not None else self._s.processed_tracks
        self._track_genres_path = (
            Path(track_genres_path) if track_genres_path is not None
            else self._s.processed_track_genres
        )

    def load(self) -> pd.DataFrame:
        s = self._s
        tracks = self._read_csv(self._tracks_path)
        track_genres = self._read_csv(self._track_genres_path)
        self._validate(tracks, track_genres)
        genres = track_genres.groupby(s.id_col)[s.track_genre_col].agg(
            lambda labels: tuple(sorted(labels))
        )
        catalog = tracks.copy()
        catalog[s.genres_col] = catalog[s.id_col].map(genres)
        return catalog

    def _read_csv(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, dtype={self._s.id_col: "string"})
        except FileNotFoundError as error:
            raise CatalogContractError(
                f"Catalog file not found: {path}. Generate the cleaned dataset with "
                "python -m clean.export_dataset first."
            ) from error
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as error:
            raise CatalogContractError(f"Cannot read catalog CSV {path}: {error}") from error

    def _validate(self, tracks: pd.DataFrame, track_genres: pd.DataFrame) -> None:
        s = self._s
        for table_name, frame, required in (
            ("tracks", tracks, [s.id_col, "track_name", "artists", s.popularity_col, *s.features]),
            ("track_genres", track_genres, [s.id_col, s.track_genre_col]),
        ):
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise CatalogContractError(f"{table_name}: missing columns {missing}.")
            if frame.empty:
                raise CatalogContractError(f"{table_name}: the table is empty.")

        for table_name, frame, columns in (
            ("tracks", tracks, [s.id_col, "track_name", "artists"]),
            ("track_genres", track_genres, [s.id_col, s.track_genre_col]),
        ):
            for column in columns:
                text = frame[column].astype("string")
                if (text.isna() | text.str.strip().eq("")).any():
                    raise CatalogContractError(f"{table_name}: {column} contains missing or blank values.")
                if text.ne(text.str.strip()).any():
                    raise CatalogContractError(f"{table_name}: {column} contains surrounding whitespace.")

        if tracks[s.id_col].duplicated().any():
            raise CatalogContractError("tracks: duplicate track IDs.")
        if track_genres.duplicated([s.id_col, s.track_genre_col]).any():
            raise CatalogContractError("track_genres: duplicate track/genre pairs.")
        if (~track_genres[s.id_col].isin(tracks[s.id_col])).any():
            raise CatalogContractError("track_genres: genre associations refer to unknown track IDs.")
        if (~tracks[s.id_col].isin(track_genres[s.id_col])).any():
            raise CatalogContractError("tracks: some tracks have no genre association.")

        for column in [s.popularity_col, *s.features]:
            values = tracks[column]
            if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                raise CatalogContractError(f"tracks: {column} must contain numeric values.")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise CatalogContractError(f"tracks: {column} contains missing or non-finite values.")
