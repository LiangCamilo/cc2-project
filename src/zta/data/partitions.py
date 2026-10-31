"""Reserve a song-level test set and cross-validation folds without fitting data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data.features import select_acoustic_features
from zta.data.loader import CatalogContractError


def _validate_config(test_size: float, n_folds: int, random_state: int) -> None:
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise CatalogContractError("test_size must be a fraction strictly between 0 and 1.")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise CatalogContractError("n_folds must be an integer of at least 2.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise CatalogContractError("random_state must be a nonnegative integer.")


@dataclass(frozen=True)
class CatalogPartitions:
    """Exact IDs in reproducible order; fold numbers start at 1.

    Each development song belongs to one validation fold. Its training roles
    in the other folds are derived, so assignments never need to be duplicated.
    """

    test_ids: tuple[str, ...]
    validation_folds: tuple[tuple[str, ...], ...]
    test_size: float
    random_state: int

    def __post_init__(self) -> None:
        _validate_config(self.test_size, self.n_folds, self.random_state)
        all_ids = self.test_ids + self.development_ids
        if any(not isinstance(value, str) or not value or value != value.strip() for value in all_ids):
            raise CatalogContractError("Partition IDs must be nonempty strings without surrounding whitespace.")
        if len(set(all_ids)) != len(all_ids):
            raise CatalogContractError("Partitions contain overlapping or duplicate track IDs.")
        if not all(self.validation_folds) or not self.test_ids:
            raise CatalogContractError("Test and every validation fold must be nonempty.")
        if len(self.test_ids) != math.ceil(len(all_ids) * self.test_size):
            raise CatalogContractError("Test count does not match ceil(catalog size * test_size).")
        sizes = [len(fold) for fold in self.validation_folds]
        if max(sizes) - min(sizes) > 1:
            raise CatalogContractError("Validation fold sizes must differ by at most one song.")

    @property
    def n_folds(self) -> int:
        return len(self.validation_folds)

    @property
    def development_ids(self) -> tuple[str, ...]:
        return tuple(track_id for fold in self.validation_folds for track_id in fold)

    def ids(self, split: str, fold: int | None = None) -> tuple[str, ...]:
        """Return development/test IDs or training/validation IDs for one fold."""
        if split in ("development", "test"):
            if fold is not None:
                raise CatalogContractError("Development and test sets do not take a fold number.")
            return self.development_ids if split == "development" else self.test_ids
        if split not in ("train", "validation"):
            raise CatalogContractError(f"Unknown split: {split}.")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 1 <= fold <= self.n_folds:
            raise CatalogContractError(f"Choose a fold between 1 and {self.n_folds}.")
        if split == "validation":
            return self.validation_folds[fold - 1]
        return tuple(
            track_id for number, ids in enumerate(self.validation_folds, start=1)
            if number != fold for track_id in ids
        )

    def select_features(
        self, matrix: pd.DataFrame, split: str, fold: int | None = None
    ) -> pd.DataFrame:
        """Select raw feature rows by ID, regardless of their incoming order."""
        _validate_coverage(self, matrix.index)
        return matrix.loc[list(self.ids(split, fold))].copy()

    def assignments(self, id_col: str = "track_id") -> pd.DataFrame:
        """One row per song; test songs have no validation fold."""
        rows = [(track_id, "test", None) for track_id in self.test_ids]
        rows.extend(
            (track_id, "development", number)
            for number, ids in enumerate(self.validation_folds, start=1) for track_id in ids
        )
        result = pd.DataFrame(rows, columns=[id_col, "split", "fold"])
        result["fold"] = result["fold"].astype("Int64")
        return result


def _validate_coverage(partitions: CatalogPartitions, ids: pd.Index) -> None:
    expected = set(partitions.test_ids + partitions.development_ids)
    if ids.has_duplicates or len(ids) != len(expected) or set(ids) != expected:
        raise CatalogContractError("Partition IDs do not match the complete catalog or feature matrix.")


def create_partitions(
    catalog: pd.DataFrame, settings: Settings | None = None
) -> CatalogPartitions:
    """Sort IDs, shuffle with NumPy's seeded generator, then reserve the test set.

    The test count is rounded up. The remaining shuffled IDs are divided into
    balanced, contiguous validation folds. Genre labels do not drive the split.
    """
    s = settings or Settings()
    _validate_config(s.test_size, s.n_folds, s.random_state)
    matrix = select_acoustic_features(catalog, s)
    if any(not isinstance(track_id, str) for track_id in matrix.index):
        raise CatalogContractError("Partition track IDs must be strings.")
    ids = sorted(matrix.index)
    n_test = math.ceil(len(ids) * s.test_size)
    if len(ids) - n_test < s.n_folds:
        raise CatalogContractError("Not enough development songs for the requested number of folds.")
    shuffled = np.random.default_rng(s.random_state).permutation(np.array(ids, dtype=object))
    return CatalogPartitions(
        test_ids=tuple(shuffled[:n_test]),
        validation_folds=tuple(tuple(fold) for fold in np.array_split(shuffled[n_test:], s.n_folds)),
        test_size=s.test_size,
        random_state=s.random_state,
    )


def _catalog_hash(catalog: pd.DataFrame, settings: Settings) -> str:
    # Includes metadata, all genre tuples and raw feature values, independent of row order.
    canonical = catalog.sort_values(settings.id_col).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def genre_representation(
    catalog: pd.DataFrame, partitions: CatalogPartitions, settings: Settings | None = None
) -> pd.DataFrame:
    """Count songs per genre, including zeros; multi-label shares can sum above 1."""
    s = settings or Settings()
    _validate_coverage(partitions, pd.Index(catalog[s.id_col]))
    genres = catalog.set_index(s.id_col)[s.genres_col]
    universe = sorted(genres.explode().unique())
    groups = [("development", None), ("test", None)]
    groups.extend((split, fold) for fold in range(1, partitions.n_folds + 1) for split in ("train", "validation"))
    rows = []
    for split, fold in groups:
        ids = partitions.ids(split, fold)
        labels = genres.loc[list(ids)].map(lambda values: tuple(set(values))).explode()
        counts = labels.value_counts().reindex(universe, fill_value=0)
        rows.extend((split, fold, genre, int(count), count / len(ids)) for genre, count in counts.items())
    result = pd.DataFrame(rows, columns=["split", "fold", "genre", "tracks", "fraction"])
    result["fold"] = result["fold"].astype("Int64")
    return result


def save_partitions(
    catalog: pd.DataFrame,
    partitions: CatalogPartitions,
    output_dir: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Save exact assignments, genre counts and an integrity manifest.

    The manifest is replaced last. Readers reject partial or mixed exports
    using file hashes. Source CSVs are never written by this operation.
    """
    s = settings or Settings()
    matrix = select_acoustic_features(catalog, s)
    _validate_coverage(partitions, matrix.index)
    if (partitions.test_size, partitions.n_folds, partitions.random_state) != (s.test_size, s.n_folds, s.random_state):
        raise CatalogContractError("Partition configuration differs from Settings.")
    report = genre_representation(catalog, partitions, s)
    destination = Path(output_dir) if output_dir is not None else s.partitions_dir
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".partitions-", dir=destination) as temporary:
        staging = Path(temporary)
        partitions.assignments(s.id_col).to_csv(staging / "assignments.csv", index=False)
        report.to_csv(staging / "genre_representation.csv", index=False)
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "sorted-ids_numpy-default-rng-permutation_array-split_v1",
            "numpy_version": np.__version__,
            "catalog_sha256": _catalog_hash(catalog, s),
            "id_column": s.id_col,
            "features": list(s.features),
            "test_size": s.test_size,
            "n_folds": s.n_folds,
            "random_state": s.random_state,
            "counts": {
                "catalog": len(matrix), "development": len(partitions.development_ids),
                "test": len(partitions.test_ids),
                "validation_folds": [len(fold) for fold in partitions.validation_folds],
            },
            "files_sha256": {
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in ("assignments.csv", "genre_representation.csv")
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in ("assignments.csv", "genre_representation.csv", "manifest.json"):
            os.replace(staging / name, destination / name)
    return destination


def load_partitions(
    catalog: pd.DataFrame, directory: Path | None = None, settings: Settings | None = None
) -> CatalogPartitions:
    """Reuse saved IDs only when the catalog, configuration and file hashes match."""
    s = settings or Settings()
    source = Path(directory) if directory is not None else s.partitions_dir
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files_sha256"), dict):
            raise CatalogContractError("Saved partition manifest has an invalid structure.")
        expected = {
            "schema_version": 1, "catalog_sha256": _catalog_hash(catalog, s),
            "id_column": s.id_col, "features": list(s.features),
            "test_size": s.test_size, "n_folds": s.n_folds, "random_state": s.random_state,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise CatalogContractError("Saved partitions belong to a different catalog or configuration.")
        for name in ("assignments.csv", "genre_representation.csv"):
            actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if actual != manifest["files_sha256"][name]:
                raise CatalogContractError(f"Saved partition file failed integrity validation: {name}.")
        assignments = pd.read_csv(
            source / "assignments.csv", dtype={s.id_col: "string", "fold": "Int64"},
            keep_default_na=False, na_values={"fold": [""]},
        )
        if assignments.columns.tolist() != [s.id_col, "split", "fold"]:
            raise CatalogContractError("Saved assignments have unexpected columns.")
        test = assignments["split"].eq("test")
        development = assignments["split"].eq("development")
        if not (test | development).all() or assignments.loc[test, "fold"].notna().any():
            raise CatalogContractError("Saved assignments have invalid split labels or test folds.")
        if not assignments.loc[development, "fold"].isin(range(1, s.n_folds + 1)).all():
            raise CatalogContractError("Saved development assignments have invalid fold numbers.")
        partitions = CatalogPartitions(
            test_ids=tuple(assignments.loc[test, s.id_col]),
            validation_folds=tuple(
                tuple(assignments.loc[development & assignments["fold"].eq(fold), s.id_col])
                for fold in range(1, s.n_folds + 1)
            ),
            test_size=s.test_size, random_state=s.random_state,
        )
        _validate_coverage(partitions, select_acoustic_features(catalog, s).index)
        return partitions
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, CatalogContractError):
            raise
        raise CatalogContractError(f"Cannot load saved partitions from {source}: {error}") from error
