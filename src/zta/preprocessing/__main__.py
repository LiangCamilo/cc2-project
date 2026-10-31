"""Normalize the five development folds using their saved song assignments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions, select_acoustic_features
from zta.preprocessing import normalize_fold


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(catalog: pd.DataFrame, settings: Settings, staging: Path | None) -> dict:
    matrix = select_acoustic_features(catalog, settings)
    partitions = load_partitions(catalog, settings=settings)
    reports = []
    summaries = []
    for fold in range(1, partitions.n_folds + 1):
        result = normalize_fold(matrix, partitions, fold, settings)
        if staging is not None:
            (staging / f"fold_{fold}.json").write_text(
                json.dumps(result.normalizer.to_dict(), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        splits = {}
        for split, transformed in (("train", result.training), ("validation", result.validation)):
            reports.append(transformed.range_report.assign(fold=fold, split=split))
            splits[split] = {
                "rows": len(transformed.matrix),
                "rows_outside_training_range": transformed.rows_outside_range,
                "values_outside_training_range": transformed.values_outside_range,
            }
        summaries.append({
            "fold": fold,
            "constant_features": [
                row.feature for row in result.training.range_report.itertuples() if row.constant_feature
            ],
            **splits,
        })
    if staging is not None:
        report = pd.concat(reports, ignore_index=True)
        columns = ["fold", "split", *report.columns.drop(["fold", "split"])]
        report.loc[:, columns].to_csv(staging / "range_report.csv", index=False)
    return {
        "status": "valid",
        "method": "min-max-clip-constant-zero-v1",
        "features": list(settings.features),
        "test_rows_reserved": len(partitions.test_ids),
        "folds": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, help="Path to the cleaned tracks.csv")
    parser.add_argument("--track-genres", type=Path, help="Path to the cleaned track_genres.csv")
    parser.add_argument(
        "--save", action="store_true",
        help="Save fitted fold parameters, range counts and manifest in data/processed/normalization",
    )
    args = parser.parse_args()
    settings = Settings()
    tracks = args.tracks if args.tracks is not None else settings.processed_tracks
    genres = args.track_genres if args.track_genres is not None else settings.processed_track_genres
    try:
        catalog = CsvCatalogSource(tracks, genres, settings).load()
        if not args.save:
            summary = _prepare(catalog, settings, staging=None)
            summary["action"] = "checked"
        else:
            destination = settings.normalization_dir
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".normalization-", dir=destination) as temporary:
                staging = Path(temporary)
                summary = _prepare(catalog, settings, staging)
                manifest = {
                    "schema_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "inputs_sha256": {
                        "tracks.csv": _sha256(tracks),
                        "track_genres.csv": _sha256(genres),
                        "partitions/assignments.csv": _sha256(settings.partitions_dir / "assignments.csv"),
                        "partitions/manifest.json": _sha256(settings.partitions_dir / "manifest.json"),
                    },
                    "code_sha256": {
                        name: _sha256(Path(__file__).parent / name)
                        for name in ("normalization.py", "__main__.py")
                    },
                    "files_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
                }
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                # Publish the manifest last, after all five parameter files and the report.
                for name in [*manifest["files_sha256"], "manifest.json"]:
                    os.replace(staging / name, destination / name)
            summary = {**summary, "action": "saved", "directory": str(destination)}
    except (CatalogContractError, OSError) as error:
        parser.exit(status=1, message=f"Normalization failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
