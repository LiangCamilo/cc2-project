"""Generate all three listening modes from a saved validation calculation example."""

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
from zta.domain import (
    AcousticDistanceModel, AcousticProfile, ListeningMode, calibrate_zones, rank_zone,
)
from zta.domain.profiles import normalization_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_example(s: Settings, fold: int, catalog: pd.DataFrame) -> tuple:
    source = s.profile_distances_dir / f"fold_{fold}"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict) or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("inputs_sha256"), dict)
        or not isinstance(manifest.get("files_sha256"), dict)
        or not isinstance(manifest.get("summary"), dict) or manifest["summary"].get("fold") != fold
    ):
        raise CatalogContractError("Invalid profile/distance example manifest.")
    inputs = {
        "tracks.csv": s.processed_tracks, "track_genres.csv": s.processed_track_genres,
        "partitions/assignments.csv": s.partitions_dir / "assignments.csv",
        "partitions/manifest.json": s.partitions_dir / "manifest.json",
        "normalization/manifest.json": s.normalization_dir / "manifest.json",
        f"normalization/fold_{fold}.json": s.normalization_dir / f"fold_{fold}.json",
    }
    if any(manifest["inputs_sha256"].get(name) != _sha256(path) for name, path in inputs.items()):
        raise CatalogContractError("The saved calculation example belongs to different input files.")
    for name in ("profile.json", "distance_model.json", "reference_distances.csv", "candidate_distances.csv"):
        if manifest["files_sha256"].get(name) != _sha256(source / name):
            raise CatalogContractError(f"Saved calculation example failed integrity validation: {name}.")
    profile = AcousticProfile.from_dict(json.loads((source / "profile.json").read_text(encoding="utf-8")))
    model = AcousticDistanceModel.from_dict(json.loads((source / "distance_model.json").read_text(encoding="utf-8")))
    if profile.normalization_sha256 != normalization_fingerprint(model.normalizer):
        raise CatalogContractError("Saved profile and model belong to different normalized spaces.")
    partitions = load_partitions(catalog, settings=s)
    training = partitions.select_features(select_acoustic_features(catalog, s), "train", fold)
    model.normalizer.validate_training(training)
    if not set(profile.seed_ids).issubset(partitions.ids("validation", fold)):
        raise CatalogContractError("Example seeds must belong to the selected validation fold.")
    reference = pd.read_csv(source / "reference_distances.csv", dtype={s.id_col: "string"}).set_index(s.id_col)
    candidates = pd.read_csv(source / "candidate_distances.csv", dtype={s.id_col: "string"}).set_index(s.id_col)
    for frame, expected in (
        (reference, set(training.index) - set(profile.seed_ids)),
        (candidates, set(partitions.ids("validation", fold)) - set(profile.seed_ids)),
    ):
        if frame.columns.tolist() != ["euclidean", "mahalanobis"] or frame.index.has_duplicates or set(frame.index) != expected:
            raise CatalogContractError("Saved distance rows do not match the selected partition IDs.")
    return profile, reference, candidates, partitions, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k", type=int, default=10, help="Requested recommendations per mode and metric")
    parser.add_argument("--popularity-lambda", type=float, help="Nonnegative within-zone popularity penalty")
    parser.add_argument("--save", action="store_true", help="Save six recommendation tables, thresholds and manifest")
    args = parser.parse_args()
    s = Settings()
    try:
        catalog = CsvCatalogSource(settings=s).load()
        profile, reference, candidates, partitions, source = _load_example(s, args.fold, catalog)
        thresholds, results = {}, []
        for metric in ("euclidean", "mahalanobis"):
            zones = calibrate_zones(reference[metric], s)
            thresholds[metric] = zones.to_dict()
            results.extend(
                rank_zone(candidates[metric], catalog, zones, mode, args.k,
                          seed_ids=profile.seed_ids, popularity_lambda=args.popularity_lambda, settings=s)
                for mode in ListeningMode
            )
        summary = {
            "status": "valid", "fold": args.fold,
            "example_type": "saved-calculation-example; no relevance labels or quality evaluation",
            "seed_ids": list(profile.seed_ids), "test_rows_reserved": len(partitions.test_ids),
            "thresholds": thresholds, "lists": [result.summary() for result in results], "action": "checked",
        }
        if args.save:
            destination = s.recommendations_dir / f"fold_{args.fold}"
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".recommendations-", dir=destination) as temporary:
                staging = Path(temporary)
                (staging / "thresholds.json").write_text(json.dumps(thresholds, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                for result in results:
                    table = result.table.copy()
                    table[s.genres_col] = table[s.genres_col].map(lambda tags: json.dumps(list(tags), ensure_ascii=False))
                    table.to_csv(staging / f"{result.thresholds.metric}_{result.mode.value}.csv", index=False)
                manifest = {
                    "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": {**summary, "action": "saved", "directory": str(destination)},
                    "inputs_sha256": {name: _sha256(source / name) for name in (
                        "manifest.json", "profile.json", "distance_model.json",
                        "reference_distances.csv", "candidate_distances.csv",
                    )},
                    "code_sha256": {name: _sha256(Path(__file__).parent / name) for name in (
                        "__main__.py", "../domain/models.py", "../domain/zones.py", "../domain/ranking.py", "../config.py",
                    )},
                    "files_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
                }
                (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                for name in [*manifest["files_sha256"], "manifest.json"]:
                    os.replace(staging / name, destination / name)
            summary = {**summary, "action": "saved", "directory": str(destination)}
    except (CatalogContractError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(status=1, message=f"Recommendation preparation failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
