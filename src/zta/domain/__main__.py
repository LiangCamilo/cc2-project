"""Build a validation example profile and compute distances without ranking."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions, select_acoustic_features
from zta.domain import AcousticDistanceModel, build_acoustic_profile
from zta.preprocessing import MinMaxNormalizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1, help="Development fold, numbered 1 to 5")
    parser.add_argument("--seeds", nargs="+", help="Distinct validation track IDs; defaults to the first five sorted IDs")
    parser.add_argument("--save", action="store_true", help="Save profile, fitted model and distance tables for this fold")
    args = parser.parse_args()
    s = Settings()
    try:
        catalog = CsvCatalogSource(settings=s).load()
        matrix = select_acoustic_features(catalog, s)
        partitions = load_partitions(catalog, settings=s)
        training = partitions.select_features(matrix, "train", args.fold)
        validation = partitions.select_features(matrix, "validation", args.fold)
        inputs = {
            "tracks.csv": s.processed_tracks,
            "track_genres.csv": s.processed_track_genres,
            "partitions/assignments.csv": s.partitions_dir / "assignments.csv",
            "partitions/manifest.json": s.partitions_dir / "manifest.json",
        }
        normalization_manifest = json.loads((s.normalization_dir / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(normalization_manifest, dict) or normalization_manifest.get("schema_version") != 1:
            raise CatalogContractError("Unsupported normalization manifest.")
        if (
            not isinstance(normalization_manifest.get("inputs_sha256"), dict)
            or not isinstance(normalization_manifest.get("files_sha256"), dict)
        ):
            raise CatalogContractError("Normalization manifest has invalid integrity records.")
        if any(normalization_manifest["inputs_sha256"].get(name) != _sha256(path) for name, path in inputs.items()):
            raise CatalogContractError("Saved normalization belongs to different catalog inputs or partitions.")
        parameter_path = s.normalization_dir / f"fold_{args.fold}.json"
        if normalization_manifest["files_sha256"].get(parameter_path.name) != _sha256(parameter_path):
            raise CatalogContractError("Normalization parameters failed integrity validation.")
        normalizer = MinMaxNormalizer.from_dict(json.loads(parameter_path.read_text(encoding="utf-8")))
        seeds = args.seeds if args.seeds is not None else sorted(validation.index)[:5]
        if any(seed not in validation.index for seed in seeds):
            raise CatalogContractError("Example seeds must belong to this fold's validation set.")
        model = AcousticDistanceModel.fit(training, normalizer, s.mahalanobis_regularization)
        profile = build_acoustic_profile(validation, seeds, normalizer)
        reference_distances = model.distances(profile, training)
        candidate_distances = model.distances(profile, validation)
        summary = {
            "status": "valid", "fold": args.fold,
            "example_type": "deterministic-calculation-example; no relevance labels or quality evaluation",
            "seed_ids": list(profile.seed_ids), "centroid": dict(zip(profile.features, profile.centroid)),
            "training_rows": len(training), "reference_rows": len(reference_distances),
            "validation_rows": len(validation), "candidate_rows": len(candidate_distances),
            "test_rows_reserved": len(partitions.test_ids), "regularization": model.regularization,
            "candidate_distance_summary": {
                metric: {
                    "min": float(candidate_distances[metric].min()) if len(candidate_distances) else None,
                    "max": float(candidate_distances[metric].max()) if len(candidate_distances) else None,
                }
                for metric in candidate_distances
            },
            "action": "checked",
        }
        if args.save:
            destination = s.profile_distances_dir / f"fold_{args.fold}"
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".distances-", dir=destination) as temporary:
                staging = Path(temporary)
                for name, state in (("profile.json", profile.to_dict()), ("distance_model.json", model.to_dict())):
                    (staging / name).write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                reference_distances.to_csv(staging / "reference_distances.csv", index=True)
                candidate_distances.to_csv(staging / "candidate_distances.csv", index=True)
                inputs.update({
                    "normalization/manifest.json": s.normalization_dir / "manifest.json",
                    f"normalization/{parameter_path.name}": parameter_path,
                })
                manifest = {
                    "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": {**summary, "action": "saved", "directory": str(destination)},
                    "inputs_sha256": {name: _sha256(path) for name, path in inputs.items()},
                    "code_sha256": {
                        name: _sha256(Path(__file__).parent / name)
                        for name in (
                            "profiles.py", "distances.py", "__main__.py",
                            "../preprocessing/normalization.py", "../config.py",
                        )
                    },
                    "files_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
                }
                (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                for name in [*manifest["files_sha256"], "manifest.json"]:
                    os.replace(staging / name, destination / name)
            summary = {**summary, "action": "saved", "directory": str(destination)}
    except (CatalogContractError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(status=1, message=f"Profile/distance preparation failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
