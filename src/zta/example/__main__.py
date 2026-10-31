"""Run seeds → normalization → profile → distances → zones → ranking."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions
from .workflow import WorkedExample, build_worked_example


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_example(example: WorkedExample, s: Settings, inputs: dict[str, Path]) -> Path:
    """Publish tables, fitted parameters, figures and a manifest replaced last."""
    from .figures import export_figures
    import matplotlib

    destination = s.worked_example_dir / f"fold_{example.fold}"
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".example-", dir=destination) as temporary:
        staging = Path(temporary)
        _json(staging / "profile.json", example.profile.to_dict())
        _json(staging / "distance_model.json", example.model.to_dict())
        _json(staging / "thresholds.json", {metric: zones.to_dict() for metric, zones in example.thresholds.items()})
        for name, table, indexed in (
            ("raw_seeds", example.raw_seeds, True),
            ("normalized_seeds", example.normalized_seeds, True),
            ("feature_parameters", example.feature_parameters, False),
            ("validation_clipping", example.clipping_report, False),
            ("reference_distances", example.reference_distances, True),
            ("candidate_distances", example.candidate_distances, True),
            ("zone_counts", example.zone_counts, False),
            *((f"{r.thresholds.metric}_{r.mode.value}", r.table, False) for r in example.recommendations),
        ):
            table = table.copy()
            if s.genres_col in table:
                table[s.genres_col] = table[s.genres_col].map(lambda tags: json.dumps(list(tags), ensure_ascii=False))
            table.to_csv(staging / f"{name}.csv", index=indexed)
        export_figures(example, staging / "figures")
        code_root = Path(__file__).resolve().parents[1]
        manifest = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {**example.summary(), "action": "saved", "directory": str(destination)},
            "fitting_source": "selected-fold-training-only; saved partition IDs reused",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
            "inputs_sha256": {name: _sha256(path) for name, path in inputs.items()},
            "code_sha256": {
                path.relative_to(code_root).as_posix(): _sha256(path)
                for path in sorted(code_root.rglob("*.py"))
            },
            "files_sha256": {
                path.relative_to(staging).as_posix(): _sha256(path)
                for path in sorted(staging.rglob("*")) if path.is_file()
            },
        }
        _json(staging / "manifest.json", manifest)
        for name in [*manifest["files_sha256"], "manifest.json"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, target)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", help="Distinct validation IDs; default: first five sorted IDs")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--popularity-lambda", type=float, help="Nonnegative popularity penalty within each zone")
    parser.add_argument("--save", action="store_true", help="Export calculation tables and three figures in PNG and PDF")
    args = parser.parse_args()
    s = Settings()
    try:
        inputs = {
            "tracks.csv": s.processed_tracks, "track_genres.csv": s.processed_track_genres,
            "partitions/manifest.json": s.partitions_dir / "manifest.json",
            "partitions/assignments.csv": s.partitions_dir / "assignments.csv",
            "partitions/genre_representation.csv": s.partitions_dir / "genre_representation.csv",
        }
        initial_hashes = {name: _sha256(path) for name, path in inputs.items()}
        catalog = CsvCatalogSource(settings=s).load()
        partitions = load_partitions(catalog, settings=s)
        example = build_worked_example(
            catalog, partitions, fold=args.fold, seed_ids=args.seeds, k=args.k,
            popularity_lambda=args.popularity_lambda, settings=s,
        )
        if initial_hashes != {name: _sha256(path) for name, path in inputs.items()}:
            raise CatalogContractError("Source files changed while building the example; run again.")
        summary = {**example.summary(), "action": "checked"}
        if args.save:
            destination = save_example(example, s, inputs)
            summary = {**summary, "action": "saved", "directory": str(destination)}
    except (CatalogContractError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(status=1, message=f"Worked example failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
