"""Validate the cleaned catalog and summarize its unscaled acoustic features."""

import argparse
import json
from pathlib import Path

from zta.config import Settings
from zta.data.features import select_acoustic_features
from zta.data.loader import CatalogContractError, CsvCatalogSource
from zta.data.partitions import create_partitions, load_partitions, save_partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, help="Path to the cleaned tracks.csv")
    parser.add_argument("--track-genres", type=Path, help="Path to the cleaned track_genres.csv")
    parser.add_argument(
        "--save-partitions", action="store_true",
        help="Save test/development IDs, validation folds and genre counts in data/processed/partitions",
    )
    parser.add_argument(
        "--check-partitions", action="store_true",
        help="Verify and reuse the saved assignments for the current catalog and configuration",
    )
    args = parser.parse_args()
    if args.save_partitions and args.check_partitions:
        parser.error("Choose either --save-partitions or --check-partitions.")
    settings = Settings()
    try:
        catalog = CsvCatalogSource(args.tracks, args.track_genres, settings).load()
        features = select_acoustic_features(catalog, settings)
        partitions = None
        if args.save_partitions:
            partitions = create_partitions(catalog, settings)
            save_partitions(catalog, partitions, settings=settings)
        elif args.check_partitions:
            partitions = load_partitions(catalog, settings=settings)
    except (CatalogContractError, OSError) as error:
        parser.exit(status=1, message=f"Catalog preparation failed: {error}\n")

    genres = catalog[settings.genres_col]
    summary = {
        "tracks": len(catalog),
        "unique_track_ids": int(catalog[settings.id_col].nunique()),
        "genre_associations": int(genres.map(len).sum()),
        "genres": int(genres.explode().nunique()),
        "feature_matrix_shape": list(features.shape),
        "features": features.columns.tolist(),
        "status": "valid",
    }
    if partitions is not None:
        summary["partitions"] = {
            "development": len(partitions.development_ids),
            "test": len(partitions.test_ids),
            "folds": [
                {"fold": fold, "train": len(partitions.ids("train", fold)),
                 "validation": len(partitions.ids("validation", fold))}
                for fold in range(1, partitions.n_folds + 1)
            ],
            "random_state": partitions.random_state,
            "directory": str(settings.partitions_dir),
            "action": "saved" if args.save_partitions else "verified",
        }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
