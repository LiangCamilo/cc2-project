"""Validate the cleaned catalog and summarize its unscaled acoustic features."""

import argparse
import json
from pathlib import Path

from zta.config import Settings
from zta.data.features import select_acoustic_features
from zta.data.loader import CatalogContractError, CsvCatalogSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, help="Path to the cleaned tracks.csv")
    parser.add_argument("--track-genres", type=Path, help="Path to the cleaned track_genres.csv")
    args = parser.parse_args()
    settings = Settings()
    try:
        catalog = CsvCatalogSource(args.tracks, args.track_genres, settings).load()
        features = select_acoustic_features(catalog, settings)
    except CatalogContractError as error:
        parser.exit(status=1, message=f"Catalog validation failed: {error}\n")

    genres = catalog[settings.genres_col]
    print(json.dumps({
        "tracks": len(catalog),
        "unique_track_ids": int(catalog[settings.id_col].nunique()),
        "genre_associations": int(genres.map(len).sum()),
        "genres": int(genres.explode().nunique()),
        "feature_matrix_shape": list(features.shape),
        "features": features.columns.tolist(),
        "status": "valid",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
