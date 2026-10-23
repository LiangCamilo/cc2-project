from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Columns from ETL
    id_col: str = "track_id"
    popularity_col: str = "popularity"
    genres_col: str = "genres"
    track_genre_col: str = "track_genre"
    features: tuple[str, ...] = (
        "danceability", "energy", "valence", "acousticness",
        "instrumentalness", "liveness", "loudness", "tempo", "speechiness",
    )
    # Validation
    test_size: float = 0.15 # C_test
    n_folds: int = 5  # Cross-validation on the remaining 85%
    random_state: int = 42
    # ZTA
    zone_percentiles: dict[str, float] = field(
        default_factory=lambda: {"familiar": 25.0, "discovery" : 60.0, "exploration" : 90.0}
    )
    popularity_lambda: float = 0.0  # 0 - ranking
    mahalanobis_regularization: float = 1e-6
    # Paths
    processed_tracks: Path = PROJECT_ROOT / "data" / "processed" / "tracks.csv"
    processed_track_genres: Path = PROJECT_ROOT / "data" / "processed" / "track_genres.csv"
    partitions_dir: Path = PROJECT_ROOT / "data" / "processed" / "partitions"
    normalization_dir: Path = PROJECT_ROOT / "data" / "processed" / "normalization"
    profile_distances_dir: Path = PROJECT_ROOT / "data" / "processed" / "profile_distances"
    recommendations_dir: Path = PROJECT_ROOT / "data" / "processed" / "recommendations"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"