import numpy as np
import pandas as pd
import pytest

from zta.config import Settings
from zta.data import CsvCatalogSource, create_partitions
from zta.models import MODEL_NAMES
from zta.models.app import MusicRecommender
from zta.models.runner import run_cv, run_final

FEATURES = Settings().features


@pytest.fixture(scope="module")
def recommender(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    rng = np.random.default_rng(1)
    n, g = 900, 5
    labels = rng.integers(0, g, n)
    X = np.clip(rng.random((g, 9))[labels] + rng.normal(0, 0.2, (n, 9)), 0, 1)
    tracks = pd.DataFrame(X, columns=FEATURES)
    tracks.insert(0, "track_id", [f"id{i:05d}" for i in range(n)])
    tracks["track_name"], tracks["artists"] = [f"song{i}" for i in range(n)], "artist"
    tracks["popularity"] = rng.integers(0, 100, n)
    genres = pd.DataFrame({"track_id": tracks.track_id, "track_genre": [f"g{x}" for x in labels]})
    tracks.to_csv(tmp / "tracks.csv", index=False)
    genres.to_csv(tmp / "track_genres.csv", index=False)
    catalog = CsvCatalogSource(tmp / "tracks.csv", tmp / "track_genres.csv").load()
    partitions = create_partitions(catalog)
    for task in ("genre", "popularity"):
        names = MODEL_NAMES if task == "genre" else ("rf",)
        run_cv(task, names, range(1, partitions.n_folds + 1), catalog, partitions, reports_dir=tmp / "reports")
        run_final(task, names, catalog, partitions, reports_dir=tmp / "reports", artifacts_dir=tmp / "models")
    return MusicRecommender(catalog, partitions, artifacts_dir=tmp / "models")


def test_sliders_become_a_valid_profile(recommender):
    profile = recommender.profile_from_sliders(energy=0.9, danceability=0.1)
    assert len(profile.centroid) == 9 and profile.centroid[FEATURES.index("energy")] == 0.9
    with pytest.raises(ValueError):
        recommender.profile_from_sliders(energy=1.5)
    with pytest.raises(ValueError):
        recommender.profile_from_sliders(not_a_feature=0.5)


def test_genre_suggestions_cover_the_four_models(recommender):
    table = recommender.suggest_genres(recommender.profile_from_sliders(energy=0.9), top=3)
    assert set(table["model"]) == set(MODEL_NAMES) and len(table) == 12
    assert 0 <= recommender.expected_popularity(recommender.profile_from_sliders()) <= 100


@pytest.mark.parametrize("mode", ["familiar", "discovery", "exploration"])
def test_zta_recommendations_respect_zone_and_k(recommender, mode):
    table = recommender.recommend(recommender.profile_from_sliders(energy=0.8), mode, k=5)
    assert len(table) <= 5 and set(table["zone"]) <= {mode}


def test_knn_returns_closest_and_genre_filter_works(recommender):
    profile = recommender.profile_from_sliders(valence=0.9)
    table = recommender.nearest_songs(profile, k=7)
    assert len(table) == 7 and table["distance"].is_monotonic_increasing
    wanted = recommender.genres[0]
    filtered = recommender.nearest_songs(profile, k=7, genres=[wanted])
    assert all(wanted in tags for tags in filtered["genres"])


def test_profile_from_songs_excludes_the_seeds(recommender):
    seeds = list(recommender.catalog["track_id"][:3])
    profile = recommender.profile_from_songs(seeds)
    assert not set(seeds) & set(recommender.nearest_songs(profile, k=10)["track_id"])
    assert not set(seeds) & set(recommender.recommend(profile, "familiar", k=10)["track_id"])
