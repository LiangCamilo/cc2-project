import numpy as np
import pandas as pd
import pytest

from zta.config import Settings
from zta.data import CsvCatalogSource, create_partitions, select_acoustic_features
from zta.models import MODEL_NAMES, build_model, genre_target, popularity_target
from zta.models.evaluation import evaluate
from zta.preprocessing import MinMaxNormalizer

FEATURES = Settings().features


@pytest.fixture()
def catalog(tmp_path):
    rng = np.random.default_rng(0)
    n, g = 600, 6
    labels = rng.integers(0, g, n)
    X = np.clip(rng.random((g, 9))[labels] + rng.normal(0, 0.2, (n, 9)), 0, 1)
    tracks = pd.DataFrame(X, columns=FEATURES)
    tracks.insert(0, "track_id", [f"id{i:05d}" for i in range(n)])
    tracks["track_name"], tracks["artists"] = "t", "a"
    tracks["popularity"] = rng.integers(0, 100, n)
    genres = pd.DataFrame({"track_id": tracks.track_id, "track_genre": [f"g{x}" for x in labels]})
    tracks.to_csv(tmp_path / "tracks.csv", index=False)
    genres.to_csv(tmp_path / "track_genres.csv", index=False)
    return CsvCatalogSource(tmp_path / "tracks.csv", tmp_path / "track_genres.csv").load()


def test_targets_align_with_catalog(catalog):
    y = genre_target(catalog)
    assert y.shape == (len(catalog), 6) and (y.sum(axis=1) >= 1).all()
    assert list(y.index) == list(catalog["track_id"])
    assert popularity_target(catalog).between(0, 100).all()


def test_normalizer_uses_training_rows_only(catalog):
    partitions = create_partitions(catalog)
    matrix = select_acoustic_features(catalog)
    train = partitions.select_features(matrix, "train", 1)
    normalizer = MinMaxNormalizer.fit(train)
    assert np.allclose(normalizer.minimum, train.min().to_numpy())
    assert not set(train.index) & set(partitions.ids("validation", 1))
    assert not set(partitions.development_ids) & set(partitions.test_ids)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("task", ["genre", "popularity"])
def test_every_model_scores_expected_shape(catalog, name, task):
    X = catalog[list(FEATURES)].to_numpy()
    y = genre_target(catalog).to_numpy() if task == "genre" else popularity_target(catalog).to_numpy()
    params = {"max_rows": 200} if name == "svm" and task == "popularity" else {}
    if name == "rf":
        params = {"n_estimators": 10}
    scores = build_model(name, task, params).fit(X, y).score(X)
    assert scores.shape == ((len(X), y.shape[1]) if task == "genre" else (len(X),))
    assert all(np.isfinite(v) for v in evaluate(task, y, scores).values())
