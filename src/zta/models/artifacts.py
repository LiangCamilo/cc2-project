"""Contracts shared by model selection, persisted estimators and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn

from zta.config import Settings
from zta.data import CatalogPartitions, select_acoustic_features
from zta.preprocessing import MinMaxNormalizer

from .registry import MODEL_NAMES, Model, TASKS

SCHEMA_VERSION = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_context(catalog: pd.DataFrame, partitions: CatalogPartitions, settings: Settings) -> dict:
    """Identify acoustic inputs, targets and song assignments without fitting."""
    s = settings
    if (partitions.n_folds, partitions.test_size, partitions.random_state) != (s.n_folds, s.test_size, s.random_state):
        raise ValueError("Partitions and training settings differ.")
    matrix = select_acoustic_features(catalog, s).astype(float)
    partitions.select_features(matrix, "development")  # Verify complete ID coverage.
    metadata = catalog.set_index(s.id_col)
    canonical = matrix.copy()
    canonical[s.popularity_col] = metadata[s.popularity_col].astype(float)
    canonical[s.genres_col] = metadata[s.genres_col].map(
        lambda tags: json.dumps(sorted(tags), ensure_ascii=False, separators=(",", ":"))
    )
    assignments = partitions.assignments(s.id_col).sort_values(s.id_col)
    return {
        "catalog_sha256": hashlib.sha256(canonical.sort_index().to_csv(lineterminator="\n").encode()).hexdigest(),
        "partitions_sha256": hashlib.sha256(assignments.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "features": list(s.features), "id_column": s.id_col,
        "genre_labels": sorted(metadata[s.genres_col].explode().unique().tolist()),
        "random_state": s.random_state, "n_folds": s.n_folds, "test_size": s.test_size,
    }


def validate_artifact(artifact: dict) -> MinMaxNormalizer:
    """Reject legacy/inconsistent bundles before their scores can be used."""
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported model artifact. Regenerate it after complete cross-validation.")
    task, name = artifact.get("task"), artifact.get("name")
    if task not in TASKS or name not in MODEL_NAMES or artifact.get("stage") != "final":
        raise ValueError("Model artifact has an invalid task, name or training stage.")
    if artifact.get("sklearn") != sklearn.__version__:
        raise ValueError("Model artifact uses a different scikit-learn version; use the training version or regenerate it.")
    normalizer = MinMaxNormalizer.from_dict(artifact.get("normalizer"))
    context = artifact.get("training_context")
    if (
        not isinstance(context, dict)
        or artifact.get("features") != list(normalizer.features)
        or artifact.get("id_column") != normalizer.id_column
        or context.get("features") != list(normalizer.features)
        or context.get("id_column") != normalizer.id_column
        or any(
            not isinstance(context.get(key), str) or len(context[key]) != 64
            or any(c not in "0123456789abcdef" for c in context[key])
            for key in ("catalog_sha256", "partitions_sha256")
        )
    ):
        raise ValueError("Model artifact has inconsistent features, normalization or training context.")
    labels = artifact.get("labels")
    if task == "genre":
        if (
            not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels) or labels != context.get("genre_labels")
        ):
            raise ValueError("Genre artifact labels do not match its training context.")
    elif labels is not None:
        raise ValueError("Popularity artifacts must not contain genre output labels.")
    model = artifact.get("model")
    if (
        not isinstance(model, Model) or (model.task, model.name) != (task, name)
        or model.params != artifact.get("params")
        or getattr(model.estimator, "n_features_in_", len(normalizer.features)) != len(normalizer.features)
    ):
        raise ValueError("Saved estimator does not match its artifact metadata.")
    return normalizer


def validate_bundle(genre_models: dict, popularity_model: dict, context: dict) -> MinMaxNormalizer:
    """All application models must describe the same catalog and fitted space."""
    if set(genre_models) != set(MODEL_NAMES):
        raise ValueError("The application requires all four genre models.")
    normalizers = []
    for task, name, artifact in [
        *(("genre", name, artifact) for name, artifact in genre_models.items()),
        ("popularity", "rf", popularity_model),
    ]:
        normalizer = validate_artifact(artifact)
        if artifact["task"] != task or artifact["name"] != name or artifact["training_context"] != context:
            raise ValueError(f"The {task}/{name} artifact belongs to a different model, catalog or partition.")
        normalizers.append(normalizer)
    reference = normalizers[0]
    if any(n.to_dict() != reference.to_dict() for n in normalizers[1:]):
        raise ValueError("Loaded models use different fitted normalizers; retrain them on the same development set.")
    return reference
