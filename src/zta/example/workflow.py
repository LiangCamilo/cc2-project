"""Connect existing ZTA operations without changing their mathematical rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from zta.config import Settings
from zta.data import CatalogPartitions, select_acoustic_features
from zta.domain import (
    AcousticDistanceModel, AcousticProfile, ListeningMode, RecommendationResult,
    ZoneThresholds, build_acoustic_profile, calibrate_zones, rank_zone,
)
from zta.preprocessing import MinMaxNormalizer


@dataclass(frozen=True)
class WorkedExample:
    fold: int
    seed_selection: str
    profile: AcousticProfile
    model: AcousticDistanceModel
    raw_seeds: pd.DataFrame
    normalized_seeds: pd.DataFrame
    feature_parameters: pd.DataFrame
    clipping_report: pd.DataFrame
    reference_distances: pd.DataFrame
    candidate_distances: pd.DataFrame
    thresholds: dict[str, ZoneThresholds]
    recommendations: tuple[RecommendationResult, ...]
    zone_counts: pd.DataFrame
    validation_rows: int
    test_rows_reserved: int

    def summary(self) -> dict:
        return {
            "status": "valid", "fold": self.fold,
            "example_type": "deterministic-calculation-example; no relevance labels or quality evaluation",
            "seed_selection": self.seed_selection,
            "seed_ids": list(self.profile.seed_ids),
            "centroid": dict(zip(self.profile.features, self.profile.centroid)),
            "training_rows": self.model.normalizer.training_rows,
            "reference_rows": len(self.reference_distances),
            "validation_rows": self.validation_rows,
            "candidate_rows": len(self.candidate_distances),
            "test_rows_reserved": self.test_rows_reserved,
            "regularization": self.model.regularization,
            "thresholds": {metric: zones.to_dict() for metric, zones in self.thresholds.items()},
            "zone_counts": self.zone_counts.to_dict(orient="records"),
            "lists": [result.summary() for result in self.recommendations],
        }


def build_worked_example(
    catalog: pd.DataFrame,
    partitions: CatalogPartitions,
    *,
    fold: int = 1,
    seed_ids: Sequence[str] | None = None,
    k: int = 10,
    popularity_lambda: float | None = None,
    settings: Settings | None = None,
) -> WorkedExample:
    """Fit on training, query validation seeds, calibrate on training and rank validation.

    The default seeds are the first five sorted validation IDs (or all if fewer).
    This deterministic choice illustrates the calculation; it does not represent
    a measured listener preference. Test songs participate in none of its stages.
    Normalization is fitted anew on the same saved training IDs, so this command
    needs no profile, distance or recommendation exports from earlier steps.
    """
    s = settings or Settings()
    matrix = select_acoustic_features(catalog, s)
    training = partitions.select_features(matrix, "train", fold)
    validation = partitions.select_features(matrix, "validation", fold)
    seeds = sorted(validation.index)[:5] if seed_ids is None else seed_ids
    normalizer = MinMaxNormalizer.fit(training, s)
    # Supplying validation only also prevents training/test IDs from becoming seeds.
    profile = build_acoustic_profile(validation, seeds, normalizer)
    model = AcousticDistanceModel.fit(training, normalizer, s.mahalanobis_regularization)
    reference = model.distances(profile, training)
    candidates = model.distances(profile, validation)
    thresholds, recommendations, counts = {}, [], []
    for metric in reference.columns:
        zones = calibrate_zones(reference[metric], s)
        thresholds[metric] = zones
        for mode in ListeningMode:
            result = rank_zone(
                candidates[metric], catalog, zones, mode, k,
                seed_ids=profile.seed_ids, popularity_lambda=popularity_lambda, settings=s,
            )
            recommendations.append(result)
            counts.append({"metric": metric, "zone": mode.value, "candidates": result.available_in_zone})
        counts.append({
            "metric": metric, "zone": "outside",
            "candidates": int(candidates[metric].gt(zones.cutoffs[-1]).sum()),
        })
    metadata = catalog.set_index(s.id_col).loc[
        list(profile.seed_ids), ["track_name", "artists", s.genres_col, s.popularity_col],
    ]
    raw_seeds = metadata.join(validation.loc[list(profile.seed_ids)])
    normalized_seeds = normalizer.transform(validation.loc[list(profile.seed_ids)]).matrix
    parameters = pd.DataFrame({
        "feature": profile.features, "training_min": normalizer.minimum,
        "training_max": normalizer.maximum, "centroid": profile.centroid,
    })
    clipping = normalizer.transform(validation).range_report.copy()
    return WorkedExample(
        fold=fold,
        seed_selection="first-five-sorted-validation-ids" if seed_ids is None else "explicit-validation-ids",
        profile=profile, model=model, raw_seeds=raw_seeds, normalized_seeds=normalized_seeds,
        feature_parameters=parameters, clipping_report=clipping,
        reference_distances=reference, candidate_distances=candidates,
        thresholds=thresholds, recommendations=tuple(recommendations),
        zone_counts=pd.DataFrame(counts), validation_rows=len(validation),
        test_rows_reserved=len(partitions.test_ids),
    )
