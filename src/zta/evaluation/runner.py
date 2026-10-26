"""Compare ZTA distances on shared synthetic profiles, without using test songs."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from zta.config import PROJECT_ROOT, Settings
from zta.data import CatalogContractError, CatalogPartitions, select_acoustic_features
from zta.domain import AcousticDistanceModel, ListeningMode, RankingCatalog, build_acoustic_profile, calibrate_zones, rank_zone
from zta.domain.profiles import normalization_fingerprint
from zta.preprocessing import MinMaxNormalizer

from .metrics import METRICS, evaluate_list
from .profiles import generate_profiles, positive_integer

REPORTS_DIR = PROJECT_ROOT / "reports" / "recommendations"
DISTANCES = ("euclidean", "mahalanobis")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _distances(values: np.ndarray, index: pd.Index, centroid: np.ndarray, cholesky: np.ndarray) -> pd.DataFrame:
    """Reuse validated normalized rows and the fitted Cholesky across queries.

    These are the same operations as AcousticDistanceModel.distances; caching
    avoids normalizing the entire training reference again for every profile.
    Seeds are removed by ID by the caller/ranker, never by acoustic distance.
    """
    delta = values - centroid
    return pd.DataFrame({
        "euclidean": np.linalg.norm(delta, axis=1),
        "mahalanobis": np.linalg.norm(np.linalg.solve(cholesky, delta.T), axis=0),
    }, index=index)


def summarize(per_profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average profiles within folds first; dispersion is then across folds.

    Report non-null observation counts explicitly: empty lists have undefined
    inverse popularity, and lists shorter than two have undefined ILD.
    """
    keys = ["fold", "metric", "mode"]
    grouped = per_profile.groupby(keys, sort=True)
    means = grouped[list(METRICS)].mean()
    counts = grouped[list(METRICS)].count().add_suffix("_count")
    sizes = grouped.size().rename("profiles")
    folds = pd.concat([sizes, means, counts], axis=1).reset_index()
    summary = folds.groupby(["metric", "mode"])[list(METRICS)].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_fold_count" if stat == "count" else f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    totals = folds.groupby(["metric", "mode"])[["profiles", *counts.columns]].sum().reset_index()
    totals = totals.rename(columns={"profiles": "total_profiles"})
    summary = summary.merge(totals, on=["metric", "mode"], validate="one_to_one")
    return folds, summary


def select_distance(summary: pd.DataFrame, k: int = 10) -> dict:
    """Familiar precision, then complete-list proportion, then Euclidean."""
    familiar = summary.loc[summary["mode"].eq("familiar")].set_index("metric")
    if set(familiar.index) != set(DISTANCES) or not np.isfinite(
        familiar[["precision_at_k_mean", "complete_mean"]].to_numpy()
    ).all():
        raise CatalogContractError("Distance selection needs finite Familiar results for both distances.")
    # Exact ties only: no post-hoc tolerance changes the documented criterion.
    ordered = sorted(DISTANCES, key=lambda metric: (
        -float(familiar.loc[metric, "precision_at_k_mean"]),
        -float(familiar.loc[metric, "complete_mean"]), metric != "euclidean",
    ))
    return {
        "metric": ordered[0], "mode": "familiar", "k": k,
        "criterion": "mean_fold_precision_at_k_then_complete_rate_then_euclidean",
        "compared": {metric: {
            "precision_at_k_mean": float(familiar.loc[metric, "precision_at_k_mean"]),
            "complete_rate_mean": float(familiar.loc[metric, "complete_mean"]),
        } for metric in DISTANCES},
    }


def _plot(summary: pd.DataFrame, path: Path, k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = tuple(mode.value for mode in ListeningMode)
    panels = (
        ("precision_at_k", f"Precision@{k}"), ("recall_at_k", f"Recall@{k}"),
        ("hit_at_k", f"Hit Rate@{k}"), ("ild", "Diversidad interna (ILD)"),
        ("inverse_popularity", "Indicador inverso de popularidad"),
        ("genre_coverage", "Cobertura de géneros"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    positions = np.arange(3)
    for ax, (name, title) in zip(axes.flat, panels):
        for offset, metric, label, color in (
            (-0.19, "euclidean", "Euclídea", "#326a9c"),
            (0.19, "mahalanobis", "Mahalanobis", "#25856b"),
        ):
            rows = summary.loc[summary["metric"].eq(metric)].set_index("mode").loc[list(modes)]
            y = rows[f"{name}_mean"].to_numpy()
            error = rows[f"{name}_std"].fillna(0).to_numpy()
            ax.bar(positions + offset, y, width=0.36, yerr=error, capsize=3, color=color, label=label)
        ax.set_title(title)
        ax.set_xticks(positions, ("Familiar", "Descubrimiento", "Exploración"), fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Evaluación ZTA · medias por fold y desviación estándar")
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_evaluation(
    catalog: pd.DataFrame,
    partitions: CatalogPartitions,
    settings: Settings | None = None,
    *,
    folds: tuple[int, ...] | None = None,
    profiles_per_genre: int = 5,
    seed_count: int = 5,
    k: int = 10,
    genres: tuple[str, ...] | None = None,
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Execute development CV only; reduced protocol runs go under smoke/.

    A profile's held-out genre and relevance IDs are used only by the evaluator.
    ZTA receives acoustic seeds, all candidate songs, and fixed configuration.
    The final test evaluation is deliberately outside this entry point.
    """
    s = settings or Settings()
    for value, name in ((profiles_per_genre, "profiles_per_genre"), (seed_count, "seed_count"), (k, "k")):
        positive_integer(value, name)
    selected_folds = tuple(range(1, partitions.n_folds + 1)) if folds is None else tuple(folds)
    if (
        not selected_folds or len(set(selected_folds)) != len(selected_folds)
        or any(isinstance(f, bool) or not isinstance(f, int) or not 1 <= f <= partitions.n_folds for f in selected_folds)
    ):
        raise CatalogContractError("Choose distinct valid development folds.")
    selected_folds = tuple(sorted(selected_folds))
    if (partitions.n_folds, partitions.random_state, partitions.test_size) != (s.n_folds, s.random_state, s.test_size):
        raise CatalogContractError("Evaluation settings must match the saved partition design.")
    # Validate full identity coverage without selecting or transforming test rows.
    matrix = select_acoustic_features(catalog, s)
    if set(matrix.index) != set(partitions.development_ids + partitions.test_ids):
        raise CatalogContractError("Catalog IDs differ from the partition IDs.")
    complete = (
        selected_folds == tuple(range(1, partitions.n_folds + 1))
        and profiles_per_genre == 5 and seed_count == 5 and k == 10 and genres is None
    )
    out = Path(reports_dir) if complete else Path(reports_dir) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    indexed = catalog.set_index(s.id_col, drop=False)
    universe = tuple(sorted({genre for tags in catalog[s.genres_col] for genre in tags}))
    # Hash logical catalog values and assignments for in-memory and CLI callers.
    context = {
        "catalog_sha256": hashlib.sha256(catalog.sort_values(s.id_col).to_csv(index=False).encode()).hexdigest(),
        "assignments_sha256": hashlib.sha256(partitions.assignments(s.id_col).sort_values(s.id_col).to_csv(index=False).encode()).hexdigest(),
    }
    source_hashes = {str(path.relative_to(PROJECT_ROOT / "src")): _sha256(path) for path in sorted((PROJECT_ROOT / "src" / "zta").rglob("*.py"))}
    print(f"[{'full_cv' if complete else 'smoke'}] reports={out}", flush=True)
    rows, eligibility_reports, fold_records = [], [], []
    recommendation_columns = [
        "fold", "profile_id", "metric", "mode", "rank", s.id_col, "distance", "score",
        "lower_bound", "upper_bound", "lower_inclusive",
    ]
    with TemporaryDirectory(prefix=".evaluation-", dir=out) as temporary:
        staging = Path(temporary)
        with (
            (staging / "profiles.jsonl").open("w", encoding="utf-8") as profile_file,
            (staging / "recommendations.csv").open("w", encoding="utf-8", newline="") as recommendation_file,
            (staging / "fold_candidates.csv").open("w", encoding="utf-8", newline="") as candidate_file,
        ):
            writer = csv.DictWriter(recommendation_file, fieldnames=recommendation_columns)
            writer.writeheader()
            candidate_writer = csv.writer(candidate_file)
            candidate_writer.writerow(["fold", s.id_col])
            for fold in selected_folds:
                fold_start = perf_counter()
                training = partitions.select_features(matrix, "train", fold).sort_index()
                validation = partitions.select_features(matrix, "validation", fold).sort_index()
                # Reuse ordinary string indexes in the query loop; identities,
                # ordering and fitted values are unchanged by this conversion.
                for frame in (training, validation):
                    frame.index = pd.Index(frame.index.to_numpy(dtype=object), name=s.id_col, dtype=object)
                test_ids = set(partitions.test_ids)
                if set(training.index) & set(validation.index) or (set(training.index) | set(validation.index)) & test_ids:
                    raise CatalogContractError("Training, validation and reserved test IDs must be disjoint.")
                evaluation_catalog = indexed.loc[validation.index].reset_index(drop=True)
                ranking_catalog = RankingCatalog(evaluation_catalog, s)
                cases, eligibility = generate_profiles(
                    evaluation_catalog, fold, s, profiles_per_genre=profiles_per_genre,
                    seed_count=seed_count, genres=genres, genre_universe=universe,
                )
                if not cases:
                    raise CatalogContractError(f"Fold {fold} has no genre with seeds and at least one target.")
                eligibility_reports.append(eligibility)
                normalizer = MinMaxNormalizer.fit(training, s)
                model = AcousticDistanceModel.fit(training, normalizer, s.mahalanobis_regularization)
                normalized_training = normalizer.transform(training)
                normalized_validation = normalizer.transform(validation)
                train_values = normalized_training.matrix.to_numpy()
                validation_values = normalized_validation.matrix.to_numpy()
                cholesky = np.linalg.cholesky(np.asarray(model.covariance) + model.regularization * np.eye(len(s.features)))
                _json(staging / f"fold_{fold}_distance_model.json", model.to_dict())
                pd.concat([
                    normalized_training.range_report.assign(split="train"),
                    normalized_validation.range_report.assign(split="validation"),
                ]).to_csv(staging / f"fold_{fold}_normalization_ranges.csv", index=False)
                candidate_writer.writerows((fold, value) for value in validation.index)
                genre_counts = Counter(genre for tags in evaluation_catalog[s.genres_col] for genre in set(tags))
                print(f"[fold {fold}] train={len(training)} validation={len(validation)} profiles={len(cases)}", flush=True)
                for number, case in enumerate(cases, start=1):
                    query_start = perf_counter()
                    profile = build_acoustic_profile(validation, case.seed_ids, normalizer)
                    profile_file.write(json.dumps({**case.to_dict(), "acoustic_profile": profile.to_dict()}, ensure_ascii=False, allow_nan=False) + "\n")
                    centroid = np.asarray(profile.centroid)
                    reference = _distances(train_values, training.index, centroid, cholesky)
                    candidates = _distances(validation_values, validation.index, centroid, cholesky).drop(index=list(case.seed_ids))
                    seed_genres = Counter(genre for tags in indexed.loc[list(case.seed_ids), s.genres_col] for genre in set(tags))
                    candidate_genres = {genre for genre, count in genre_counts.items() if count > seed_genres[genre]}
                    distance_seconds = perf_counter() - query_start
                    for metric in DISTANCES:
                        calibration_start = perf_counter()
                        zones = calibrate_zones(reference[metric], s)
                        calibration_seconds = perf_counter() - calibration_start
                        selected_ids: set[str] = set()
                        for mode in ListeningMode:
                            ranking_start = perf_counter()
                            result = rank_zone(
                                candidates[metric], ranking_catalog, zones, mode, k,
                                seed_ids=case.seed_ids, settings=s,
                            )
                            ranking_seconds = perf_counter() - ranking_start
                            ids = result.table[s.id_col].tolist()
                            if (
                                len(ids) != len(set(ids)) or set(ids) & set(case.seed_ids)
                                or not set(ids).issubset(candidates.index) or set(ids) & selected_ids
                                or not zones.band(mode).contains(result.table["distance"].to_numpy()).all()
                            ):
                                raise CatalogContractError("Recommendations violated candidate, seed or disjoint-band invariants.")
                            selected_ids.update(ids)
                            metrics = evaluate_list(result.table, case.relevant_ids, k, normalized_validation.matrix, candidate_genres, s)
                            rows.append({
                                "fold": fold, "profile_id": case.profile_id, "genre": case.genre,
                                "replicate": case.replicate, **result.summary(), **metrics,
                                "seed_ids": json.dumps(list(case.seed_ids)),
                                "q_familiar": zones.cutoffs[0], "q_discovery": zones.cutoffs[1], "q_exploration": zones.cutoffs[2],
                                "empty_zone": int(result.available_in_zone == 0),
                                "distance_seconds": distance_seconds, "calibration_seconds": calibration_seconds,
                                "ranking_seconds": ranking_seconds,
                            })
                            for item in result.table.to_dict(orient="records"):
                                writer.writerow({
                                    "fold": fold, "profile_id": case.profile_id,
                                    **{column: item[column] for column in recommendation_columns[2:]},
                                })
                    if number % 100 == 0 or number == len(cases):
                        print(f"[fold {fold}] profiles={number}/{len(cases)} elapsed={perf_counter() - fold_start:.1f}s", flush=True)
                fold_records.append({
                    "fold": fold, "training_rows": len(training), "validation_rows": len(validation),
                    "profiles": len(cases), "normalization_sha256": normalization_fingerprint(normalizer),
                    "normalizer_training_sha256": normalizer.training_sha256,
                    "test_overlap": 0, "elapsed_seconds": perf_counter() - fold_start,
                })
        per_profile = pd.DataFrame(rows)
        fold_metrics, summary = summarize(per_profile)
        per_profile.to_csv(staging / "profile_metrics.csv", index=False)
        fold_metrics.to_csv(staging / "fold_metrics.csv", index=False)
        summary.to_csv(staging / "cv_summary.csv", index=False)
        pd.concat(eligibility_reports, ignore_index=True).to_csv(staging / "genre_eligibility.csv", index=False)
        selection = select_distance(summary, k)
        _json(staging / "selection.json", {**selection, "eligible_for_final": complete})
        _plot(summary, staging / "cv_comparison.png", k)
        manifest = {
            "schema_version": 1, "execution_type": "full_cv" if complete else "smoke",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "synthetic-genre-affinity-v1", "profile_sampling": "sha256(seed,fold,genre,replicate)_numpy-default-rng_sorted-ids-v1",
            "candidate_rule": "all_validation_fold_ids_except_profile_seeds",
            "relevance_rule": "all_candidates_carrying_evaluation_genre; evaluator_only",
            "profile_pairing": "same_saved_seeds_and_targets_for_both_distances_and_all_modes",
            "folds": list(selected_folds), "profiles_per_genre": profiles_per_genre,
            "seed_count": seed_count, "k": k, "genres": list(genres) if genres is not None else None,
            "random_state": s.random_state, "features": list(s.features),
            "zone_percentiles": s.zone_percentiles, "popularity_lambda": s.popularity_lambda,
            "mahalanobis_regularization": s.mahalanobis_regularization,
            "aggregation": "equal_weight_profiles_within_fold; equal_weight_fold_means; sample_std_ddof1",
            "metric_definitions": {
                "precision_at_k": "hits / requested_k (also for incomplete lists)",
                "recall_at_k": "hits / all_relevant_candidates",
                "hit_at_k": "at_least_one_relevant_recommendation",
                "ild": "mean_pairwise_normalized_euclidean / sqrt(number_of_features); undefined below 2 songs",
                "inverse_popularity": "mean(-log2((observed_popularity+1)/101)); undefined for empty lists",
                "genre_coverage": "union_of_recommended_genres / union_of_candidate_genres",
                "completeness": "returned_k / requested_k",
                "complete": "returned_k == requested_k", "empty_zone": "available_in_zone == 0",
            },
            "timing": "distance_seconds shared by six lists; calibration_seconds shared by three modes; ranking_seconds per list; do not sum repeated shared times",
            "test_rows_reserved": len(partitions.test_ids), "test_used_for_fitting_or_evaluation": False,
            "fold_records": fold_records, "evaluation_profiles": len(per_profile) // 6,
            "evaluated_lists": len(per_profile), "elapsed_seconds": perf_counter() - start,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "numeric_thread_environment": {name: os.environ.get(name) for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )},
            "training_context": context, "code_sha256": source_hashes,
            "files_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        _json(staging / "manifest.json", manifest)
        # Replace the manifest last; its hashes identify incomplete/mixed exports.
        for filename in [*manifest["files_sha256"], "manifest.json"]:
            os.replace(staging / filename, out / filename)
    print(f"[done] profiles={len(per_profile) // 6} lists={len(per_profile)} selected={selection['metric']} elapsed={perf_counter() - start:.1f}s", flush=True)
    return summary
