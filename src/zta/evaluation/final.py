"""Evaluate the frozen ZTA selection once on the reserved song-level test set."""

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
from .profiles import generate_profiles
from .runner import REPORTS_DIR, DISTANCES, _distances, _json, _sha256, select_distance


def validate_final_selection(catalog, partitions, settings, reports_dir):
    """Reject reduced, stale or modified CV selections before touching test rows."""
    s, source = settings, Path(reports_dir)
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1, "execution_type": "full_cv",
            "protocol": "synthetic-genre-affinity-v1",
            "profile_sampling": "sha256(seed,fold,genre,replicate)_numpy-default-rng_sorted-ids-v1",
            "folds": list(range(1, partitions.n_folds + 1)), "profiles_per_genre": 5,
            "seed_count": 5, "k": 10, "genres": None, "random_state": s.random_state,
            "features": list(s.features), "zone_percentiles": s.zone_percentiles,
            "popularity_lambda": s.popularity_lambda,
            "mahalanobis_regularization": s.mahalanobis_regularization,
            "test_rows_reserved": len(partitions.test_ids), "test_used_for_fitting_or_evaluation": False,
        }
        context = {
            "catalog_sha256": hashlib.sha256(catalog.sort_values(s.id_col).to_csv(index=False).encode()).hexdigest(),
            "assignments_sha256": hashlib.sha256(partitions.assignments(s.id_col).sort_values(s.id_col).to_csv(index=False).encode()).hexdigest(),
        }
        if (
            any(manifest.get(key) != value for key, value in expected.items())
            or manifest.get("training_context") != context
            or manifest.get("versions", {}).get("numpy") != np.__version__
            or manifest.get("versions", {}).get("pandas") != pd.__version__
            or not isinstance(manifest.get("files_sha256"), dict)
            or not {"selection.json", "cv_summary.csv", "fold_metrics.csv"}.issubset(manifest["files_sha256"])
        ):
            raise CatalogContractError("Final ZTA evaluation requires compatible complete development CV and its frozen settings.")
        for name, digest in manifest["files_sha256"].items():
            if Path(name).name != name or _sha256(source / name) != digest:
                raise CatalogContractError(f"ZTA selection failed integrity validation: {name}.")
        # Adding the final runner and test sampling does not change the CV
        # protocol. Its fitted geometry, ranking and metric implementations must
        # still be the ones used to obtain the stored selection.
        for name in (
            "zta/config.py", "zta/data/features.py", "zta/preprocessing/normalization.py",
            "zta/domain/profiles.py", "zta/domain/distances.py", "zta/domain/models.py",
            "zta/domain/zones.py", "zta/domain/ranking.py", "zta/evaluation/metrics.py",
        ):
            if manifest.get("code_sha256", {}).get(name) != _sha256(PROJECT_ROOT / "src" / name):
                raise CatalogContractError(f"ZTA geometry or metric code changed after selection: {name}.")
        summary = pd.read_csv(source / "cv_summary.csv", float_precision="round_trip")
        expected_rows = {(metric, mode.value) for metric in DISTANCES for mode in ListeningMode}
        if (
            len(summary) != len(expected_rows)
            or set(zip(summary["metric"], summary["mode"])) != expected_rows
            or not summary["precision_at_k_fold_count"].eq(partitions.n_folds).all()
        ):
            raise CatalogContractError("ZTA selection must contain both distances, all modes and every validation fold.")
        selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
        if selection != {**select_distance(summary, 10), "eligible_for_final": True}:
            raise CatalogContractError("Saved ZTA choice differs from the documented CV selection rule.")
        return manifest, selection, summary
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, CatalogContractError):
            raise
        raise CatalogContractError(f"Missing or incompatible complete ZTA selection: {error}") from error


def _plot_comparison(comparison, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("precision_at_k", "Precision@10"), ("recall_at_k", "Recall@10"),
        ("hit_at_k", "Hit Rate@10"), ("ild", "Diversidad interna (ILD)"),
        ("inverse_popularity", "Indicador inverso de popularidad"), ("genre_coverage", "Cobertura de géneros"),
    )
    data = comparison.set_index("mode").loc[[mode.value for mode in ListeningMode]]
    x = np.arange(3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, (name, title) in zip(axes.flat, panels):
        ax.bar(x - .19, data[f"validation_{name}_mean"], .36, label="Validación", color="#326a9c")
        ax.bar(x + .19, data[f"{name}_mean"], .36, label="Prueba", color="#25856b")
        ax.set_title(title)
        ax.set_xticks(x, ("Familiar", "Descubrimiento", "Exploración"), fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("ZTA · configuración seleccionada · medias de validación y prueba")
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_final_evaluation(
    catalog: pd.DataFrame, partitions: CatalogPartitions, settings: Settings | None = None,
    *, reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Fit geometry on all development songs and evaluate only the chosen distance.

    Test contains one held-out set, not five new folds. Its summary averages
    synthetic profiles; profile standard deviations are descriptive and are not
    validation-fold dispersion or confidence intervals.
    """
    s = settings or Settings()
    source, out = Path(reports_dir), Path(reports_dir) / "final"
    if (out / "manifest.json").exists():
        raise CatalogContractError("Final ZTA results already exist; preserve them instead of evaluating test again.")
    if (partitions.n_folds, partitions.test_size, partitions.random_state) != (s.n_folds, s.test_size, s.random_state):
        raise CatalogContractError("Final ZTA settings differ from the partition design.")
    cv_manifest, selection, cv_summary = validate_final_selection(catalog, partitions, s, source)
    metric, k = selection["metric"], selection["k"]
    selection_hashes = {name: _sha256(source / name) for name in ("manifest.json", "selection.json", "cv_summary.csv")}
    matrix = select_acoustic_features(catalog, s)
    development = partitions.select_features(matrix, "development").sort_index()
    test = partitions.select_features(matrix, "test").sort_index()
    for frame in (development, test):
        frame.index = pd.Index(frame.index.to_numpy(dtype=object), name=s.id_col, dtype=object)
    if set(development.index) & set(test.index):
        raise CatalogContractError("Development and test song IDs overlap.")
    start = perf_counter()
    normalizer = MinMaxNormalizer.fit(development, s)
    model = AcousticDistanceModel.fit(development, normalizer, s.mahalanobis_regularization)
    normalized_development, normalized_test = normalizer.transform(development), normalizer.transform(test)
    indexed = catalog.set_index(s.id_col, drop=False)
    test_catalog = indexed.loc[test.index].reset_index(drop=True)
    cache = RankingCatalog(test_catalog, s)
    universe = tuple(sorted({genre for tags in catalog[s.genres_col] for genre in tags}))
    cases, eligibility = generate_profiles(test_catalog, None, s, genre_universe=universe, split="test")
    if not cases:
        raise CatalogContractError("No eligible test genres for the frozen profile protocol.")
    counts = Counter(genre for tags in test_catalog[s.genres_col] for genre in set(tags))
    cholesky = np.linalg.cholesky(np.asarray(model.covariance) + model.regularization * np.eye(len(s.features)))
    out.mkdir(parents=True, exist_ok=True)
    print(f"[final ZTA] development={len(development)} test={len(test)} profiles={len(cases)} metric={metric}", flush=True)
    rows = []
    columns = ["profile_id", "metric", "mode", "rank", s.id_col, "distance", "score", "lower_bound", "upper_bound", "lower_inclusive"]
    with TemporaryDirectory(prefix=".final-", dir=out) as temporary:
        staging = Path(temporary)
        with (staging / "profiles.jsonl").open("w", encoding="utf-8") as profiles_file, (staging / "recommendations.csv").open("w", encoding="utf-8", newline="") as recommendations_file:
            writer = csv.DictWriter(recommendations_file, fieldnames=columns)
            writer.writeheader()
            for number, case in enumerate(cases, start=1):
                query_start = perf_counter()
                profile = build_acoustic_profile(test, case.seed_ids, normalizer)
                profiles_file.write(json.dumps({**case.to_dict(), "acoustic_profile": profile.to_dict()}, ensure_ascii=False, allow_nan=False) + "\n")
                reference = _distances(normalized_development.matrix.to_numpy(), development.index, np.asarray(profile.centroid), cholesky)
                candidates = _distances(normalized_test.matrix.to_numpy(), test.index, np.asarray(profile.centroid), cholesky).drop(index=list(case.seed_ids))
                zones = calibrate_zones(reference[metric], s)
                seed_counts = Counter(genre for tags in indexed.loc[list(case.seed_ids), s.genres_col] for genre in set(tags))
                candidate_genres = {genre for genre, count in counts.items() if count > seed_counts[genre]}
                geometry_seconds = perf_counter() - query_start
                used = set()
                for mode in ListeningMode:
                    ranking_start = perf_counter()
                    result = rank_zone(candidates[metric], cache, zones, mode, k, seed_ids=case.seed_ids, settings=s)
                    ranking_seconds = perf_counter() - ranking_start
                    ids = result.table[s.id_col].tolist()
                    if (len(ids) != len(set(ids)) or set(ids) & (set(case.seed_ids) | used)
                            or not set(ids).issubset(candidates.index)
                            or not zones.band(mode).contains(result.table["distance"].to_numpy()).all()):
                        raise CatalogContractError("Final ZTA list violated candidate, seed or band invariants.")
                    used.update(ids)
                    rows.append({
                        "split": "test", "profile_id": case.profile_id, "genre": case.genre,
                        "replicate": case.replicate, **result.summary(),
                        **evaluate_list(result.table, case.relevant_ids, k, normalized_test.matrix, candidate_genres, s),
                        "seed_ids": json.dumps(list(case.seed_ids)), "empty_zone": int(result.available_in_zone == 0),
                        "q_familiar": zones.cutoffs[0], "q_discovery": zones.cutoffs[1], "q_exploration": zones.cutoffs[2],
                        "geometry_seconds": geometry_seconds, "ranking_seconds": ranking_seconds,
                    })
                    for item in result.table.to_dict(orient="records"):
                        writer.writerow({"profile_id": case.profile_id, **{name: item[name] for name in columns[1:]}})
                if number % 100 == 0 or number == len(cases):
                    print(f"[final ZTA] profiles={number}/{len(cases)} elapsed={perf_counter() - start:.1f}s", flush=True)
        per_profile = pd.DataFrame(rows)
        summary = per_profile.groupby(["metric", "mode"])[list(METRICS)].agg(["mean", "std", "count"])
        summary.columns = [f"{name}_profile_std" if stat == "std" else f"{name}_{stat}" for name, stat in summary.columns]
        summary = summary.reset_index()
        validation = cv_summary.loc[cv_summary["metric"].eq(metric), ["metric", "mode", *[f"{name}_mean" for name in METRICS]]]
        comparison = summary.merge(validation.rename(columns={f"{name}_mean": f"validation_{name}_mean" for name in METRICS}), on=["metric", "mode"], validate="one_to_one")
        per_profile.to_csv(staging / "profile_metrics.csv", index=False)
        summary.to_csv(staging / "test_metrics.csv", index=False)
        comparison.to_csv(staging / "validation_test_comparison.csv", index=False)
        eligibility.to_csv(staging / "genre_eligibility.csv", index=False)
        pd.DataFrame({s.id_col: test.index}).to_csv(staging / "test_candidates.csv", index=False)
        _json(staging / "distance_model.json", model.to_dict())
        _json(staging / "selection.json", selection)
        pd.concat([normalized_development.range_report.assign(split="development"), normalized_test.range_report.assign(split="test")]).to_csv(staging / "normalization_ranges.csv", index=False)
        _plot_comparison(comparison, staging / "test_comparison.png")
        manifest = {
            "schema_version": 1, "execution_type": "final_test", "protocol": cv_manifest["protocol"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "selected_metric": metric,
            "selection_source_sha256": selection_hashes, "training_context": cv_manifest["training_context"],
            "features": list(s.features), "random_state": s.random_state, "k": k,
            "seed_count": 5, "profiles_per_genre": 5, "zone_percentiles": s.zone_percentiles,
            "popularity_lambda": s.popularity_lambda, "mahalanobis_regularization": s.mahalanobis_regularization,
            "development_rows": len(development), "test_rows": len(test), "development_test_overlap": 0,
            "test_used_for_fitting": False, "test_used_for_selection": False,
            "evaluation_profiles": len(cases), "evaluated_lists": len(rows),
            "normalization_sha256": normalization_fingerprint(normalizer), "normalizer_training_sha256": normalizer.training_sha256,
            "profile_sampling": "sha256(seed,'test',genre,replicate)_numpy-default-rng_sorted-ids-v1",
            "candidate_rule": "all_test_ids_except_profile_seeds", "relevance_rule": cv_manifest["relevance_rule"],
            "aggregation": "profile_means_within_one_test_set; profile_std_is_descriptive_not_fold_std_or_confidence_interval",
            "timing": "geometry_seconds shared by three modes; ranking_seconds per list; do not sum shared times",
            "elapsed_seconds": perf_counter() - start,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "code_sha256": {str(path.relative_to(PROJECT_ROOT / "src")): _sha256(path) for path in sorted((PROJECT_ROOT / "src" / "zta").rglob("*.py"))},
            "files_sha256": {path.name: _sha256(path) for path in sorted(staging.iterdir())},
        }
        _json(staging / "manifest.json", manifest)
        for name in [*manifest["files_sha256"], "manifest.json"]:
            os.replace(staging / name, out / name)
    return summary
