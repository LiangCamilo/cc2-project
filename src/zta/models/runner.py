"""Cross-validation on development folds, then one final fit/test run.

Protocol (same as the project's partition design):
  * the Min-Max normalizer is fitted only on the training rows of each run;
  * every model/configuration sees the same folds;
  * configurations are chosen with validation folds only;
  * the test set is used once, in `run_final`, with the chosen configurations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from zta.config import PROJECT_ROOT, Settings
from zta.data import CatalogPartitions, select_acoustic_features
from zta.preprocessing import MinMaxNormalizer

from .evaluation import PRIMARY, evaluate
from .artifacts import SCHEMA_VERSION, sha256, training_context
from .registry import CONFIGS, MODEL_NAMES, build_model
from .targets import build_target

REPORTS_DIR = PROJECT_ROOT / "reports" / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "models"


def _model_names(task, model_names) -> tuple[str, ...]:
    names = tuple(model_names)
    if (
        task not in PRIMARY or not names or any(name not in MODEL_NAMES for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Choose a known task and distinct model names.")
    return names


def _validate_selection(out, task, names, context, partitions):
    """Check eligibility before any final fitting or test-set prediction."""
    try:
        manifest = json.loads((out / "cv_manifest.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": SCHEMA_VERSION, "execution_type": "full_cv",
            "task": task, "folds": list(range(1, partitions.n_folds + 1)),
            "max_train_rows": None, "training_context": context,
            "primary_metric": PRIMARY[task][0], "sklearn": sklearn.__version__,
        }
        if (
            not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items())
            or not isinstance(manifest.get("models"), list) or not set(names).issubset(manifest["models"])
            or not isinstance(manifest.get("files_sha256"), dict)
        ):
            raise ValueError("Final training requires complete CV on all folds with the current catalog and partitions.")
        for filename in ("best_configs.json", "cv_folds.csv", "cv_summary.csv"):
            if manifest["files_sha256"].get(filename) != sha256(out / filename):
                raise ValueError(f"CV selection failed integrity validation: {filename}.")
        best = json.loads((out / "best_configs.json").read_text(encoding="utf-8"))
        for name in names:
            chosen = best[name]
            config_id = chosen["config_id"]
            if (
                isinstance(config_id, bool) or not isinstance(config_id, int)
                or not 0 <= config_id < len(CONFIGS[task][name])
                or chosen["params"] != CONFIGS[task][name][config_id]
            ):
                raise ValueError("Selected configuration no longer matches the model registry; repeat complete CV.")
        return best
    except (OSError, KeyError, TypeError) as error:
        raise ValueError("Missing or incompatible complete CV selection. Run --mode cv on all folds without a row limit.") from error


def _prepare(matrix, target, partitions, train_split, eval_split, fold, settings, max_train_rows=None):
    train_raw = partitions.select_features(matrix, train_split, fold)
    eval_raw = partitions.select_features(matrix, eval_split, fold)
    if max_train_rows is not None and max_train_rows < len(train_raw):
        train_raw = train_raw.sample(n=max_train_rows, random_state=settings.random_state)
    normalizer = MinMaxNormalizer.fit(train_raw, settings)  # training rows only
    x_train = normalizer.transform(train_raw).matrix
    x_eval = normalizer.transform(eval_raw).matrix
    return normalizer, x_train, x_eval, target.loc[x_train.index], target.loc[x_eval.index]


def run_cv(
    task: str, model_names, folds, catalog: pd.DataFrame, partitions: CatalogPartitions,
    settings: Settings | None = None, reports_dir: Path = REPORTS_DIR, max_train_rows: int | None = None,
) -> pd.DataFrame:
    s = settings or Settings()
    model_names = _model_names(task, model_names)
    folds = tuple(folds)
    if (
        not folds or len(set(folds)) != len(folds)
        or any(isinstance(f, bool) or not isinstance(f, int) or not 1 <= f <= partitions.n_folds for f in folds)
    ):
        raise ValueError("Choose distinct valid folds.")
    if max_train_rows is not None and (
        isinstance(max_train_rows, bool) or not isinstance(max_train_rows, int) or max_train_rows < 1
    ):
        raise ValueError("max_train_rows must be a positive integer.")
    folds = tuple(sorted(folds))
    complete = folds == tuple(range(1, partitions.n_folds + 1)) and max_train_rows is None
    context = training_context(catalog, partitions, s)
    matrix = select_acoustic_features(catalog, s)
    target = build_target(task, catalog, s)
    out = Path(reports_dir) / task
    if not complete:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{'full_cv' if complete else 'smoke'}] reports={out}", flush=True)
    rows = []
    for fold in folds:
        _, x_tr, x_va, y_tr, y_va = _prepare(matrix, target, partitions, "train", "validation", fold, s, max_train_rows)
        print(f"[fold {fold}] train={len(x_tr)} validation={len(x_va)}", flush=True)
        for name in ("baseline", *model_names):
            for config_id, params in enumerate(CONFIGS[task][name]):
                model = build_model(name, task, params, s.random_state)
                start = perf_counter()
                model.fit(x_tr.to_numpy(), y_tr.to_numpy())
                fit_seconds = perf_counter() - start
                metrics = evaluate(task, y_va.to_numpy(), model.score(x_va.to_numpy()))
                rows.append({
                    "model": name, "config_id": config_id, "params": json.dumps(params, sort_keys=True),
                    "fold": fold, **metrics, "fit_seconds": round(fit_seconds, 2),
                })
                primary, _ = PRIMARY[task]
                print(f"  {name:8s} cfg{config_id} {primary}={metrics[primary]:.4f} ({fit_seconds:.1f}s)", flush=True)

    per_fold = pd.DataFrame(rows)
    metric_cols = [c for c in per_fold.columns if c not in ("model", "config_id", "params", "fold")]
    summary = per_fold.groupby(["model", "config_id", "params"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    primary, higher_is_better = PRIMARY[task]
    best = {}
    for name in model_names:
        candidates = summary[summary["model"] == name]
        pick = candidates[f"{primary}_mean"].idxmax() if higher_is_better else candidates[f"{primary}_mean"].idxmin()
        best[name] = {"config_id": int(candidates.loc[pick, "config_id"]), "params": json.loads(candidates.loc[pick, "params"])}

    with TemporaryDirectory(prefix=".cv-", dir=out) as temporary:
        staging = Path(temporary)
        per_fold.to_csv(staging / "cv_folds.csv", index=False)
        summary.to_csv(staging / "cv_summary.csv", index=False)
        (staging / "best_configs.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        _plot_summary(summary, task, staging / "cv_comparison.png")
        manifest = {
            "schema_version": SCHEMA_VERSION, "execution_type": "full_cv" if complete else "smoke",
            "task": task, "folds": list(folds), "models": list(model_names), "primary_metric": primary,
            "random_state": s.random_state, "max_train_rows": max_train_rows, "training_context": context,
            "sklearn": sklearn.__version__, "python": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files_sha256": {p.name: sha256(p) for p in sorted(staging.iterdir())},
        }
        (staging / "cv_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in [*manifest["files_sha256"], "cv_manifest.json"]:
            os.replace(staging / filename, out / filename)
    return summary


def run_final(
    task: str, model_names, catalog: pd.DataFrame, partitions: CatalogPartitions,
    settings: Settings | None = None, reports_dir: Path = REPORTS_DIR, artifacts_dir: Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Refit the selected configurations on all development songs and score the test set once."""
    s = settings or Settings()
    model_names = _model_names(task, model_names)
    out = Path(reports_dir) / task
    if (out / "final_manifest.json").exists():
        raise ValueError("Final supervised results already exist; preserve them instead of evaluating test again.")
    context = training_context(catalog, partitions, s)
    best = _validate_selection(out, task, model_names, context, partitions)
    selection_hashes = {name: sha256(out / name) for name in ("cv_manifest.json", "best_configs.json", "cv_summary.csv")}
    run_start = perf_counter()
    matrix = select_acoustic_features(catalog, s)
    target = build_target(task, catalog, s)
    normalizer, x_dev, x_test, y_dev, y_test = _prepare(matrix, target, partitions, "development", "test", None, s)
    print(f"[final] development={len(x_dev)} test={len(x_test)}", flush=True)
    labels = list(target.columns) if task == "genre" else None
    destination = Path(artifacts_dir) / task
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with TemporaryDirectory(prefix=".final-", dir=destination) as temporary:
        staging = Path(temporary)
        for name in ("baseline", *model_names):
            params = {} if name == "baseline" else best[name]["params"]
            model = build_model(name, task, params, s.random_state)
            start = perf_counter()
            model.fit(x_dev.to_numpy(), y_dev.to_numpy())
            fit_seconds = perf_counter() - start
            metrics = evaluate(task, y_test.to_numpy(), model.score(x_test.to_numpy()))
            rows.append({"model": name, "params": json.dumps(params, sort_keys=True), **metrics, "fit_seconds": round(fit_seconds, 2)})
            if name != "baseline":
                joblib.dump({
                    "schema_version": SCHEMA_VERSION, "stage": "final", "training_context": context,
                    "task": task, "name": name, "params": params, "model": model,
                    "features": list(s.features), "id_column": s.id_col, "labels": labels,
                    "normalizer": normalizer.to_dict(), "sklearn": sklearn.__version__,
                }, staging / f"{name}.joblib", compress=3)
            print(f"  {name:8s} test: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
        result = pd.DataFrame(rows)
        result.to_csv(staging / "test_metrics.csv", index=False)
        (staging / "final_normalizer.json").write_text(json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8")
        pd.concat([
            normalizer.transform(matrix.loc[x_dev.index]).range_report.assign(split="development"),
            normalizer.transform(matrix.loc[x_test.index]).range_report.assign(split="test"),
        ]).to_csv(staging / "final_normalization_ranges.csv", index=False)
        artifact_hashes = {path.name: sha256(path) for path in sorted(staging.glob("*.joblib"))}
        report_names = ("test_metrics.csv", "final_normalizer.json", "final_normalization_ranges.csv")
        manifest = {
            "schema_version": SCHEMA_VERSION, "execution_type": "final_test", "task": task,
            "models": list(model_names), "baseline_evaluated": True,
            "development_rows": len(x_dev), "test_rows": len(x_test),
            "normalizer_training_sha256": normalizer.training_sha256, "training_context": context,
            "selection_source_sha256": selection_hashes, "test_used_for_fitting_or_selection": False,
            "random_state": s.random_state, "sklearn": sklearn.__version__, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": perf_counter() - run_start,
            "artifacts_sha256": artifact_hashes,
            "files_sha256": {name: sha256(staging / name) for name in report_names},
            "code_sha256": {str(path.relative_to(PROJECT_ROOT / "src")): sha256(path) for path in sorted((PROJECT_ROOT / "src" / "zta").rglob("*.py"))},
        }
        (staging / "final_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in artifact_hashes:
            os.replace(staging / name, destination / name)
        for name in (*report_names, "final_manifest.json"):
            os.replace(staging / name, out / name)
    return result


def _plot_summary(summary: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric, higher_is_better = PRIMARY[task]
    pick = summary.groupby("model")[f"{metric}_mean"].agg("idxmax" if higher_is_better else "idxmin")
    best = summary.loc[pick.values].sort_values(f"{metric}_mean", ascending=not higher_is_better)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(best["model"], best[f"{metric}_mean"], yerr=best[f"{metric}_std"].fillna(0), capsize=4)
    ax.set_ylabel(f"{metric} (validation, mean ± std over folds)")
    ax.set_title(f"Model comparison — task: {task}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
