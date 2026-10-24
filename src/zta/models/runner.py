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
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from zta.config import PROJECT_ROOT, Settings
from zta.data import CatalogPartitions, select_acoustic_features
from zta.preprocessing import MinMaxNormalizer

from .evaluation import PRIMARY, evaluate
from .registry import CONFIGS, build_model
from .targets import build_target

REPORTS_DIR = PROJECT_ROOT / "reports" / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "models"


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
    matrix = select_acoustic_features(catalog, s)
    target = build_target(task, catalog, s)
    out = Path(reports_dir) / task
    out.mkdir(parents=True, exist_ok=True)
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

    per_fold.to_csv(out / "cv_folds.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)
    (out / "best_configs.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    (out / "cv_manifest.json").write_text(json.dumps({
        "task": task, "folds": list(folds), "models": list(model_names), "primary_metric": primary,
        "random_state": s.random_state, "max_train_rows": max_train_rows,
        "sklearn": sklearn.__version__, "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    _plot_summary(summary, task, out / "cv_comparison.png")
    return summary


def run_final(
    task: str, model_names, catalog: pd.DataFrame, partitions: CatalogPartitions,
    settings: Settings | None = None, reports_dir: Path = REPORTS_DIR, artifacts_dir: Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Refit the selected configurations on all development songs and score the test set once."""
    s = settings or Settings()
    out = Path(reports_dir) / task
    best_path = out / "best_configs.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} not found. Run --mode cv first; test data is never used to choose.")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    matrix = select_acoustic_features(catalog, s)
    target = build_target(task, catalog, s)
    normalizer, x_dev, x_test, y_dev, y_test = _prepare(matrix, target, partitions, "development", "test", None, s)
    print(f"[final] development={len(x_dev)} test={len(x_test)}", flush=True)
    labels = list(target.columns) if task == "genre" else None
    destination = Path(artifacts_dir) / task
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in model_names:
        params = best[name]["params"]
        model = build_model(name, task, params, s.random_state)
        start = perf_counter()
        model.fit(x_dev.to_numpy(), y_dev.to_numpy())
        fit_seconds = perf_counter() - start
        metrics = evaluate(task, y_test.to_numpy(), model.score(x_test.to_numpy()))
        rows.append({"model": name, "params": json.dumps(params, sort_keys=True), **metrics, "fit_seconds": round(fit_seconds, 2)})
        joblib.dump({
            "task": task, "name": name, "params": params, "model": model,
            "features": list(s.features), "id_column": s.id_col, "labels": labels,
            "normalizer": normalizer.to_dict(), "sklearn": sklearn.__version__,
        }, destination / f"{name}.joblib", compress=3)
        print(f"  {name:8s} test: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    result = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "test_metrics.csv", index=False)
    (out / "final_manifest.json").write_text(json.dumps({
        "task": task, "development_rows": len(x_dev), "test_rows": len(x_test),
        "normalizer_training_sha256": normalizer.training_sha256,
        "random_state": s.random_state, "sklearn": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
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
