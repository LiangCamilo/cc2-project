"""Casos de prueba 1 y 2 como tablas Markdown para el informe.

  python -m zta.models.casos_prueba [--full-rows 60916]

Caso 1: entrenamiento reducido (corrida rápida: `--folds 1 --max-train-rows 5000`, carpeta `smoke/`).
Caso 2: entrenamiento con el dataset real completo (validación cruzada completa).
Ambos casos se miden en el MISMO fold 1 y sobre las MISMAS canciones de validación; el conjunto de prueba no se toca.
En cada caso se toma, para cada modelo, la variante con mejor métrica principal en ese fold.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluation import PRIMARY
from .runner import REPORTS_DIR

METRICS = {"genre": ["lrap", "macro_ap", "top1_hit", "top5_hit"], "popularity": ["rmse", "mae", "r2"]}
NAMES = {"baseline": "Referencia (media)", "rf": "Random Forest", "knn": "KNN", "svm": "SVM", "linear": "Regresión lineal"}
TASK_TITLE = {"genre": "Géneros (114 etiquetas)", "popularity": "Popularidad (0-100)"}


def _best_fold1(path: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["fold"] == 1]
    primary, higher_is_better = PRIMARY[task]
    pick = frame.groupby("model")[primary].agg("idxmax" if higher_is_better else "idxmin")
    best = frame.loc[pick.values].set_index("model")
    return best.sort_values(primary, ascending=not higher_is_better)


def _table(frame: pd.DataFrame, task: str) -> str:
    cols = METRICS[task]
    lines = ["| Posición | Modelo | " + " | ".join(cols) + " | Ajuste (s) |", "|---|---|" + "---|" * (len(cols) + 1)]
    for i, (model, row) in enumerate(frame.iterrows(), start=1):
        lines.append(f"| {i} | {NAMES[model]} | " + " | ".join(f"{row[c]:.4f}" for c in cols) + f" | {row['fit_seconds']:.2f} |")
    return "\n".join(lines)


def _comparison(small: pd.DataFrame, full: pd.DataFrame, task: str) -> str:
    primary, higher_is_better = PRIMARY[task]
    rank_small = {m: i for i, m in enumerate(small.index, start=1)}
    rank_full = {m: i for i, m in enumerate(full.index, start=1)}
    lines = [f"| Modelo | {primary} (Caso 1) | {primary} (Caso 2) | Cambio | Posición C1 → C2 | Tiempo de ajuste C1 → C2 (s) |", "|---|---|---|---|---|---|"]
    for model in full.index:
        a, b = small.loc[model, primary], full.loc[model, primary]
        change = (b - a) / abs(a) * 100 if a else float("nan")
        better = (b > a) if higher_is_better else (b < a)
        lines.append(
            f"| {NAMES[model]} | {a:.4f} | {b:.4f} | {change:+.1f} % ({'mejora' if better else 'empeora'}) | "
            f"{rank_small[model]} → {rank_full[model]} | {small.loc[model, 'fit_seconds']:.2f} → {full.loc[model, 'fit_seconds']:.2f} |"
        )
    return "\n".join(lines)


def _n(value: int) -> str:
    return f"{value:,}".replace(",", ".")  # separador de miles en español


def build_markdown(reports_dir: Path = REPORTS_DIR, full_rows: int | None = 60916) -> str:
    parts = []
    for task in ("genre", "popularity"):
        out = Path(reports_dir) / task
        small_path, full_path = out / "smoke" / "cv_folds.csv", out / "cv_folds.csv"
        for path in (small_path, full_path):
            if not path.exists():
                raise FileNotFoundError(f"Falta {path}. Corre la validación cruzada completa y la corrida rápida de {task}.")
        small, full = _best_fold1(small_path, task), _best_fold1(full_path, task)
        manifest = json.loads((out / "smoke" / "cv_manifest.json").read_text(encoding="utf-8"))
        small_rows = manifest.get("max_train_rows")
        primary, higher = PRIMARY[task]
        parts += [
            f"### {TASK_TITLE[task]}",
            f"**Caso 1** (entrenamiento con {_n(small_rows)} canciones, fold 1; métrica principal: {primary}, {'mayor' if higher else 'menor'} es mejor):",
            "", _table(small, task), "",
            f"**Caso 2** (entrenamiento con el dataset real, fold 1{f', {_n(full_rows)} canciones' if full_rows else ''}; mismas canciones de validación):",
            "", _table(full, task), "",
            "**Comparación entre casos**", "", _comparison(small, full, task), "",
        ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-rows", type=int, default=60916, help="Canciones de entrenamiento del fold 1 en el Caso 2")
    args = parser.parse_args()
    text = build_markdown(full_rows=args.full_rows)
    (REPORTS_DIR / "casos_prueba.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
