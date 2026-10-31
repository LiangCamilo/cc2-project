"""Train and compare linear regression, KNN, random forest and SVM.

  python -m zta.models --task genre --mode cv
  python -m zta.models --task genre --mode final --confirm-test
"""

import argparse
from pathlib import Path

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions
from .registry import MODEL_NAMES, TASKS
from .runner import run_cv, run_final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, default="genre")
    parser.add_argument("--mode", choices=("cv", "final"), default="cv")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Default: all folds; subsets are saved separately as smoke runs")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Marks a separate smoke run, ineligible for final selection (cv mode)")
    parser.add_argument("--confirm-test", action="store_true", help="Required for --mode final; permits evaluation on the reserved test set")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--track-genres", type=Path)
    args = parser.parse_args()
    if args.mode == "final" and not args.confirm_test:
        parser.error("--mode final evaluates on the reserved test set; add --confirm-test when the CV decisions are closed.")
    if args.mode == "final" and (args.folds is not None or args.max_train_rows is not None):
        parser.error("--folds and --max-train-rows are only valid in cv mode.")
    settings = Settings()
    try:
        catalog = CsvCatalogSource(args.tracks, args.track_genres, settings).load()
        partitions = load_partitions(catalog, settings=settings)
        folds = args.folds if args.folds is not None else list(range(1, partitions.n_folds + 1))
        if args.mode == "cv":
            summary = run_cv(args.task, args.models, folds, catalog, partitions, settings, max_train_rows=args.max_train_rows)
        else:
            summary = run_final(args.task, args.models, catalog, partitions, settings)
        print(summary.to_string(index=False))
    except (CatalogContractError, ValueError, OSError) as error:
        parser.exit(status=1, message=f"Model training failed: {error}\n")


if __name__ == "__main__":
    main()
