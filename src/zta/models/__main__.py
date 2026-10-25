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
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Default: all folds")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Quick smoke runs only (cv mode)")
    parser.add_argument("--confirm-test", action="store_true", help="Required for --mode final (uses the test set once)")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--track-genres", type=Path)
    args = parser.parse_args()
    if args.mode == "final" and not args.confirm_test:
        parser.error("--mode final evaluates on the reserved test set; add --confirm-test when the CV decisions are closed.")
    settings = Settings()
    try:
        catalog = CsvCatalogSource(args.tracks, args.track_genres, settings).load()
        partitions = load_partitions(catalog, settings=settings)
    except CatalogContractError as error:
        parser.exit(status=1, message=f"Cannot prepare data: {error}\n")
    folds = args.folds or list(range(1, partitions.n_folds + 1))
    if args.mode == "cv":
        summary = run_cv(args.task, args.models, folds, catalog, partitions, settings, max_train_rows=args.max_train_rows)
        print(summary.to_string(index=False))
    else:
        print(run_final(args.task, args.models, catalog, partitions, settings).to_string(index=False))


if __name__ == "__main__":
    main()
