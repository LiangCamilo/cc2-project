"""Evaluate ZTA development folds or the frozen selection on reserved test."""

import argparse
from dataclasses import replace
from pathlib import Path

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions

from .runner import REPORTS_DIR, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cv", "final"), default="cv")
    parser.add_argument("--confirm-test", action="store_true", help="Required for final evaluation on the reserved test set")
    parser.add_argument("--folds", type=int, nargs="+", help="Default: all development folds; subsets are smoke runs")
    parser.add_argument("--genres", nargs="+", help="Restrict genres for smoke checks only")
    parser.add_argument("--profiles-per-genre", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--popularity-lambda", type=float, help="Optional fixed within-zone popularity penalty")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    if args.mode == "final" and not args.confirm_test:
        parser.error("Final evaluation uses reserved test; add --confirm-test after closing development decisions.")
    if args.mode == "final" and (
        args.folds is not None or args.genres is not None
        or (args.profiles_per_genre, args.seed_count, args.k) != (5, 5, 10)
    ):
        parser.error("Final evaluation uses the complete frozen protocol; reduced profile options are only valid in cv mode.")
    s = Settings()
    if args.popularity_lambda is not None:
        s = replace(s, popularity_lambda=args.popularity_lambda)
    try:
        catalog = CsvCatalogSource(settings=s).load()
        partitions = load_partitions(catalog, settings=s)
        if args.mode == "final":
            from .final import run_final_evaluation
            print(run_final_evaluation(catalog, partitions, s, reports_dir=args.reports_dir).to_string(index=False))
        else:
            run_evaluation(
                catalog, partitions, s, folds=tuple(args.folds) if args.folds is not None else None,
                profiles_per_genre=args.profiles_per_genre, seed_count=args.seed_count, k=args.k,
                genres=tuple(args.genres) if args.genres is not None else None, reports_dir=args.reports_dir,
            )
    except (CatalogContractError, OSError) as error:
        parser.exit(1, f"Recommendation evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
