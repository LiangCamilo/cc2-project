"""Evaluate ZTA recommendations on development folds; keep test reserved."""

import argparse
from dataclasses import replace
from pathlib import Path

from zta.config import Settings
from zta.data import CatalogContractError, CsvCatalogSource, load_partitions

from .runner import REPORTS_DIR, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", help="Default: all development folds; subsets are smoke runs")
    parser.add_argument("--genres", nargs="+", help="Restrict genres for smoke checks only")
    parser.add_argument("--profiles-per-genre", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--popularity-lambda", type=float, help="Optional fixed within-zone popularity penalty")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    s = Settings()
    if args.popularity_lambda is not None:
        s = replace(s, popularity_lambda=args.popularity_lambda)
    try:
        catalog = CsvCatalogSource(settings=s).load()
        partitions = load_partitions(catalog, settings=s)
        run_evaluation(
            catalog, partitions, s, folds=tuple(args.folds) if args.folds is not None else None,
            profiles_per_genre=args.profiles_per_genre, seed_count=args.seed_count, k=args.k,
            genres=tuple(args.genres) if args.genres is not None else None, reports_dir=args.reports_dir,
        )
    except (CatalogContractError, OSError) as error:
        parser.exit(1, f"Recommendation evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
