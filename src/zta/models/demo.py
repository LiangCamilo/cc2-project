"""Try the application layer from the terminal.

  python -m zta.models.demo --danceability 0.8 --energy 0.7 --acousticness 0.1
  python -m zta.models.demo --energy 0.2 --acousticness 0.9 --mode familiar --genres jazz blues
  python -m zta.models.demo --seeds <track_id> <track_id>
"""

import argparse

import pandas as pd

from zta.config import Settings
from zta.data import CsvCatalogSource, load_partitions
from zta.domain import ListeningMode

from .app import MusicRecommender


def main() -> None:
    s = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in s.features:
        parser.add_argument(f"--{name}", type=float, metavar="0-1", help=f"slider for {name}")
    parser.add_argument("--seeds", nargs="+", help="Build the profile from liked songs instead of sliders")
    parser.add_argument("--mode", choices=[m.value for m in ListeningMode], default="discovery")
    parser.add_argument("--metric", choices=["euclidean", "mahalanobis"], default="mahalanobis")
    parser.add_argument("--genres", nargs="+", help="Only recommend songs of these genres")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    catalog = CsvCatalogSource(settings=s).load()
    app = MusicRecommender(catalog, load_partitions(catalog, settings=s), s)
    if args.seeds:
        profile = app.profile_from_songs(args.seeds)
    else:
        profile = app.profile_from_sliders(**{n: getattr(args, n) for n in s.features if getattr(args, n) is not None})

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 38)
    print("\n== Perfil (slider 0-1 y valor real) ==\n", app.describe_profile(profile), sep="")
    print(f"\nPopularidad esperada para este sonido: {app.expected_popularity(profile):.0f}/100 (orientativa)")
    print("\n== Géneros que encajan (por modelo) ==")
    print(app.suggest_genres(profile, top=3).pivot(index="rank", columns="model", values="genre")[["rf", "knn", "svm", "linear"]])
    print(f"\n== Recomendación ZTA (modo {args.mode}, distancia {args.metric}) ==")
    print(app.recommend(profile, args.mode, args.k, args.metric, args.genres).drop(columns="track_id").to_string(index=False))
    print("\n== Vecinos más cercanos (KNN) ==")
    print(app.nearest_songs(profile, args.k, args.genres).drop(columns="track_id").to_string(index=False))


if __name__ == "__main__":
    main()
