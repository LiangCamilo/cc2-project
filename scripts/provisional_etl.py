from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zta.config import PROJECT_ROOT, Settings

def main(raw: Path = PROJECT_ROOT / "data" / "raw" / "dataset.csv") -> None:
    s = Settings()
    df = pd.read_csv(raw).dropna(subset=list(s.features) + ["track_name", "artists"])
    genres = df.groupby(s.id_col)["track_genre"].agg(lambda g: "|".join(sorted(set(g))))

    first = df.drop_duplicates(s.id_col).set_index(s.id_col)
    
    out = first[["track_name", "artists", s.popularity_col, *s.features]].copy()
    out[s.genres_col] = genres
    out[s.primary_genre_col] = first["track_genre"]
    out.reset_index().to_csv(s.processed_catalog, index=False)

    print(f"Provisional catalog: {len(out):,} songs -> {s.processed_catalog}")


if __name__ == "__main__":
    main()