"""Application layer: sliders -> profile -> genre fit and song recommendations.

Function 1  "Build my profile"
    The user sets the nine acoustic sliders (0-1). The four trained genre models
    say which genres fit that profile; the popularity forest says how mainstream
    such a sound usually is.
Function 2  "Recommend songs"
    * recommend()      -> ZTA (team's mathematical model): tolerance zones
                          (familiar / discovery / exploration) and ranking.
    * nearest_songs()  -> KNN: the k closest songs to the profile.

All models share the same Min-Max normalizer (fitted on the development songs),
so a profile is just nine numbers in [0, 1] that every model understands.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from zta.config import Settings
from zta.data import CatalogPartitions, select_acoustic_features
from zta.domain import (
    AcousticDistanceModel, AcousticProfile, ListeningMode, build_acoustic_profile,
    calibrate_zones, rank_zone,
)
from zta.domain.profiles import normalization_fingerprint

from .inference import load_artifact
from .artifacts import training_context, validate_bundle
from .registry import MODEL_NAMES
from .runner import ARTIFACTS_DIR

MANUAL_PROFILE_ID = "manual_profile"
COLUMNS = ["rank", "track_id", "track_name", "artists", "genres", "popularity", "distance"]


class MusicRecommender:
    def __init__(
        self, catalog: pd.DataFrame, partitions: CatalogPartitions,
        settings: Settings | None = None, artifacts_dir: Path = ARTIFACTS_DIR,
    ):
        s = self.settings = settings or Settings()
        self.catalog = catalog
        root = Path(artifacts_dir)
        self.genre_models = {name: load_artifact(root / "genre" / f"{name}.joblib") for name in MODEL_NAMES}
        self.popularity_model = load_artifact(root / "popularity" / "rf.joblib")
        context = training_context(catalog, partitions, s)
        self.normalizer = validate_bundle(self.genre_models, self.popularity_model, context)
        self.genres = list(self.genre_models["rf"]["labels"])

        self._raw = select_acoustic_features(catalog, s)
        development = partitions.select_features(self._raw, "development", None)
        # Fails loudly if the saved models were trained on a different dataset/partition.
        self.normalizer.validate_training(development)
        self._development = development
        self.distance_model = AcousticDistanceModel.fit(development, self.normalizer, s.mahalanobis_regularization)
        self._normalized = self.normalizer.transform(self._raw).matrix
        self._development_mean = self.normalizer.transform(development).matrix.mean()
        self._info = catalog.set_index(s.id_col)[["track_name", "artists", s.genres_col, s.popularity_col]]

    # ------------------------------------------------------------------ profile
    def profile_from_sliders(self, **sliders: float) -> AcousticProfile:
        """Each slider is a value in [0, 1] (position inside the dataset's range).

        Sliders left out take the dataset average, i.e. "no preference".
        """
        features = self.normalizer.features
        unknown = sorted(set(sliders) - set(features))
        if unknown:
            raise ValueError(f"Unknown sliders {unknown}. Valid names: {list(features)}")
        centroid = {name: float(self._development_mean[name]) for name in features}
        for name, value in sliders.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Slider '{name}' must be a number between 0 and 1, got {value!r}.")
            centroid[name] = float(value)
        return AcousticProfile(
            features=tuple(features), centroid=tuple(centroid[name] for name in features),
            seed_ids=(MANUAL_PROFILE_ID,), normalization_sha256=normalization_fingerprint(self.normalizer),
        )

    def profile_from_songs(self, seed_ids: Sequence[str]) -> AcousticProfile:
        """Profile = average of liked songs (the team's definition)."""
        return build_acoustic_profile(self._raw, list(seed_ids), self.normalizer)

    def describe_profile(self, profile: AcousticProfile) -> pd.DataFrame:
        """Slider value and the same value in real units (e.g. BPM, dB)."""
        low, high = np.asarray(self.normalizer.minimum, float), np.asarray(self.normalizer.maximum, float)
        slider = np.asarray(profile.centroid, float)
        return pd.DataFrame({"slider_0_1": slider.round(3), "real_value": (low + slider * (high - low)).round(3)},
                            index=list(profile.features))

    def feature_info(self) -> pd.DataFrame:
        """Return slider ranges and default positions for the API metadata."""
        features = self.normalizer.features
        return pd.DataFrame({
            "feature": features,
            "min": self.normalizer.minimum,
            "max": self.normalizer.maximum,
            "default_slider": self._development_mean.reindex(features).to_numpy(),
        })

    # --------------------------------------------------------- function 1: genres
    def suggest_genres(self, profile: AcousticProfile, top: int = 5) -> pd.DataFrame:
        """Top genres per model. Random Forest is the best-validated one.

        Scores are only comparable inside one model (forest/KNN ~ probability-like,
        SVM = margin distance, linear = unbounded), so read them as rankings.
        """
        x = np.asarray(profile.centroid, dtype=float).reshape(1, -1)
        rows = []
        for name, artifact in self.genre_models.items():
            scores = pd.Series(artifact["model"].score(x)[0], index=self.genres)
            for rank, (genre, value) in enumerate(scores.nlargest(top).items(), start=1):
                rows.append({"model": name, "rank": rank, "genre": genre, "score": round(float(value), 4)})
        return pd.DataFrame(rows)

    def expected_popularity(self, profile: AcousticProfile) -> float:
        """Popularity (0-100) a song with this sound usually has. Modest accuracy (R2 ~ 0.22)."""
        x = np.asarray(profile.centroid, dtype=float).reshape(1, -1)
        return float(np.clip(self.popularity_model["model"].score(x)[0], 0, 100))

    # --------------------------------------------------------- function 2: songs
    def _allowed(self, genres: Sequence[str] | None) -> pd.Index | None:
        if not genres:
            return None
        wanted = set(genres)
        unknown = sorted(wanted - set(self.genres))
        if unknown:
            raise ValueError(f"Unknown genres: {unknown}")
        keep = self.catalog[self.settings.genres_col].map(lambda tags: bool(wanted & set(tags)))
        return pd.Index(self.catalog.loc[keep, self.settings.id_col])

    def recommend(
        self, profile: AcousticProfile, mode: str | ListeningMode = "discovery", k: int = 10,
        metric: str = "mahalanobis", genres: Sequence[str] | None = None,
        popularity_lambda: float | None = None, dedupe: bool = True,
    ) -> pd.DataFrame:
        """ZTA: keep songs inside the chosen tolerance zone, rank by distance (and optional popularity penalty)."""
        s = self.settings
        reference = self.distance_model.distances(profile, self._development)[metric]
        zones = calibrate_zones(reference, s)
        candidates = self.distance_model.distances(profile, self._raw)[metric]
        allowed = self._allowed(genres)
        if allowed is not None:
            candidates = candidates.loc[candidates.index.isin(allowed)]
        # When deduplication is requested, rank the full eligible zone first so
        # repeated releases do not leave the response with fewer than k songs.
        result = rank_zone(candidates, self.catalog, zones, mode,
                           max(1, len(candidates)) if dedupe else k,
                           seed_ids=profile.seed_ids, popularity_lambda=popularity_lambda, settings=s)
        table = result.table.rename(columns={s.genres_col: "genres", s.popularity_col: "popularity"})
        if dedupe:
            table = table.drop_duplicates(subset=["track_name", "artists"], keep="first").head(k).copy()
            table["rank"] = np.arange(1, len(table) + 1)
        return table[COLUMNS].assign(zone=result.mode.value)

    def search_songs(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Find catalog songs whose title or artist contains the query (case-insensitive)."""
        import re

        if not isinstance(query, str) or len(query.strip()) < 2:
            raise ValueError("Search query must contain at least 2 characters.")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Search limit must be a positive integer.")
        s = self.settings
        q = re.escape(query.strip())
        mask = (self.catalog["track_name"].astype("string").str.contains(q, case=False, na=False) |
                self.catalog["artists"].astype("string").str.contains(q, case=False, na=False))
        columns = [s.id_col, "track_name", "artists", s.genres_col, s.popularity_col]
        found = self.catalog.loc[mask, columns].copy()
        found = found.sort_values([s.popularity_col, "track_name"], ascending=[False, True], kind="stable").head(limit)
        return found.rename(columns={s.id_col: "track_id", s.genres_col: "genres", s.popularity_col: "popularity"}).reset_index(drop=True)

    def nearest_songs(
        self, profile: AcousticProfile, k: int = 10, genres: Sequence[str] | None = None,
        dedupe: bool = True,
    ) -> pd.DataFrame:
        """KNN: the k songs closest (Euclidean, normalized space) to the profile."""
        s = self.settings
        space = self._normalized
        allowed = self._allowed(genres)
        if allowed is not None:
            space = space.loc[space.index.isin(allowed)]
        seeds = [i for i in profile.seed_ids if i in space.index]
        if space.empty:
            return pd.DataFrame(columns=COLUMNS)
        # Query all candidates so deduplication can still return k unique tracks.
        index = NearestNeighbors(n_neighbors=len(space)).fit(space.to_numpy())
        distance, position = index.kneighbors(np.asarray(profile.centroid, float).reshape(1, -1))
        table = pd.DataFrame({"track_id": space.index[position[0]], "distance": distance[0]})
        table = table[~table["track_id"].isin(seeds)]
        info = self._info.loc[table["track_id"]].reset_index(drop=True)
        table = pd.concat([table.reset_index(drop=True), info.rename(columns={s.genres_col: "genres", s.popularity_col: "popularity"})], axis=1)
        if dedupe:
            table = table.drop_duplicates(subset=["track_name", "artists"], keep="first")
        table = table.head(k).reset_index(drop=True)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        return table[COLUMNS]
