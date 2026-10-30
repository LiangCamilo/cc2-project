"""Create shared evaluation cases without passing relevance labels to ZTA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from zta.config import Settings
from zta.data import CatalogContractError


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CatalogContractError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class EvaluationProfile:
    profile_id: str
    fold: int | None
    genre: str
    replicate: int
    seed_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    split: str = "validation"

    def to_dict(self) -> dict:
        state = {
            "profile_id": self.profile_id, "fold": self.fold, "genre": self.genre,
            "replicate": self.replicate, "seed_ids": list(self.seed_ids),
            "relevant_ids": list(self.relevant_ids),
        }
        if self.split == "test":
            state.pop("fold")
            state["split"] = "test"
        return state


def generate_profiles(
    evaluation_catalog: pd.DataFrame,
    fold: int | None,
    settings: Settings | None = None,
    *,
    profiles_per_genre: int = 5,
    seed_count: int = 5,
    genres: tuple[str, ...] | None = None,
    genre_universe: tuple[str, ...] | None = None,
    split: str = "validation",
) -> tuple[tuple[EvaluationProfile, ...], pd.DataFrame]:
    """Sample sorted song IDs, independently by fold, genre and replicate.

    Candidates are every evaluation song except the five seeds. All remaining
    songs carrying the chosen genre are relevant, including multi-label songs.
    A reduced genre selection retains exactly the same cases as a complete run.
    """
    s = settings or Settings()
    if split not in ("validation", "test") or (split == "test" and fold is not None):
        raise CatalogContractError("Test profiles use split='test' without a validation fold.")
    if split == "validation":
        positive_integer(fold, "fold")
    for value, name in ((profiles_per_genre, "profiles_per_genre"), (seed_count, "seed_count")):
        positive_integer(value, name)
    if isinstance(s.random_state, bool) or not isinstance(s.random_state, int) or s.random_state < 0:
        raise CatalogContractError("random_state must be a nonnegative integer.")
    if (
        not isinstance(evaluation_catalog, pd.DataFrame) or evaluation_catalog.empty
        or evaluation_catalog.columns.has_duplicates
        or not {s.id_col, s.genres_col}.issubset(evaluation_catalog.columns)
    ):
        raise CatalogContractError("Profiles require evaluation song IDs and genre collections.")
    ids = evaluation_catalog[s.id_col]
    if ids.duplicated().any() or any(
        not isinstance(value, str) or not value or value != value.strip() for value in ids
    ):
        raise CatalogContractError("Evaluation song IDs must be distinct, nonblank strings.")
    groups: dict[str, list[str]] = {}
    for track_id, labels in evaluation_catalog[[s.id_col, s.genres_col]].itertuples(index=False, name=None):
        if not isinstance(labels, (tuple, list)) or not labels or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise CatalogContractError("Evaluation genres must be nonempty label collections.")
        for label in set(labels):
            groups.setdefault(label, []).append(track_id)
    universe = tuple(sorted(set(genre_universe) if genre_universe is not None else groups))
    if genres is not None and (
        not isinstance(genres, tuple) or not genres or len(set(genres)) != len(genres)
        or any(genre not in universe for genre in genres)
    ):
        raise CatalogContractError("Choose distinct genres from the catalog's genre universe.")
    chosen = tuple(sorted(genres)) if genres is not None else universe
    cases, eligibility = [], []
    for genre in chosen:
        songs = tuple(sorted(groups.get(genre, ())))
        eligible = len(songs) >= seed_count + 1
        eligibility.append({
            **({"split": "test"} if split == "test" else {"fold": fold}),
            "genre": genre, "songs": len(songs), "eligible": eligible,
            "profiles": profiles_per_genre if eligible else 0,
            "reason": "eligible" if eligible else "insufficient_songs_for_seeds_and_target",
        })
        if not eligible:
            continue
        for replicate in range(1, profiles_per_genre + 1):
            # Preserve every existing CV sample; test has its own RNG namespace.
            namespace = "test" if split == "test" else fold
            key = json.dumps([s.random_state, namespace, genre, replicate], ensure_ascii=False).encode("utf-8")
            digest = hashlib.sha256(key).hexdigest()
            rng = np.random.default_rng(int(digest, 16))
            seeds = tuple(sorted(str(value) for value in rng.choice(songs, size=seed_count, replace=False)))
            seed_set = set(seeds)
            cases.append(EvaluationProfile(
                profile_id=f"test_{digest[:24]}" if split == "test" else f"fold_{fold}_{digest[:24]}",
                fold=fold, genre=genre, split=split,
                replicate=replicate, seed_ids=seeds,
                relevant_ids=tuple(value for value in songs if value not in seed_set),
            ))
    return tuple(cases), pd.DataFrame(eligibility)
