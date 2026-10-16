"""Shared cleaning rules for the Streamlit app and reproducible exports."""

import numpy as np
import pandas as pd

from common.audit_checks import TEXT_COLUMNS


def remove_incomplete_records(data):
    mask = (
        data[["artists", "album_name", "track_name"]].isna().all(axis=1)
        & data["duration_ms"].eq(0)
    )
    return data.loc[~mask].copy(), data.loc[mask].copy()


def clean_text_fields(data):
    cleaned = data.copy()
    counts, changes = [], []
    for column in TEXT_COLUMNS:
        text = data[column].astype("string")
        stripped = text.str.strip()
        changed = (text.notna() & text.ne(stripped)).fillna(False)
        counts.append({"Column": column, "Values corrected": int(changed.sum())})
        if changed.any():
            cleaned.loc[changed, column] = stripped.loc[changed]
            changes.append(pd.DataFrame({
                "Source index": data.index[changed],
                "track_id": data.loc[changed, "track_id"].to_numpy(),
                "Column": column,
                "Before": text.loc[changed].map(repr).to_numpy(),
                "After": stripped.loc[changed].map(repr).to_numpy(),
            }))
    change_log = (
        pd.concat(changes, ignore_index=True) if changes else
        pd.DataFrame(columns=["Source index", "track_id", "Column", "Before", "After"])
    )
    return cleaned, pd.DataFrame(counts), change_log


def build_track_genres(data):
    return (
        data[["track_id", "track_genre"]].drop_duplicates()
        .sort_values(["track_id", "track_genre"]).reset_index(drop=True)
    )


def consolidate_tracks(data):
    popularity = (
        data[["track_id", "popularity"]].drop_duplicates()
        .groupby("track_id", as_index=False)["popularity"].mean()
    )
    details = data.drop(columns=["track_genre", "popularity"]).drop_duplicates().copy()
    if details["track_id"].duplicated().any():
        raise ValueError("There are unresolved conflicts in other track fields.")
    return details.merge(popularity, on="track_id", validate="one_to_one")


def separate_unavailable_tempo(tracks):
    # Missing/non-numeric values remain for the final validator to reject explicitly.
    tempo = pd.to_numeric(tracks["tempo"], errors="coerce")
    excluded = tempo.le(0)
    return tracks.loc[~excluded].copy(), tracks.loc[excluded].copy()


def mark_time_signature(tracks):
    reviewed = tracks.copy()
    reviewed["time_signature_review"] = ~tracks["time_signature"].isin([3, 4, 5, 6, 7])
    return reviewed


def mark_loudness(tracks):
    reviewed = tracks.copy()
    loudness = pd.to_numeric(tracks["loudness"], errors="coerce")
    finite = loudness.notna() & np.isfinite(loudness)
    reviewed["loudness_review"] = finite & ~loudness.between(-60, 0)
    reviewed["loudness_invalid"] = ~finite
    return reviewed
