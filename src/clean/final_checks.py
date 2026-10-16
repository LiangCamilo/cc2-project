"""Checks for the consolidated catalog; review flags are distinct from failures."""

import numpy as np
import pandas as pd

from common.audit_checks import NUMERIC_COLUMNS, TEXT_COLUMNS, UNIT_INTERVAL_COLUMNS


def inspect_distributions(tracks):
    statistics = []
    flags = pd.DataFrame({"track_id": tracks["track_id"]})
    for column in NUMERIC_COLUMNS:
        if column in ("key", "mode", "time_signature"):
            continue
        values = pd.to_numeric(tracks[column], errors="coerce")
        values = values.where(np.isfinite(values))
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        extreme = (values.lt(lower) | values.gt(upper)).fillna(False)
        flags[column + "_iqr_review"] = extreme
        statistics.append({
            "variable": column, "finite_count": int(values.count()),
            "min": values.min(), "p01": values.quantile(0.01),
            "q1": q1, "median": values.median(), "q3": q3,
            "p99": values.quantile(0.99), "max": values.max(),
            "iqr_lower": lower, "iqr_upper": upper,
            "flagged_tracks": int(extreme.sum()),
            "flagged_pct": float(extreme.mean() * 100) if len(tracks) else 0.0,
        })
    return pd.DataFrame(statistics), flags


def compare_genres(source, tracks, track_genres):
    before = source.groupby("track_genre")["track_id"].nunique()
    after = track_genres.groupby("track_genre")["track_id"].nunique()
    report = pd.DataFrame({"tracks_before": before, "tracks_after": after}).fillna(0).astype(int)
    report["tracks_excluded"] = report["tracks_before"] - report["tracks_after"]
    report["retention_pct"] = report["tracks_after"].div(report["tracks_before"]).mul(100)
    report["catalog_pct"] = report["tracks_after"].div(len(tracks) or 1).mul(100)
    return report.rename_axis("track_genre").reset_index().sort_values(
        "tracks_after", ascending=False, kind="stable"
    ).reset_index(drop=True)


def validate_catalog(tracks, track_genres, source_relations, excluded_tracks):
    checks = []

    def add(name, affected, detail, review=False):
        affected = int(affected)
        checks.append({
            "check": name, "affected": affected,
            "status": ("review" if review else "fail") if affected else "pass",
            "detail": detail,
        })

    required = set(NUMERIC_COLUMNS + [c for c in TEXT_COLUMNS if c != "track_genre"] + [
        "explicit", "time_signature_review", "loudness_review", "loudness_invalid",
    ])
    missing = sorted(required.difference(tracks.columns))
    add("track_schema", len(missing), "Missing columns: " + ", ".join(missing))
    relation_missing = sorted({"track_id", "track_genre"}.difference(track_genres.columns))
    add("genre_schema", len(relation_missing), "Missing columns: " + ", ".join(relation_missing))
    if missing or relation_missing:
        return pd.DataFrame(checks)

    add("empty_catalog", not len(tracks), "The catalog must contain tracks.")
    add("unique_track_ids", tracks["track_id"].duplicated().sum(), "One row per track_id.")
    add("unique_genre_pairs", track_genres.duplicated(["track_id", "track_genre"]).sum(),
        "One row per track_id/track_genre pair.")
    for frame, columns, prefix in [
        (tracks, [c for c in TEXT_COLUMNS if c != "track_genre"], "tracks"),
        (track_genres, ["track_id", "track_genre"], "genres"),
    ]:
        for column in columns:
            text = frame[column].astype("string")
            add(prefix + ":" + column + ":missing_or_blank",
                (text.isna() | text.str.strip().eq("")).sum(), "Required non-blank text.")
            add(prefix + ":" + column + ":surrounding_whitespace",
                (text.notna() & text.ne(text.str.strip())).sum(), "Outer spaces removed; internal text preserved.")

    ids = set(tracks["track_id"])
    excluded_ids = set(excluded_tracks["track_id"].dropna())
    add("orphan_genre_associations", (~track_genres["track_id"].isin(ids)).sum(),
        "Every association must refer to a retained track.")
    add("tracks_without_genres", (~tracks["track_id"].isin(track_genres["track_id"])).sum(),
        "Every retained track must have at least one genre.")
    add("retained_and_excluded_overlap", len(ids & excluded_ids), "Retained and excluded IDs must be disjoint.")
    source_ids = set(source_relations["track_id"].dropna())
    add("track_identity_reconciliation", len(source_ids.symmetric_difference(ids | excluded_ids)),
        "Every source track must be retained or explicitly excluded.")
    expected = source_relations.loc[source_relations["track_id"].isin(ids), ["track_id", "track_genre"]]
    expected_pairs = pd.MultiIndex.from_frame(expected.drop_duplicates())
    actual_pairs = pd.MultiIndex.from_frame(track_genres[["track_id", "track_genre"]].drop_duplicates())
    add("genre_association_reconciliation",
        len(expected_pairs.difference(actual_pairs)) + len(actual_pairs.difference(expected_pairs)),
        "Preserve every original genre association of every retained song.")

    numeric = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(tracks[column], errors="coerce")
        numeric[column] = values
        add(column + ":missing_or_invalid", (~np.isfinite(values)).sum(), "Finite numeric values required.")
    for column in UNIT_INTERVAL_COLUMNS + ["popularity"]:
        upper = 100 if column == "popularity" else 1
        add(column + ":domain", (~numeric[column].between(0, upper)).sum(), f"Values in [0, {upper}].")
    for column in ["duration_ms", "tempo"]:
        add(column + ":positive", numeric[column].le(0).sum(), "Strictly positive values required.")
    for column in ["duration_ms", "time_signature"]:
        add(column + ":integer", numeric[column].mod(1).ne(0).sum(), "Integer representation required.")
    for column, allowed in {"key": range(-1, 12), "mode": [0, 1]}.items():
        add(column + ":categories", (~numeric[column].isin(allowed)).sum(), "Documented category codes.")
    add("explicit:boolean", (~tracks["explicit"].isin([True, False])).sum(), "Boolean explicit flag.")
    # Popularity is intentionally allowed to be fractional after consolidation.
    expected_flags = {
        "time_signature_review": ~tracks["time_signature"].isin([3, 4, 5, 6, 7]),
        "loudness_review": np.isfinite(numeric["loudness"]) & ~numeric["loudness"].between(-60, 0),
        "loudness_invalid": ~np.isfinite(numeric["loudness"]),
    }
    for column, expected_flag in expected_flags.items():
        add(column + ":consistency", (~tracks[column].eq(expected_flag).fillna(False)).sum(),
            "Review flag matches its rule and is not missing.")
        add(column + ":boolean_type", not pd.api.types.is_bool_dtype(tracks[column]),
            "Review columns must have boolean dtype.")
    add("time_signature:documented_range", expected_flags["time_signature_review"].sum(),
        "Keep original codes; exclude time_signature and review flags from the initial distance features.", review=True)
    add("loudness:typical_range", expected_flags["loudness_review"].sum(),
        "Retain atypical finite values without clipping.", review=True)
    add("hito2:record_count", len(tracks) <= 35000,
        "The highest dataset-size rubric level requires more than 35,000 track records.")
    lost = set(source_relations["track_genre"]) - set(track_genres["track_genre"])
    add("genre_coverage", len(lost), "Genres lost after exclusions: " + ", ".join(sorted(lost)), review=True)
    return pd.DataFrame(checks)
