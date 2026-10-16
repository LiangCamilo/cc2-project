"""Read-only dataset checks. No cleaning rules are applied to the input."""

import numpy as np
import pandas as pd


UNIT_INTERVAL_COLUMNS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]
NUMERIC_COLUMNS = [
    "popularity", "duration_ms", *UNIT_INTERVAL_COLUMNS,
    "key", "loudness", "mode", "tempo", "time_signature",
]
TEXT_COLUMNS = ["track_id", "artists", "album_name", "track_name", "track_genre"]


def audit_dataset(data):
    """Return full-data counts and masks; the UI alone limits row previews."""
    exact_extra = data.duplicated()
    exact_members = data.duplicated(keep=False)

    # Missing IDs are reported by the missing-value check, not grouped as a song.
    identified = data.loc[data["track_id"].notna()]
    grouped = identified.groupby("track_id", sort=True)
    track_counts = grouped.size()
    distinct = grouped.nunique(dropna=False)
    comparison_columns = [c for c in data.columns if c not in ("track_id", "track_genre")]
    repeated_tracks = pd.DataFrame({
        "Rows": track_counts,
        "Genres": grouped["track_genre"].nunique(),
        "Conflicting fields (excluding genre)": distinct[comparison_columns].gt(1).sum(axis=1),
    })
    repeated_tracks = repeated_tracks.loc[track_counts.gt(1)].sort_values(
        "Rows", ascending=False, kind="stable"
    ).rename_axis("track_id").reset_index()
    conflicts = distinct[comparison_columns].gt(1)
    conflict_counts = conflicts.sum().rename("Track IDs with conflicting values").rename_axis(
        "Variable"
    ).reset_index()

    # Domain checks and review heuristics remain distinct in the report.
    checks = []
    masks = {}

    def add_check(label, column, kind, mask, action):
        mask = mask.fillna(False).astype(bool)
        masks[label] = mask
        checks.append({
            "Check": label, "Variable": column, "Category": kind,
            "Affected rows": int(mask.sum()),
            "Rows (%)": round(float(mask.mean() * 100), 6) if len(data) else 0.0,
            "Proposed review": action,
        })

    numeric = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        numeric[column] = values
        add_check(
            f"{column}: non-numeric or infinite", column, "Invalid numeric representation",
            data[column].notna() & ~np.isfinite(values),
            "Inspect the source value; missing values are counted separately.",
        )

    for column in UNIT_INTERVAL_COLUMNS + ["popularity"]:
        upper = 100 if column == "popularity" else 1
        values = numeric[column]
        add_check(
            f"{column}: outside [0, {upper}]", column, "Documented domain",
            np.isfinite(values) & ~values.between(0, upper),
            "Verify against the source before correcting or excluding.",
        )

    for column, allowed in {
        "key": list(range(-1, 12)), "mode": [0, 1], "time_signature": list(range(3, 8)),
    }.items():
        values = numeric[column]
        add_check(
            f"{column}: outside documented categories", column, "Documentation discrepancy",
            np.isfinite(values) & ~values.isin(allowed),
            "Investigate the coding and documentation; do not automatically remove.",
        )

    add_check(
        "explicit: outside boolean values", "explicit", "Documented domain",
        data["explicit"].notna() & ~data["explicit"].isin([True, False]),
        "Check boolean encoding in the source.",
    )
    for column in ["popularity", "duration_ms"]:
        values = numeric[column]
        add_check(
            f"{column}: fractional value", column, "Documented integer representation",
            np.isfinite(values) & values.mod(1).ne(0),
            "Inspect values before selecting an integer representation.",
        )
    for column in ["duration_ms", "tempo"]:
        add_check(
            f"{column}: zero or negative", column, "Project review rule",
            numeric[column].le(0),
            "Investigate zero or negative values; this is not an isna() check.",
        )
    add_check(
        "key: not detected (-1)", "key", "Documented sentinel",
        numeric["key"].eq(-1), "Keep distinct from a detected pitch and from pandas missing values.",
    )
    add_check(
        "loudness: outside typical [-60, 0] dB", "loudness", "Typical range only",
        np.isfinite(numeric["loudness"]) & ~numeric["loudness"].between(-60, 0),
        "Atypical does not mean invalid; review without automatic clipping.",
    )
    for column in TEXT_COLUMNS:
        text = data[column].astype("string")
        add_check(
            f"{column}: blank text", column, "Text quality",
            text.notna() & text.str.strip().eq(""), "Inspect empty or whitespace-only text.",
        )
        add_check(
            f"{column}: surrounding whitespace", column, "Text quality",
            text.notna() & text.ne(text.str.strip()),
            "Review formatting; preserve meaningful spelling and punctuation.",
        )

    # Category codes are not treated as continuous quantities for IQR screening.
    distribution_columns = [c for c in NUMERIC_COLUMNS if c not in ("key", "mode", "time_signature")]
    statistics = []
    extreme_masks = {}
    for column in distribution_columns:
        values = numeric[column].where(np.isfinite(numeric[column]))
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        extreme = (values.lt(lower) | values.gt(upper)).fillna(False)
        extreme_masks[column] = extreme
        statistics.append({
            "Variable": column, "Finite count": int(values.count()),
            "Min": values.min(), "P01": values.quantile(0.01), "Q1": q1,
            "Median": values.median(), "Mean": values.mean(), "Q3": q3,
            "P99": values.quantile(0.99), "Max": values.max(), "IQR": iqr,
            "Lower IQR fence": lower, "Upper IQR fence": upper,
            "Outside IQR fences": int(extreme.sum()),
        })

    genres = data.groupby("track_genre", dropna=True).agg(
        Rows=("track_id", "size"),
        **{"Unique tracks": ("track_id", "nunique")},
    )
    genres["Rows (%)"] = genres["Rows"] / len(data) * 100 if len(data) else 0.0
    # This deliberately demonstrates an unsafe shortcut, on a temporary view only.
    first_only = identified.drop_duplicates("track_id", keep="first")
    first_counts = first_only.groupby("track_genre")["track_id"].nunique()
    genres["Tracks if keeping first row only"] = first_counts.reindex(genres.index, fill_value=0)
    genres["Lost track–genre associations"] = (
        genres["Unique tracks"] - genres["Tracks if keeping first row only"]
    )
    genres = genres.sort_values("Unique tracks", ascending=False, kind="stable").rename_axis(
        "Genre"
    ).reset_index()
    lost_genres = genres.loc[genres["Tracks if keeping first row only"].eq(0), "Genre"].tolist()

    validation = pd.DataFrame(checks)
    findings = []

    def add_finding(topic, count, unit, interpretation, decision):
        findings.append({
            "Finding": topic, "Count": int(count), "Unit": unit,
            "Status": "Pending review" if count else "No cases detected",
            "Interpretation": interpretation, "Proposed decision (not applied)": decision,
        })

    add_finding("Missing metadata or values", data.isna().any(axis=1).sum(), "rows",
                "At least one pandas missing value.", "Inspect incomplete records and document exclusion or recovery.")
    add_finding("Exact duplicate surplus", exact_extra.sum(), "rows beyond first occurrences",
                "All data columns match; the exported index is excluded.", "Confirm and remove redundant copies during cleaning.")
    add_finding("Repeated track IDs", len(repeated_tracks), "track IDs",
                "An ID can have multiple genre labels.", "Define one song record while preserving its genre associations.")
    add_finding("Multiple genres per song", distinct["track_genre"].gt(1).sum(), "track IDs",
                "Includes distinct missing/non-missing labels if present.", "Preserve valid labels as a set or a separate relationship table.")
    add_finding("Conflicting non-genre fields", conflicts.any(axis=1).sum(), "track IDs",
                "Different values for the same ID; missing versus present also counts.", "Define consolidation rules for each conflicting field.")
    for check in checks:
        if check["Affected rows"]:
            add_finding(check["Check"], check["Affected rows"], "rows",
                        check["Category"], check["Proposed review"])
    extreme_union = pd.DataFrame(extreme_masks).any(axis=1)
    add_finding("Outside IQR fences in at least one variable", extreme_union.sum(), "rows",
                "Statistical screening; skewed features can produce many flags.", "Inspect distributions; do not equate extremes with errors.")
    add_finding("Genres lost by keeping the first row", len(lost_genres), "genres",
                "Hypothetical, order-dependent deduplication; the dataset is unchanged.", "Avoid discarding secondary genre associations.")

    return {
        "exact_extra": exact_extra, "exact_members": exact_members,
        "unique_tracks": int(track_counts.size), "repeated_tracks": repeated_tracks,
        "conflicts": conflict_counts, "validation": validation, "validation_masks": masks,
        "statistics": pd.DataFrame(statistics), "extreme_masks": extreme_masks,
        "genres": genres, "lost_genres": lost_genres,
        "findings": pd.DataFrame(findings),
    }
