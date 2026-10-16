"""Build the cleaned catalog and evidence reports without a Streamlit runtime."""

from dataclasses import dataclass

import pandas as pd

from clean.final_checks import compare_genres, inspect_distributions, validate_catalog
from clean.rules import (
    build_track_genres, clean_text_fields, consolidate_tracks, mark_loudness,
    mark_time_signature, remove_incomplete_records, separate_unavailable_tempo,
)
from common.audit_checks import NUMERIC_COLUMNS, TEXT_COLUMNS


@dataclass
class CleaningResult:
    tables: dict
    source_sha256: str

    @property
    def exportable(self):
        return not self.tables["validation"]["status"].eq("fail").any()


def finalize_dataset(source, deduplicated, consolidated, tracks, track_genres, excluded_tempo, fingerprint):
    remaining, excluded_nulls = remove_incomplete_records(source)
    normalized, _, text_changes = clean_text_fields(remaining)
    source_text, _, _ = clean_text_fields(source)
    source_relations = build_track_genres(source_text)
    excluded = pd.concat([
        excluded_nulls.drop(columns="track_genre").assign(
            exclusion_reason="missing_artist_album_title_and_zero_duration"),
        excluded_tempo.assign(exclusion_reason="nonpositive_tempo"),
    ], ignore_index=True)
    excluded_genres = source_relations.loc[
        source_relations["track_id"].isin(excluded["track_id"])
    ].copy()
    summary, outlier_flags = inspect_distributions(tracks)
    validation = validate_catalog(tracks, track_genres, source_relations, excluded)
    genre_report = compare_genres(source_text, tracks, track_genres)

    observations = deduplicated[["track_id", "popularity"]].drop_duplicates()
    popularity = observations.groupby("track_id")["popularity"].agg(
        distinct_values="count", observed_min="min", observed_max="max", consolidated_mean="mean"
    )
    popularity = popularity.loc[popularity["distinct_values"].gt(1)].reset_index()

    stages = []

    def stage(name, before, after, decision):
        stages.append({"stage": name, "input_rows": before, "output_rows": after,
                       "removed_rows": before - after, "decision": decision})

    stage("source", len(source), len(source), "Immutable raw CSV; hash recorded in manifest.")
    stage("incomplete_records", len(source), len(remaining),
          "Exclude only records missing artist, album and title with zero duration.")
    stage("text_whitespace", len(remaining), len(normalized),
          f"Strip outer whitespace only; {len(text_changes)} corrected cells.")
    stage("exact_duplicates", len(normalized), len(deduplicated), "Keep one identical full row; source index excluded.")
    stage("track_consolidation", len(deduplicated), len(consolidated),
          "One track per ID; preserve all genres separately; average distinct popularity observations. Removed rows are representations, not excluded songs.")
    stage("unavailable_tempo", len(consolidated), len(tracks),
          "Exclude nonpositive tempo from the first distance-based catalog; keep excluded songs and their genres.")
    stage("time_signature", len(tracks), len(tracks),
          "Keep original codes; flag discrepancies; omit this field and review flags from initial model features.")
    stage("loudness", len(tracks), len(tracks), "Keep finite atypical values; flag without clipping.")
    stage("iqr_review", len(tracks), len(tracks), "IQR is a screening rule; no deletion or winsorization.")
    log = pd.DataFrame(stages)
    reconciled = len(tracks) + len(excluded) + int(log.loc[
        log["stage"].isin(["exact_duplicates", "track_consolidation"]), "removed_rows"
    ].sum())
    validation.loc[len(validation)] = {
        "check": "source_row_reconciliation", "affected": abs(len(source) - reconciled),
        "status": "pass" if len(source) == reconciled else "fail",
        "detail": "Source rows = final tracks + excluded tracks + redundant/consolidated representations.",
    }
    return CleaningResult({
        "tracks": tracks, "track_genres": track_genres,
        "excluded_tracks": excluded, "excluded_track_genres": excluded_genres,
        "validation": validation, "outlier_summary": summary, "outlier_flags": outlier_flags,
        "genre_representation": genre_report, "cleaning_log": log,
        "text_changes": text_changes, "popularity_conflicts": popularity,
    }, fingerprint)


def run_cleaning(source, fingerprint=""):
    required = set(NUMERIC_COLUMNS + TEXT_COLUMNS + ["explicit"])
    missing = sorted(required.difference(source.columns))
    if missing or source.empty:
        raise ValueError(f"Cannot clean an empty dataset or one missing columns: {missing}")
    remaining, _ = remove_incomplete_records(source)
    normalized, _, _ = clean_text_fields(remaining)
    deduplicated = normalized.drop_duplicates().copy()
    relations = build_track_genres(deduplicated)
    consolidated = consolidate_tracks(deduplicated)
    tracks, excluded_tempo = separate_unavailable_tempo(consolidated)
    tracks = mark_loudness(mark_time_signature(tracks))
    relations = relations.loc[relations["track_id"].isin(tracks["track_id"])].copy()
    return finalize_dataset(source, deduplicated, consolidated, tracks, relations, excluded_tempo, fingerprint)
