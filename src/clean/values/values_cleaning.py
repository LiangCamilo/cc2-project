import numpy as np
import pandas as pd
import streamlit as st


def render_tempo_review(tracks):
    st.divider()
    st.subheader("Tracks with zero or negative tempo")

    flagged_tracks = tracks.loc[
        tracks["tempo"].le(0)
    ]

    st.write("Affected tracks:", len(flagged_tracks))

    st.dataframe(
        flagged_tracks[[
            "track_id",
            "track_name",
            "artists",
            "duration_ms",
            "tempo",
            "time_signature",
            "energy",
            "speechiness",
        ]]
    )

    st.write(
        "Time signature counts:",
        flagged_tracks["time_signature"].value_counts(),
    )

    excluded_tracks = flagged_tracks.copy()

    valid_tracks = tracks.loc[
        tracks["tempo"].gt(0)
    ].copy()

    st.write("Tracks excluded for unavailable tempo:", len(excluded_tracks))
    st.write("Remaining tracks:", len(valid_tracks))

    return valid_tracks, excluded_tracks

def render_time_signature_review(tracks):
    st.divider()
    st.subheader("Time signature review")

    st.write("Tracks per time signature:")
    st.dataframe(
        tracks["time_signature"]
        .value_counts(dropna=False)
        .sort_index()
        .rename("Tracks")
    )

    flagged_tracks = tracks.loc[
        ~tracks["time_signature"].isin([3, 4, 5, 6, 7])
    ]

    st.write("Tracks requiring review:", len(flagged_tracks))

    st.dataframe(
        flagged_tracks[[
            "track_id",
            "track_name",
            "time_signature",
            "tempo",
        ]]
    )
    
    reviewed_tracks = tracks.copy()

    reviewed_tracks["time_signature_review"] = (
        ~reviewed_tracks["time_signature"].isin([3, 4, 5, 6, 7])
    )

    st.write(
        "Tracks marked for time signature review:",
        int(reviewed_tracks["time_signature_review"].sum()),
    )

    return reviewed_tracks


def render_loudness_review(tracks):
    st.divider()
    st.subheader("Loudness review")
    st.write(
        "The typical range is -60 to 0 dB. Values outside this range "
        "are kept for review; they are not automatically errors."
    )

    reviewed_tracks = tracks.copy()
    loudness = pd.to_numeric(tracks["loudness"], errors="coerce")
    finite = loudness.notna() & np.isfinite(loudness)

    reviewed_tracks["loudness_review"] = (
        finite & ~loudness.between(-60, 0)
    )
    reviewed_tracks["loudness_invalid"] = ~finite

    col1, col2, col3 = st.columns(3)
    col1.metric("Below -60 dB", int((finite & loudness.lt(-60)).sum()))
    col2.metric("Above 0 dB", int((finite & loudness.gt(0)).sum()))
    col3.metric("Missing or invalid loudness", int((~finite).sum()))

    st.write("Finite loudness summary (dB):")
    st.dataframe(
        loudness.loc[finite].agg(["min", "median", "max"]).rename("Loudness (dB)")
    )

    flagged = (
        reviewed_tracks["loudness_review"]
        | reviewed_tracks["loudness_invalid"]
    )
    st.write("Tracks requiring review:", int(flagged.sum()))
    st.dataframe(
        reviewed_tracks.loc[flagged, [
            "track_id", "track_name", "artists", "loudness", "energy",
            "loudness_review", "loudness_invalid",
        ]]
    )
    if reviewed_tracks["loudness_invalid"].any():
        st.warning(
            "Missing, non-numeric or infinite loudness values need a treatment "
            "decision before loudness is used to calculate distances."
        )
    st.write("Tracks retained:", len(reviewed_tracks))
    st.caption("No tracks are removed and original loudness values are preserved.")

    return reviewed_tracks
