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