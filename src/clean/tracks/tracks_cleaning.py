import streamlit as st

from common.data_access import compute_audit


def render_track_conflicts(data):
    st.divider()
    st.subheader("Conflicting values for the same track")

    audit = compute_audit(data)

    conflicts = audit["conflicts"]
    st.dataframe(
        conflicts.loc[
            conflicts["Track IDs with conflicting values"].gt(0)
        ]
    )

    repeated = audit["repeated_tracks"]
    conflicting_ids = repeated.loc[
        repeated["Conflicting fields (excluding genre)"].gt(0),
        "track_id",
    ]

    st.write("Tracks with conflicting values:", len(conflicting_ids))

    if not conflicting_ids.empty:
        selected_id = st.selectbox(
            "Select a track to inspect",
            conflicting_ids.tolist(),
        )

        st.dataframe(
            data.loc[data["track_id"].eq(selected_id)]
        )

def render_tracks(data):
    st.divider()
    st.subheader("One record per track")

    popularity_by_track = (
        data[["track_id", "popularity"]]
        .drop_duplicates()
        .groupby("track_id", as_index=False)["popularity"]
        .mean()
    )

    track_details = (
        data.drop(columns=["track_genre", "popularity"])
        .drop_duplicates()
        .copy()
    )

    if track_details["track_id"].duplicated().any():
        st.error(
            "There are unresolved conflicts in other track fields."
        )
        st.stop()

    tracks = track_details.merge(
        popularity_by_track,
        on="track_id",
        validate="one_to_one",
    )

    st.write("Tracks:", len(tracks))
    st.write(
        "Repeated track IDs:",
        int(tracks["track_id"].duplicated().sum()),
    )
    st.dataframe(tracks)

    return tracks