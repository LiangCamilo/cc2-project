import streamlit as st

from clean.rules import consolidate_tracks


def render_track_conflicts(data):
    from common.data_access import compute_audit

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

    try:
        tracks = consolidate_tracks(data)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.write("Tracks:", len(tracks))
    st.write(
        "Repeated track IDs:",
        int(tracks["track_id"].duplicated().sum()),
    )
    st.dataframe(tracks)

    return tracks
