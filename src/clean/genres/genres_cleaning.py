import streamlit as st


def render_track_genres(data):
    st.divider()
    st.subheader("Track–genre associations")

    track_genres = (
        data[["track_id", "track_genre"]]
        .drop_duplicates()
        .sort_values(["track_id", "track_genre"])
        .reset_index(drop=True)
    )

    st.write("Associations:", len(track_genres))
    st.write("Unique tracks:", track_genres["track_id"].nunique())
    st.write("Unique genres:", track_genres["track_genre"].nunique())
    st.dataframe(track_genres)

    return track_genres