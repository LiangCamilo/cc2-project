import streamlit as st

from clean.rules import build_track_genres


def render_track_genres(data):
    st.divider()
    st.subheader("Track–genre associations")

    track_genres = build_track_genres(data)

    st.write("Associations:", len(track_genres))
    st.write("Unique tracks:", track_genres["track_id"].nunique())
    st.write("Unique genres:", track_genres["track_genre"].nunique())
    st.dataframe(track_genres)

    return track_genres
