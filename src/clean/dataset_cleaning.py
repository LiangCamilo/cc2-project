from clean.tracks.tracks_cleaning import render_track_conflicts, render_tracks
import streamlit as st
from clean.duplicates.duplicates_cleaning import (
    render_duplicates,
)
from clean.genres.genres_cleaning import (
    render_track_genres,
)
from clean.texts.texts_cleaning import render_text_cleaning
from clean.final_review import render_final_review
from clean.pipeline import finalize_dataset
from clean.values.values_cleaning import (
    render_loudness_review,
    render_tempo_review,
    render_time_signature_review,
)

from common.data_access import load_dataset
from clean.nulls.nulls_cleaning import (
    render_nulls,
)


def main():
    st.set_page_config(
        page_title="Spotify Tracks Dataset Cleaning",
        page_icon="🧹",
        layout="wide",
    )
    try:
        data, fingerprint = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()
    st.title("Spotify Tracks Dataset Cleaning")
    st.write("This page shows the cleaning steps applied to the Spotify Tracks dataset.")

    #Cleaning process 
    cleaned_data = render_nulls(data)
    cleaned_data = render_text_cleaning(cleaned_data)
    cleaned_data = render_duplicates(cleaned_data)

    track_genres = render_track_genres(cleaned_data)

    render_track_conflicts(cleaned_data)
    consolidated_tracks = render_tracks(cleaned_data)

    # Continue with value reviews for cleaning
    tracks, excluded_tempo_tracks = render_tempo_review(consolidated_tracks)
    tracks = render_time_signature_review(tracks)
    tracks = render_loudness_review(tracks)

    #Modified Genre Table
    genres_before = set(track_genres["track_genre"])

    track_genres = track_genres.loc[
        track_genres["track_id"].isin(tracks["track_id"])
    ].copy()

    genres_after = set(track_genres["track_genre"])

    st.write("Remaining genres:", len(genres_after))
    st.write("Genres lost:", sorted(genres_before - genres_after))

    result = finalize_dataset(
        data, cleaned_data, consolidated_tracks, tracks, track_genres,
        excluded_tempo_tracks, fingerprint,
    )
    render_final_review(result)

if __name__ == "__main__":
    main()
