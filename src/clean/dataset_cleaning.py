from clean.tracks.tracks_cleaning import render_track_conflicts, render_tracks
import streamlit as st
from clean.duplicates.duplicates_cleaning import (
    render_duplicates,
)
from clean.genres.genres_cleaning import (
    render_track_genres,
)
from clean.texts.texts_cleaning import render_text_cleaning
from clean.values.values_cleaning import (
    render_loudness_review,
    render_tempo_review,
    render_time_signature_review,
)

from pathlib import Path

import pandas as pd
import streamlit as st
from common.data_access import (
    load_dataset,
    load_data,
)
from clean.nulls.nulls_cleaning import (
    render_nulls,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "dataset.csv"

def main():

    cleaned_data = None
    
    try:
        data, fingerprint = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.set_page_config(
        page_title="Spotify Tracks Dataset Cleaning",
        page_icon="🧹",
        layout="wide",
    )
    st.title("Spotify Tracks Dataset Cleaning")
    st.write("This page shows the cleaning steps applied to the Spotify Tracks dataset.")

    #Cleaning process 
    cleaned_data = render_nulls(data)
    cleaned_data = render_text_cleaning(cleaned_data)
    cleaned_data = render_duplicates(cleaned_data)

    track_genres = render_track_genres(cleaned_data)

    render_track_conflicts(cleaned_data)
    tracks = render_tracks(cleaned_data)

    # Continue with value reviews for cleaning
    tracks, excluded_tempo_tracks = render_tempo_review(tracks)
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

if __name__ == "__main__":
    main()
