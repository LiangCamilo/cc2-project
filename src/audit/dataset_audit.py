from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="Spotify Tracks Dataset Audit",
    page_icon="🎹",
    layout="centered"
)

# Using cache for loading dataset
@st.cache_data
def load_data():
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_PATH = ROOT_DIR / "data" / "raw" / "dataset.csv"
    return pd.read_csv(DATASET_PATH, index_col=0)

# loading dataset
ds = load_data()

# Streamlit title
st.markdown("""
    <h1 style='text-align: center; color: #1DB954;'>
        🎹 Dataset Audit
    </h1>
    <p style='text-align: center; color: gray;'>
        Exploratory Data Analysis - Profiling
    </p>
""", unsafe_allow_html=True)

st.divider()

# Dataset Dimensions
st.subheader('Dataset Dimensions')
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Total Rows", value=f"{ds.shape[0]:,}")

with col2:
    st.metric(label="Total Columns", value=f"{ds.shape[1]}")

st.divider()

# Data Dictionary
metadata = {
    "track_id": {
        "Description": "The unique Spotify ID for the track.",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Alphanumeric string",
    },
    "artists": {
        "Description": "The artists who performed the track. Multiple artists are separated by a semicolon (;).",
        "Type": "Categorical string (List)",
        "Unit or scale": "Unitless",
        "Documented domain": "Text",
    },
    "album_name": {
        "Description": "The album name in which the track appears.",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Text",
    },
    "track_name": {
        "Description": "The name of the track.",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Text",
    },
    "popularity": {
        "Description": "The popularity of a track. Calculated by algorithm and based, in the most part, on the total number of plays the track has had and how recent those plays are.",
        "Type": "Numeric, recorded as integers",
        "Unit or scale": "Popularity score",
        "Documented domain": "Between 0 and 100 (100 being the most popular)",
    },
    "duration_ms": {
        "Description": "Duration of the song.",
        "Type": "Numeric, recorded as integers",  # Cambiado a Type para consistencia, o puedes volver a usar 'Nature'
        "Unit or scale": "Milliseconds",
        "Documented domain": "Limits not specified (Starts at >0)",
    }, 
    "explicit": {
        "Description": "Whether or not the track has explicit lyrics.",
        "Type": "Boolean",
        "Unit or scale": "Unitless",
        "Documented domain": "True or False",
    },
    "danceability": {
        "Description": "Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "energy": {
        "Description": "Perceived intensity and activity of the song. Energetic tracks feel fast, loud, and noisy.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "key": {
        "Description": "The key the track is in. Integers map to pitches using standard Pitch Class notation (e.g., 0 = C, 1 = C♯/D♭, 2 = D).",
        "Type": "Categorical integer",
        "Unit or scale": "Pitch class",
        "Documented domain": "Between -1 and 11 (-1 if no key was detected)",
    },
    "loudness": {
        "Description": "The overall loudness of a track in decibels (dB), averaged across the entire track. Useful for comparing relative loudness of tracks.",
        "Type": "Continuous numeric",
        "Unit or scale": "Decibels (dB)",
        "Documented domain": "Typically between -60 and 0",
    },
    "mode": {
        "Description": "Indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived.",
        "Type": "Binary integer",
        "Unit or scale": "Unitless",
        "Documented domain": "1 for Major, 0 for Minor",
    },
    "speechiness": {
        "Description": "Detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book), the closer to 1.0.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0 (Values > 0.66 represent entirely spoken words)",
    },
    "acousticness": {
        "Description": "A confidence measure of whether the track is acoustic.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0 (1.0 represents high confidence)",
    },
    "instrumentalness": {
        "Description": "Predicts whether a track contains no vocals. 'Ooh' and 'aah' sounds are treated as instrumental. Rap or spoken word are vocal.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0 (Values closer to 1.0 indicate no vocal content)",
    },
    "liveness": {
        "Description": "Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "valence": {
        "Description": "A measure describing the musical positiveness conveyed by a track. High valence sounds more positive (happy, cheerful), low valence sounds negative (sad, depressed).",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "tempo": {
        "Description": "The overall estimated tempo of a track in beats per minute (BPM).",
        "Type": "Continuous numeric",
        "Unit or scale": "Beats per minute (BPM)",
        "Documented domain": "Limits not specified (Typically between 50 and 200+)",
    },
    "time_signature": {
        "Description": "An estimated time signature. It specifies how many beats are in each bar (or measure).",
        "Type": "Categorical integer",
        "Unit or scale": "Beats per bar",
        "Documented domain": "Between 3 and 7 (e.g., 3/4 to 7/4 time signature)",
    },
    "track_genre": {
        "Description": "The genre to which the track belongs.",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Predefined list of 114 specific genres",
    }
}
st.subheader('🔍 Data Dictionary')

dsColumnsTable = pd.DataFrame(
    list(ds.dtypes.astype(str).items()), 
    columns=['Column Name', 'Data Type']
)

#Streamlit dataset
st.dataframe(
    dsColumnsTable, 
    use_container_width=True,
    hide_index=True,
)

st.divider()

#Null verify

st.subheader('Row Count Of Null/NaN Values')
dsNullNanTable = pd.DataFrame(
    ds.isna().sum()
)

st.dataframe(
    dsNullNanTable, 
    use_container_width=True,
    hide_index=True,
)
