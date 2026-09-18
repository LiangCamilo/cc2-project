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
