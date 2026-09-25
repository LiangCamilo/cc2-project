"""Source context, dimensions, data dictionary and missing values."""

import pandas as pd
import streamlit as st

from .common import show_rows


def render_header(fingerprint):
    """Explain the audited source and read-only scope."""
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

    st.info("Read-only audit: findings and proposed decisions do not change the source dataset.")
    with st.expander("Source and interpretation"):
        st.write("Source: data/raw/dataset.csv. The first CSV column is loaded as the source index.")
        st.write("A row is a dataset record; multiple rows may refer to the same Spotify track ID.")
        st.write("Source index identifies a record in the input; it is not the physical CSV line number.")
        st.code(fingerprint, language=None)
        st.caption("SHA-256 fingerprint of the audited file, for reproducibility.")
        st.markdown("Definitions: [dataset author](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/blob/main/README.md) "
                    "and [Spotify audio features](https://developer.spotify.com/documentation/web-api/reference/get-audio-features). "
                    "Documentation can differ from observed values; discrepancies require review.")


def render_dimensions(ds):
    """Show the size of the loaded dataset."""
    # Dataset Dimensions
    st.subheader('Dataset Dimensions')
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Total Rows", value=f"{ds.shape[0]:,}")

    with col2:
        st.metric(label="Total Columns", value=f"{ds.shape[1]}")

    st.divider()


def render_dictionary(ds, metadata):
    """Combine observed pandas types with documented meanings."""
    st.subheader('🔍 Data Dictionary')

    data_dictionary = pd.DataFrame({
        "Variable": ds.columns,
        "Pandas Type": ds.dtypes.astype(str).to_numpy(),
        "Description": [metadata.get(col, {}).get("Description", "No documentado") for col in ds.columns],
        "Type": [metadata.get(col, {}).get("Type", "No documentado") for col in ds.columns],
        "Unit or scale": [metadata.get(col, {}).get("Unit or scale", "No documentado") for col in ds.columns],
        "Documented domain": [metadata.get(col, {}).get("Documented domain", "No documentado") for col in ds.columns],
    })

    st.dataframe(
        data_dictionary,
        hide_index=True,
        width="stretch",
    )

    st.divider()


def render_missing_values(ds):
    """Distinguish missing cells from incomplete source records."""
    st.subheader('Missing Values by Column')

    missing = ds.isna()

    missing_by_column = pd.DataFrame({
        "Variable": ds.columns,
        "Missing count": missing.sum().to_numpy(),
        "Missing (%)": (missing.mean() * 100).round(6).to_numpy(),
    })

    st.dataframe(
        missing_by_column,
        hide_index=True,
        width="stretch",
    )

    rows_with_missing = ds.loc[ds.isna().any(axis=1)]

    st.write("Rows with at least one missing value:", len(rows_with_missing))
    show_rows(rows_with_missing)
    st.caption("Missing cells and incomplete rows are different counts. Zero and documented sentinel values are reviewed below.")
