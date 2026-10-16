import streamlit as st
from clean.duplicates.duplicates_cleaning import (
    render_duplicates,
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

    cleaned_data = render_nulls(data)
    render_duplicates(cleaned_data)




if __name__ == "__main__":
    main()
