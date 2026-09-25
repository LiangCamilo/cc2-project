"""Streamlit entry point: load the source and display the audit sections."""

import streamlit as st

from data_access import compute_audit, load_dataset
from data_dictionary import COLUMN_METADATA
from ui.exploration import render_distributions, render_genres
from ui.overview import (
    render_dictionary,
    render_dimensions,
    render_header,
    render_missing_values,
)
from ui.quality import (
    render_duplicates,
    render_findings,
    render_repeated_tracks,
    render_validation,
)


def main():
    st.set_page_config(
        page_title="Spotify Tracks Dataset Audit",
        page_icon="🎹",
        layout="wide",
    )
    try:
        data, fingerprint = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    render_header(fingerprint)
    render_dimensions(data)
    render_dictionary(data, COLUMN_METADATA)
    render_missing_values(data)

    audit = compute_audit(data)
    render_duplicates(data, audit)
    render_repeated_tracks(data, audit)
    render_validation(data, audit)
    render_distributions(data, audit, COLUMN_METADATA)
    render_genres(audit)
    render_findings(audit)


if __name__ == "__main__":
    main()
