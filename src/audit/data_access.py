"""Source loading, validation and caching for the Streamlit application."""

from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

from audit_checks import NUMERIC_COLUMNS, TEXT_COLUMNS, audit_dataset


@st.cache_data
def load_data(dataset_path, modified_ns):
    # modified_ns invalidates the cache if the local source file changes.
    return pd.read_csv(dataset_path, index_col=0)


@st.cache_data
def compute_audit(data):
    return audit_dataset(data)


@st.cache_data
def source_fingerprint(dataset_path, modified_ns):
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def load_dataset():
    """Load and validate the local source, returning data and its fingerprint."""
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run src/download_data.py from the project environment first."
        )
    modified_ns = dataset_path.stat().st_mtime_ns
    data = load_data(str(dataset_path), modified_ns)
    required_columns = set(NUMERIC_COLUMNS + TEXT_COLUMNS + ["explicit"])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns or data.empty:
        raise ValueError(
            f"Cannot audit this dataset: missing columns {sorted(missing_columns)} or no rows."
        )
    return data, source_fingerprint(str(dataset_path), modified_ns)
