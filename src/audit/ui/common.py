"""Shared display helpers for audit sections."""

import streamlit as st


def show_rows(rows, limit=200):
    """Keep source indices for tracing findings, without sending huge tables."""
    st.caption(f"Showing {min(len(rows), limit):,} of {len(rows):,} rows. Counts use all rows.")
    st.dataframe(rows.head(limit).rename_axis("Source index").reset_index(),
                 hide_index=True, width="stretch")
