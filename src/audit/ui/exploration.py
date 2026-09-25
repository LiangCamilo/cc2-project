"""Numeric distributions, extremes and representation by genre."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .common import show_rows


def render_distributions(ds, audit, metadata):
    """Plot distributions and expose statistical extremes."""
    st.divider()
    st.subheader("Distributions and Extreme Values")
    statistics = audit["statistics"]
    st.dataframe(statistics.round(6), hide_index=True, width="stretch")
    st.caption("Statistics use finite, non-missing values from all source rows, including duplicates. "
               "Categorical codes (key, mode and time_signature) are excluded from this numeric screening.")
    st.write("IQR = Q3 − Q1. Values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are flagged. "
             "This is an exploratory rule: skewed distributions and zero-heavy features can have many legitimate extremes. "
             "If IQR is zero, every value different from the quartiles can be flagged.")
    variable = st.selectbox("Numeric variable", statistics["Variable"].tolist())
    values = pd.to_numeric(ds[variable], errors="coerce")
    finite_values = values.loc[np.isfinite(values)]
    if not finite_values.empty:
        # Aggregate the full dataset before plotting (no sampling or Altair row limit).
        counts, edges = np.histogram(finite_values, bins=40)
        histogram = pd.DataFrame({"From": edges[:-1], "To": edges[1:], "Rows": counts})
        chart = alt.Chart(histogram).mark_bar().encode(
            x=alt.X("From:Q", bin="binned", title=f"{variable} ({metadata[variable]['Unit or scale']})"),
            x2="To:Q", y=alt.Y("Rows:Q", title="Rows"),
            tooltip=[alt.Tooltip("From:Q", format=".4f"), alt.Tooltip("To:Q", format=".4f"), "Rows:Q"],
        )
        st.altair_chart(chart, width="stretch")
        st.write("Rows outside IQR fences:", int(audit["extreme_masks"][variable].sum()))
        with st.expander("Inspect statistical extremes"):
            show_rows(ds.loc[audit["extreme_masks"][variable]].sort_values(variable, kind="stable"))
        with st.expander("Smallest and largest finite values"):
            ordered = values.loc[np.isfinite(values)].sort_values(kind="stable")
            st.write("Ten smallest")
            show_rows(ds.loc[ordered.head(10).index])
            st.write("Ten largest")
            show_rows(ds.loc[ordered.tail(10).index])
    else:
        st.info("No finite numeric values available for this distribution.")


def render_genres(audit):
    """Compare source representation with a hypothetical deduplication."""
    st.divider()
    st.subheader("Genre Representation")
    genres = audit["genres"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed genres", len(genres))
    col2.metric("Minimum unique tracks per genre", int(genres["Unique tracks"].min()) if len(genres) else 0)
    col3.metric("Maximum unique tracks per genre", int(genres["Unique tracks"].max()) if len(genres) else 0)
    st.write("Counts distinguish source rows from distinct tracks within each genre. "
             "A song may belong to several genres, so unique-track counts across genres must not be summed as catalog size.")
    st.caption("Rows (%) uses all source rows as denominator. Missing genres are counted in Missing Values, not as a genre.")
    st.dataframe(genres.round(4), hide_index=True, width="stretch")
    if not genres.empty:
        genre_order = st.selectbox("Genres to plot", ["Fewest unique tracks", "Most unique tracks", "Largest loss if keeping first row"])
        sort_column = "Lost track–genre associations" if genre_order.startswith("Largest") else "Unique tracks"
        plot_genres = genres.sort_values(sort_column, ascending=genre_order.startswith("Fewest"), kind="stable").head(20)
        genre_chart = alt.Chart(plot_genres).mark_bar().encode(
            x=alt.X(f"{sort_column}:Q", title=sort_column),
            y=alt.Y("Genre:N", sort=plot_genres["Genre"].tolist()),
            tooltip=["Genre:N", "Rows:Q", "Unique tracks:Q", "Lost track–genre associations:Q"],
        )
        st.altair_chart(genre_chart, width="stretch")
    st.write("The last two table columns simulate keeping only the first occurrence of each track ID, "
             "in the current file order. This simulation does not modify the dataset.")
    if audit["lost_genres"]:
        st.warning("Genres that would disappear under that shortcut: " + ", ".join(audit["lost_genres"]))
    st.caption("Equal source-row counts alone do not establish a balanced final catalog. "
               "Reassess representation after deciding how to preserve all valid genre associations.")


