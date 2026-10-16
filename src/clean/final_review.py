"""Final distribution, integrity and genre review in Streamlit."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from clean.export_dataset import export_result


def render_final_review(result):
    tables = result.tables
    tracks = tables["tracks"]
    st.divider()
    st.subheader("Numeric distributions and statistical extremes")
    st.write("IQR flags are retained for inspection. No statistical extremes are removed or clipped.")
    st.dataframe(tables["outlier_summary"], hide_index=True)
    variable = st.selectbox("Inspect a numeric distribution", tables["outlier_summary"]["variable"].tolist())
    values = pd.to_numeric(tracks[variable], errors="coerce")
    finite = values.loc[np.isfinite(values)]
    if not finite.empty:
        counts, edges = np.histogram(finite, bins=40)
        histogram = pd.DataFrame({variable: (edges[:-1] + edges[1:]) / 2, "Tracks": counts})
        st.bar_chart(histogram, x=variable, y="Tracks")
    selected_ids = tables["outlier_flags"].loc[
        tables["outlier_flags"][variable + "_iqr_review"], "track_id"
    ]
    st.write("Tracks outside the IQR fences:", len(selected_ids))
    st.dataframe(tracks.loc[tracks["track_id"].isin(selected_ids), [
        "track_id", "track_name", "artists", variable,
    ]].sort_values(variable), hide_index=True)

    st.subheader("Genre representation after cleaning")
    genres = tables["genre_representation"]
    st.dataframe(genres, hide_index=True)
    st.write("The catalog is multi-label: genre counts overlap and percentages need not sum to 100%.")
    st.write("No resampling was applied. Counts describe this catalog, not the entire music population.")
    st.write("Lost genres:", genres.loc[genres["tracks_after"].eq(0), "track_genre"].tolist())
    st.write("Smallest / largest genre:", int(genres["tracks_after"].min()), int(genres["tracks_after"].max()))
    st.write("Twenty genres with the most retained tracks:")
    st.bar_chart(
        genres.head(20).set_index("track_genre")[["tracks_before", "tracks_after"]],
        stack=False,
    )

    st.subheader("Final validation and cleaning log")
    validation = tables["validation"]
    st.dataframe(validation, hide_index=True)
    st.dataframe(tables["cleaning_log"], hide_index=True)
    st.write("Failed integrity checks:", int(validation["status"].eq("fail").sum()))
    st.write("Checks with documented review cases:", int(validation["status"].eq("review").sum()))
    st.write("Excluded tracks:", len(tables["excluded_tracks"]))
    st.dataframe(tables["excluded_tracks"], hide_index=True)

    st.subheader("Export cleaned tables and reports")
    st.write("Save the catalog, associations, exclusions and evidence to data/processed.")
    st.write("Feature selection and scaling remain part of model preparation.")
    if st.button("Save cleaned dataset and reports", disabled=not result.exportable):
        output_dir = Path(__file__).resolve().parents[2] / "data/processed"
        try:
            export_result(result, output_dir)
        except (ValueError, OSError) as error:
            st.error(str(error))
        else:
            st.write("Saved to:", str(output_dir))
