import streamlit as st

def render_nulls(data):
    st.divider()
    st.write("Rows and Columns:", data.shape)

    incomplete_rows = data.loc[data.isna().any(axis=1)]
    st.write("Rows with missing data:", len(incomplete_rows))
    st.dataframe(incomplete_rows)

    incomplete_ids = incomplete_rows["track_id"].dropna()

    related_rows = data.loc[
        data["track_id"].isin(incomplete_ids)
    ]

    st.write("All rows for tracks with missing data:")
    st.write("Matching rows:", len(related_rows))
    st.dataframe(related_rows)

    rows_to_remove = (
        data[["artists", "album_name", "track_name"]]
        .isna()
        .all(axis=1)
        & data["duration_ms"].eq(0)
    )

    cleaned_data = data.loc[~rows_to_remove].copy()

    st.write("Rows removed:", int(rows_to_remove.sum()))
    st.write("Remaining rows:", len(cleaned_data))
    st.write(
        "Remaining rows with missing data:",
        int(cleaned_data.isna().any(axis=1).sum()),
    )

    return cleaned_data