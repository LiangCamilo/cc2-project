import streamlit as st

from clean.rules import remove_incomplete_records

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

    cleaned_data, removed_rows = remove_incomplete_records(data)

    st.write("Rows removed:", len(removed_rows))
    st.write("Remaining rows:", len(cleaned_data))
    st.write(
        "Remaining rows with missing data:",
        int(cleaned_data.isna().any(axis=1).sum()),
    )

    return cleaned_data
