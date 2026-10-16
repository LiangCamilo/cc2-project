import streamlit as st


def render_duplicates(data):
    st.divider()
    st.subheader("Exact duplicates")

    extra_copies = data.duplicated()
    all_duplicates = data.duplicated(keep=False)

    st.write("Redundant copies:", int(extra_copies.sum()))

    st.dataframe(
        data.loc[all_duplicates]
        .sort_values("track_id")
    )

    cleaned_data = data.drop_duplicates().copy()

    st.write("Rows removed:", len(data) - len(cleaned_data))
    st.write("Remaining rows:", len(cleaned_data))
    st.write(
        "Remaining exact duplicates:",
        int(cleaned_data.duplicated().sum()),
    )

    return cleaned_data