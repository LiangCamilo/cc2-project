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