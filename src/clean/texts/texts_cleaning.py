
import streamlit as st

from clean.rules import clean_text_fields


def render_text_cleaning(data):
    st.divider()
    st.subheader("Surrounding whitespace in text")
    st.write(
        "Remove whitespace at the beginning and end of text fields. "
        "Internal spacing, artist separators, capitalization and punctuation are preserved."
    )

    cleaned_data, counts, changes = clean_text_fields(data)
    st.dataframe(counts, hide_index=True)
    st.write("Rows with corrected text:", int(changes["Source index"].nunique()))
    if not changes.empty:
        st.caption("Quotes make surrounding spaces visible in the before/after table.")
        st.dataframe(changes, hide_index=True)
    else:
        st.write("No surrounding whitespace found.")

    st.write("Rows retained:", len(cleaned_data))
    return cleaned_data
