
import pandas as pd
import streamlit as st

from common.audit_checks import TEXT_COLUMNS


def render_text_cleaning(data):
    st.divider()
    st.subheader("Surrounding whitespace in text")
    st.write(
        "Remove whitespace at the beginning and end of text fields. "
        "Internal spacing, artist separators, capitalization and punctuation are preserved."
    )

    cleaned_data = data.copy()
    changed_rows = pd.Series(False, index=data.index)
    counts = []
    changes = []

    for column in TEXT_COLUMNS:
        text = data[column].astype("string")
        stripped = text.str.strip()
        changed = (text.notna() & text.ne(stripped)).fillna(False)
        counts.append({"Column": column, "Values corrected": int(changed.sum())})
        changed_rows |= changed

        if changed.any():
            cleaned_data.loc[changed, column] = stripped.loc[changed]
            changes.append(pd.DataFrame({
                "Source index": data.index[changed],
                "track_id": data.loc[changed, "track_id"].to_numpy(),
                "Column": column,
                "Before": text.loc[changed].map(repr).to_numpy(),
                "After": stripped.loc[changed].map(repr).to_numpy(),
            }))

    st.dataframe(pd.DataFrame(counts), hide_index=True)
    st.write("Rows with corrected text:", int(changed_rows.sum()))
    if changes:
        st.caption("Quotes make surrounding spaces visible in the before/after table.")
        st.dataframe(pd.concat(changes, ignore_index=True), hide_index=True)
    else:
        st.write("No surrounding whitespace found.")

    st.write("Rows retained:", len(cleaned_data))
    return cleaned_data
