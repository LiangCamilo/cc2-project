"""Record identity, domain checks and pending cleaning decisions."""

import streamlit as st

from .common import show_rows


def render_duplicates(ds, audit):
    """Show exact duplicate counts and examples."""
    st.divider()
    st.subheader("Exact Duplicates")
    col1, col2, col3 = st.columns(3)
    col1.metric("Extra identical rows", f"{audit['exact_extra'].sum():,}")
    col2.metric("Rows in duplicate groups (including originals)", f"{audit['exact_members'].sum():,}")
    col3.metric("Distinct duplicated records", f"{len(ds.loc[audit['exact_members']].drop_duplicates()):,}")
    st.caption("All data columns are compared. The exported source index is excluded. Extra rows count all copies after the first.")
    with st.expander("Inspect exact duplicate groups"):
        show_rows(ds.loc[audit["exact_members"]].sort_values("track_id", kind="stable"))


def render_repeated_tracks(ds, audit):
    """Inspect repeated identities and conflicting fields."""
    st.divider()
    st.subheader("Repeated Track IDs and Consistency")
    repeated = audit["repeated_tracks"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique track IDs", f"{audit['unique_tracks']:,}")
    col2.metric("Track IDs appearing more than once", f"{len(repeated):,}")
    col3.metric("Extra rows per track ID", f"{ds['track_id'].notna().sum() - audit['unique_tracks']:,}")
    st.write("Repeated IDs are not automatically redundant: a song may have multiple genres or conflicting values.")
    with st.expander("Repeated IDs overview"):
        st.caption(f"Showing {min(len(repeated), 100):,} of {len(repeated):,} repeated IDs, ordered by row count.")
        st.dataframe(repeated.head(100), hide_index=True, width="stretch")
    st.write("Conflicting fields for the same track ID (genre excluded)")
    st.dataframe(audit["conflicts"], hide_index=True, width="stretch")
    st.caption("A missing value and a present value count as different. Each count refers to track IDs, not rows.")
    if not repeated.empty:
        track_id = st.text_input("Inspect a track ID", value=str(repeated.iloc[0]["track_id"]))
        track_rows = ds.loc[ds["track_id"].eq(track_id.strip())]
        if track_rows.empty:
            st.info("No records found for this ID.")
        else:
            show_rows(track_rows)


def render_validation(ds, audit):
    """Inspect the records flagged by each validation rule."""
    st.divider()
    st.subheader("Values and Valid Categories")
    st.write("Documented domains, typical ranges, sentinel values and project review rules are labeled separately. "
             "Flags identify records to inspect; they do not authorize automatic deletion.")
    validation = audit["validation"]
    only_flagged = st.checkbox("Show only checks with affected rows", value=True)
    st.dataframe(validation.loc[validation["Affected rows"].gt(0)] if only_flagged else validation,
                 hide_index=True, width="stretch")
    flagged_checks = validation.loc[validation["Affected rows"].gt(0), "Check"].tolist()
    if flagged_checks:
        selected_check = st.selectbox("Inspect flagged records", flagged_checks)
        show_rows(ds.loc[audit["validation_masks"][selected_check]])
    else:
        st.success("No records flagged by the configured checks.")
    with st.expander("Observed category counts"):
        for column in ["key", "mode", "time_signature", "explicit"]:
            st.write(column)
            counts = ds[column].value_counts(dropna=False).rename_axis("Value").reset_index(name="Rows")
            st.dataframe(counts, hide_index=True, width="stretch")


def render_findings(audit):
    """Summarize evidence and decisions that have not been applied."""
    st.divider()
    st.subheader("Findings and Pending Cleaning Decisions")
    st.write("This is an evidence summary, not a cleaning log. Proposed decisions have not been applied. "
             "Counts use different units and can overlap; do not add them to estimate rows to remove.")
    st.dataframe(audit["findings"], hide_index=True, width="stretch")
    st.info("Before cleaning: agree on incomplete records, exact duplicates, one-song identity, genre preservation, "
            "conflicting fields and exceptional values. Then verify the resulting counts and genre coverage.")


