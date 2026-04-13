import streamlit as st
import yaml
from datetime import datetime
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent.parent / "experiments" / "results.yaml"


def load_results():
    if not RESULTS_FILE.exists():
        return []

    with open(RESULTS_FILE, "r") as f:
        data = yaml.safe_load(f)

    if data is None or "results" not in data:
        return []

    return data.get("results", [])


st.title("Results")

st.markdown("---")

results = load_results()

if not results:
    st.info("No results yet - check back later as experiments progress.")

    st.markdown("""
    To add a result, edit `experiments/results.yaml`:

    ```yaml
    results:
      - date: "2025-04-13"
        title: "Your result title"
        description: "Description of what was done"
        metric: 0.85
    ```
    """)
else:
    results_sorted = sorted(results, key=lambda x: x.get("date", ""), reverse=True)

    st.markdown(f"**{len(results_sorted)} result(s) recorded**")

    for i, res in enumerate(results_sorted):
        with st.expander(
            f"{res.get('date', 'Unknown date')} - {res.get('title', 'Untitled')}",
            expanded=True,
        ):
            st.markdown(f"**Date:** {res.get('date', 'N/A')}")

            st.markdown(f"**Title:** {res.get('title', 'N/A')}")

            st.markdown(f"**Description:** {res.get('description', 'No description')}")

            metric = res.get("metric")
            if metric is not None:
                st.metric("Metric Value", metric)

st.markdown("---")

st.caption(f"Results loaded from: {RESULTS_FILE}")

st.sidebar.markdown("""
**Results**

Track experiment outcomes over time.

[Home](../Home)
[Progress](../Progress)
""")

st.sidebar.page_link("app.py", label="← Back to Home")
