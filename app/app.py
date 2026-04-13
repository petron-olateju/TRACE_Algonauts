import streamlit as st

st.set_page_config(
    page_title="TRACE: Contrastive Embeddings",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e88e5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .info-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("TRACE: Contrastive Embeddings")

st.markdown("---")

st.markdown("""
**TRACE** (Time series Representation Analysis through Contrastive Embeddings) is a contrastive 
learning framework applied to human fMRI data.

Originally developed for cellular recordings in mice, TRACE now extends to the macroscopic scale 
of human neuroimaging to visualize functional organization.

Use the sidebar to navigate between pages:
""")

st.markdown("""
| Page | Description |
|------|------------|
| **Home** | Project overview |
| **Results** | Experiment outcomes |
| **Progress** | Methodology for review |
""")

st.markdown("---")
st.caption("Use `streamlit run app/app.py` to launch the app.")
