import streamlit as st

st.title(
    "Learning human cortical representations of multimodal stimuli using contrastive embeddings"
)

st.markdown("---")

st.header("About the Project")

st.markdown("""
**TRACE** (Time series Representation Analysis through Contrastive Embeddings) is a contrastive 
learning framework that exploits the multi-trial structure of neuroscience experiments.

Originally developed for cellular recordings (calcium imaging, Neuropixels) in mouse brains, TRACE has been 
shown to capture response characteristics and identify two-dimensional structure.

This project extends TRACE to the **macroscopic scale of human neuroimaging** using large-scale fMRI data.
""")

st.markdown("---")

st.header("Objective")

st.markdown("""
Produce two-dimensional embeddings that:
- Capture functionally relevant variations
- Identify clusters of regions responsive to specific stimulus modalities
- Group spatially close or functionally connected parcels together

This establishes TRACE as a novel tool for neuroimaging, moving beyond the original domain 
of cellular recordings.
""")

st.markdown("---")

st.header("Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Subjects", "4")

with col2:
    st.metric("Movie Duration", "~80 hours")

with col3:
    st.metric("Brain Regions", "1,000")

st.markdown("""
- **Data source**: Algonauts naturalistic movie dataset
- **Parcellation**: Schaefer 2018 (7 networks, 1000 parcels)
- **Modalities**: Visual, auditory, language (multimodal)
""")

st.markdown("---")

st.header("Current Status")

st.info("Project in early stages - TRACE framework being applied to human fMRI data.")

st.markdown("---")

st.caption("Based on: TRACE original paper + Algonauts 2025 dataset")

st.sidebar.markdown("""
**Home**

Overview of the TRACE project for human fMRI.

[Results](../Results)
[Progress](../Progress)
""")

st.sidebar.page_link("app.py", label="← Back to Home")
