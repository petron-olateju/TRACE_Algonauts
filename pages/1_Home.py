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

This project extends TRACE to the **macroscopic scale of human neuroimaging** using large-scale fMRI data from both naturalistic stimuli (**Algonauts**) and controlled cognitive tasks (**HCP Test-Retest**).
""")

st.markdown("---")

st.header("Objective")

st.markdown("""
Produce two-dimensional embeddings that:
- Capture functionally relevant variations across diverse cognitive domains.
- Identify "Universal Functional Fingerprints" that are stable across tasks.
- Group spatially close or functionally connected parcels together using surface-based anatomy.

This establishes TRACE as a novel tool for neuroimaging, moving beyond the original domain 
of cellular recordings.
""")

st.markdown("---")

st.header("Datasets")

tab1, tab2 = st.tabs(["Algonauts (Naturalistic)", "HCPTRT (Cross-Task)"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Subjects", "4")
    with col2:
        st.metric("Movie Duration", "~80 hours")
    with col3:
        st.metric("Brain Regions", "1,000")
    st.markdown("""
    - **Focus**: Continuous responses to complex, multimodal stimuli (Friends, Movie10).
    - **Parcellation**: Schaefer 2018 (Volumetric).
    """)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Subjects", "46 (Test-Retest)")
    with col2:
        st.metric("Tasks Stacked", "6 + Rest")
    with col3:
        st.metric("Grayordinates", "91,282")
    st.markdown("""
    - **Focus**: Universal functional fingerprinting by stacking signals horizontally across time.
    - **Methodology**: **Horizontal Stacking** of Motor, Gambling, WM, Social, Emotion, and Rest signals.
    - **Parcellation**: Glasser MMP (360), Cole-Anticevic, and Schaefer (Surface/CIFTI).
    """)

st.markdown("---")

st.header("Current Status")

st.success("Universal functional fingerprinting successfully validated on the HCPTRT dataset.")

st.markdown("---")

st.caption("Based on: TRACE original paper + Algonauts 2025 dataset + HCP Test-Retest")

st.sidebar.markdown("""
**Home**

Overview of the TRACE project for human fMRI.

[Results](../Results)
[Progress](../Progress)
""")

st.sidebar.page_link("app.py", label="← Back to Home")
