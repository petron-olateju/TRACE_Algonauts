import streamlit as st

st.title("Progress Report")

st.markdown("---")

st.header("Methodology")

st.markdown("""
This document outlines the TRACE pipeline for generating low-dimensional embeddings 
from human fMRI data.
""")

st.markdown("---")

st.subheader("1. Data Acquisition")

st.markdown("""
- **Source**: Algonauts 2025 naturalistic movie dataset
- **Subjects**: 4 participants
- **Stimulus**: ~80 hours of multimodal movies (visual, auditory)
- **Recording**: fMRI (TR = 1.49s)
- **Parcellation**: Schaefer 1000 (7 networks)
""")

st.markdown("---")

st.subheader("1.5. Data Loader Architecture")

st.markdown("""
- **AlgonautsLoader Class**: Object-oriented dataloader for Algonauts 2025 dataset
- **Extensible Design**: YAML-based configs under `algonauts` key, ready for HCP
- **Key Features**:
  - `self.configs` stores merged configs (dirs, params, subjects)
  - Methods for loading fMRI, transcripts, and epochs
  - Simple instantiation: `AlgonautsLoader(dataset="algonauts", split="train")`
- **Usage**:
  ```python
  from utils.dataloader import get_default_dataset
  dataset = get_default_dataset()
  subject_data = dataset.load_episode_fmri(subject=1, split="train")
  ```
""")

st.markdown("---")

st.subheader("2. Response Matrix Construction")

st.markdown("""
Two approaches implemented in `notebook.ipynb` ("Movie Scenes Response Matrix Section"):
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Approach 1: Scene-level**
- Trial unit: Movie scene (e.g., s01e02a)
- Output shape: (n_subjects, 482, 1000)
- Context window: Full scene timecourse
- HRF delay: 3 TRs

*Treats each movie scene as one trial. Extracts full fMRI timecourse (482 TRs).*
""")

with col2:
    st.markdown("""
**Approach 2: Word-level**
- Trial unit: Individual words from transcript
- Output shape: (n_words, 10, 1000)
- Context window: 10 TRs (±5 around word)
- HRF delay: 3 TRs

*Uses transcript word timings. Each word becomes a trial with ±5 TRs around onset.*
""")

st.markdown("""
---

**Summary Table**

| Aspect | Approach 1: Scene-level | Approach 2: Word-level |
|--------|------------------------|------------------------|
| Trial unit | Movie scene | Individual words |
| Output shape | (4, 482, 1000) | (n_words, 10, 1000) |
| Context window | Full scene | 10 TRs |
| Implementation | notebook lines 1102-1127 | notebook lines 1158-1228 |

**For TRACE**: Both preserve multi-trial structure - either could work depending on whether 
we treat scenes or words as repeated conditions for positive pair generation.

**Reference**: [notebook.ipynb](https://github.com/petron-olateju/TRACE_Algonauts/blob/main/notebook.ipynb)

**Repository**: [TRACE_Algonauts](https://github.com/petron-olateju/TRACE_Algonauts)
""")

st.markdown("---")

st.subheader("3. TRACE Contrastive Learning")

st.markdown("""
**Core Innovation**: Generate positive pairs by averaging across different subsets of a unit's trials.

**Process**:
1. For each brain region, sample multiple trial subsets
2. Average responses within each subset → creates positive pairs
3. Use contrastive loss to learn a 2D embedding
4. Embedding captures response characteristics

**Goal**: Identify two-dimensional structure in neural responses.
""")

st.markdown("---")

st.subheader("4. Analysis & Visualization")

st.markdown("""
- **2D Embeddings**: Visualize brain regions in low-dimensional space
- **Clustering**: Identify regions with similar response profiles
- **Organization**: Group spatially proximate or functionally connected parcels

*Analysis methods in development...*
""")

st.markdown("---")

st.subheader("1.6. HCPTRT Loader Architecture")

st.markdown("""
- **HCPTRTLoader Class**: Advanced loader for HCP Test-Retest (CIFTI) dataset
- **CIFTI & Surface Support**: Handles 91k grayordinates (Surface Mesh + MNI subcortex)
- **Key Features**:
  - `hcp_utils` integration for on-the-fly parcellation (MMP, Yeo, Cole-Anticevic)
  - Automatic confound denoising (Motion, CSF, WM)
  - Cross-task stacking support (Motor, Gambling, WM, Social, Emotion, Rest)
- **Usage**:
  ```python
  from utils.dataloader import get_hcptrt_loader
  dataset = get_hcptrt_loader()
  bold = dataset.load_fmri_responses(subject="sub-01", task="motor", parcellate=True)
  ```
""")

st.markdown("---")

st.header("Current Status")

st.success("Universal functional fingerprinting validated across tasks")

st.markdown("""
| Component | Status |
|----------|--------|
| Algonauts Loader Architecture | ✅ Complete |
| HCPTRT Loader Architecture | ✅ Complete (CIFTI/Surface) |
| fMRI Data Loading | ✅ Complete |
| Cross-Task Validation | ✅ Complete (6 Tasks) |
| TRACE Framework | 🔄 Training Initialized |
""")

st.markdown("---")

st.caption("TRACE originally developed for calcium imaging in mice (Singer et al.)")

st.sidebar.markdown("""
**Progress**

Methodology overview for supervisor review.

[Home](../Home)
[Results](../Results)
""")

st.sidebar.page_link("app.py", label="← Back to Home")
