# Results: 2026-04-22 10:00 (Cross-Task Integration & HCPTRT Pipeline)

## 📊 Summary of Findings
The implementation and validation of the **HCP Test-Retest (HCPTRT)** pipeline marks a transition from stimulus-specific analysis (Algonauts) to **universal functional fingerprinting**. By stacking neural activity across six distinct cognitive tasks, we have demonstrated that brain regions maintain a stable "functional identity" regardless of the cognitive state (e.g., motor vs. social). This validates the use of multi-task data for generating robust, state-invariant contrastive embeddings.

### 1. Cross-Task Anatomical Stability
Experiments show that even when parcels are represented by their activity across a diverse range of tasks (**Motor, Gambling, WM, Social, Emotion, and Resting State**), they cluster primarily by **Anatomical Lobe**. This indicates that the "where" (anatomical location) is a stronger driver of the long-term functional signal than the "what" (current task-specific demand).

### 2. Multi-Task "Fingerprinting"
The `hcptrt_across_tasks` experiment stacked temporal activity across different domains. The resulting 2D embeddings show that parcels from the same MMP region (e.g., V1, MST) cluster tightly together even though the data includes completely different cognitive demands. This stability is the prerequisite for the **TRACE** framework to learn meaningful cross-task representations.

### 3. Surface-Based High Resolution (MMP)
Transitioning to the **Glasser MMP (360 regions)** parcellation on the CIFTI 91k grayordinate surface has significantly increased the "cleanness" of the clusters compared to volume-based parcellations. The surface-based denoising (regressing out motion and physiological noise) has yielded a high signal-to-noise ratio suitable for deep contrastive learning.

### 4. Space Distinction: Grayordinate (HCP) vs. Volumetric (Algonauts)
A key driver of the improved results in this update is the shift in data representation:
*   **Algonauts (Pure Volumetric):** Uses MNI152 3D voxels for the entire brain. This is prone to "signal bleeding" across cortical folds where two functionally distinct regions touch in 3D space.
*   **HCPTRT (Hybrid Grayordinate):** Uses a high-resolution **Surface Mesh** for the cortex (~59k vertices) and **MNI voxels** only for subcortical structures (~32k voxels). Measuring distance along the cortical ribbon rather than through 3D space prevents signal contamination, resulting in the sharper, more biologically accurate clusters seen in the `hcptrt_across_tasks` embeddings.

---

## 📽️ Presentation Strategy: "Beyond Stimulus-Specific Patterns"

### Slide 1: Universal Functional Identity
*   **Objective:** Show that brain regions have a signature that transcends specific tasks.
*   **Primary Image:** `experiments/hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png`
*   **Talking Point:** "By stacking 6 tasks, we see that the brain's fundamental organization is preserved. A visual parcel in a 'Social' task still looks like a visual parcel in a 'Motor' task. This universal fingerprint is what our model will learn."

### Slide 2: Embedding Robustness (t-SNE vs UMAP)
*   **Objective:** Demonstrate that the organizational findings are not an artifact of a single algorithm.
*   **Compare:**
    *   `experiments/hcptrt_across_tasks/tSNE_all_tasks@perplexity=15.png`
    *   `experiments/hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png`
*   **Talking Point:** "Both t-SNE and UMAP consistently recover the lobar architecture (Occipital, Parietal, Temporal, etc.) from multi-task data. This confirms that our data representation is capturing genuine biological structure."

### Slide 3: The Advantage of Surface Data
*   **Objective:** Explain why moving to HCPTRT (CIFTI) was necessary.
*   **Talking Point:** "Unlike the volume-based data in Algonauts, the CIFTI grayordinate space allows us to parcellate exactly on the cortical ribbon. This reduces signal bleeding between adjacent folds and results in the highly distinct clusters we see in our multi-task embeddings."

---

## 🚀 Technical Implementation (HCPTRT Pipeline)

### utils/loaders/hcptrt.py
*   **CIFTI Support**: Native loading of `.dtseries.nii` files (91,282 grayordinates).
*   **Denoising**: Integrated regression of 24 motion parameters + CSF/White Matter signals.
*   **Atlas Mapping**: Built-in `MMP_LOBE` mapping to categorize Glasser regions into 8 broad lobes (Occipital, Parietal, Somatomotor, Insular, Temporal, Prefrontal, Cingulate, Orbitofrontal).

### utils/preprocessing.py
*   **`parcel_samples` Switchboard**: Updated to detect the loader type and route to `parcel_samples_hcptrt`.
*   **HCP Sampling Modes**:
    *   `trials="continuous"`: Full run analysis used in the current cross-task stacking.
    *   `time_collapse="windowed_mean"`: Essential for stabilizing the embedding by reducing temporal noise.

---

## ⚠️ Current Observations
*   **Subcortical Accuracy**: Some subcortical regions show higher variance in clustering, likely due to lower SNR in surface-targeted sequences.
*   **Task Weighting**: We are currently weighting all tasks equally. Future iterations might explore if specific tasks (like Resting State) provide a better "baseline" for contrastive learning.

## 🚀 Next Steps
*   **TRACE Training on HCPTRT**: Initialize the contrastive model using the cross-task stacked matrices.
*   **Intersubject Validation**: Test if a model trained on Subject 1's multi-task fingerprints can correctly identify parcels in Subject 2.
*   **Algonauts-HCP Bridge**: Use the universal fingerprints learned here to decode specific movie-watching activity in the Algonauts dataset.

---

# Results: 2026-04-16 14:30 (Analysis of Pipeline Validation)

## 📊 Summary of Findings
The exploratory experiments conducted in the `experiments/` folder successfully validate the dataloader and preprocessing pipeline. By projecting high-dimensional parcel activity into 2D space, we have confirmed that the current data representation captures fundamental neuroanatomical and functional properties. This provides a solid foundation for the implementation of the TRACE contrastive learning framework.

### 1. Neuroanatomical Consistency
Experiments show that brain parcels cluster naturally by **Lobe** and **Hemisphere**. This indicates that the functional signal extracted by the dataloader is "clean" enough to preserve known brain architecture, even after windowed averaging.

### 2. Temporal Stability
By subsampling episodes into multiple windows, we observed that parcels maintain their functional identity over time. This temporal consistency is vital for contrastive learning, as it justifies using different time segments of the same parcel as "positive pairs."

### 3. Scaling Impact (Global vs. Local)
Comparing the main experiments (`minmax_time` scaling) with the `v1` folders (`minmax` global scaling) reveals how normalization affects clustering:
*   **Local Scaling (`minmax_time`):** Normalizes each parcel's activity relative to its own range. This emphasizes *relative* temporal dynamics.
*   **Global Scaling (`minmax`):** Normalizes activity across the entire dataset. This preserves the *absolute* magnitude differences between parcels, which often results in tighter anatomical clusters but may mask subtle temporal features.

---

## 📽️ Presentation Strategy: "From Anatomy to Embeddings"

If presenting these results, use the following sequence to demonstrate pipeline validity.

### Slide 1: The Baseline - Anatomical Organization
*   **Objective:** Show that the pipeline captures the "ground truth" of brain structure.
*   **Primary Image:** `experiments/subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png`
*   **Talking Point:** "Using a single episode and Subject 1, our preprocessing preserves clear clustering by Lobe and Hemisphere. The data 'knows' where it came from in the brain."

### Slide 2: Scaling Sensitivity (The "v1" Comparison)
*   **Objective:** Explain how data normalization affects our feature space.
*   **Compare:**
    *   `experiments/subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png` (Local/Time Scaling)
    *   **vs.**
    *   `experiments/subject_episode+parcel_trials_clustering/v1/UMAP@n_neighbors=50.png` (Global Scaling)
*   **Talking Point:** "Global scaling (v1) tightens anatomical clusters by preserving absolute signal differences. Local scaling focus more on the 'shape' of activity over time. For TRACE, we must choose the scaling that highlights the features we want our model to learn."

### Slide 3: Temporal Persistence
*   **Objective:** Prove that regions have a stable "fingerprint" across a movie episode.
*   **Primary Image:** `experiments/subject_episode+parcel+window_subsample_trials/UMAP@n_neighbors=50.png`
*   **Talking Point:** "When we subsample the episode, parcels still cluster with their anatomical neighbors. This stability allows us to define positive samples for contrastive learning across different time-windows."

### Slide 4: Robustness Across Stimuli
*   **Objective:** Show that these findings aren't specific to one movie scene.
*   **Primary Image:** `experiments/subject+multi_episodes+parcel_trials_clustering/UMAP@n_neighbors=50.png`
*   **Talking Point:** "Scaling up to 10 episodes, the functional-anatomical organization remains robust. The pipeline generalizes across diverse visual and auditory stimuli."

### Slide 5: The Path Forward (TRACE)
*   **Conclusion:** The pipeline is validated.
*   **Next Steps:** Proceed to train the Contrastive Embedding model using these validated positive/negative sampling strategies.

---

## ⚠️ Current Limitations
*   **Single Trial Data:** The Algonauts 2025 (CNeuroMod) dataset naturally lacks multiple trials for the same subject-scene. In contrastive learning, multiple trials are ideal for defining "positive pairs" (the same brain responding to the same stimulus twice).
*   **Subsampling Workaround:** Our current approach uses windowed subsampling within a single episode to simulate "trials." While this has validated the functional stability of our regions, it is a temporal proxy rather than a true stimulus-repetition baseline.
*   **Scaling Ambiguity:** We are still evaluating whether global or local normalization provides the most "contrastive-friendly" signal for the model.

---

## 🚀 Next Steps
*   **Incorporate Human Connectome Project (HCP) Data:** We will scale TRACE to the HCP movie dataset, which contains approximately **15 trials per subject-scene**. This will provide the necessary redundancy to move from temporal-workarounds to true trial-based contrastive embeddings.
*   **TR-Level Alignment:** Further refine the alignment of movie transcripts with TR-level activity to investigate if fine-grained semantic features (e.g., specific words) result in distinct clustering within the functional-anatomical regions.
*   **TRACE Embedding Training:** Initialize the contrastive learning architecture using the windowed preprocessing settings that yielded the best anatomical separation in these validation steps.

---

# Results: 2026-04-15 10:30 (Loading and Preprocessing Pipeline)

## Overview
The loading and preprocessing pipeline provides utilities for extracting and preparing fMRI data from the Algonauts 2025 (CNeuroMod) naturalistic movie dataset. The pipeline supports loading continuous brain responses across multiple episodes, extracting word-locked epochs, and transforming the data into formats suitable for downstream machine learning analysis. The dataset comprises 4 subjects (1, 2, 3, 5) watching ~80 hours of multimodal movies (Friends TV series + Movie10 clips) with fMRI recorded at TR = 1.49s. Brain responses are parcellated using the Schaefer 1000 (7 networks) atlas.

## utils/dataloader.py
The dataloader module provides functions for loading fMRI responses, movie transcripts, and brain atlas metadata.

### Data Loading Functions
- **list_fmri_sessions()**: Lists available fMRI session keys in an HDF5 file for a given subject and split (train = Friends, test = Movie10).
- **load_fmri_responses()**: Loads the fMRI response matrix (timepoints × parcels) for a specific stimuli session.
- **get_fmri_file_path()**: Builds the file path for an fMRI HDF5 file based on subject and split.

### Transcript Loading Functions
- **load_transcript()**: Loads movie transcript for a given stimuli session. Automatically routes to Friends or Movie10 loader based on split type.
- **_load_friends_transcript()**: Loads transcript for a Friends episode (season/episode structure).
- **_load_movie10_transcript()**: Loads transcript for a Movie10 clip.

### Epoching Functions
- **epoch_fmri_by_words()**: Creates epoched fMRI response windows aligned to word onsets. For each word in the transcript, extracts an fMRI window centered around the HRF-delayed timepoint (default 3 TRs delay). Each epoch has a configurable context window (default ±5 TRs around the word).

### High-Level Loading Functions
- **load_episode_fmri()**: Loads continuous fMRI timeseries for all episodes for a given subject. Returns a dictionary with scene responses, parcel coordinates, parcel IDs, and parcel descriptions.
- **load_episode_word_epochs()**: Loads word-locked fMRI epochs for all episodes. Each epoch is time-locked to a spoken word onset, shifted by HRF delay.

### Atlas Functions
- **load_atlas_for_subject()**: Loads the Schaefer 1000-parcel atlas for a specific subject. Extracts parcel centroids and metadata (hemisphere, region, region index) from the NIfTI parcellation file.
- **get_parcel_label()**: Returns the Schaefer atlas label for a 3D coordinate.

## utils/preprocessing.py
The preprocessing module provides signal processing functions for transforming fMRI data into sample matrices.

### Signal Processing Functions
- **pad_to_width()**: Pads or truncates response arrays to a fixed width for consistent dimensionality.
- **window_mean()**: Averages fMRI responses across temporal windows. Reduces the time dimension by computing mean within N windows.
- **signal_windows()**: Splits fMRI responses into temporal windows without averaging. Returns the windowed array for downstream processing.

### Main Extraction Function
- **parcel_samples()**: The primary function for extracting parcel-wise fMRI response samples. Provides multiple modes for different analysis approaches:

**Sampling Modes (trials parameter):**
- `episodes`: One row per stimulus episode. Each row contains the full timecourse for all parcels.
- `within_episodes`: Each stimulus is split into N subsample windows, producing one row per parcel per window. This enables temporal analysis within episodes.

**Time Collapse Options:**
- `None`: Pad/truncate responses to fixed width (default 500 TRs).
- `windowed_mean`: Average within N temporal windows, producing a fixed number of timepoints per sample.

**Returns:**
- X: Response array (n_samples, n_features) - parcel timecourses or windowed averages.
- Y: Labels DataFrame with hemisphere, region, parcel identity, 3D coordinates, radius, and stimulus identifiers (season, episode, episode_split).

## Data Loading Modes in Practice

### Mode 1: Continuous Episode Response
Used for scene-level analysis where each movie scene is one trial. The dataloader loads full fMRI timeseries for each episode, and preprocessing can either pad to fixed width or apply windowed mean for dimensionality reduction.

### Mode 2: Word-Locked Epochs
Used for fine-grained temporal analysis. The dataloader epochs fMRI to individual word onsets with HRF delay (±5 TRs context). Each word becomes a trial with a short temporal window around its onset.

### Mode 3: Windowed Subsampling
Within an episode, the data can be split into multiple temporal windows. This is useful for simulating "trials" within a single long movie, where each window of the same parcel can serve as a positive pair in contrastive learning.

---

# Results: 2026-04-18 10:30 (AlgonautsLoader Class Implementation)

## Summary
Refactored the dataloader module to use an object-oriented design with the `AlgonautsLoader` class, making it extensible for future datasets (e.g., HCP).

## Changes Made

### 1. Created `AlgonautsLoader` Class (`utils/dataloader.py`)
- Converted module-level functions into class methods
- Added `_load_configs()` method that loads YAML configs into `self.configs`
- Added `_init_atlas()` method for Schaefer atlas initialization

### 2. Restructured YAML Configs
- **`configs/dirs.yaml`**:
  ```yaml
  algonauts:
    dirs:
      fmri: "data/algonauts_2025.competitors/fmri"
      transcript: "data/algonauts_2025.competitors/stimuli/transcripts"
  ```

- **`configs/configs.yaml`**:
  ```yaml
  algonauts:
    params:
      tr: 1.49
      hrf_delay: 3
      context_trs: 5
      atlas_name: '1000Parcels_7Networks_order'
      num_parcels: 1000
    subjects: [1, 2, 3, 5]
  ```

### 3. Simplified `AlgonautsLoader.__init__`
- Now only takes `dataset` and `split` arguments
- All config values loaded from YAML via `_load_configs()`
- `self.configs` stores merged configs (dirs, params, subjects)
- Instance attributes (`fmri_dir`, `transcript_dir`, `hrf_delay`, etc.) set from configs

### 4. Updated `load_fmri_responses` Method
- Added required `subject: int` argument
- Fixed bug where `subject=None` was hardcoded

### 5. Updated `utils/preprocessing.py`
- Changed import to `from utils.dataloader import AlgonautsLoader, get_default_dataset`
- Updated `parcel_samples()` to take `dataset: AlgonautsLoader` as first argument

## Usage Example
```python
from utils.dataloader import get_default_dataset
from utils.preprocessing import parcel_samples

dataset = get_default_dataset()
X, Y = parcel_samples(dataset, subject=1, split="train")
```
