"""Data loading utilities for fMRI encoding models.

This module provides functions for loading fMRI responses, movie transcripts,
and brain atlas data from the Algonauts 2025 Challenge dataset.
"""

import ast
import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py
import yaml
import nibabel as nib
from nilearn import datasets


with open("./configs/dirs.yaml", "r") as f:
    dir_configs = yaml.safe_load(f)
TRANSCRIPT_DIR = dir_configs["TRANSCRIPT_DIR"]
FMRI_DIR = dir_configs["FMRI_DIR"]

with open("./configs/configs.yaml", "r") as f:
    configs = yaml.safe_load(f)
HRF_DELAY = configs["hrf_delay"]
TR = configs["tr"]
CONTEXT_TRS = configs["context_trs"]
SUBJECTS = configs["subjects"]

ATLAS_NAME = configs["atlas_name"]
N_PARCELS = configs["num_parcels"]
ATLAS_DESC = datasets.fetch_atlas_schaefer_2018(n_rois=N_PARCELS)
ATLAS_IMG = nib.load(ATLAS_DESC.maps)
ATLAS_LABELS = ATLAS_DESC.labels
ATLAS_DATA = ATLAS_IMG.get_fdata()
INV_AFFINE = np.linalg.inv(ATLAS_IMG.affine)


def _read_tsv(file_path: str) -> pd.DataFrame:
    """Read a TSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed TSV data.
    """
    return pd.read_csv(file_path, sep="\t")


def get_fmri_file_path(dir: str, subject: int, split: str = "train") -> str:
    """Build the file path for an fMRI data file.

    Parameters
    ----------
    dir : str
        Base directory containing fMRI data.
    subject : int
        Subject ID (1, 2, 3, or 5).
    split : str, optional
        'train' for Friends dataset or 'test' for Movie10 dataset.
        Default is 'train'.

    Returns
    -------
    str
        Full path to the fMRI HDF5 file.
    """
    fmri_file_path = f"{dir}/sub-0{subject}/func/"
    if split == "train":
        fmri_file_path += f"sub-0{subject}_task-friends_"
        fmri_file_path += "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    else:
        fmri_file_path += f"sub-0{subject}_task-movie10_"
        fmri_file_path += (
            "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
        )

    return fmri_file_path


def list_fmri_sessions(dir: str, subject: int, split: str = "train") -> List[str]:
    """List available fMRI session keys in an HDF5 file.

    Parameters
    ----------
    dir : str
        Base directory containing fMRI data.
    subject : int
        Subject ID (1, 2, 3, or 5).
    split : str, optional
        'train' for Friends dataset or 'test' for Movie10 dataset.
        Default is 'train'.

    Returns
    -------
    List[str]
        List of session keys (e.g., 'ses-001_task-s01e02a').
    """
    fmri_file_path = get_fmri_file_path(dir, subject, split)

    with h5py.File(fmri_file_path, "r") as fmri_file:
        fmri_keys = list(fmri_file.keys())
    return fmri_keys


def load_fmri_responses(
    dir: str, subject: int, stimuli_name: str, split: str = "train"
) -> np.ndarray:
    """Load fMRI response data for a specific stimuli session.

    Parameters
    ----------
    dir : str
        Base directory containing fMRI data.
    subject : int
        Subject ID (1, 2, 3, or 5).
    stimuli_name : str
        Session key (e.g., 'ses-001_task-s01e02a').
    split : str, optional
        'train' for Friends dataset or 'test' for Movie10 dataset.
        Default is 'train'.

    Returns
    -------
    np.ndarray
        fMRI response matrix of shape (n_timepoints, n_parcels).
    """
    fmri_file_path = get_fmri_file_path(dir, subject, split)

    with h5py.File(fmri_file_path, "r") as fmri_file:
        fmri_data = fmri_file[stimuli_name][()]
    return fmri_data


def _load_friends_transcript(
    dir: str, stimuli_name: str, ignore_nans: bool = False
) -> pd.DataFrame:
    """Load transcript for a Friends episode.

    Parameters
    ----------
    dir : str
        Base directory containing transcript data.
    stimuli_name : str
        Session key (e.g., 'ses-001_task-s01e02a').
    ignore_nans : bool, optional
        If True, drop rows where 'text_per_tr' is NaN. Default is False.

    Returns
    -------
    pd.DataFrame
        Transcript DataFrame with columns:
        text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr.
    """
    season = stimuli_name[14:16]
    episode_name = stimuli_name[13:]
    transcript_path = f"{dir}/friends/s{int(season)}/friends_{episode_name}.tsv"
    df = _read_tsv(transcript_path)
    if ignore_nans:
        df = df.dropna(subset=["text_per_tr"])
    return df


def _load_movie10_transcript(
    dir: str, stimuli_name: str, ignore_nans: bool = False
) -> pd.DataFrame:
    """Load transcript for a Movie10 clip.

    Parameters
    ----------
    dir : str
        Base directory containing transcript data.
    stimuli_name : str
        Session key (e.g., 'ses-001_task-movie10_clip01').
    ignore_nans : bool, optional
        If True, drop rows where 'text_per_tr' is NaN. Default is False.

    Returns
    -------
    pd.DataFrame
        Transcript DataFrame with columns:
        text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr.
    """
    movie_id = stimuli_name[13:-2]
    clip_name = stimuli_name[13:]
    transcript_path = f"{dir}/movie10/{movie_id}/movie10_{clip_name}.tsv"
    df = _read_tsv(transcript_path)
    if ignore_nans:
        df = df.dropna(subset=["text_per_tr"])
    return df


def load_transcript(
    dir: str, stimuli_name: str, split: str = "train", ignore_nans: bool = False
) -> pd.DataFrame:
    """Load transcript for a given stimuli session.

    Automatically routes to the appropriate loader based on split type.

    Parameters
    ----------
    dir : str
        Base directory containing transcript data.
    stimuli_name : str
        Session key identifying the stimuli.
    split : str, optional
        'train' for Friends episodes or 'test' for Movie10 clips.
        Default is 'train'.
    ignore_nans : bool, optional
        If True, drop rows where 'text_per_tr' is NaN. Default is False.

    Returns
    -------
    pd.DataFrame
        Transcript DataFrame with columns:
        text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr.
    """
    if split == "train":
        return _load_friends_transcript(dir, stimuli_name, ignore_nans)
    else:
        return _load_movie10_transcript(dir, stimuli_name, ignore_nans)


def epoch_fmri_by_words(
    stimuli_name: str,
    fmri_data: np.ndarray,
    transcript_dir: str = TRANSCRIPT_DIR,
    fmri_dir: str = FMRI_DIR,
    hrf_delay: int = HRF_DELAY,
    tr: float = TR,
    context_trs: int = CONTEXT_TRS,
    split: str = "train",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Create epoched fMRI response windows aligned to word onsets.

    For each transcript row containing words, extracts an fMRI response window
    centered around the HRF-delayed timepoint, accounting for the hemodynamic
    response delay.

    Parameters
    ----------
    stimuli_name : str
        Session key identifying the stimuli.
    fmri_data : np.ndarray
        fMRI response matrix of shape (n_timepoints, n_parcels).
    transcript_dir : str, optional
        Directory containing transcript data. Default is TRANSCRIPT_DIR.
    fmri_dir : str, optional
        Directory containing fMRI data. Default is FMRI_DIR.
    hrf_delay : int, optional
        Number of TRs to shift for hemodynamic response delay. Default is HRF_DELAY.
    tr : float, optional
        Repetition time in seconds. Default is TR.
    context_trs : int, optional
        Number of TRs before and after the target to include in each epoch.
        Default is CONTEXT_TRS.
    split : str, optional
        'train' for Friends episodes or 'test' for Movie10 clips. Default is 'train'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray] or Tuple[None, None, None]
        epochs : np.ndarray or None
            Stacked epoched fMRI responses of shape (n_epochs, 2*context_trs, n_parcels).
        start : np.ndarray or None
            Start indices for each epoch.
        end : np.ndarray or None
            End indices for each epoch.
        Returns None tuple if no valid epochs found.
    """
    try:
        transcript_df = load_transcript(
            transcript_dir, stimuli_name, split, ignore_nans=True
        )
    except FileNotFoundError as e:
        print(FileNotFoundError(e))
        return None, None, None

    epochs = []
    start = []
    end = []

    for tr_idx, row in transcript_df.iterrows():
        words = row["words_per_tr"]
        if isinstance(words, str):
            words = ast.literal_eval(words)
        if not isinstance(words, list):
            continue

        if len(words) > 0:
            fmri_idx = tr_idx + hrf_delay
            start_idx = fmri_idx - context_trs
            end_idx = fmri_idx + context_trs

            if (start_idx >= 0) and (end_idx < len(fmri_data)):
                epoch = fmri_data[start_idx:end_idx, :]
                epochs.append(epoch)
                start.append(start_idx)
                end.append(end_idx)

    if len(epochs) > 0:
        return np.stack(epochs), np.array(start), np.array(end)
    return None, None, None


def get_parcel_label(coord: np.ndarray) -> Optional[str]:
    """Get the Schaefer atlas label for a 3D coordinate.

    Parameters
    ----------
    coord : np.ndarray
        World coordinates of shape (3,).

    Returns
    -------
    str or None
        Parcel label (e.g., 'LH_Vis_1') or None if outside brain/background.
    """
    voxel = nib.affines.apply_affine(INV_AFFINE, coord).astype(int)

    if (
        0 <= voxel[0] < ATLAS_DATA.shape[0]
        and 0 <= voxel[1] < ATLAS_DATA.shape[1]
        and 0 <= voxel[2] < ATLAS_DATA.shape[2]
    ):
        label_idx = ATLAS_DATA[voxel[0], voxel[1], voxel[2]]

        if label_idx == 0:
            print("Background/Non-Brain")
            return None

        return ATLAS_LABELS[int(label_idx)]

    print("Outside Volume Bounds")
    return None


def get_atlas_file_path(subject: int, dir: str = FMRI_DIR) -> str:
    """Build the file path for a subject's atlas file.

    Parameters
    ----------
    subject : int
        Subject ID (1, 2, 3, or 5).
    dir : str, optional
        Base directory containing fMRI data. Default is FMRI_DIR.

    Returns
    -------
    str
        Full path to the atlas NIfTI file.
    """
    atlas_path = f"{dir}/sub-0{subject}/atlas/"
    atlas_path += f"sub-0{subject}"
    atlas_path += "_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    return atlas_path


def load_atlas_for_subject(
    subject: int, dir: str = FMRI_DIR
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
    """Load brain atlas data for a specific subject.

    Extracts parcel centroids and metadata (hemisphere, region, region index)
    from the subject's Schaefer parcellation.

    Parameters
    ----------
    subject : int
        Subject ID (1, 2, 3, or 5).
    dir : str, optional
        Base directory containing fMRI data. Default is FMRI_DIR.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict]
        parcel_coords : np.ndarray
            Centroid coordinates for each parcel, shape (n_parcels, 3).
        parcel_ids : np.ndarray
            Unique parcel IDs.
        parcel_desc : Dict
            Dictionary mapping parcel_id to metadata dict with keys:
            'hemisphere', 'region', 'region_idx'.
    """
    atlas_path = get_atlas_file_path(subject, dir)
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    subject_voxel_indices = np.argwhere(atlas_data > 0)
    world_coords = nib.affines.apply_affine(affine, subject_voxel_indices)
    labels = atlas_data[atlas_data > 0]
    labels = np.round(labels).astype(int)

    unique_parcels = np.unique(labels)
    parcel_coords_matrix = np.zeros((len(unique_parcels), 3))
    parcel_desc = {}

    for i, parcel_id in enumerate(unique_parcels):
        mask = labels == parcel_id
        centroid = world_coords[mask].mean(axis=0)
        parcel_coords_matrix[i] = centroid

        parcel_name = get_parcel_label(centroid)[10:]
        parcel_name = parcel_name.split("_")

        parcel_desc[parcel_id] = {
            "hemisphere": parcel_name[0],
            "region": "".join(parcel_name[1:-1]),
            "region_idx": parcel_name[-1],
        }

    return parcel_coords_matrix, unique_parcels, parcel_desc
