import ast
import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py
import yaml
import nibabel as nib
from nilearn import datasets

from tqdm import tqdm

from utils import dataloader


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


def pad_to_width(arr, target_width, pad_value=0):
    current_width = arr.shape[1]

    if current_width >= target_width:
        return arr  # or slice if you want truncation

    pad_amount = target_width - current_width

    return np.pad(
        arr,
        pad_width=((0, 0), (0, pad_amount)),  # (rows, cols)
        mode="constant",
        constant_values=pad_value,
    )


def window_mean(arr, num_windows=4):
    N, T = arr.shape

    # Trim T so it divides evenly
    T_trim = (T // num_windows) * num_windows
    arr_trim = arr[:, :T_trim]

    # Reshape and average
    reshaped = arr_trim.reshape(N, num_windows, T_trim // num_windows)
    return reshaped.mean(axis=2)


def signal_windows(arr, num_windows=4):
    N, T = arr.shape

    # Trim T so it divides evenly
    T_trim = (T // num_windows) * num_windows
    arr_trim = arr[:, :T_trim]

    # Reshape
    reshaped = arr_trim.reshape(N, num_windows, T_trim // num_windows)

    return reshaped


def parcel_samples(
    subject,
    split: str = "train",
    fmri_dir: str = FMRI_DIR,
    trials: str = "episodes",
    time_collapse: Optional[str] = None,
    pad_width: int = 500,
    n_windows: int = 4,
    n_subsamples: int = 4,
    n_episodes=1
):
    """
    Extract parcel-wise fMRI response samples for a given subject.

    Parameters
    ----------
    subject : str
        Subject identifier.
    split : str, default "train"
        Data split, either "train" or "test".
    fmri_dir : str, default FMRI_DIR
        Directory containing fMRI data.
    trials : str, default "episodes"
        Sampling mode:
        - "episodes": one row per stimulus episode (rows are parcels).
        - "within_episodes": each stimulus is split into `n_subsamples` windows,
          producing one row per parcel per window (C * W rows per stimulus).
    time_collapse : str or None, default None
        Post-processing applied after mode-specific reshaping:
        - None: pad/truncate responses to fixed width `pad_width`.
        - "windowed_mean": average within `n_windows` temporal windows.
    pad_width : int, default 500
        Target width for padding when `time_collapse` is None.
    n_windows : int, default 4
        Number of windows for `time_collapse == "windowed_mean"`.
    n_subsamples : int, default 4
        Number of subsample windows per stimulus when `trials == "within_episodes"`.

    Returns
    -------
    X : ndarray
        Response array of shape (n_samples, n_features).
        - `episodes` mode: (n_parcels * n_stimuli, n_features)
        - `within_episodes` mode: (n_parcels * n_subsamples * n_stimuli, n_features)
    Y : DataFrame
        Labels with one row per sample:
        - `hemisphere`, `region`, `parcel`: parcel anatomical metadata.
        - `x`, `y`, `z`, `radius`: parcel centroid coordinates.
        - `season`, `episode`, `episode_split`: stimulus identifiers (train split only).
        - `time`: subsample window index per sample (only in `within_episodes` mode).

    Notes
    -----
    - In `within_episodes` mode, the `time` column reflects the original subsample
      window index (0 to n_subsamples-1) before any `time_collapse` transformation.
    - Label ordering (hemisphere, region, parcel) follows C-major flattening:
      all parcels for window 0, then all parcels for window 1, etc.
      This matches the reshape order from `signal_windows()`.
    """
    assert trials in ["episodes", "within_episodes"]
    assert time_collapse in [None, "windowed_mean"]
    if time_collapse is None:
        assert pad_width != 0
    if time_collapse == "windowed_mean":
        assert n_windows >= 1
    assert n_episodes >= 1

    subject_data = dataloader.load_episode_fmri(subject, split, fmri_dir)
    scenes_response = subject_data["scenes_response"]
    parcel_desc = subject_data["parcel_desc"]
    stimulus = list(scenes_response.keys())[0:n_episodes]

    fmri_response: List[np.ndarray] = []
    coordinates: List[np.ndarray] = []
    hemisphere: List[str] = []
    region: List[str] = []
    parcel: List[str] = []
    season: List[int] = []
    episode: List[int] = []
    episode_split: List[str] = []
    # time: List[int] = []

    for stimuli in stimulus:
        _response = scenes_response[stimuli].T
        n_parcels = scenes_response[stimuli].shape[-1]

        if trials == "within_episodes":
            _response = signal_windows(_response, num_windows=n_subsamples)
            n_w, n_t = _response.shape[1], _response.shape[2]
            _response = _response.reshape(-1, n_t)
            # time.extend([i for i in range(n_w) for _ in range(n_parcels)])

        if time_collapse is None:
            _response = pad_to_width(_response, pad_width)
        elif time_collapse == "windowed_mean":
            _response = window_mean(_response, n_windows)

        fmri_response.append(_response)
        coordinates.append(subject_data["parcel_coords"])

        if trials == "within_episodes":
            multiplier = n_w
        else:
            multiplier = 1

        for p in range(n_parcels):
            h = parcel_desc[p + 1]["hemisphere"]
            r = parcel_desc[p + 1]["region"]
            hemisphere.extend([h] * multiplier)
            region.extend([r] * multiplier)
            parcel.extend([f"{h}_{r}"] * multiplier)

        if split == "train":
            n_rows = _response.shape[0]
            season.extend([int(stimuli[1:3])] * n_rows)
            episode.extend([int(stimuli[4:6])] * n_rows)
            episode_split.extend([stimuli[-1]] * n_rows)

    X = np.vstack(fmri_response)
    coords = np.vstack(coordinates)

    if trials == "within_episodes":
        coords = np.tile(coords, (n_subsamples, 1))

    Y = pd.DataFrame(
        {
            "hemisphere": hemisphere,
            "region": region,
            "parcel": parcel,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "radius": np.sqrt(
                coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2
            ),
            "season": season,
            "episode": episode,
            "episode_split": episode_split,
        }
    )

    # if trials == "within_episodes":
    #     Y["time"] = [t for group in time for t in group]

    return X, Y
