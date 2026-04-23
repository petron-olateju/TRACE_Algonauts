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

from utils.dataloader import AlgonautsLoader, HCPTRTLoader
from utils.loaders.parcel_maps import SCHAEFER_LOBE, get_lobe


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


def parcel_samples_algonauts(
    dataset: AlgonautsLoader,
    subject,
    split: str = "train",
    trials: str = "episodes",
    time_collapse: Optional[str] = None,
    pad_width: int = 500,
    n_windows: int = 4,
    n_subsamples: int = 4,
    n_episodes=1
):
    """
    Extract parcel-wise fMRI response samples for AlgonautsLoader.

    Parameters
    ----------
    dataset : AlgonautsLoader
        Dataset loader instance.
    subject : str
        Subject identifier.
    split : str, default "train"
        Data split, either "train" or "test".
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

    subject_data = dataset.load_episode_fmri(subject, split)
    scenes_response = subject_data["scenes_response"]
    parcel_desc = subject_data["parcel_desc"]
    stimulus = list(scenes_response.keys())[0:n_episodes]

    fmri_response: List[np.ndarray] = []
    coordinates: List[np.ndarray] = []
    hemisphere: List[str] = []
    region: List[str] = []
    lobe: List[str] = []
    structure_type: List[str] = []
    parcel: List[str] = []
    season: List[int] = []
    episode: List[int] = []
    episode_split: List[str] = []

    for stimuli in stimulus:
        _response = scenes_response[stimuli].T
        n_parcels = scenes_response[stimuli].shape[-1]

        if trials == "within_episodes":
            _response = signal_windows(_response, num_windows=n_subsamples)
            n_w, n_t = _response.shape[1], _response.shape[2]
            _response = _response.reshape(-1, n_t)

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
            # Schaefer labels may include a numeric sub-index suffix (e.g. "Vis_1").
            # Strip the trailing "_N" to get the clean network key for SCHAEFER_LOBE.
            r_key = r.rsplit("_", 1)[0] if r[-1].isdigit() else r
            l  = SCHAEFER_LOBE.get(r_key, r_key)
            hemisphere.extend([h] * multiplier)
            region.extend([r] * multiplier)
            lobe.extend([l] * multiplier)
            structure_type.extend(["cortical"] * multiplier)
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
            "hemisphere":     hemisphere,
            "region":         region,
            "lobe":           lobe,
            "structure_type": structure_type,
            "parcel":         parcel,
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

    return X, Y


def parcel_samples_hcptrt(
    dataset: HCPTRTLoader,
    subject: str,
    task: str,
    session: Optional[str] = None,
    run: Optional[int] = None,
    trials: str = "continuous",
    time_collapse: Optional[str] = None,
    pad_width: int = 500,
    n_windows: int = 4,
):
    """
    Extract parcel-wise fMRI response samples for HCPTRTLoader.

    Parameters
    ----------
    dataset : HCPTRTLoader
        Dataset loader instance.
    subject : str
        Subject identifier (e.g., "sub-01").
    task : str
        Task name (e.g., "motor", "wm", "emotion").
    session : str, optional
        Specific session to load (e.g., "ses-001"). If None, loads all sessions.
    run : int, optional
        Specific run number to load (e.g., 1). If None, loads all runs for
        the specified session(s).
    trials : str, default "continuous"
        Sampling mode:
        - "continuous": load continuous BOLD timeseries via load_task_fmri().
        - "events": load block-epoched BOLD via load_task_epochs().
    time_collapse : str or None, default None
        Post-processing applied after mode-specific reshaping:
        - None: pad/truncate responses to fixed width `pad_width`.
        - "windowed_mean": average within `n_windows` temporal windows.
    pad_width : int, default 500
        Target width for padding when `time_collapse` is None.
    n_windows : int, default 4
        Number of windows for `time_collapse == "windowed_mean"`.

    Returns
    -------
    X : ndarray
        Response array of shape (n_samples, n_features).
        - `continuous` mode: (n_parcels * n_timepoints, n_features)
        - `events` mode: (n_parcels * n_blocks, n_features)
    Y : DataFrame
        Labels with one row per sample:
        - `hemisphere`, `region`, `parcel`: parcel anatomical metadata.
        - `x`, `y`, `z`, `radius`: parcel centroid coordinates.
        - `session`: session number (from "ses-XXX").
        - `run`: run number (from "run-XX").
        - `task`: task name (e.g., "motor", "wm").
        - `block_label`: condition label (only in `events` mode, e.g., "left_hand").

    Notes
    -----
    - Label ordering (hemisphere, region, parcel) follows C-major flattening.
    """
    assert trials in ["continuous", "events"]
    assert time_collapse in [None, "windowed_mean"]
    if time_collapse is None:
        assert pad_width != 0
    if time_collapse == "windowed_mean":
        assert n_windows >= 1

    if trials == "continuous":
        subject_data = dataset.load_task_fmri(subject, task, session=session, run=run, parcellate=True)
    else:
        subject_data = dataset.load_task_epochs(subject, task, session=session, run=run, parcellate=True)

    scenes_response = subject_data["scenes_response"]
    parcel_desc = subject_data["parcel_desc"]

    fmri_response: List[np.ndarray] = []
    coordinates: List[np.ndarray] = []
    hemisphere: List[str] = []
    region: List[str] = []
    lobe: List[str] = []
    structure_type: List[str] = []
    parcel: List[str] = []
    session_list: List[int] = []
    run_list: List[int] = []
    block_label: List[str] = []

    parcel_ids = sorted(parcel_desc.keys())  # authoritative ordered parcel IDs

    for key, data in scenes_response.items():
        if trials == "events":
            _response = data["trials"]          # (n_blocks, window_trs, n_parcels)
            _labels   = data["labels"]
            n_blocks     = _response.shape[0]
            n_timepoints = _response.shape[1]
            n_parcels    = _response.shape[2]
            # reshape to (n_blocks * n_parcels, n_timepoints)
            _response = _response.transpose(0, 2, 1).reshape(-1, n_timepoints)
        else:
            _response = data.T                  # (n_parcels, T)
            n_parcels = data.shape[-1]
            n_blocks  = 1
            _labels   = None

        if time_collapse is None:
            _response = pad_to_width(_response, pad_width)
        elif time_collapse == "windowed_mean":
            _response = window_mean(_response, n_windows)

        # n_rows: n_parcels (continuous) or n_blocks*n_parcels (events)
        n_rows = _response.shape[0]
        assert n_rows == n_blocks * n_parcels, (
            f"Shape mismatch: n_rows={n_rows}, n_blocks={n_blocks}, n_parcels={n_parcels}"
        )

        fmri_response.append(_response)
        # coords: one (x,y,z) per row — tile by n_blocks for events mode
        coordinates.append(np.tile(subject_data["parcel_coords"], (n_blocks, 1)))

        session_num = int(key.split("_")[0].replace("ses-", ""))
        run_num     = int(key.split("_")[1].replace("run-", ""))
        session_list.extend([session_num] * n_rows)
        run_list.extend([run_num] * n_rows)

        # block_label: one entry per row
        if trials == "events" and _labels is not None:
            for blk in range(n_blocks):
                block_label.extend([_labels[blk]] * n_parcels)
        else:
            block_label.extend([None] * n_rows)

        # hemisphere/region/parcel: one entry per row
        # continuous: one entry per parcel
        # events:     repeat the parcel list once per block
        for _ in range(n_blocks):
            for pid in parcel_ids:
                h = parcel_desc[pid]["hemisphere"]
                r = parcel_desc[pid]["region"]
                l = parcel_desc[pid]["lobe"]
                st = parcel_desc[pid]["structure_type"]
                hemisphere.append(h)
                region.append(r)
                lobe.append(l)
                structure_type.append(st)
                parcel.append(f"{h}_{r}")

    X      = np.vstack(fmri_response)
    coords = np.vstack(coordinates)

    # Sanity check before building DataFrame
    n_total = X.shape[0]
    assert len(hemisphere) == n_total, f"hemisphere len {len(hemisphere)} != X rows {n_total}"
    assert coords.shape[0] == n_total, f"coords rows {coords.shape[0]} != X rows {n_total}"
    assert len(session_list) == n_total
    assert len(run_list) == n_total

    Y_data = {
        "hemisphere":     hemisphere,
        "region":         region,
        "lobe":           lobe,
        "structure_type": structure_type,
        "parcel":         parcel,
        "x":      coords[:, 0],
        "y":      coords[:, 1],
        "z":      coords[:, 2],
        "radius": np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2),
        "session": session_list,
        "run":     run_list,
        "task":   [task] * n_total,
    }

    if trials == "events":
        Y_data["block_label"] = block_label

    Y = pd.DataFrame(Y_data)

    return X, Y


def parcel_samples(
    dataset,
    subject,
    split: str = "train",
    task: Optional[str] = None,
    session: Optional[str] = None,
    run: Optional[int] = None,
    trials: str = "episodes",
    time_collapse: Optional[str] = None,
    pad_width: int = 500,
    n_windows: int = 4,
    n_subsamples: int = 4,
    n_episodes: int = 1,
):
    """
    Extract parcel-wise fMRI response samples for a given subject.

    Automatically dispatches to parcel_samples_algonauts() or
    parcel_samples_hcptrt() based on the dataset type.

    Parameters
    ----------
    dataset : AlgonautsLoader or HCPTRTLoader
        Dataset loader instance.
    subject : str
        Subject identifier.
    split : str, default "train"
        Data split for Algonauts (either "train" or "test").
    task : str, optional
        Task name for HCPTRT (e.g., "motor", "wm"). Required for HCPTRTLoader.
    session : str, optional
        Specific session for HCPTRT (e.g., "ses-001").
    run : int, optional
        Specific run number for HCPTRT (e.g., 1).
    trials : str, default varies
        Sampling mode:
        - Algonauts: "episodes" or "within_episodes"
        - HCPTRT: "continuous" or "events"
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
    n_episodes : int, default 1
        Number of episodes to load for Algonauts.

    Returns
    -------
    X : ndarray
        Response array of shape (n_samples, n_features).
    Y : DataFrame
        Labels with one row per sample. Columns vary by dataset type.

    Notes
    -----
    - For AlgonautsLoader, use `trials="episodes"` or `trials="within_episodes"`.
    - For HCPTRTLoader, use `trials="continuous"` or `trials="events"`.
      The `task` parameter is required for HCPTRTLoader.
    """
    if isinstance(dataset, HCPTRTLoader):
        assert task is not None, "task is required for HCPTRTLoader"
        return parcel_samples_hcptrt(
            dataset, subject, task,
            session=session, run=run,
            trials=trials,
            time_collapse=time_collapse,
            pad_width=pad_width,
            n_windows=n_windows,
        )
    else:
        return parcel_samples_algonauts(
            dataset, subject, split,
            trials=trials,
            time_collapse=time_collapse,
            pad_width=pad_width,
            n_windows=n_windows,
            n_subsamples=n_subsamples,
            n_episodes=n_episodes,
        )