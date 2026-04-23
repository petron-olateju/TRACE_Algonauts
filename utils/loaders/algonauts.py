"""Data loading utilities for Algonauts 2025 Challenge dataset.

This module provides the AlgonautsLoader class for loading fMRI responses,
movie transcripts, and brain atlas data.
"""

import ast
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py
import yaml
import nibabel as nib
from nilearn import datasets
from tqdm import tqdm

from .parcel_maps import SCHAEFER_LOBE, get_lobe


def _read_tsv(file_path: str) -> pd.DataFrame:
    """Read a TSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, sep="\t")


class AlgonautsLoader:
    """Data loader for Algonauts 2025 Challenge dataset.

    Provides methods for loading fMRI responses, transcripts, and brain atlas data.
    """

    def __init__(
        self,
        dataset: str = "algonauts",
        split: str = "train"
    ):
        """Initialize the Algonauts loader.

        Parameters
        ----------
        dataset : str
            Dataset name matching key in configs (e.g., "algonauts").
        split : str, optional
            'train' for Friends dataset or 'test' for Movie10 dataset.
            Default is 'train'.
        """
        self.dataset = dataset
        self.split = split

        self._load_configs()

        self.fmri_dir = self.configs["dirs"]["fmri"]
        self.transcript_dir = self.configs["dirs"]["transcript"]
        self.hrf_delay = self.configs["params"]["hrf_delay"]
        self.tr = self.configs["params"]["tr"]
        self.context_trs = self.configs["params"]["context_trs"]
        self.subjects = self.configs["subjects"]

        self._init_atlas()

    def _load_configs(self) -> dict:
        """Load configuration files and store in self.configs."""
        with open("./configs/dirs.yaml", "r") as f:
            dir_configs = yaml.safe_load(f)
        with open("./configs/configs.yaml", "r") as f:
            configs = yaml.safe_load(f)

        self.configs = {
            "dirs": dir_configs[self.dataset]["dirs"],
            "params": configs[self.dataset]["params"],
            "subjects": configs[self.dataset]["subjects"]
        }
        return self.configs

    def _init_atlas(self):
        """Initialize the Schaefer atlas."""
        n_parcels = self.configs["params"]["num_parcels"]
        atlas_desc = datasets.fetch_atlas_schaefer_2018(n_rois=n_parcels)
        self.atlas_img = nib.load(atlas_desc.maps)
        self.atlas_labels = atlas_desc.labels
        self.atlas_data = self.atlas_img.get_fdata()
        self.inv_affine = np.linalg.inv(self.atlas_img.affine)

    def get_fmri_file_path(self, subject: int, split: Optional[str] = None) -> str:
        """Build the file path for an fMRI data file."""
        if split is None:
            split = self.split

        fmri_file_path = f"{self.fmri_dir}/sub-0{subject}/func/"
        if split == "train":
            fmri_file_path += f"sub-0{subject}_task-friends_"
            fmri_file_path += "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        else:
            fmri_file_path += f"sub-0{subject}_task-movie10_"
            fmri_file_path += (
                "space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
            )

        return fmri_file_path

    def list_fmri_sessions(self, subject: int, split: Optional[str] = None) -> List[str]:
        """List available fMRI session keys in an HDF5 file."""
        if split is None:
            split = self.split

        fmri_file_path = self.get_fmri_file_path(subject, split)

        with h5py.File(fmri_file_path, "r") as fmri_file:
            fmri_keys = list(fmri_file.keys())
        return fmri_keys

    def load_fmri_responses(
        self, stimuli_name: str, subject: int, split: Optional[str] = None
    ) -> np.ndarray:
        """Load fMRI response data for a specific stimuli session."""
        if split is None:
            split = self.split

        fmri_file_path = self.get_fmri_file_path(subject, split)

        with h5py.File(fmri_file_path, "r") as fmri_file:
            fmri_data = fmri_file[stimuli_name][()]
        return fmri_data

    def _load_friends_transcript(
        self, stimuli_name: str, ignore_nans: bool = False
    ) -> pd.DataFrame:
        """Load transcript for a Friends episode."""
        season = stimuli_name[14:16]
        episode_name = stimuli_name[13:]
        transcript_path = f"{self.transcript_dir}/friends/s{int(season)}/friends_{episode_name}.tsv"
        df = _read_tsv(transcript_path)
        if ignore_nans:
            df = df.dropna(subset=["text_per_tr"])
        return df

    def _load_movie10_transcript(
        self, stimuli_name: str, ignore_nans: bool = False
    ) -> pd.DataFrame:
        """Load transcript for a Movie10 clip."""
        movie_id = stimuli_name[13:-2]
        clip_name = stimuli_name[13:]
        transcript_path = f"{self.transcript_dir}/movie10/{movie_id}/movie10_{clip_name}.tsv"
        df = _read_tsv(transcript_path)
        if ignore_nans:
            df = df.dropna(subset=["text_per_tr"])
        return df

    def load_transcript(
        self, stimuli_name: str, split: Optional[str] = None, ignore_nans: bool = False
    ) -> pd.DataFrame:
        """Load transcript for a given stimuli session."""
        if split is None:
            split = self.split

        if split == "train":
            return self._load_friends_transcript(stimuli_name, ignore_nans)
        else:
            return self._load_movie10_transcript(stimuli_name, ignore_nans)

    def epoch_fmri_by_words(
        self,
        stimuli_name: str,
        fmri_data: np.ndarray,
        transcript_dir: Optional[str] = None,
        fmri_dir: Optional[str] = None,
        hrf_delay: Optional[int] = None,
        tr: Optional[float] = None,
        context_trs: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Create epoched fMRI response windows aligned to word onsets."""
        if transcript_dir is None:
            transcript_dir = self.transcript_dir
        if hrf_delay is None:
            hrf_delay = self.hrf_delay
        if tr is None:
            tr = self.tr
        if context_trs is None:
            context_trs = self.context_trs
        if split is None:
            split = self.split

        try:
            transcript_df = self.load_transcript(stimuli_name, split, ignore_nans=True)
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

    def get_atlas_file_path(self, subject: int) -> str:
        """Build the file path for a subject's atlas file."""
        atlas_path = f"{self.fmri_dir}/sub-0{subject}/atlas/"
        atlas_path += f"sub-0{subject}"
        atlas_path += "_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
        return atlas_path

    def get_parcel_label(self, coord: np.ndarray) -> Optional[str]:
        """Get the Schaefer atlas label for a 3D coordinate."""
        voxel = nib.affines.apply_affine(self.inv_affine, coord).astype(int)

        if (
            0 <= voxel[0] < self.atlas_data.shape[0]
            and 0 <= voxel[1] < self.atlas_data.shape[1]
            and 0 <= voxel[2] < self.atlas_data.shape[2]
        ):
            label_idx = self.atlas_data[voxel[0], voxel[1], voxel[2]]

            if label_idx == 0:
                print("Background/Non-Brain")
                return None

            return self.atlas_labels[int(label_idx)]

        print("Outside Volume Bounds")
        return None

    def load_atlas_for_subject(
        self, subject: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
        """Load brain atlas data for a specific subject."""
        atlas_path = self.get_atlas_file_path(subject)
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

            parcel_name = self.get_parcel_label(centroid)[10:]
            parcel_name = parcel_name.split("_")

            parcel_desc[parcel_id] = {
                "hemisphere": parcel_name[0],
                "region": "".join(parcel_name[1:-1]),
                "region_idx": parcel_name[-1],
            }

        return parcel_coords_matrix, unique_parcels, parcel_desc

    def load_episode_fmri(
        self,
        subject: int,
        split: Optional[str] = None,
        fmri_dir: Optional[str] = None
    ) -> dict:
        """Load continuous fMRI timeseries for all episodes for a given subject."""
        if split is None:
            split = self.split
        if fmri_dir is None:
            fmri_dir = self.fmri_dir

        print(f"Loading fMRI timeseries for s-0{subject}")

        fmri_stimuli_info = self.list_fmri_sessions(subject, split)

        scenes_response_matrix = {}
        for stimuli in tqdm(fmri_stimuli_info, total=len(fmri_stimuli_info)):
            fmri_response = self.load_fmri_responses(stimuli, subject, split)
            stimuli_name = stimuli[13:]
            if stimuli_name not in scenes_response_matrix:
                scenes_response_matrix[stimuli_name] = [fmri_response]
            else:
                scenes_response_matrix[stimuli_name].append(fmri_response)

        for stimuli_name in scenes_response_matrix:
            scenes_response_matrix[stimuli_name] = np.stack(
                scenes_response_matrix[stimuli_name], axis=0
            ).squeeze(0)

        print(f"Loading parcel coordinates for s-0{subject}")
        parcel_coords, parcel_ids, parcel_desc = self.load_atlas_for_subject(subject)

        return {
            "scenes_response": scenes_response_matrix,
            "parcel_coords": parcel_coords,
            "parcel_ids": parcel_ids,
            "parcel_desc": parcel_desc
        }

    def load_episode_word_epochs(
        self,
        subject: int,
        split: Optional[str] = None,
        fmri_dir: Optional[str] = None,
        transcript_dir: Optional[str] = None
    ) -> dict:
        """Load word-locked fMRI epochs for all episodes for a given subject."""
        if split is None:
            split = self.split
        if fmri_dir is None:
            fmri_dir = self.fmri_dir
        if transcript_dir is None:
            transcript_dir = self.transcript_dir

        print(f"Loading word epochs for s-0{subject}")

        fmri_stimuli_info = self.list_fmri_sessions(subject, split)

        scenes_response_matrix = {}
        for stimuli in tqdm(fmri_stimuli_info, total=len(fmri_stimuli_info)):
            stimuli_fmri_response = self.load_fmri_responses(stimuli, subject, split)
            trials, start, end = self.epoch_fmri_by_words(
                stimuli, stimuli_fmri_response,
                transcript_dir, fmri_dir,
                None, None, None,
                split=split
            )
            if trials is None:
                continue

            stimuli_name = stimuli[13:]
            if stimuli_name not in scenes_response_matrix:
                scenes_response_matrix[stimuli_name] = {
                    'trials': [trials],
                    'start': [start],
                    'end': [end]
                }
            else:
                scenes_response_matrix[stimuli_name]['trials'].append(trials)
                scenes_response_matrix[stimuli_name]['start'].append(start)
                scenes_response_matrix[stimuli_name]['end'].append(end)

        for stimuli_name in scenes_response_matrix:
            scenes_response_matrix[stimuli_name]['trials'] = np.stack(
                scenes_response_matrix[stimuli_name]['trials'], axis=0
            ).squeeze(0)
            scenes_response_matrix[stimuli_name]['start'] = np.stack(
                scenes_response_matrix[stimuli_name]['start'], axis=0
            ).squeeze(0)
            scenes_response_matrix[stimuli_name]['end'] = np.stack(
                scenes_response_matrix[stimuli_name]['end'], axis=0
            ).squeeze(0)

        print(f"Loading parcel coordinates for s-0{subject}")
        parcel_coords, parcel_ids, parcel_desc = self.load_atlas_for_subject(subject)

        return {
            "scenes_response": scenes_response_matrix,
            "parcel_coords": parcel_coords,
            "parcel_ids": parcel_ids,
            "parcel_desc": parcel_desc
        }


def get_default_dataset(dataset: str = "algonauts") -> AlgonautsLoader:
    """Create and return default AlgonautsLoader instance."""
    return AlgonautsLoader(dataset=dataset, split="train")