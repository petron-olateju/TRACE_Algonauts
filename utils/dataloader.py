import ast
import numpy as np
import pandas as pd
import h5py

import os
import yaml

from nilearn import datasets
from nilearn.image import coord_transform
import nibabel as nib

with open('./configs/dirs.yaml', 'r') as f:
    dir_configs = yaml.safe_load(f)
TRANSCRIPT_DIR = dir_configs['TRANSCRIPT_DIR']
FMRI_DIR = dir_configs['FMRI_DIR']

with open('./configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)
HRF_DELAY = configs['HRF_DELAY']
TR = configs['TR']
CONTEXT_TRS = configs['CONTEXT_TRS']
SUBJECTS = configs['subjects']

def load_tsv(file_path):
    return pd.read_csv(file_path, sep="\t")

def fmri_file_name(dir, subject, split='train'):
    fmri_file_path = f'{dir}/sub-0{subject}/func/'
    if split=='train':
        fmri_file_path += f'sub-0{subject}_task-friends_'
        fmri_file_path += 'space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    else:
        fmri_file_path += f'sub-0{subject}_task-movie10_'
        fmri_file_path += 'space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'

    return fmri_file_path

def load_fmri_data(dir, subject, stimuli_name, split='train'):
    fmri_file_path = fmri_file_name(dir, subject, split)
    
    with h5py.File(fmri_file_path, 'r') as fmri_file:
        fmri_data = fmri_file[stimuli_name][()] # type: ignore
    return fmri_data

def get_fmri_sessions(dir, subject, split='train'):
    fmri_file_path = fmri_file_name(dir, subject, split)

    with h5py.File(fmri_file_path, 'r') as fmri_file:
        fmri_keys = list(fmri_file.keys())
    return fmri_keys

def load_transcript(dir, stimuli_name, split='train', ignore_nans=False):
    if split=='train':
        return load_train_transcript(
            dir, stimuli_name, ignore_nans
        )
    else:
        return load_test_transcript(
            dir, stimuli_name, ignore_nans
        )

def load_train_transcript(dir, stimuli_name, ignore_nans=False):
    season = stimuli_name[14:16]
    stimuli_name = stimuli_name[13:]
    DIR = f'{dir}/friends/s{int(season)}/friends_{stimuli_name}.tsv'
    df = load_tsv(DIR)
    if ignore_nans:
        df = df.dropna(subset=['text_per_tr'])
    return df

def load_test_transcript(dir, stimuli_name, ignore_nans=False):
    test_movie = stimuli_name[13:-2]
    stimuli_name = stimuli_name[13:]
    DIR = f'{dir}/movie10/{test_movie}/movie10_{stimuli_name}.tsv'
    df = load_tsv(DIR)
    if ignore_nans:
        df = df.dropna(subset=['text_per_tr'])
    return df


def load_transcript_scenes(stimuli_name, fmri_data, 
                           transcript_dir=TRANSCRIPT_DIR, FMRI_DIR=FMRI_DIR, 
                           HRF_DELAY=HRF_DELAY, TR=TR, CONTEXT_TRS=CONTEXT_TRS, split='train'):
    try:
        transcript_df = load_transcript(transcript_dir, stimuli_name, split, ignore_nans=True)
    except FileNotFoundError as v:
        print(FileNotFoundError(v))
        return None, None, None

    epochs = []
    start = []
    end = []

    for tr_idx, row in transcript_df.iterrows():
        words = row['words_per_tr']
        if isinstance(words, str):
            words = ast.literal_eval(words)
        if not isinstance(words, list):
            continue

        if len(words)>0:
            fmri_idx = tr_idx + HRF_DELAY
            start_idx = fmri_idx - CONTEXT_TRS
            end_idx = fmri_idx +  CONTEXT_TRS

            if (start_idx >= 0) and (end_idx < len(fmri_data)):
                epoch = fmri_data[start_idx:end_idx, :]
                epochs.append(epoch)
                start.append(start_idx)
                end.append(end_idx)
    
    if len(epochs) > 0:
        return np.stack(epochs), np.array(start), np.array(end)
    return None, None, None




ATLAS_NAME = configs['ATLAS_NAME']
N_PARCELS = configs['N_PARCELS']
ATLAS_DESC = datasets.fetch_atlas_schaefer_2018(n_rois=N_PARCELS)
ATLAS_IMG = nib.load(ATLAS_DESC.maps)
ATLAS_LABELS = ATLAS_DESC.labels
ATLAS_DATA = ATLAS_IMG.get_fdata()
INV_AFFINE = np.linalg.inv(ATLAS_IMG.affine)
def get_parcel_name(coord):
    voxel = nib.affines.apply_affine(INV_AFFINE, coord).astype(int)

    if (0 <= voxel[0] < ATLAS_DATA.shape[0] and 
        0 <= voxel[1] < ATLAS_DATA.shape[1] and 
        0 <= voxel[2] < ATLAS_DATA.shape[2]):
        
        label_idx = ATLAS_DATA[voxel[0], voxel[1], voxel[2]]
        
        if label_idx == 0:
            print("Backgrpund/Non-Brain")
            return  None
        
        return ATLAS_LABELS[int(label_idx)]
    
    print("Outside Volume Bounds")
    return None

def subject_atlas_file_name(subject, dir=FMRI_DIR):
    atlas_path = f'{dir}/sub-0{subject}/atlas/'
    atlas_path += f'sub-0{subject}'
    atlas_path += '_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    return atlas_path

def load_subject_atlas(subject, dir=FMRI_DIR):
    atlas_path = subject_atlas_file_name(subject, dir)
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
        mask = (labels == parcel_id)
        centroid = world_coords[mask].mean(axis=0)
        parcel_coords_matrix[i] = centroid

        parcel_name = get_parcel_name(centroid)[10:]
        parcel_name = parcel_name.split('_')

        parcel_desc[parcel_id] = {
            'hemisphere': parcel_name[0],
            'region': ''.join(parcel_name[1:-1]),
            'region_idx': parcel_name[-1]
        }

    return parcel_coords_matrix, unique_parcels, parcel_desc