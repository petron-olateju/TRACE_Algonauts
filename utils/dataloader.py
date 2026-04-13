import ast
import numpy as np
import pandas as pd
import h5py

import os
import yaml

with open('./configs/dirs.yaml', 'r') as f:
    dir_configs = yaml.safe_load(f)
TRANSCRIPT_DIR = dir_configs['TRANSCRIPT_DIR']
h5_DIR = dir_configs['h5_DIR']

with open('./configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)
HRF_DELAY = configs['HRF_DELAY']
TR = configs['TR']
CONTEXT_TRS = configs['CONTEXT_TRS']
SUBJECTS = configs['subjects']

def load_tsv(file_path):
    return pd.read_csv(file_path, sep="\t")

def fmri_file_name(dir, subject, split='train'):
    h5_path = f'{dir}/sub-0{subject}/func/'
    if split=='train':
        h5_path += f'sub-0{subject}_task-friends_'
        h5_path += 'space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    else:
        h5_path += f'sub-0{subject}_task-movie10_'
        h5_path += 'space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'

    return h5_path

def load_fmri_data(dir, subject, stimuli_name, split='train'):
    h5_path = fmri_file_name(dir, subject, split)
    
    with h5py.File(h5_path, 'r') as h5_file:
        fmri_data = h5_file[stimuli_name][()] # type: ignore
    return fmri_data

def get_fmri_sessions(dir, subject, split='train'):
    h5_path = fmri_file_name(dir, subject, split)

    with h5py.File(h5_path, 'r') as h5_file:
        fmri_keys = list(h5_file.keys())
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
                           transcript_dir=TRANSCRIPT_DIR, h5_dir=h5_DIR, 
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
