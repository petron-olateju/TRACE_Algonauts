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

def load_fmri_data(dir, subject, dataset_name, split='train'):
    h5_path = fmri_file_name(dir, subject, split)
    
    with h5py.File(h5_path, 'r') as h5_file:
        fmri_data = h5_file[dataset_name][()] # type: ignore
    return fmri_data

def get_fmri_sessions(dir, subject, split='train'):
    h5_path = fmri_file_name(dir, subject, split)

    with h5py.File(h5_path, 'r') as h5_file:
        fmri_keys = list(h5_file.keys())
    return fmri_keys

def load_transcript(dir, season, episode, episode_split, split='train', ignore_nans=False):
    if split=='train':
        return load_train_transcript(
            dir, season, episode, episode_split, ignore_nans
        )
    else:
        return load_test_transcript(
            dir, season, episode, episode_split, ignore_nans
        )

def load_train_transcript(dir, season, episode, episode_split, ignore_nans=False):
    DIR = f'{dir}/friends/s{int(season)}/friends_s{season}e{episode}{episode_split}.tsv'
    df = load_tsv(DIR)
    if ignore_nans:
        df = df.dropna(subset=['text_per_tr'])
    return df

def load_test_transcript(dir, movie, movie_split, ignore_nans=False):
    DIR = f'{dir}movie_10_{movie}{movie_split}.tsv'
    df = load_tsv(DIR)
    if ignore_nans:
        df = df.dropna(subset=['text_per_tr'])
    return df


def load_transcript_scene(subject, transcript_index, dataset_name, transcript_dir=TRANSCRIPT_DIR, h5_dir=h5_DIR, HRF_DELAY=HRF_DELAY, TR=TR):
    pass