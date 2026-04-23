#!/bin/bash
set -euo pipefail

mkdir -p data
cd data/

datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git
cd algonauts_2025.competitors
datalad get stimuli/transcripts/
datalad get fmri/

cd ../
datalad install https://github.com/courtois-neuromod/cneuromod.processed.git
cd cneuromod.processed
datalad get fmriprep/hcptrt/sub-*
datalad get fmriprep/hcptrt/sourcedata/hcptrt/sub-*/ses-*/func/sub-*_ses-*_task-*_events.tsv
