#!/bin/bash
set -euo pipefail


# Install NeuroDebian repo
. /etc/os-release
wget -O- http://neuro.debian.net/lists/${VERSION_CODENAME}.de-fzj.libre | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo wget -q -O/etc/apt/trusted.gpg.d/neuro.debian.net.asc https://neuro.debian.net/_static/neuro.debian.net.asc

sudo apt-get update
sudo apt-get install -y datalad git-annex

mkdir -p data