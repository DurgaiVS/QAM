#!/bin/bash

PWD=$pwd
cd "$(realpath "$(dirname "$0")")" || exit 0

if [[ ! -n $(command -v conda) ]]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ./miniconda.sh
    bash ./miniconda.sh -b -u
    conda init
    conda config --set report_errors false
fi

if [[ ! -n $(conda env list | grep qam) ]]; then
    conda env create -n qam -f environment.yml
fi

if [[ -f "./miniconda.sh" ]]; then
    rm -f ./miniconda.sh
fi
cd $PWD
