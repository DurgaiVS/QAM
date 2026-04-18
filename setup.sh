#!/bin/bash

PWD=$pwd
cd "$(realpath "$(dirname "$0")")" || exit 0

if [[ ! -n $(command -v conda) ]]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ./miniconda.sh
    chmod +x ./miniconda.sh
    ./miniconda.sh -b -u
    source ~/.bashrc
    conda init
    conda config --set report_errors false
fi

if [[ ! -n $(conda env list | grep qam) ]]; then
    conda env create -n qam -f environment.yml
fi

cat <<EOF > ./.env
QAM_ROOT="$(realpath "$(dirname "$0")")"
EOF

pip install poetry==2.3.4
poetry install
pre-commit install

sudo apt-get install libssl-dev libboost-all-dev


if [[ -f "./miniconda.sh" ]]; then
    rm -f ./miniconda.sh
fi

cd $PWD
