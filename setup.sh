#!/bin/bash

PWD=$pwd
cd "$(realpath "$(dirname "$0")")" || exit 2

if [[ ! -n $(command -v poetry) ]]; then
    curl -sSL https://install.python-poetry.org > ./get-poetry.py
    python ./get-poetry.py -y
    export PATH="$PATH:$HOME/.local/bin"

    rm -f "./get-poetry.py"
fi

cat <<EOF > ./.env
QAM_ROOT=$(pwd)
EOF

poetry install
pre-commit install

sudo apt-get install libssl-dev libboost-all-dev

cd $PWD || exit 1
