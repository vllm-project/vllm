#!/bin/bash
set -ex

ln -sf "$(which python3)" /usr/local/bin/python

WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    WORKSPACE=$(pwd)/deepgemm_workspace
fi
mkdir -p "$WORKSPACE"

pushd "$WORKSPACE"

if [ ! -d DeepGEMM ]; then
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM
fi
cd DeepGEMM
./develop.sh
PIP_NO_BUILD_ISOLATION=0 pip install -vvv -e .

popd
