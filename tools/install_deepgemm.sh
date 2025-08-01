#!/bin/bash
set -ex

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
PIP_NO_BUILD_ISOLATION=0 pip install -vvv -e .

popd
