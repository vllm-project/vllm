#! /bin/bash
# This script sets up vLLM without compilation.
# It re-install the vLLM of the given commit using the public wheel,
# and link the cloned Python source code to the installed vLLM for development.
set -e

# Check the number of arguments to be 1 or 2.
if [[ $# -ne 1 && $# -ne 2 ]]; then
    echo "Usage: $0 <vLLM_PATH> <COMMIT (optional))>"
    exit 1
fi

VLLM_PATH=$1
if [ -z "$2" ]; then
    # If the commit is not provided, use merge-base to find the
    # commmon ancestor of the current branch and main.
    COMMIT=$(git merge-base main `git branch --show-current`)
else
    COMMIT=$2
fi

pushd $VLLM_PATH
pip uninstall -y vllm
VLLM_PRECOMPILED_WHEEL_LOCATION=https://vllm-wheels.s3.us-west-2.amazonaws.com/${COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
    pip install -e .
popd
