#!/bin/bash
# This script tests building and installing vllm (without kernels) and vllm-kernels separately
# to test they can work together to replace a vllm package built with kernels.

set -e
set -x

if [ -d "/vllm-workspace" ]; then
    cd /vllm-workspace
    # restore the original files
    mv src/vllm ./vllm

    export MAX_JOBS=12
    export NVCC_THREADS=8
fi

pip uninstall -y vllm

cd kernels
# install vllm-kernels and vllm without kernels packages
pip install -v --no-build-isolation --editable .
cd ..
VLLM_NO_EXTENSION=1 pip install -v --no-build-isolation --editable .

# simple test to confirm kernels can be loaded
python -c "import vllm._C as c"
# eval
pytest -s -v ./tests/evals/gsm8k/test_gsm8k_correctness.py --model-config-file=Llama-3-8B-Instruct-nonuniform-CT.yaml --tp-size=1
