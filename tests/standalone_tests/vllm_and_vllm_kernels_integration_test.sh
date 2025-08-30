#!/bin/bash
# This script tests building and installing vllm (without kernels) and vllm-kernels separately
# to test they can work together to replace a vllm package built with kernels.

set -e
set -x

TEST_IMAGE="${TEST_IMAGE:-1}"

if [ "$TEST_IMAGE" -eq 1 ]; then
    cd /vllm-workspace/
    # restore the original files
    mv src/vllm ./vllm
    pip uninstall -y vllm
fi

cd kernels
# install vllm-kernels and vllm without kernels packages
pip install -v --no-build-isolation --editable .
cd ..
VLLM_TARGET_DEVICE=empty pip install -v --editable .

# simple test to confirm kernels can be loaded
python -c "import vllm._C as c; import vllm.vllm_flash_attn as f; assert f.is_fa_version_supported(3)"
# eval
pytest -s -v ./tests/evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-small.txt --tp-size=1
