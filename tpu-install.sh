#!/bin/bash

DATE="+20240601"

pip uninstall torch -y
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly${DATE}-cp310-cp310-linux_x86_64.whl
pip uninstall torch-xla -y
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly${DATE}-cp310-cp310-linux_x86_64.whl
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

VLLM_TARGET_DEVICE="tpu" pip install -e .
