# SPDX-License-Identifier: Apache-2.0

msg = """Old style python only build (without compilation) is deprecated, please check https://docs.vllm.ai/en/latest/getting_started/installation.html#python-only-build-without-compilation for the new way to do python only build (without compilation).

TL;DR:

VLLM_USE_PRECOMPILED=1 pip install -e .

or

export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install -e .
""" # noqa

print(msg)
