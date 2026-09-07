# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes

MAX_MODEL_LEN = 1024
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "robertgshaw2/zephyr-7b-beta-channelwise-gptq"
)
REVISION = os.environ.get("REVISION", "main")
QUANTIZATION = os.environ.get("QUANTIZATION", "gptq_marlin")
MIN_CAPABILITY = os.environ.get("MIN_CAPABILITY", "80")
SAFETENSORS_LOAD_STRATEGY = os.environ.get(
    "WEIGHT_LOADING_TEST_SAFETENSORS_LOAD_STRATEGY"
)


@pytest.mark.skipif(
    MODEL_NAME == "casperhansen/deepseek-coder-v2-instruct-awq", reason="OOM in the CI"
)
@pytest.mark.skipif(
    not current_platform.has_device_capability(int(MIN_CAPABILITY)),
    reason="Current system does not have minimum capability.",
)
def test_weight_loading(vllm_runner):
    """
    Test parameter weight loading with tp>1.
    """

    # MoE models need fp16.
    NEEDS_FP16 = (
        QUANTIZATION == "gptq"
        or MODEL_NAME == "nm-testing/test-w4a16-mixtral-actorder-group"
    )
    with vllm_runner(
        model_name=MODEL_NAME,
        revision=REVISION,
        dtype=torch.half if NEEDS_FP16 else "auto",
        quantization=None if QUANTIZATION == "None" else QUANTIZATION,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=2,
        safetensors_load_strategy=SAFETENSORS_LOAD_STRATEGY,
        # Generating 20 tokens needs a tiny KV cache, while sizing and
        # allocating a full-size one dominates this test on large devices.
        kv_cache_memory_bytes=2 * GiB_bytes,
    ) as model:
        output = model.generate_greedy("Hello world!", max_tokens=20)
        print(output)
        assert output
