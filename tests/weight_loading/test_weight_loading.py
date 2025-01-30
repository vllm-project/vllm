import os

import pytest
import torch

from vllm.platforms import current_platform

MAX_MODEL_LEN = 1024
MODEL_NAME = os.environ.get("MODEL_NAME",
                            "robertgshaw2/zephyr-7b-beta-channelwise-gptq")
REVISION = os.environ.get("REVISION", "main")
QUANTIZATION = os.environ.get("QUANTIZATION", "gptq_marlin")
MIN_CAPABILITY = os.environ.get("MIN_CAPABILITY", "89")


@pytest.mark.skipif(
    not current_platform.has_device_capability(int(MIN_CAPABILITY)),
    reason="Current system does not have minimum capability.")
def test_weight_loading(vllm_runner):
    """
    Test parameter weight loading with tp>1.
    """
    with vllm_runner(
            model_name=MODEL_NAME,
            revision=REVISION,
            dtype=torch.half if QUANTIZATION == "gptq" else "auto",
            quantization=None if QUANTIZATION == "None" else QUANTIZATION,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=2) as model:

        output = model.generate_greedy("Hello world!", max_tokens=20)
        print(output)
        assert output
