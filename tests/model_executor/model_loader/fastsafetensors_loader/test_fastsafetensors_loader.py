# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

test_model = "openai-community/gpt2"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_model_loader_download_files(vllm_runner):
    with vllm_runner(test_model, load_format="fastsafetensors") as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs
