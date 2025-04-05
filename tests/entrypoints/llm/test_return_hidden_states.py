# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm import LLM, SamplingParams


@pytest.mark.skip_global_cleanup
def test_return_hidden_states():
    model = LLM("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    sampling_params = SamplingParams(skip_special_tokens=False,
                                     return_hidden_states=True)
    prompt = "Now, tell me about Aspect's experiment \
    related to EPR and quantum physics"

    o = model.generate(
        prompt,
        sampling_params=sampling_params,
    )

    assert isinstance(o[0].hidden_states, torch.Tensor)
