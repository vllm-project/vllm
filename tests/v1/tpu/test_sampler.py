# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from vllm import LLM, envs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-1.5B-Instruct"])
@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test needs a TPU")
def test_sampler_different(model_name: str):
    """
    Test significantly different sampling params to assert the model produces 
    different results.
    """
    llm = LLM(model_name,
              enforce_eager=False,
              max_num_seqs=1,
              max_model_len=512,
              max_num_batched_tokens=512)
    prompts = [
        "Write a short story about a robot that dreams for the first time."
    ]
    sampling_params = SamplingParams(temperature=0.9, min_p=0.2, max_tokens=64)
    output = llm.generate(prompts, sampling_params)

    sampling_params = SamplingParams(temperature=0.1, min_p=0.8, max_tokens=64)
    output2 = llm.generate(prompts, sampling_params)
    assert output[0].outputs[0].text != output2[0].outputs[0].text

    with pytest.raises(ValueError):
        # Unsupported `seed` param.
        sampling_params = SamplingParams(temperature=0.3, seed=42)
        output2 = llm.generate(prompts, sampling_params)

    # Batch-case with TopK/P
    for B in [4, 16]:
        p = prompts * B
        sampling_params = [
            SamplingParams(
                temperature=0.1,
                min_p=0.8,
                max_tokens=64,
                # Vary number of ks
                top_k=random.randint(4, 12),
                top_p=random.random()) for _ in range(B)
        ]
        # Make sure first two reqs have the same K/P
        sampling_params[0] = sampling_params[1]
        output = llm.generate(p, sampling_params)
        assert output[0].outputs[0].text == output[1].outputs[0].text
