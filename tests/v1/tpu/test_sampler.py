# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
              max_num_batched_tokens=256)
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
        # There are natural numerical instabilities that make it difficult
        # to have deterministic results over many tokens, tests the first ~20
        # tokens match.
        assert output[0].outputs[0].text[:20] == output[1].outputs[0].text[:20]


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-1.5B-Instruct"])
# TODO TPU will appear busy if we fan-out test params here
@pytest.mark.parametrize("n_prompts", [1])
@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test needs a TPU")
def test_logprobs(model_name: str, n_prompts: int):
    """
    Request top logprobs with different sampling settings and check
    that results contains the requested number, ordered ascendingly.  
    """

    def check_num_logprobs(logprobs, expected_num: int):
        for step in logprobs:
            prev_logp = 1.0
            # order by rank
            sorted_step = dict(
                sorted(step.items(), key=lambda item: item[1].rank))

            # Can contain the sampled token
            assert len(step) == expected_num or len(step) == expected_num + 1
            # Check results are ordered by prob value
            for rankno, (tid, logp) in enumerate(sorted_step.items()):
                assert logp.logprob <= prev_logp
                prev_logp = logp.logprob
                assert logp.rank == rankno + 1

    llm = LLM(model_name,
              enforce_eager=False,
              max_num_seqs=1,
              max_model_len=128,
              max_num_batched_tokens=128)
    prompts = [
        "Write a short story about a robot that dreams for the first time."
    ] * n_prompts
    greedy_sampling_params = SamplingParams(temperature=0.0, max_tokens=64,\
         logprobs=4)
    regular_sampling_params = SamplingParams(temperature=0.4, max_tokens=64,\
         logprobs=4)
    topkp_sampling_params = SamplingParams(temperature=0.4, max_tokens=64,\
         logprobs=4, top_k=12, top_p=0.5)

    for sp in [greedy_sampling_params, regular_sampling_params, \
               topkp_sampling_params]:
        output = llm.generate(prompts, sp)
        for o in output:
            check_num_logprobs(o.outputs[0].logprobs, 4)
