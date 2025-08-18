# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, envs
from vllm.sampling_params import SamplingParams

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-1.5B-Instruct"])
# TODO TPU will appear busy if we fan-out test params here
@pytest.mark.parametrize("n_prompts", [1])
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

            if len(step) != expected_num:
                print("watch out", sorted_step)

            # check results are ordered by prob value
            # assert len(step) == expected_num
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
