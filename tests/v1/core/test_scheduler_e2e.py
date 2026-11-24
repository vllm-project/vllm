# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM
from vllm.sampling_params import SamplingParams

MODEL = "vllm-project/tiny-random-LlamaForCausalLM"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module", params=[False, True])
def llm(request) -> LLM:
    return LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        long_prefill_token_threshold=2,
        max_num_batched_tokens=6,
        max_num_seqs=3,
        block_size=16,
        adaptive_prefill=request.param,
    )


def test_concurrent_partial_prefill(llm):
    outputs = llm.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1


def test_prefix_cache_stats_is_recorded(llm):
    # 17 tokens will make sure first 16 tokens are cached in a block
    input_tokens = {"prompt_token_ids": [101] * 17}
    _ = llm.generate([input_tokens])
    outputs = llm.generate([input_tokens])
    assert outputs[0].num_cached_tokens == 16


def test_mixed_prefill_requests_complete(llm):
    prompts = [{"prompt_token_ids": [101] * length} for length in (384, 384, 16)]
    outputs = llm.generate(
        prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    )
    assert len(outputs) == len(prompts)
    assert all(len(output.outputs[0].token_ids) == 8 for output in outputs)
