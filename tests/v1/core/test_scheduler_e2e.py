# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams

MODEL = "vllm-project/tiny-random-LlamaForCausalLM"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        long_prefill_token_threshold=2,
        max_num_batched_tokens=6,
        max_num_seqs=3,
        block_size=16,
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


def test_duplicate_prefixes_survive_cache_pressure(vllm_runner):
    """Scoring copies stay reusable until cache pressure evicts their blocks."""
    with vllm_runner(
        MODEL,
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=160,
        max_num_seqs=16,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.2,
        kv_cache_memory_bytes=16 * 1024 * 1024,
    ) as runner:
        engine = runner.get_llm()
        score = SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=1
        )
        scoring = engine.generate(
            [{"prompt_token_ids": [101] * 16}] * 640, score, use_tqdm=False
        )
        assert all(output.num_cached_tokens == 0 for output in scoring)
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        probe = {"prompt_token_ids": [101] * 16 + [102]}
        hot = engine.generate([probe], params, use_tqdm=False)[0]
        assert hot.num_cached_tokens == 16
        engine.generate(
            [{"prompt_token_ids": [200 + i] * 128} for i in range(96)],
            params,
            use_tqdm=False,
        )
        cold = engine.generate([probe], params, use_tqdm=False)[0]
        warm = engine.generate([probe], params, use_tqdm=False)[0]
        assert cold.num_cached_tokens == 0
        assert warm.num_cached_tokens == 16
        assert hot.outputs[0].token_ids == cold.outputs[0].token_ids
        assert hot.outputs[0].token_ids == warm.outputs[0].token_ids
