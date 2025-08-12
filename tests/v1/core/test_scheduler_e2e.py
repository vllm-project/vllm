# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from vllm import LLM

if os.getenv("VLLM_USE_V1", "0") != "1":
    pytest.skip("Test package requires V1", allow_module_level=True)

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def model() -> LLM:
    return LLM(MODEL,
               enforce_eager=True,
               enable_prefix_caching=True,
               long_prefill_token_threshold=2,
               max_num_batched_tokens=6,
               max_num_seqs=3,
               block_size=16)


def test_concurrent_partial_prefill(model):
    outputs = model.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1


def test_prefix_cache_stats_is_recorded(model):
    # 17 tokens will make sure first 16 tokens are cached in a block
    input_tokens = {"prompt_token_ids": [101] * 17}
    _ = model.generate([input_tokens])
    outputs = model.generate([input_tokens])
    assert outputs[0].num_cached_tokens == 16
