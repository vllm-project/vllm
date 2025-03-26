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
               max_num_seqs=3)


def test_concurrent_partial_prefill(model):
    outputs = model.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1
