# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import pytest

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_LEN_LEN = [
    # Example models with sliding window.
    ("bigcode/starcoder2-3b", 4096, 16384),
    # ("mistralai/Mistral-7B-v0.1", 4096, 32768), << OOM in CI

    # Confirm model with sliding window works.
    # config has "use_sliding_window": false
    ("Qwen/Qwen1.5-0.5B-Chat", 32768, 32768),
    # config has no sliding window attribute.
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2048, 2048),
]


@pytest.mark.parametrize("model_len_len", MODEL_LEN_LEN)
def test_disable_sliding_window(model_len_len, ):
    model, sliding_len, full_len = model_len_len
    disabled_llm = LLM(model, disable_sliding_window=True)
    disabled_llm.generate("Hi my name is")
    model_config = disabled_llm.llm_engine.model_config
    assert model_config.max_model_len == sliding_len, (
        "Max len expected to equal sliding_len of %s, but got %s", sliding_len,
        model_config.max_model_len)

    del disabled_llm
    cleanup_dist_env_and_memory()

    enabled_llm = LLM(model,
                      enforce_eager=True,
                      disable_sliding_window=False,
                      enable_prefix_caching=False)
    enabled_llm.generate("Hi my name is")
    model_config = enabled_llm.llm_engine.model_config
    assert model_config.max_model_len == full_len, (
        "Max len expected to equal full_len of %s, but got %s", full_len,
        model_config.max_model_len)

    del enabled_llm
    cleanup_dist_env_and_memory()
