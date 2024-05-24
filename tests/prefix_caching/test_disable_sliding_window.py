"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import pytest

MODEL_LEN_LEN = [
    ("bigcode/starcoder2-3b", 4096, 16384),
    # too big for automation
]

@pytest.mark.parametrize("model_len_len", MODEL_LEN_LEN)
def test_disable_sliding_window(
    vllm_runner,
    model_len_len,
):  
    model, sliding_len, full_len = model_len_len
    vllm_disabled_model = vllm_runner(model, disable_sliding_window=True)
    _ = vllm_disabled_model.generate_greedy(["Hi my name is"], max_tokens=10)
    model_config = vllm_disabled_model.model.llm_engine.model_config
    assert model_config.max_model_len == sliding_len, (
        "Max len expected to equal sliding_len of %s, but got %s",
        sliding_len, model_config.max_model_len
    )

    del vllm_disabled_model

    vllm_enabled_model = vllm_runner(model, disable_sliding_window=True)
    _ = vllm_enabled_model.generate_greedy(["Hi my name is"], max_tokens=10)
    model_config = vllm_enabled_model.model.llm_engine.model_config
    assert model_config.max_model_len == full_len, (
        "Max len expected equal sliding_len of %s, but got %s",
        full_len, model_config.max_model_len
    )

    del vllm_enabled_model
