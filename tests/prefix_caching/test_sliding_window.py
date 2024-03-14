from vllm import LLM


# For qwen1.5 the default value of sliding_window is not None,
# the default value of use_sliding_window is False
# This test aims to verify two scenarios.
# 1. Verify their default values, and whether the return value of
#    get_sliding_window function is right when model is qwen1.5 and
#    the value of use_sliding_window is different
# 2. whether the return value of get_sliding_window function is right
#    when using other model
def test_sliding_window_value():
    qwen = LLM(
        model="Qwen/Qwen-0.5B",
        enforce_eager=True,
        download_dir=None,
        dtype="float16",
        tokenizer_mode='auto'
    )

    # verify default values
    hf_config = qwen.llm_engine.model_config.hf_config
    default_sliding_window = hf_config.sliding_window
    default_use_sliding_window = hf_config.use_sliding_window
    assert default_sliding_window is not None, \
        ("In Qwen1.5, sliding_window default value should not be None")
    assert default_use_sliding_window is False, \
        ("In Qwen1.5, use_sliding_window default value should be True")

    # verify the return value of get_sliding_window function
    get_sliding_window = qwen.llm_engine.model_config.get_sliding_window()
    assert get_sliding_window is None, \
        ("In Qwen1.5, sliding_window should be None, "
         "because use_sliding_window is False")

    # verify the return value of get_sliding_window function
    qwen.llm_engine.model_config.hf_config.use_sliding_window = True
    get_sliding_window = qwen.llm_engine.model_config.get_sliding_window()
    assert get_sliding_window == default_sliding_window, \
        ("In Qwen1.5, sliding_window should be "
         f"{default_sliding_window}, because use_sliding_window is True")


def test_sliding_window_with_other_model():
    facebook = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        download_dir=None,
        dtype="float16",
        tokenizer_mode='auto'
    )

    # verify the return value of get_sliding_window function
    get_sliding_window = facebook.llm_engine.model_config.get_sliding_window()
    assert get_sliding_window is None, \
        ("In facebook/opt-125m, sliding_window should be None, "
         "because the default value of sliding_window is None")

    sliding_window = 4096
    facebook.llm_engine.model_config.hf_config.sliding_window = sliding_window
    get_sliding_window = facebook.llm_engine.model_config.get_sliding_window()
    assert get_sliding_window == sliding_window, \
        (f"In facebook/opt-125m, sliding_window should be {sliding_window}, ")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
