import pytest

from vllm.config import ModelConfig

MODEL_IDS_EXPECTED = [
    ("Qwen/Qwen1.5-7B", 32768),
    ("mistralai/Mistral-7B-v0.1", 4096),
    ("mistralai/Mistral-7B-Instruct-v0.2", 32768),
]


@pytest.mark.parametrize("model_id_expected", MODEL_IDS_EXPECTED)
def test_disable_sliding_window(model_id_expected):
    model_id, expected = model_id_expected
    model_config = ModelConfig(
        model_id,
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        disable_sliding_window=True,
    )
    assert model_config.max_model_len == expected


def test_get_sliding_window():
    TEST_SLIDING_WINDOW = 4096
    # Test that the sliding window is correctly computed.
    # For Qwen1.5/Qwen2, get_sliding_window() should be None
    # when use_sliding_window is False.
    qwen2_model_config = ModelConfig(
        "Qwen/Qwen1.5-7B",
        "Qwen/Qwen1.5-7B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )

    qwen2_model_config.hf_config.use_sliding_window = False
    qwen2_model_config.hf_config.sliding_window = TEST_SLIDING_WINDOW
    assert qwen2_model_config.get_sliding_window() is None

    qwen2_model_config.hf_config.use_sliding_window = True
    assert qwen2_model_config.get_sliding_window() == TEST_SLIDING_WINDOW

    mistral_model_config = ModelConfig(
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-v0.1",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )
    mistral_model_config.hf_config.sliding_window = None
    assert mistral_model_config.get_sliding_window() is None

    mistral_model_config.hf_config.sliding_window = TEST_SLIDING_WINDOW
    assert mistral_model_config.get_sliding_window() == TEST_SLIDING_WINDOW


def test_rope_customization():
    TEST_ROPE_SCALING = {"type": "dynamic", "factor": 2.0}
    TEST_ROPE_THETA = 16_000_000.0
    LONGCHAT_ROPE_SCALING = {"type": "linear", "factor": 8.0}

    llama_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )
    assert getattr(llama_model_config.hf_config, "rope_scaling", None) is None
    assert getattr(llama_model_config.hf_config, "rope_theta", None) == 500_000
    assert llama_model_config.max_model_len == 8192

    llama_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        rope_scaling=TEST_ROPE_SCALING,
        rope_theta=TEST_ROPE_THETA,
    )
    assert getattr(llama_model_config.hf_config, "rope_scaling",
                   None) == TEST_ROPE_SCALING
    assert getattr(llama_model_config.hf_config, "rope_theta",
                   None) == TEST_ROPE_THETA
    assert llama_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig(
        "lmsys/longchat-13b-16k",
        "lmsys/longchat-13b-16k",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )
    assert getattr(longchat_model_config.hf_config, "rope_scaling",
                   None) == LONGCHAT_ROPE_SCALING
    assert longchat_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig(
        "lmsys/longchat-13b-16k",
        "lmsys/longchat-13b-16k",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        rope_scaling=TEST_ROPE_SCALING,
    )
    assert getattr(longchat_model_config.hf_config, "rope_scaling",
                   None) == TEST_ROPE_SCALING
    assert longchat_model_config.max_model_len == 4096
