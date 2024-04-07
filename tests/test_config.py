import pytest

from vllm.config import ModelConfig

MODEL_IDS = ["Qwen/Qwen1.5-7B", "mistralai/Mistral-7B-v0.1"]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_disable_sliding_window(model_id):
    model_config = ModelConfig(
        model_id,
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
        disable_sliding_window=True,
    )

    assert model_config.max_model_len <= model_config.hf_config.sliding_window


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
        download_dir=None,
        load_format="dummy",
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
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
    )
    mistral_model_config.hf_config.sliding_window = None
    assert mistral_model_config.get_sliding_window() is None

    mistral_model_config.hf_config.sliding_window = TEST_SLIDING_WINDOW
    assert mistral_model_config.get_sliding_window() == TEST_SLIDING_WINDOW
