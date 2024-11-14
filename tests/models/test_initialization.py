import pytest

from vllm import LLM

from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    if not model_info.is_available_online:
        pytest.skip("Model is not available online")

    # Avoid OOM
    text_config_overrides = {
        "num_layers": 1,
        "num_hidden_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 2,
        "num_local_experts": 2,
    }

    try:
        LLM(
            model_info.default,
            trust_remote_code=model_info.trust_remote_code,
            max_model_len=512,
            max_num_seqs=1,
            load_format="dummy",
            hf_overrides={
                **text_config_overrides,
                "text_config": text_config_overrides,  # For multi-modal models
            })
    except Exception:
        raise
