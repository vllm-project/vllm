import pytest

from vllm import LLM

from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_model_can_initialize(model_arch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)

    if not model_info.is_available_online:
        pytest.skip("Model is not available online")

    try:
        LLM(model_info.default,
            trust_remote_code=model_info.trust_remote_code,
            load_format="dummy")
    except Exception:
        raise
