from unittest.mock import patch

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

    # Avoid calling model.forward()
    def _initialize_kv_caches(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    with patch.object(LLM.get_engine_class(), "_initialize_kv_caches",
                      _initialize_kv_caches):
        LLM(
            model_info.default,
            trust_remote_code=model_info.trust_remote_code,
            load_format="dummy",
            hf_overrides={
                **text_config_overrides,
                "text_config": text_config_overrides,  # For multi-modal models
            },
        )
