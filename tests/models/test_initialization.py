from unittest.mock import patch

import pytest
import transformers
from transformers import PretrainedConfig

from vllm import LLM

from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch):
    if (model_arch == "Idefics3ForConditionalGeneration"
            and transformers.__version__ < "4.46.0"):
        pytest.skip(reason="Model introduced in HF >= 4.46.0")

    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    if not model_info.is_available_online:
        pytest.skip("Model is not available online")

    # Avoid OOM
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(hf_config, "text_config"):
            text_config: PretrainedConfig = hf_config.text_config
        else:
            text_config = hf_config

        text_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 2,
            "num_local_experts": 2,
        })

        return hf_config

    # Avoid calling model.forward()
    def _initialize_kv_caches(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    with patch.object(LLM.get_engine_class(), "_initialize_kv_caches",
                      _initialize_kv_caches):
        LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            speculative_model=model_info.speculative_model,
            num_speculative_tokens=1 if model_info.speculative_model else None,
            trust_remote_code=model_info.trust_remote_code,
            load_format="dummy",
            hf_overrides=hf_overrides,
        )
