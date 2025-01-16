from unittest.mock import patch

import pytest
from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm import LLM

from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    if not model_info.is_available_online:
        pytest.skip("Model is not available online")
    if model_info.min_transformers_version is not None:
        current_version = TRANSFORMERS_VERSION
        required_version = model_info.min_transformers_version
        if Version(current_version) < Version(required_version):
            pytest.skip(
                f"You have `transformers=={current_version}` installed, but "
                f"`transformers>={required_version}` is required to run this "
                "model")

    # Avoid OOM
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type == "deepseek_vl_v2":
            hf_config.update({"architectures": ["DeepseekVLV2ForCausalLM"]})

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
