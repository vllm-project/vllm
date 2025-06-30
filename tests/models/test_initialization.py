# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from vllm import LLM
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.utils import GiB_bytes
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.engine.core import EngineCore as V1EngineCore

from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch: str, monkeypatch: pytest.MonkeyPatch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # FIXME: Possible memory leak in the previous tests?
    if model_arch == "GraniteSpeechForConditionalGeneration":
        pytest.skip("Avoid OOM")

    # Avoid OOM and reduce initialization time by only using 1 layer
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        hf_config.update(model_info.hf_overrides)

        text_config = hf_config.get_text_config()

        # Ensure at least 2 expert per group
        # Since `grouped_topk` assums top-2
        num_experts = getattr(text_config, 'n_group', 1) * 2

        text_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "num_local_experts": num_experts,
            # Otherwise there will not be any expert layers
            "first_k_dense_replace": 0,
            # To avoid OOM on DeepSeek-V3
            "n_routed_experts": num_experts,
        })

        if hasattr(hf_config, "vision_config"):
            hf_config.vision_config.update({
                "num_layers": 1,
                "num_hidden_layers": 1,
            })

        # e.g.: ibm-granite/granite-speech-3.3-2b
        if hasattr(hf_config, "encoder_config"):
            hf_config.encoder_config.update({
                "num_layers": 1,
                "num_hidden_layers": 1,
            })

        return hf_config

    # Avoid calling model.forward()
    def _initialize_kv_caches_v0(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    def _initialize_kv_caches_v1(self, vllm_config):
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        scheduler_kv_cache_config = get_kv_cache_config(
            vllm_config,
            kv_cache_specs[0],
            10 * GiB_bytes,
        )

        # gpu_blocks (> 0), cpu_blocks, scheduler_kv_cache_config
        return 1, 0, scheduler_kv_cache_config

    with (patch.object(V0LLMEngine, "_initialize_kv_caches",
                       _initialize_kv_caches_v0),
          patch.object(V1EngineCore, "_initialize_kv_caches",
                       _initialize_kv_caches_v1), monkeypatch.context() as m):
        if model_info.v0_only:
            m.setenv("VLLM_USE_V1", "0")
        LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            revision=model_info.revision,
            speculative_config={
                "model": model_info.speculative_model,
                "num_speculative_tokens": 1,
            } if model_info.speculative_model else None,
            trust_remote_code=model_info.trust_remote_code,
            max_model_len=model_info.max_model_len,
            # these tests seem to produce leftover memory
            gpu_memory_utilization=0.80,
            load_format="dummy",
            hf_overrides=hf_overrides,
        )
