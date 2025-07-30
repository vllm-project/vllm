# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from ...conftest import VllmRunner
from ..registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS

ARCH_TO_SKIP = {
    "MolmoForCausalLM": "incompatible requirements",
}


@pytest.mark.core_model
@pytest.mark.parametrize("model_arch", list(_MULTIMODAL_EXAMPLE_MODELS.keys()))
def test_model_tensor_schema(model_arch: str, vllm_runner: type[VllmRunner],
                             monkeypatch):
    if model_arch in ARCH_TO_SKIP:
        pytest.skip(f"Skipping {model_arch} due to {ARCH_TO_SKIP[model_arch]}")

    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")

    model_id = model_info.default

    # Avoid OOM and reduce initialization time by only using 1 layer
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        text_config = hf_config.get_text_config()
        # Ensure at least 2 expert per group
        # Since `grouped_topk` assumes top-2
        n_group = getattr(text_config, 'n_group', None)
        num_experts = n_group * 2 if n_group is not None else 2
        # we use three layers for Gemma-3n to check
        # both normal layer and kv_shared_layer
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
            # For Gemma-3n
            "num_kv_shared_layers": 1,
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

        if model_info.hf_overrides:
            hf_config.update(model_info.hf_overrides)

        return hf_config

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)

    if not any(
            hasattr(model_cls, f"_parse_and_validate_{m}_input")
            for m in ["image", "video", "audio"]):
        pytest.skip(f"{model_arch} does not support tensor schema validation.")

    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    processor = factories.build_processor(ctx)
    processing_info = processor.info
    dummy_inputs = processor.dummy_inputs
    supported_mm_limits = processing_info.get_supported_mm_limits()
    mm_counts = {
        modality: 3 if limit is None else limit
        for modality, limit in supported_mm_limits.items()
    }
    processor_inputs = dummy_inputs.get_dummy_processor_inputs(
        seq_len=model_config.max_model_len,
        mm_counts=mm_counts,
    )
    mm_kwargs = processor.apply(
        prompt=processor_inputs.prompt,
        mm_data=processor_inputs.mm_data,
        hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
        tokenization_kwargs=processor_inputs.tokenization_kwargs,
    )["mm_kwargs"]
    mm_kwargs = MultiModalKwargs.batch([mm_kwargs])

    # Avoid calling model.forward()
    def _initialize_kv_caches_v0(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    with patch.object(V0LLMEngine, "_initialize_kv_caches",
                      _initialize_kv_caches_v0):
        monkeypatch.setenv("VLLM_USE_V1", "0")
        with vllm_runner(
                model_id,
                tokenizer_name=model_info.tokenizer,
                tokenizer_mode=model_info.tokenizer_mode,
                revision=model_info.revision,
                trust_remote_code=model_info.trust_remote_code,
                max_model_len=model_info.max_model_len,
                load_format="dummy",
                hf_overrides=hf_overrides,
        ) as vllm_model:

            def validate_model_input(model):
                for modality in ("audio", "image", "video"):
                    method_name = f"_parse_and_validate_{modality}_input"
                    if hasattr(model, method_name):
                        getattr(model, method_name)(**mm_kwargs)

            vllm_model.llm.apply_model(validate_model_input)
