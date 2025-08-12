# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from unittest.mock import patch

import pytest

from vllm.config import ModelConfig
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.utils import GiB_bytes, set_default_torch_num_threads
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ...conftest import VllmRunner
from ..registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS
from ..utils import dummy_hf_overrides

ARCH_TO_SKIP = {
    "MolmoForCausalLM": "incompatible requirements",
    "MiniMaxVL01ForConditionalGeneration": "broken model",
}


def create_batched_mm_kwargs(
    model_config: ModelConfig,
    processor: BaseMultiModalProcessor,
) -> MultiModalKwargs:
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
    return mm_kwargs


@pytest.mark.core_model
@pytest.mark.parametrize("model_arch", list(_MULTIMODAL_EXAMPLE_MODELS.keys()))
def test_model_tensor_schema(model_arch: str, vllm_runner: type[VllmRunner],
                             monkeypatch):
    if model_arch in ARCH_TO_SKIP:
        pytest.skip(f"Skipping {model_arch} due to {ARCH_TO_SKIP[model_arch]}")

    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip",
                                          check_max_version=False)

    model_id = model_info.default

    hf_overrides_fn = partial(dummy_hf_overrides,
                              model_arch=model_arch,
                              exist_overrides=model_info.hf_overrides)

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]

    if not any(
            hasattr(model_cls, f"_parse_and_validate_{m}_input")
            for m in ["image", "video", "audio"]):
        pytest.skip(f"{model_arch} does not support tensor schema validation.")

    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()
    limit_mm_per_prompt = {
        modality: 3 if limit is None else limit
        for modality, limit in supported_mm_limits.items()
    }

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
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        if model_info.v0_only:
            m.setenv("VLLM_USE_V1", "0")

        with (
                set_default_torch_num_threads(1),
                vllm_runner(
                    model_id,
                    tokenizer_name=model_info.tokenizer,
                    tokenizer_mode=model_info.tokenizer_mode,
                    revision=model_info.revision,
                    trust_remote_code=model_info.trust_remote_code,
                    max_model_len=model_info.max_model_len,
                    load_format="dummy",
                    hf_overrides=hf_overrides_fn,
                    limit_mm_per_prompt=limit_mm_per_prompt,
                    enforce_eager=True,
                ) as vllm_model,
        ):
            model_config = vllm_model.llm.llm_engine.model_config
            llm_engine = vllm_model.llm.llm_engine

            if hasattr(llm_engine, "processor"):
                # v1 processor
                mm_registry = llm_engine.processor.mm_registry
            else:
                # v0 input_preprocessor
                mm_registry = llm_engine.input_preprocessor.mm_registry

            processor = mm_registry.create_processor(model_config)
            mm_kwargs = create_batched_mm_kwargs(model_config, processor)

            def validate_model_input(model):
                for modality in ("audio", "image", "video"):
                    method_name = f"_parse_and_validate_{modality}_input"
                    if hasattr(model, method_name):
                        getattr(model, method_name)(**mm_kwargs)

            vllm_model.apply_model(validate_model_input)
