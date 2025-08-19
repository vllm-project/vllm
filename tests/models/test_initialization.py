# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from unittest.mock import patch

import pytest

from vllm import LLM
from vllm.config import ModelImpl
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.utils import GiB_bytes
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ..utils import create_new_process_for_each_test
from .registry import (_TRANSFORMERS_BACKEND_MODELS, AUTO_EXAMPLE_MODELS,
                       HF_EXAMPLE_MODELS, HfExampleModels)
from .utils import dummy_hf_overrides


@create_new_process_for_each_test()
def can_initialize(model_arch: str, monkeypatch: pytest.MonkeyPatch,
                   EXAMPLE_MODELS: HfExampleModels):
    """The reason for using create_new_process_for_each_test is to avoid
    the WARNING:
        "We must use the 'spawn' multiprocessing start method. Overriding
        VLLM_WORKER_MULTIPROC_METHOD to 'spawn'."
    The spawn process causes the _initialize_kv_caches_v1 function below to
    become ineffective.
    """

    model_info = EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    hf_overrides_fn = partial(dummy_hf_overrides,
                              model_arch=model_arch,
                              exist_overrides=model_info.hf_overrides)

    if model_arch in ("Llama4ForCausalLM", "EagleLlama4ForCausalLM"):
        from vllm.model_executor.models.llama4 import Llama4ForCausalLM
        from vllm.model_executor.models.registry import ModelRegistry
        ModelRegistry.register_model("Llama4ForCausalLM", Llama4ForCausalLM)

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
        if model_arch == "Phi4FlashForCausalLM":
            # Phi4FlashForCausalLM only supports DIFFERENTIAL_FLASH_ATTN backend
            m.setenv("VLLM_ATTENTION_BACKEND", "DIFFERENTIAL_FLASH_ATTN")
        if model_arch == "GptOssForCausalLM":
            # FIXME: A hack to bypass FA3 assertion because our CI's L4 GPU
            # has cc==8.9 which hasn't supported FA3 yet. Remove this hack when
            # L4 supports FA3.
            m.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
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
            model_impl=ModelImpl.TRANSFORMERS
            if model_arch in _TRANSFORMERS_BACKEND_MODELS else ModelImpl.VLLM,
            hf_overrides=hf_overrides_fn,
        )


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch: str, monkeypatch: pytest.MonkeyPatch):
    can_initialize(model_arch, monkeypatch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch",
                         AUTO_EXAMPLE_MODELS.get_supported_archs())
def test_implicit_converted_models(model_arch: str,
                                   monkeypatch: pytest.MonkeyPatch):
    can_initialize(model_arch, monkeypatch, AUTO_EXAMPLE_MODELS)
