# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from unittest.mock import patch

import pytest

from vllm import LLM
from vllm.utils import GiB_bytes
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ..utils import create_new_process_for_each_test
from .registry import (_TRANSFORMERS_BACKEND_MODELS, AUTO_EXAMPLE_MODELS,
                       HF_EXAMPLE_MODELS, HfExampleModels)
from .utils import dummy_hf_overrides

# This minimal list of model architectures is smaller than the total list of
# supported models. The intention is that in the "typical" regression testing
# scenario, we only test initializing these models. This subset was chosen
# to include representative examples of model varieties/workloads (conditional
# generation, sequence classification, causal LM, ranking, chat, reward model,
# multimodal, geospatial, voice, embedding, MTP)
MINIMAL_MODEL_ARCH_LIST = [
    "LlavaForConditionalGeneration", "Llama4ForConditionalGeneration",
    "BertForSequenceClassification", "Gemma3nForCausalLM", "JinaVLForRanking",
    "InternVLChatModel", "InternLM2ForRewardModel",
    "TransformersForMultimodalLM", "PrithviGeoSpatialMAE", "UltravoxModel",
    "DeepSeekMTPModel", "XLMRobertaModel"
]

# This list is the complement of the minimal list above. The intention is that
# this list of models is only tested in a "special case" i.e. most PRs should
# not test these models
OTHER_MODEL_ARCH_LIST = (set(HF_EXAMPLE_MODELS.get_supported_archs()) -
                         set(MINIMAL_MODEL_ARCH_LIST))


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
                              exist_overrides=model_info.hf_overrides,
                              use_original_num_layers=getattr(
                                  model_info, 'use_original_num_layers',
                                  False))

    # Avoid calling model.forward()
    def _initialize_kv_caches_v1(self, vllm_config):
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        scheduler_kv_cache_config = get_kv_cache_configs(
            vllm_config,
            kv_cache_specs,
            [10 * GiB_bytes],
        )[0]

        # gpu_blocks (> 0), cpu_blocks, scheduler_kv_cache_config
        return 1, 0, scheduler_kv_cache_config

    with (patch.object(V1EngineCore, "_initialize_kv_caches",
                       _initialize_kv_caches_v1), monkeypatch.context() as m):
        if model_info.v0_only:
            # NOTE(woosuk): skip the test for V0-only models
            return

        if model_arch in ("Phi4FlashForCausalLM", "MotifForCausalLM"):
            pytest.skip(
                "Differential Flash Attention backend has been removed.")
        if model_arch == "GptOssForCausalLM":
            # FIXME: A hack to bypass FA3 assertion because our CI's L4 GPU
            # has cc==8.9 which hasn't supported FA3 yet. Remove this hack when
            # L4 supports FA3.
            m.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
        if model_arch == "WhisperForConditionalGeneration":
            m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            revision=model_info.revision,
            enforce_eager=model_info.enforce_eager,
            skip_tokenizer_init=model_info.skip_tokenizer_init,
            dtype=model_info.dtype,
            speculative_config={
                "model": model_info.speculative_model,
                "num_speculative_tokens": 1,
            } if model_info.speculative_model else None,
            trust_remote_code=model_info.trust_remote_code,
            max_model_len=model_info.max_model_len,
            # these tests seem to produce leftover memory
            gpu_memory_utilization=0.80,
            load_format="dummy",
            model_impl="transformers"
            if model_arch in _TRANSFORMERS_BACKEND_MODELS else "vllm",
            hf_overrides=hf_overrides_fn,
            max_num_seqs=model_info.max_num_seqs)


@pytest.mark.parametrize("model_arch", MINIMAL_MODEL_ARCH_LIST)
def test_can_initialize_small_subset(model_arch: str,
                                     monkeypatch: pytest.MonkeyPatch):
    """Test initializing small subset of supported models"""
    if model_arch == "Lfm2ForCausalLM":
        pytest.skip("Skipping until test supports V1-only models")
    can_initialize(model_arch, monkeypatch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch", OTHER_MODEL_ARCH_LIST)
def test_can_initialize_large_subset(model_arch: str,
                                     monkeypatch: pytest.MonkeyPatch):
    """Test initializing large subset of supported models
    
    This test covers the complement of the tests covered in the "small subset"
    test.
    """
    if model_arch == "Lfm2ForCausalLM":
        pytest.skip("Skipping until test supports V1-only models")
    can_initialize(model_arch, monkeypatch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch",
                         AUTO_EXAMPLE_MODELS.get_supported_archs())
def test_implicit_converted_models(model_arch: str,
                                   monkeypatch: pytest.MonkeyPatch):
    can_initialize(model_arch, monkeypatch, AUTO_EXAMPLE_MODELS)
