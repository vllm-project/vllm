# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator, Iterable
from typing import cast

import pytest
import torch
import torch.nn as nn
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

from vllm import EngineArgs
from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import _has_online_quant
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
    validate_weights_loading,
)
from vllm.renderers import ChatParams, renderer_from_config
from vllm.transformers_utils.config import get_safetensors_params_metadata

from .registry import (
    _TRANSFORMERS_BACKEND_MODELS,
    AUTO_EXAMPLE_MODELS,
    HF_EXAMPLE_MODELS,
    HfExampleModels,
)
from .utils import initialize_dummy_model

logger = init_logger(__name__)

# This minimal list of model architectures is smaller than the total list of
# supported models. The intention is that in the "typical" regression testing
# scenario, we only test initializing these models. This subset was chosen
# to include representative examples of model varieties/workloads (conditional
# generation, sequence classification, causal LM, ranking, chat, reward model,
# multimodal, geospatial, voice, embedding, MTP)
MINIMAL_MODEL_ARCH_LIST = [
    "LlavaForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "BertForSequenceClassification",
    "Gemma3nForCausalLM",
    "JinaVLForRanking",
    "InternVLChatModel",
    "InternLM2ForRewardModel",
    "TransformersMultiModalForCausalLM",
    "Terratorch",
    "UltravoxModel",
    "DeepSeekMTPModel",
    "XLMRobertaModel",
]

# This list is the complement of the minimal list above. The intention is that
# this list of models is only tested in a "special case" i.e. most PRs should
# not test these models
OTHER_MODEL_ARCH_LIST = set(HF_EXAMPLE_MODELS.get_supported_archs()) - set(
    MINIMAL_MODEL_ARCH_LIST
)


class _SkipValidation(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)

        self.reason = reason


def _get_weights_iterator(
    source: DefaultModelLoader.Source,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    metadata = get_safetensors_params_metadata(
        source.model_or_path,
        revision=source.revision,
    )
    if not metadata:
        raise _SkipValidation("Missing safetensors metadata")

    for name, info in metadata.items():
        if "dtype" not in info:
            raise _SkipValidation(f"Missing safetensors dtype metadata for {name=}")

        dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(info["dtype"])
        if dtype is None:
            raise _SkipValidation(
                f"Unrecognized safetensors dtype for {name=}: {info['dtype']}"
            )

        weight = torch.empty(info["shape"], dtype=dtype)

        yield name, weight


def _get_dummy_weights(model: nn.Module, model_config: ModelConfig):
    primary_weights = DefaultModelLoader.Source(
        model_config.model,
        model_config.revision,
        prefix="",
        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    yield from _get_weights_iterator(primary_weights)

    secondary_weights = cast(
        Iterable[DefaultModelLoader.Source],
        getattr(model, "secondary_weights", ()),
    )
    for source in secondary_weights:
        yield from _get_weights_iterator(source)


def _load_dummy_weights(vllm_config: VllmConfig):
    """
    Imitate `DefaultModelLoader.load_weights` so we can use dummy weights
    to validate the weight mapping.
    """
    device_config = vllm_config.device_config
    load_config = vllm_config.load_config
    load_device = (
        device_config.device if load_config.device is None else load_config.device
    )
    target_device = torch.device(load_device)
    model_config = vllm_config.model_config

    model = initialize_model(
        vllm_config=vllm_config,
        model_config=model_config,
        prefix="",
    )

    weights_it = _get_dummy_weights(model, model_config)
    loaded_weights = model.load_weights(weights_it)
    validate_weights_loading(model, loaded_weights)

    if _has_online_quant(model):
        finalize_layerwise_processing(model, model_config)

    process_weights_after_loading(model, model_config, target_device)

    return model


def can_initialize(model_arch: str, EXAMPLE_MODELS: HfExampleModels):
    """
    create_new_process_for_each_test can avoid CUDA re-initialization error.
    """
    model_info = EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    if model_arch == "MoonshotKimiaForCausalLM":
        pytest.skip(
            "Kimi-Audio requires SpeechToTextConfig "
            "which is not configured in test environment"
        )

    if model_arch == "Terratorch":
        import importlib.util

        if importlib.util.find_spec("terratorch") is None:
            pytest.skip(
                "terratorch is not installed; "
                "temporarily skipped while PyPI has `lightning` quarantined "
                "(see #41376)"
            )

    if model_arch in ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]:
        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability()
        if capability and capability.major < 9:
            pytest.skip(
                f"DeepseekV32 requires Hopper (9.0+) or Blackwell (10.0+) "
                f"for FLASHMLA_SPARSE backend. Current device has compute "
                f"capability {capability.major}.{capability.minor}"
            )

    # FIXME: A hack to bypass FA3 assertion because our CI's L4 GPU
    # has cc==8.9 which hasn't supported FA3 yet. Remove this hack when
    # L4 supports FA3.
    # Step1ForCausalLM requires TRITON_ATTN for use_alibi_sqrt support.
    attention_config = (
        AttentionConfig(backend="TRITON_ATTN")
        if model_arch in ("GptOssForCausalLM", "Step1ForCausalLM")
        else AttentionConfig()
    )

    engine_args = EngineArgs(
        model=model_info.default,
        tokenizer=model_info.tokenizer,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        enforce_eager=model_info.enforce_eager,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        dtype=model_info.dtype,
        speculative_config={
            "model": model_info.speculative_model,
            "method": model_info.speculative_method,
            "num_speculative_tokens": 1,
        }
        if model_info.speculative_model
        else None,
        trust_remote_code=model_info.trust_remote_code,
        enable_prefix_caching=model_info.enable_prefix_caching,
        max_model_len=model_info.max_model_len,
        max_num_batched_tokens=model_info.max_num_batched_tokens,
        load_format="dummy",
        model_impl="transformers"
        if model_arch in _TRANSFORMERS_BACKEND_MODELS
        else "vllm",
        hf_overrides=model_info.hf_overrides,
        max_num_seqs=model_info.max_num_seqs,
        attention_config=attention_config,
    )
    vllm_config = engine_args.create_engine_config()

    renderer = renderer_from_config(vllm_config)
    renderer.warmup(ChatParams(chat_template=load_chat_template(None)))

    try:
        # TODO: Handle speculative model
        with initialize_dummy_model(_load_dummy_weights, vllm_config, device="meta"):
            pass
    except _SkipValidation as e:
        logger.warning(
            "Skipping validation when loading dummy weights for %s. Reason: %s",
            vllm_config.model_config.model,
            e.reason,
        )


@pytest.mark.parametrize("model_arch", MINIMAL_MODEL_ARCH_LIST)
def test_can_initialize_small_subset(model_arch: str):
    """Test initializing small subset of supported models"""
    can_initialize(model_arch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch", OTHER_MODEL_ARCH_LIST)
def test_can_initialize_large_subset(model_arch: str):
    """Test initializing large subset of supported models

    This test covers the complement of the tests covered in the "small subset"
    test.
    """
    can_initialize(model_arch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch", AUTO_EXAMPLE_MODELS.get_supported_archs())
def test_implicit_converted_models(model_arch: str):
    can_initialize(model_arch, AUTO_EXAMPLE_MODELS)
