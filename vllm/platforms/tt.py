# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs import ProcessorInputs, PromptType
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = object

logger = init_logger(__name__)


def _register_model_if_missing(ModelRegistry, model_arch: str, model_path: str) -> None:
    """Register `model_arch` only if not already registered.

    This keeps TT model registration idempotent across multiple call sites
    (e.g. APIServer pre-register, TT worker import, and platform config hook).
    """
    if model_arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(model_arch, model_path)


def _should_pre_register_tt_test_models_from_cli() -> bool:
    """Return True iff `--override-tt-config` enables test models.

    `TTPlatform.pre_register_and_update()` runs before `VllmConfig` (and thus
    `ModelConfig.override_tt_config`) is constructed, but ModelConfig may
    inspect architectures early.
    """
    argv = list(sys.argv[1:])

    def _parse_override_tt_config(raw: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    # Users may pass either `--override-tt-config` or `--override_tt_config`.
    # Arg name normalization happens later during argparse processing, but this
    # function runs before parsing, so we normalize locally and compare against
    # the canonical `--override-tt-config`.
    canonical_flag = "--override-tt-config"
    for i, arg in enumerate(argv):
        if "=" in arg:
            flag, value = arg.split("=", 1)
            if flag.replace("_", "-") == canonical_flag:
                cfg = _parse_override_tt_config(value)
                return bool(cfg and cfg.get("register_test_models") is True)
        else:
            if arg.replace("_", "-") == canonical_flag and i + 1 < len(argv):
                cfg = _parse_override_tt_config(argv[i + 1])
                return bool(cfg and cfg.get("register_test_models") is True)

    return False


def register_tt_models(register_test_models=False) -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif llama_text_version == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]"
        )

    # Llama3.1/3.2 - Text
    _register_model_if_missing(ModelRegistry, "TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    _register_model_if_missing(ModelRegistry, "TTQwen2ForCausalLM", path_qwen_text)

    # Qwen3 - Text
    qwen3_text_version = os.getenv("TT_QWEN3_TEXT_VER", "tt_transformers")
    if qwen3_text_version == "tt_transformers":
        path_qwen3_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    elif qwen3_text_version == "qwen3_32b_galaxy":
        path_qwen3_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:QwenForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Qwen3 version: {qwen3_text_version}, "
            "pick one of [tt_transformers, qwen3_32b_galaxy]"
        )

    _register_model_if_missing(ModelRegistry, "TTQwen3ForCausalLM", path_qwen3_text)

    # Qwen2.5 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # Qwen3 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen3VLForConditionalGeneration",
        "models.demos.qwen3_vl.tt.generator_vllm:Qwen3VLForConditionalGeneration",
    )

    # Mistral
    _register_model_if_missing(
        ModelRegistry,
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    # Gemma3
    _register_model_if_missing(
        ModelRegistry,
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

    # DeepseekV3
    _register_model_if_missing(
        ModelRegistry,
        "TTDeepseekV3ForCausalLM",
        "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    )

    # GPT-OSS
    _register_model_if_missing(
        ModelRegistry,
        "TTGptOssForCausalLM",
        "models.tt_transformers.tt.generator_vllm:GptOssForCausalLM",
    )

    # Optionally register test models if explicitly enabled
    if register_test_models:
        register_tt_test_models()


def register_tt_test_models():
    """Register non-production TT models which are only used for testing."""
    from vllm.model_executor.models.registry import ModelRegistry

    # Fake model for testing multi-process inference on T3000
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyT3000MultiProcessModel",
        "models.vllm_test_utils.t3000_multiproc_test.test_model:DummyT3000MultiProcessModel",
    )

    # Fake model which does nothing, for measuring vLLM host overheads
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyNoOpModel",
        "models.vllm_test_utils.no_op_test.test_model:DummyNoOpModel",
    )


class TTPlatform(Platform):
    _enum = PlatformEnum.TT
    device_name: str = "tt"
    device_type: str = "tt"
    # Disable torch.compile on TT platform - the triton version in tt-metal
    # is incompatible with torch's inductor backend.
    simple_compile_backend: str = "eager"

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        # Called during CLI/parser setup (APIServer). ModelConfig may
        # validate/inspect architectures before VllmConfig is constructed in
        # this process, so we must ensure TT test models are registered early
        # when explicitly requested via CLI override.
        super().pre_register_and_update(parser)
        if _should_pre_register_tt_test_models_from_cli():
            register_tt_test_models()

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C or vllm._moe_C
        pass

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        assert not vllm_config.scheduler_config.chunked_prefill_enabled, (
            "Chunked prefill is not yet supported for TT backend"
        )
        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend"
        )
        assert (
            vllm_config.parallel_config.tensor_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
        ), "TT backend does not support distributed execution"
        assert not vllm_config.lora_config, "LoRA is not supported for TT backend"

        # Import and register models from tt-metal.
        #
        # NOTE: We also register TT models early in `vllm/v1/worker/tt_worker.py`
        # (at module import time). That registration is required to handle
        # engine/worker subprocess startup ordering where model architectures
        # may be inspected (e.g. multimodal processor cache init) before this
        # `check_and_update_config()` hook is reached in that process.
        override_tt_config = vllm_config.model_config.override_tt_config
        register_test_models = False
        if override_tt_config and "register_test_models" in override_tt_config:
            register_test_models = override_tt_config["register_test_models"]
            assert register_test_models in [True, False], (
                f"Invalid option register_test_models: {register_test_models}"
            )
        register_tt_models(register_test_models)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.tt_worker.TTWorker"
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm.v1.core.sched.ascend_scheduler.AscendScheduler"
            )

        # For TT models, prepend "TT" to the architecture name,
        # e.g. "TTLlamaForCausalLM"
        arch_names = vllm_config.model_config.hf_config.architectures
        for i in range(len(arch_names)):
            if not arch_names[i].startswith("TT"):
                arch_names[i] = "TT" + arch_names[i]

        # Verify that the TT architecture is registered in the model registry
        from vllm.model_executor.models.registry import ModelRegistry

        supported_archs = ModelRegistry.get_supported_archs()
        if not any(arch_name in supported_archs for arch_name in arch_names):
            tt_archs = sorted(
                [arch for arch in supported_archs if arch.startswith("TT")]
            )
            raise ValueError(
                f"No TT model architecture is registered for "
                f"model: '{vllm_config.model_config.model}'. "
                f"Available TT architectures: {tt_archs}"
            )

        # Setting attributes on the class level is kind of hacky, but
        # it's the only way to make validate_request depend on vllm_config
        # This is needed to catch incompatible requests early enough
        # to return an error instead of crashing.
        # TODO move this to tt_model_runner when request validation
        # stops depending on vllm_config

        if (
            override_tt_config is not None
            and "sample_on_device_mode" in override_tt_config
        ):
            sample_on_device_mode = override_tt_config["sample_on_device_mode"]
            assert sample_on_device_mode in [
                "all",
                "decode_only",
            ], f"Invalid sample_on_device_mode: {sample_on_device_mode}"
        else:
            sample_on_device_mode = None
        cls.sample_on_device_mode = sample_on_device_mode  # type: ignore[attr-defined]

        # Compat sampling uses the full vLLM sampling pipeline,
        # with logit processors and sampler, instead of our custom sampling.
        # It is enabled only if any of the requests in the batch requires it,
        # or if always_compat_sampling is enabled.

        always_compat_sampling = False
        if (
            override_tt_config is not None
            and "always_compat_sampling" in override_tt_config
        ):
            always_compat_sampling = override_tt_config["always_compat_sampling"]
            assert always_compat_sampling in [True, False], (
                "always_compat_sampling must be a boolean"
            )
            if always_compat_sampling:
                raise ValueError(
                    "always_compat_sampling is not yet supported for V1 TT backend."
                )
        cls.always_compat_sampling = always_compat_sampling  # type: ignore[attr-defined]

        # must perform local import to get around circular import
        from vllm.model_executor.model_loader.utils import get_model_architecture

        model_class, _ = get_model_architecture(vllm_config.model_config)

        # infer if non-greedy decoding is supported on-device
        # based on model implementation, and update platform
        # TODO: this should come from the class itself as an attribute
        cls.non_greedy_decoding_on_device = False  # type: ignore[attr-defined]
        if model_class.__module__.startswith(
            "models.demos.llama3_70b_galaxy.tt.generator_vllm"
        ):
            cls.non_greedy_decoding_on_device = True  # type: ignore[attr-defined]

        if model_class.__module__.startswith(
            "models.tt_transformers.tt.generator_vllm"
        ):
            cls.non_greedy_decoding_on_device = True  # type: ignore[attr-defined]

        # Get model capabilities from the class
        model_capabilities: dict | None = getattr(
            model_class, "model_capabilities", None
        )

        if vllm_config.cache_config.enable_prefix_caching:
            # Check prefix caching support from capabilities (default to False)
            supports_prefix_caching = (
                model_capabilities.get("supports_prefix_caching", False)
                if model_capabilities
                else False
            )

            if not supports_prefix_caching:
                vllm_config.cache_config.enable_prefix_caching = False
                logger.warning(
                    "Prefix caching is not supported in TT backend for %s, "
                    "disabling it",
                    model_class.__module__,
                )
            else:
                # Check if the model architecture uses sliding window
                uses_sliding_window = (
                    vllm_config.model_config.get_sliding_window() is not None
                )
                if uses_sliding_window:
                    vllm_config.cache_config.enable_prefix_caching = False
                    logger.warning(
                        "Prefix caching is not supported in TT backend for "
                        "models with sliding window, disabling it"
                    )

        logger.info(
            "Automatic prefix caching is %s",
            "enabled" if vllm_config.cache_config.enable_prefix_caching else "disabled",
        )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # The regular v0 vLLM sampling code tries
        # to use pinned memory in case we're using GPUs.
        return False

    # Require DP ranks to gather batches to a single driver
    # before executing (used by core.py to gate DP-gather behavior).
    @classmethod
    def requires_gathered_batch_dp(cls) -> bool:
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: "PromptType",
        params: "SamplingParams | PoolingParams",
        processed_inputs: "ProcessorInputs",
    ) -> None:
        """Raises if this request is unsupported on this platform"""

        dev = cls.device_name

        if params.best_of is not None:
            raise ValueError(f"Not yet supporting best_of on {dev}")
        if params.prompt_logprobs is not None:
            raise ValueError(f"Not yet supporting prompt_logprobs on {dev}")
        if params.logits_processors:
            raise ValueError(f"Custom logits_processors not supported on {dev} in V1")

    @staticmethod
    def compat_sampling_required(sampling_params, num_devices) -> bool:
        # Device logprobs only supported on multi-device setups and only
        # the sampled token's logprob is returned (not top-k alternatives).
        # Single device: any logprobs require host sampling.
        # Multi-device: logprobs > 1 requires host sampling because device
        # can only return the sampled token's logprob.
        # https://github.com/tenstorrent/tt-metal/issues/34077
        if (
            sampling_params.logprobs is not None
            and sampling_params.logprobs > 0
            and (num_devices == 1 or sampling_params.logprobs > 1)
        ):
            return True

        # all of the following sampling params require compat sampling
        return (
            sampling_params.min_p != 0.0
            or (
                sampling_params.bad_words is not None
                and len(sampling_params.bad_words) > 0
            )
            or sampling_params.prompt_logprobs is not None
            or sampling_params.logits_processors is not None
            or sampling_params.guided_decoding is not None
            or sampling_params.logit_bias is not None
            or sampling_params.allowed_token_ids is not None
            or sampling_params.min_tokens != 0
        )
