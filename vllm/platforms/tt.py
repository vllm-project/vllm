# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional, Union

import torch

import vllm.envs as envs
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
else:
    ModelConfig = None
    VllmConfig = None
    PoolingParams = None

logger = init_logger(__name__)


class TTPlatform(Platform):
    _enum = PlatformEnum.TT
    device_name: str = "tt"
    device_type: str = "tt"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        assert not vllm_config.scheduler_config.chunked_prefill_enabled, (
            "Chunked prefill is not yet supported for TT backend")
        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend")
        assert (vllm_config.parallel_config.tensor_parallel_size == 1
                and vllm_config.parallel_config.pipeline_parallel_size
                == 1), "TT backend does not support distributed execution"
        assert not vllm_config.lora_config, (
            "LoRA is not supported for TT backend")
        assert not vllm_config.cache_config.enable_prefix_caching, (
            "Automatic prefix caching is not yet supported for TT backend")

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "vllm.v1.worker.tt_worker.TTWorker"
                vllm_config.scheduler_config.scheduler_cls = (
                    "vllm.v1.core.sched.ascend_scheduler.AscendScheduler")
            else:
                parallel_config.worker_cls = "vllm.worker.tt_worker.TTWorker"

        # For TT models, prepend "TT" to the architecture name,
        # e.g. "TTLlamaForCausalLM"
        arch_names = vllm_config.model_config.hf_config.architectures
        for i in range(len(arch_names)):
            if not arch_names[i].startswith("TT"):
                arch_names[i] = "TT" + arch_names[i]

        # Setting attributes on the class level is kind of hacky, but
        # it's the only way to make validate_request depend on vllm_config
        # This is needed to catch incompatible requests early enough
        # to return an error instead of crashing.
        # TODO move this to tt_model_runner when request validation
        # stops depending on vllm_config

        override_tt_config = vllm_config.model_config.override_tt_config
        if (override_tt_config is not None
                and "sample_on_device_mode" in override_tt_config):
            sample_on_device_mode = override_tt_config["sample_on_device_mode"]
            assert sample_on_device_mode in [
                "all", "decode_only"
            ], f"Invalid sample_on_device_mode: {sample_on_device_mode}"
        else:
            sample_on_device_mode = None
        cls.sample_on_device_mode = sample_on_device_mode  # type: ignore[attr-defined]

        # Compat sampling uses the full vLLM sampling pipeline,
        # with logit processors and sampler, instead of our custom sampling.
        # It is off by default, and enabled only on request
        # or if any of the requests in the batch require it.
        # For now, it is only supported with host-side sampling.
        cls.compat_sampling_possible = (  # type: ignore[attr-defined]
            sample_on_device_mode is None)

        if cls.compat_sampling_possible and envs.VLLM_USE_V1:  # type: ignore[attr-defined]
            cls.compat_sampling_possible = False  # type: ignore[attr-defined]
            logger.warning(
                "Disabling compatibility sampling as it's not yet support for "
                "V1 TT backend.")

        always_compat_sampling = False
        if override_tt_config is not None \
            and "always_compat_sampling" in override_tt_config:
            always_compat_sampling = override_tt_config[
                "always_compat_sampling"]
            assert always_compat_sampling in [
                True, False
            ], "always_compat_sampling must be a boolean"
            if always_compat_sampling:
                if envs.VLLM_USE_V1:
                    raise ValueError(
                        "always_compat_sampling is not yet supported for "
                        "V1 TT backend.")
                logger.info(
                    "Compatibility sampling mode enabled for all requests")
        cls.always_compat_sampling = always_compat_sampling  # type: ignore[attr-defined]

        if cls.always_compat_sampling and not cls.compat_sampling_possible:  # type: ignore[attr-defined]
            raise ValueError("Compatibility sampling mode only works with"
                             "sample_on_device_mode=None")

        # must perform local import to get around circular import
        from vllm.model_executor.model_loader.utils import (
            get_model_architecture)

        # infer if non-greedy decoding is supported on-device
        # based on model implementation, and update platform
        model_class, _ = get_model_architecture(vllm_config.model_config)
        # TODO: this should come from the class itself as an attribute
        cls.non_greedy_decoding_on_device = False  # type: ignore[attr-defined]
        if model_class.__module__.startswith(
                "models.demos.llama3_70b_galaxy.tt.generator_vllm"):
            cls.non_greedy_decoding_on_device = True  # type: ignore[attr-defined]

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TT is experimental.
        # Allow users to opt in, but give a warning.
        if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
            if model_config.is_encoder_decoder:
                raise ValueError(
                    "VLLM_USE_V1=1 was set but encoder-decoder models aren't "
                    "yet supported in V1 for TT")
            logger.warning(
                "Enabling V1 since VLLM_USE_V1=1, however V1 is still "
                "experimental for TT backend.")
            return envs.VLLM_USE_V1
        return False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # The sampling code tries to use pinned memory in case we're using GPUs.
        return False

    # Require DP ranks to gather batches to a single driver
    # before executing (used by core.py to gate DP-gather behavior).
    @classmethod
    def requires_gathered_batch_dp(cls) -> bool:
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""

        if isinstance(params, SamplingParams):
            if params.n != 1:
                raise ValueError(
                    f"Currently only supporting n=1 on {cls.device_name}.")
            if params.best_of is not None:
                raise ValueError(
                    f"Currently not supporting best_of on {cls.device_name}")
            if params.prompt_logprobs is not None:
                raise ValueError(
                    f"Currently not supporting prompt_logprobs on "
                    f"{cls.device_name}")
            if cls.compat_sampling_required(
                    params
            ) and not cls.compat_sampling_possible:  # type: ignore[attr-defined]
                raise ValueError(
                    "Sampling params beyond temperature, "
                    "top_k, top_p require compatibility sampling mode"
                    " which is only available with"
                    "sample_on_device_mode=None. "
                    f"Supplied params: {params}")
            sample_mode = cls.sample_on_device_mode  # type: ignore[attr-defined]
            non_greedy = cls.non_greedy_decoding_on_device  # type: ignore[attr-defined]
            if (params.temperature > 0.0 and sample_mode is not None
                    and not non_greedy):
                raise ValueError(
                    "Non-greedy decoding on-device is not supported by this "
                    f"model implementation. Supplied params: {params}")

    @staticmethod
    def compat_sampling_required(sampling_params) -> bool:
        return (sampling_params.presence_penalty != 0.0
                or sampling_params.frequency_penalty != 0.0
                or sampling_params.repetition_penalty != 1.0
                or sampling_params.min_p != 0.0
                or (sampling_params.bad_words is not None
                    and len(sampling_params.bad_words) > 0)
                or sampling_params.logprobs is not None
                or sampling_params.prompt_logprobs is not None
                or sampling_params.logits_processors is not None
                or sampling_params.truncate_prompt_tokens is not None
                or sampling_params.guided_decoding is not None
                or sampling_params.logit_bias is not None
                or sampling_params.allowed_token_ids is not None)
