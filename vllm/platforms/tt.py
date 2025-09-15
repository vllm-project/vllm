# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.pooling_params import PoolingParams
else:
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

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.worker.tt_worker.TTWorker"

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

        always_compat_sampling = False
        if override_tt_config is not None \
            and "always_compat_sampling" in override_tt_config:
            logger.info("Compatibility sampling mode enabled for all requests")
            always_compat_sampling = override_tt_config[
                "always_compat_sampling"]
            assert always_compat_sampling in [
                True, False
            ], "always_compat_sampling must be a boolean"
        cls.always_compat_sampling = always_compat_sampling  # type: ignore[attr-defined]

        if cls.always_compat_sampling and not cls.compat_sampling_possible:  # type: ignore[attr-defined]
            raise ValueError("Compatibility sampling mode only works with"
                             "sample_on_device_mode=None")

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # The sampling code tries to use pinned memory in case we're using GPUs.
        return False

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
