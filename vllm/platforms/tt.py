# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Optional, Union

import torch

from vllm.inputs import ProcessorInputs, PromptType
from vllm.sampling_params import SamplingParams

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.pooling_params import PoolingParams
else:
    VllmConfig = None
    PoolingParams = None


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
            if params.logprobs is not None:
                raise ValueError(
                    f"Currently not supporting logprobs on {cls.device_name}")
            if params.prompt_logprobs is not None:
                raise ValueError(
                    f"Currently not supporting prompt_logprobs on "
                    f"{cls.device_name}")
            if params.guided_decoding is not None:
                raise ValueError(
                    f"Currently not supporting guided decoding on "
                    f"{cls.device_name}")
