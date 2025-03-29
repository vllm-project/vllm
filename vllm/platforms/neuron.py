# SPDX-License-Identifier: Apache-2.0
import os
from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_name: str = "neuron"
    device_type: str = "neuron"
    ray_device_key: str = "neuron_cores"
    dispatch_key: str = "XLA"
    supported_quantization: list[str] = ["neuron_quant"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
    ) -> str:
        if not use_v1:
            logger.info("Neuron backend is only supported in V1")
        logger.info("Using NKI flash-attention backend.")
        return "vllm.v1.attention.backends.neuron_attn.NeuronAttentionBackend"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if envs.VLLM_USE_V1:
            cls.check_and_update_config_v1(vllm_config)
        else:
            cls.check_and_update_config_v0(vllm_config)

        assert (vllm_config.lora_config
                is None), "LoRA is not supported for Neuron backend."
        assert (not vllm_config.speculative_config
                ), "Speculative decoding not yet supported for Neuron backend."

    @classmethod
    def check_and_update_config_v1(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm.v1.worker.neuron_worker.NeuronWorker")

        cache_config = vllm_config.cache_config
        if cache_config:
            # neuron V1 only: the preferred block_size is 32
            vllm_config.cache_config.block_size = 32

        model_config = vllm_config.model_config
        if model_config:
            model_config.enforce_eager = False

        compilation_config = vllm_config.compilation_config
        if compilation_config:
            compilation_config.custom_ops = ["silu_and_mul"]
            compilation_config.level = CompilationLevel.DYNAMO_AS_IS
            compilation_config.cudagraph_capture_sizes = [
                vllm_config.scheduler_config.max_num_batched_tokens
                if vllm_config.scheduler_config else 128
            ]

        # Disable functionalization on xla, due to https://github.com/pytorch/xla/issues/8640
        os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
        os.environ["NEURON_CC_FLAGS"] = " -O1 --model-type=transformer  "

    @classmethod
    def check_and_update_config_v0(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm.worker.neuron_worker.NeuronWorker")

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        cache_config = vllm_config.cache_config
        if cache_config:
            # neuron V0 only: needs block_size = max_model_len
            vllm_config.cache_config.block_size = (
                vllm_config.model_config.max_model_len)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if envs.VLLM_USE_V1:
            return "vllm.distributed.device_communicators.neuron_communicator.NeuronCommunicator"  # noqa
        else:
            return Platform.get_device_communicator_cls()

    @classmethod
    def use_all_gather(cls) -> bool:
        return True
