# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker base classes for vLLM v1.

This module provides minimal implementations of the worker interfaces that were
previously part of the legacy ``vllm.worker`` package.  The old package has been
removed, so these classes are re-defined here for the components that still rely
on them.
"""

import os
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, run_method,
                        update_environment_variables)
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)


class WorkerBase:
    """Base class for device specific workers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        # vLLM configuration
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config
        from vllm.platforms import current_platform
        self.current_platform = current_platform

        # Distributed information
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device/model state
        self.device: Optional[torch.device] = None
        self.model_runner: Optional[nn.Module] = None

    # --- Methods expected to be overridden by subclasses -----------------
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        raise NotImplementedError

    def init_device(self) -> None:
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def load_model(self) -> None:
        raise NotImplementedError

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[list[SamplerOutput]]:
        raise NotImplementedError

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    # --- Optional hooks ---------------------------------------------------
    def start_profile(self) -> None:
        return

    def stop_profile(self) -> None:
        return

    def sleep(self, level: int = 1) -> None:
        return

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        return

    def start_worker_execution_loop(self) -> None:
        with self.current_platform.inference_mode():
            while True:
                output = self.execute_model()
                if output is None:
                    return

    def check_health(self) -> None:
        return

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class DelegateWorkerBase(WorkerBase):
    """A WorkerBase that delegates all methods to another Worker."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config: VllmConfig = kwargs.get("vllm_config")
        cls = resolve_obj_by_qualname(vllm_config.parallel_config.worker_cls)
        self.worker = cls(*args, **kwargs)

    def init_device(self) -> None:
        self.worker.init_device()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def load_model(self) -> None:
        self.worker.load_model()

    def get_model(self) -> nn.Module:
        return self.worker.get_model()

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[list[SamplerOutput]]:
        return self.worker.execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        return self.worker.get_cache_block_size_bytes()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.worker.list_loras()

    def __getattr__(self, attr):
        return getattr(self.worker, attr)


class LoRANotSupportedWorkerBase(WorkerBase):
    """Partial implementation of WorkerBase that raises for LoRA methods."""

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def remove_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def pin_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def list_loras(self) -> Set[int]:
        raise ValueError(f"{type(self)} does not support LoRA")


class WorkerWrapperBase:
    """Lightweight wrapper that lazily constructs a worker instance."""

    def __init__(self, vllm_config: VllmConfig, rpc_rank: int = 0) -> None:
        self.rpc_rank = rpc_rank
        self.worker: Optional[WorkerBase] = None
        self.vllm_config: Optional[VllmConfig] = None
        if vllm_config.model_config is not None and \
                vllm_config.model_config.trust_remote_code:
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

    def adjust_rank(self, rank_mapping: Dict[int, int]) -> None:
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(self, envs_list: list[dict[str, str]]) -> None:
        envs = envs_list[self.rpc_rank]
        key = "CUDA_VISIBLE_DEVICES"
        if key in envs and key in os.environ:
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        kwargs = all_kwargs[self.rpc_rank]
        self.vllm_config = kwargs.get("vllm_config")
        assert self.vllm_config is not None, (
            "vllm_config is required to initialize the worker")
        enable_trace_function_call_for_thread(self.vllm_config)

        worker_cls = self.vllm_config.parallel_config.worker_cls
        if isinstance(worker_cls, str):
            worker_cls = resolve_obj_by_qualname(worker_cls)
        with set_current_vllm_config(self.vllm_config):
            self.worker = worker_cls(**kwargs)  # type: ignore

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.rpc_rank]
        assert self.worker is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self) -> None:
        assert self.worker is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.init_device()  # type: ignore

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        try:
            return run_method(self, method, args, kwargs)
        except Exception as e:  # pragma: no cover - best effort logging
            msg = (f"Error executing method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)


def extract_previous_hidden_states(
        data: Union[ExecuteModelRequest, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Extract previous hidden states from request data if present."""
    output: Dict[str, torch.Tensor] = {}
    if isinstance(data, dict):
        if "previous_hidden_states" in data:
            output["previous_hidden_states"] = data["previous_hidden_states"]
    elif getattr(data, "previous_hidden_states", None) is not None:
        output["previous_hidden_states"] = data.previous_hidden_states.hidden_states
    return output
