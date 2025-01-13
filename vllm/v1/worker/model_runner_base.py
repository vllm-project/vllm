from typing import TYPE_CHECKING, List, Optional

import enum

import torch
import torch.distributed
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv,
                        is_pin_memory_available)
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class ExecutionMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PREFIX_PREFILL = enum.auto()

    def is_prefill(self) -> bool:
        return self in (ExecutionMode.PREFILL, ExecutionMode.PREFIX_PREFILL)


class ModelRunnerBase:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        raise NotImplementedError()

    def load_model(self) -> None:
        raise NotImplementedError()

    def dummy_run(
        self,
        kv_caches,
        batch_size: int,
        seq_len: Optional[int] = None,
        exec_mode: Optional[ExecutionMode] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def profile_run(self) -> None:
        raise NotImplementedError()

    def capture_model(self) -> None:
        raise NotImplementedError()

    def initialize_kv_cache(self, num_blocks: int) -> None:
        raise NotImplementedError()
