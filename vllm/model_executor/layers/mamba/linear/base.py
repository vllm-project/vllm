# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PretrainedConfig

from vllm.config import (
    VllmConfig,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class LinearAttention(PluggableLayer, MambaBase):
    """Base class for Linear attention layer."""

    def __init__(
        self, config: PretrainedConfig, vllm_config: VllmConfig, prefix: str = ""
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

        self.BLOCK = getattr(config, "block", 256)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // self.num_heads
        )
        self.hidden_inner_size = self.head_dim * self.num_heads

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        assert self.num_heads % self.tp_size == 0

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, int, int], ...]:
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=self.num_heads, tp_size=self.tp_size, head_dim=self.head_dim
        )
