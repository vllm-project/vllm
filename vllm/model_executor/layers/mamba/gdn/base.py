# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PretrainedConfig

from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """Base class for GatedDeltaNet attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        if self.cache_config.use_replayssm_spec:
            return MambaStateDtypeCalculator.gated_delta_net_replayssm_spec_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )
        elif self.cache_config.use_replayssm:
            return MambaStateDtypeCalculator.gated_delta_net_replayssm_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
