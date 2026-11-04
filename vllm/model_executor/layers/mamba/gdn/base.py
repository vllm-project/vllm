# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PreTrainedConfig

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
    gdn_replayssm_geometry,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """Base class for GatedDeltaNet attention layer."""

    def __init__(
        self,
        config: PreTrainedConfig,
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
        self.use_replayssm = vllm_config.is_gdn_replayssm_enabled()
        self.use_flashinfer_replayssm = self.use_replayssm
        self.state_alignment = 16 if self.use_flashinfer_replayssm else 1
        self.replayssm_buffer_len = (
            self.cache_config.replayssm_buffer_len if self.use_replayssm else None
        )
        self.replayssm_executed_query_width: int | None = None
        self.replayssm_ring_slots: int | None = None
        if self.use_replayssm:
            self.replayssm_executed_query_width, self.replayssm_ring_slots = (
                gdn_replayssm_geometry(self.num_spec)
            )
        if self.replayssm_executed_query_width is not None:
            # STP executes one token but still uses four staging rows.
            packed_width = max(self.replayssm_executed_query_width, 4)
            self.register_buffer(
                "_replayssm_offsets",
                torch.arange(packed_width, dtype=torch.int32),
                persistent=False,
            )
        self._replayssm_ring_start = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes: tuple[torch.dtype, ...] = (
            MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )
        )

        if self.use_replayssm:
            dtypes = (
                *dtypes,
                *MambaStateDtypeCalculator.gated_delta_net_replayssm_ring_dtypes(
                    self.model_config.dtype
                ),
            )
        return dtypes
