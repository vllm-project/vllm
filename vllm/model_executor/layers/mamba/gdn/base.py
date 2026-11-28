# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PreTrainedConfig

from vllm import envs
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

    # GDNAttentionBackend is shared by every GDN_ATTN implementation (Qwen,
    # Kimi, OLMo, Bailing), but the batch-invariant per-request dispatch is
    # only implemented for Qwen. Subclasses opt in by setting this to True.
    supports_batch_invariant: bool = False

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

        # Fail at init rather than part-way through model execution.
        if envs.VLLM_BATCH_INVARIANT:
            if not self.supports_batch_invariant:
                raise NotImplementedError(
                    "VLLM_BATCH_INVARIANT=1 is not supported for "
                    f"{type(self).__name__}. Batch-invariant GDN_ATTN is "
                    "currently implemented only for the Qwen GatedDeltaNet "
                    "layer."
                )
            if self.num_spec > 0:
                raise NotImplementedError(
                    "VLLM_BATCH_INVARIANT=1 is not supported together with "
                    "speculative decoding on GDN_ATTN. Disable one of "
                    "VLLM_BATCH_INVARIANT or "
                    "speculative_config.num_speculative_tokens."
                )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
