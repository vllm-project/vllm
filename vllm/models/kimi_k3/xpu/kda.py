# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 KDA implementation."""

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as BaseKimiGatedDeltaNetAttention,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


class XPUKimiGatedDeltaNetAttention(BaseKimiGatedDeltaNetAttention):
    """Kimi KDA using the fused XPU convolution and recurrence op."""

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        attn_metadata_raw = get_forward_context().attn_metadata
        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        num_actual_tokens = attn_metadata.num_actual_tokens
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_state_indices = attn_metadata.non_spec_state_indices_tensor
        if non_spec_state_indices is not None:
            non_spec_state_indices = non_spec_state_indices.contiguous()
        if attn_metadata.spec_sequence_masks is not None:
            if non_spec_token_indx is not None:
                non_spec_token_indx = non_spec_token_indx.to(torch.int32)
            if spec_token_indx is not None:
                spec_token_indx = spec_token_indx.to(torch.int32)

        mixed_qkv = mixed_qkv[:num_actual_tokens].view(
            num_actual_tokens,
            3,
            self.local_num_heads,
            self.head_dim,
        )
        q_proj, k_proj, v_proj = mixed_qkv.unbind(dim=1)
        q_proj = q_proj.flatten(1)
        k_proj = k_proj.flatten(1)
        v_proj = v_proj.flatten(1)

        conv_state, recurrent_state = self.kv_cache
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )

        torch.ops._xpu_C.kda_attention(
            core_attn_out,
            q_proj,
            k_proj,
            v_proj,
            g1[:, :num_actual_tokens],
            beta[:, :num_actual_tokens].to(dtype=torch.float32).contiguous(),
            conv_state,
            recurrent_state,
            q_conv_weight,
            k_conv_weight,
            v_conv_weight,
            self.A_log,
            self.dt_bias,
            attn_metadata.num_prefills,
            attn_metadata.num_decodes,
            attn_metadata.num_spec_decodes,
            attn_metadata.has_initial_state,
            attn_metadata.non_spec_query_start_loc,
            non_spec_token_indx,
            non_spec_state_indices,
            attn_metadata.spec_query_start_loc,
            spec_token_indx,
            attn_metadata.spec_state_indices_tensor,
            attn_metadata.num_accepted_tokens,
            num_actual_tokens,
            self.gate_lower_bound,
        )
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))


class KimiK3DeltaAttention(XPUKimiGatedDeltaNetAttention):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        if not self.use_full_rank_gate:
            raise ValueError("XPU Kimi-K3 KDA requires a full-rank gate")
        if self.gate_lower_bound is None:
            raise ValueError("XPU Kimi-K3 KDA requires a bounded sigmoid gate")


class KimiLinearDeltaAttention(XPUKimiGatedDeltaNetAttention):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        if self.use_full_rank_gate:
            raise ValueError("XPU Kimi-Linear KDA requires a low-rank gate")
        if self.gate_lower_bound is not None:
            raise ValueError("XPU Kimi-Linear KDA requires an unbounded softplus gate")
