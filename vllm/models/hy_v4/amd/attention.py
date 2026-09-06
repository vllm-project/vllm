# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention as BaseAttention

from .ops import hy4_rocm_bf16_sigmoid_mul
from .rocm import HYV4ROCMAiterMLASparseBackend


class HYV4MLAAttention(BaseAttention):
    """HY V4 MLA attention using the sink-capable ROCm AITER backend."""

    def _can_use_fused_mla_gate(
        self, attn_out: torch.Tensor, gate_score: torch.Tensor
    ) -> bool:
        return (
            self.config.gating_type == "elementwise"
            and attn_out.device.type == "cuda"
            and gate_score.device == attn_out.device
            and attn_out.dtype == torch.bfloat16
            and gate_score.dtype == torch.bfloat16
            and attn_out.ndim == 2
            and attn_out.shape == gate_score.shape
            and attn_out.shape[-1] == 2048
            and attn_out.is_contiguous()
            and gate_score.is_contiguous()
        )

    def _apply_mla_gate(
        self, attn_out: torch.Tensor, gate_score: torch.Tensor
    ) -> torch.Tensor:
        if self._can_use_fused_mla_gate(attn_out, gate_score):
            return hy4_rocm_bf16_sigmoid_mul(attn_out, gate_score)
        return super()._apply_mla_gate(attn_out, gate_score)

    def _resolve_sink_backend(self, kv_cache_dtype: str):
        return HYV4ROCMAiterMLASparseBackend
