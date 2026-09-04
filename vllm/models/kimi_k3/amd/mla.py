# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MLA wrapper: fuse the sigmoid output gate with PTPC FP8 o_proj quant.

Wired from ``amd/linear.py`` so constructing the Kimi-K3 AMD MLA layer uses
this class. Shared ``layers/mla.py`` stays free of vendor checks; this only
overrides ``_gated_o_proj``.

``g_proj`` (CK ``hgemm_bf16``) always runs — there is no FP8 epilogue hook on
that GEMM. Fusion replaces ATen sigmoid, ATen mul, and the standalone AITER
``dynamic_per_token_scaled_quant`` with one Triton producer. ``mla_a8w8`` /
``kn_mla_reduce`` are not touched (wrong tensor).

When ``o_proj`` is not PTPC, fall through to the shared implementation so a
BF16 gate fusion (#50664) can still apply.
"""

import torch

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.amd.ops.sigmoid_mul_fp8_per_token import (
    maybe_fused_mla_oproj_ptpc,
    o_proj_is_ptpc_fp8,
)


class KimiK3MultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    def _gated_o_proj(
        self,
        attn_out: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.g_proj is None or not o_proj_is_ptpc_fp8(self.o_proj):
            return super()._gated_o_proj(attn_out, hidden_states)
        gate = self.g_proj(hidden_states)[0]
        fused = maybe_fused_mla_oproj_ptpc(attn_out, gate, self.o_proj)
        if fused is not None:
            return self.o_proj(fused)[0]
        return self.o_proj(attn_out * gate.sigmoid())[0]
