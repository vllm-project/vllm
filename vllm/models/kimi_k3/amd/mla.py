# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MLA wrapper: fuse the sigmoid output gate with MXFP4 o_proj quant.

Registered via ``PluggableLayer.register_oot`` so constructing
``MultiHeadLatentAttentionWrapper`` instantiates this class once this
module has been imported (see ``amd/linear.py``). Shared ``layers/mla.py``
stays free of vendor checks; this only overrides ``_gated_o_proj``.

``g_proj`` (CK ``hgemm_bf16``) always runs — there is no MXFP4 epilogue
hook on that GEMM. Fusion replaces ATen sigmoid, ATen mul, and the
standalone MXFP4 quant with one Triton producer. ``mla_a8w8`` /
``kn_mla_reduce`` are not touched (wrong tensor).
"""

import torch

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.amd.ops.mla_quant import maybe_fused_mla_oproj_quant
from vllm.platforms import current_platform


class ROCmMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    def _gated_o_proj(
        self,
        attn_out: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.g_proj is None:
            return self.o_proj(attn_out)[0]
        gate = self.g_proj(hidden_states)[0]
        fused = maybe_fused_mla_oproj_quant(attn_out, gate, self.o_proj)
        if fused is not None:
            return self.o_proj(fused)[0]
        return self.o_proj(attn_out * gate.sigmoid())[0]


if current_platform.is_rocm():
    MultiHeadLatentAttentionWrapper.register_oot(
        ROCmMultiHeadLatentAttentionWrapper
    )
