# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time int8 quantization of an unquantized lm_head on RDNA (gfx11/gfx12).

Decode reads the whole lm_head every step, so on bandwidth-bound RDNA GPUs an
fp16/bf16 head is one of the largest per-token costs even when the checkpoint
quantizes every other linear layer. ``Int8LMHeadMethod`` quantizes the loaded
weight to symmetric int8 with one scale per output channel and runs it on the
RDNA W8A16 op (HIP skinny GEMM for decode, Triton for prefill). It cuts the
head's traffic 2x (int4: ~3.4x) but flips ~7x fewer top-1 tokens.
"""

import torch
from torch.nn import Module, Parameter

# Importing rdna_w8a16 registers torch.ops.vllm.rdna_w8a16_apply.
from vllm.model_executor.kernels.linear.mixed_precision.rdna_w8a16 import (
    empty_int8_weight,
)
from vllm.model_executor.layers.quantization.online.rdna_lm_head import (
    RdnaLMHeadMethod,
)
from vllm.utils.platform_utils import num_compute_units

# -128 is left unused so the code range is symmetric around 0.
_INT8_MAX = 127
# wvSplitK_int8 loads K in 16-byte chunks and, at large vocab, tiles rows by 4.
_K_ALIGN = 16
_N_ALIGN = 4


@torch.no_grad()
def quantize_lm_head_int8(
    weight: torch.Tensor, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization of ``weight`` [N, K].

    Returns the codes in the RDNA W8A16 layout (int8 [N, K], see
    ``empty_int8_weight``) and the scales [N] in the weight dtype, such that
    ``codes * scale[:, None]`` approximates ``weight``. Rows are processed in
    chunks to bound the fp32 working set.
    """
    n, k = weight.shape
    codes = empty_int8_weight(n, k, weight.device)
    scales = torch.empty(n, dtype=weight.dtype, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        w = weight[rows].float()
        # Quantize against the stored scale so the kernel dequantizes exactly
        # the values the codes were rounded for.
        scale = (
            (w.abs().amax(dim=-1) / _INT8_MAX)
            .clamp(min=torch.finfo(weight.dtype).tiny)
            .to(weight.dtype)
        )
        scales[rows] = scale
        codes[rows] = (
            (w / scale.float().unsqueeze(-1)).round().clamp(-_INT8_MAX, _INT8_MAX)
        )
    return codes, scales


class Int8LMHeadMethod(RdnaLMHeadMethod):
    """int8 per-channel lm_head on the RDNA W8A16 op."""

    scheme = "int8_per_channel"
    n_align = _N_ALIGN
    k_align = _K_ALIGN

    def quantize_weight(self, layer: Module, weight: torch.Tensor) -> None:
        w_q, w_s = quantize_lm_head_int8(weight)
        self.set_weight(layer, w_q)
        layer.weight_scale = Parameter(w_s, requires_grad=False)

    def matmul(
        self, layer: Module, x_2d: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return torch.ops.vllm.rdna_w8a16_apply(
            x_2d, layer.weight, layer.weight_scale, bias, num_compute_units()
        )
