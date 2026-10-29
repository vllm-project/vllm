# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time int4 quantization of an unquantized lm_head on RDNA (gfx11/gfx12).

Decode reads the whole lm_head every step, so on bandwidth-bound RDNA GPUs an
fp16/bf16 head is one of the largest per-token costs even when the checkpoint
quantizes every other linear layer. ``Int4LMHeadMethod`` quantizes the loaded
weight to asymmetric uint4 with one scale and zero point per 32 input channels
and runs it on the RDNA hybrid W4A16 op (HIP skinny GEMM for decode, Triton
for prefill).
"""

import torch
from torch.nn import Module, Parameter

from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    pack_int4_exllama_shuffle,
    skinny_int4_layout,
)
from vllm.model_executor.layers.quantization.online.rdna_lm_head import (
    RdnaLMHeadMethod,
)
from vllm.utils.platform_utils import num_compute_units

GROUP_SIZE = 32
_UINT4_MAX = 15


def quantize_int4_rows(
    weight: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uint4 quantization of the rows of ``weight`` [N, K].

    Returns the codes [N, K] uint8, the scales [N, K // group_size] in the
    weight dtype and the zero points [N, K // group_size] uint8, such that
    ``(codes - zero_point) * scale`` approximates ``weight``. Each group's range
    is widened to include 0, so the zero point always fits in uint4.
    """
    n, k = weight.shape
    groups = weight.float().view(n, k // group_size, group_size)
    w_min = groups.amin(dim=-1).clamp(max=0)
    w_max = groups.amax(dim=-1).clamp(min=0)
    # Quantize against the stored scale so the kernel dequantizes exactly the
    # values the codes were rounded for.
    scale = (
        ((w_max - w_min) / _UINT4_MAX)
        .clamp(min=torch.finfo(weight.dtype).tiny)
        .to(weight.dtype)
    )
    scale_f = scale.float()
    zero_point = (-w_min / scale_f).round().clamp(0, _UINT4_MAX)
    codes = ((groups / scale_f.unsqueeze(-1)).round() + zero_point.unsqueeze(-1)).clamp(
        0, _UINT4_MAX
    )
    return (
        codes.to(torch.uint8).view(n, k),
        scale,
        zero_point.to(torch.uint8),
    )


def pack_zero_points(zero_point: torch.Tensor) -> torch.Tensor:
    """Pack uint4 zero points along rows: [N, G] -> [N // 8, G] int32.

    Row n lands in word n // 8 at bits 4 * (n % 8).
    """
    n, g = zero_point.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=zero_point.device)
    words = zero_point.to(torch.int32).view(n // 8, 8, g) << shifts.view(1, 8, 1)
    # The nibbles are disjoint, so the (wrapping) sum is their bitwise OR.
    return words.sum(dim=1, dtype=torch.int32)


@torch.no_grad()
def quantize_lm_head_int4(
    weight: torch.Tensor, group_size: int = GROUP_SIZE, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` [N, K] into the RDNA hybrid W4A16 layout.

    Returns the packed weight (int8 [N, K // 2], ExLlama shuffled), the scales
    [N, K // group_size] and the packed zero points [N // 8, K // group_size].
    Rows are processed in chunks to bound the fp32 working set.
    """
    n, k = weight.shape
    shuffled = torch.empty(n, k // 8, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    zero_points = torch.empty(
        n, k // group_size, dtype=torch.uint8, device=weight.device
    )
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        codes, scales[rows], zero_points[rows] = quantize_int4_rows(
            weight[rows], group_size
        )
        shuffled[rows] = pack_int4_exllama_shuffle(codes)
    return skinny_int4_layout(shuffled), scales, pack_zero_points(zero_points)


class Int4LMHeadMethod(RdnaLMHeadMethod):
    """int4 per-group-32 lm_head on the RDNA hybrid W4A16 op."""

    scheme = "int4_per_group_32"
    # Zero points pack 8 rows per int32 word.
    n_align = 8
    k_align = GROUP_SIZE

    def quantize_weight(self, layer: Module, weight: torch.Tensor) -> None:
        w_q, w_s, w_zp = quantize_lm_head_int4(weight)
        self.set_weight(layer, w_q)
        layer.weight_scale = Parameter(w_s, requires_grad=False)
        layer.weight_zero_point = Parameter(w_zp, requires_grad=False)

    def matmul(
        self, layer: Module, x_2d: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return torch.ops.vllm.rdna_hybrid_w4a16_apply(
            x_2d,
            layer.weight,
            layer.weight_scale,
            layer.weight_zero_point,
            bias,
            num_compute_units(),
            GROUP_SIZE,
        )
