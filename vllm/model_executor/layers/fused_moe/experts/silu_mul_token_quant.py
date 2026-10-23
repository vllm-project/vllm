# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiLU-and-mul fused with dynamic per-token FP8 quantization.

Replaces `silu_and_mul` + `scaled_fp8_quant(use_per_token_if_dynamic=True)` in the
Triton MoE experts when the second GEMM takes per-token-quantized activations (for
example PTPC-FP8). Under expert parallelism the first GEMM only writes the rows whose
expert is local. This kernel reads the same `expert_map[topk_ids]` mask and skips the
other rows instead of activating and quantizing them. The second GEMM and the
finalize never read those rows.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _silu_mul_token_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    H,
    stride_x,
    stride_q,
    fp8_max,
    min_scale,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if HAS_MASK:
        expert = tl.load(topk_ids_ptr + row)
        if tl.load(expert_map_ptr + expert) < 0:
            return
    offs = tl.arange(0, BLOCK)
    m = offs < H
    x_row = x_ptr + row * stride_x
    gate = tl.load(x_row + offs, mask=m, other=0.0).to(tl.float32)
    up = tl.load(x_row + H + offs, mask=m, other=0.0).to(tl.float32)
    # Round where `silu_and_mul` does, after the SiLU and after the product, so the
    # scales and FP8 values match the unfused path bit for bit.
    dt = x_ptr.dtype.element_ty
    silu = (gate * tl.sigmoid(gate)).to(dt).to(tl.float32)
    act = (silu * up).to(dt).to(tl.float32)
    amax = tl.max(tl.where(m, tl.abs(act), 0.0), axis=0)
    scale = tl.maximum(amax / fp8_max, min_scale)
    q = tl.clamp(act / scale, -fp8_max, fp8_max)
    tl.store(q_ptr + row * stride_q + offs, q.to(q_ptr.dtype.element_ty), mask=m)
    tl.store(scale_ptr + row, scale)


def silu_mul_token_quant(
    x: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SiLU-and-mul of `x` (`[rows, 2H]`, gate then up), quantized to FP8 per row.

    Returns the FP8 activation `[rows, H]` and float32 scales `[rows, 1]`. With
    `topk_ids` (`[rows]` once flattened) and `expert_map`, rows routed to a remote
    expert are skipped and their outputs are left uninitialized.
    """
    assert x.dim() == 2 and x.stride(-1) == 1 and x.size(1) % 2 == 0
    rows, H = x.size(0), x.size(1) // 2
    _, fp8_max = get_fp8_min_max()
    q = torch.empty((rows, H), dtype=current_platform.fp8_dtype(), device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    if rows == 0:
        return q, scale
    has_mask = False
    ids = emap = x
    if topk_ids is not None and expert_map is not None:
        has_mask = True
        ids, emap = topk_ids.reshape(-1), expert_map
    _silu_mul_token_quant_kernel[(rows,)](
        x,
        q,
        scale,
        ids,
        emap,
        H,
        x.stride(0),
        q.stride(0),
        fp8_max,
        1.0 / (fp8_max * 512.0),
        HAS_MASK=has_mask,
        BLOCK=triton.next_power_of_2(H),
        num_warps=4 if H <= 1024 else 8,
    )
    return q, scale
