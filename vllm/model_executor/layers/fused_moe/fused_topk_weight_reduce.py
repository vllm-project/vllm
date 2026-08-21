# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for MoE topk-weighted output reduction.

Replaces the ``at::sum_out`` ATen fallback that ``ops.moe_sum`` dispatches to
when ``topk > 4``.  The generic ATen reduce_kernel fires one CTA-level kernel
launch per MoE layer; at 40 layers × 8 topk × BS=8 this is 360 launches/step
at ~2 µs each.  The Triton kernel accumulates in FP32 registers and writes
the reduced output in the original dtype (BF16/FP16) in a single pass.

Fires unconditionally whenever ``topk > 4`` — the range where ``ops.moe_sum``
falls back to the slow path.  Models with ``topk <= 4`` continue to use the
optimised C++ path unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_topk_reduce_kernel(
    input_ptr,   # [M, topk, K] expert outputs — bf16/fp16
    output_ptr,  # [M, K]       reduced output  — bf16/fp16
    M,
    topk: tl.constexpr,
    K,
    stride_im: tl.constexpr,  # input stride along M
    stride_it: tl.constexpr,  # input stride along topk
    stride_ik: tl.constexpr,  # input stride along K (usually 1)
    stride_om: tl.constexpr,  # output stride along M
    stride_ok: tl.constexpr,  # output stride along K (usually 1)
    BLOCK_K: tl.constexpr,
):
    """Reduce [M, topk, K] → [M, K] by summing the topk dimension in FP32."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    base = pid_m.to(tl.int64) * stride_im
    for t in tl.static_range(topk):
        ptr = base + t * stride_it + k_off * stride_ik
        acc += tl.load(input_ptr + ptr, mask=k_mask, other=0.0).to(tl.float32)

    out_ptr = pid_m * stride_om + k_off * stride_ok
    tl.store(output_ptr + out_ptr, acc.to(output_ptr.type.element_ty),
             mask=k_mask)


def fused_topk_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Sum ``input`` over its topk dimension and write into ``output``.

    Parameters
    ----------
    input:
        Shape ``[M, topk, K]``, dtype bf16 or fp16.  The routing weights
        must already be multiplied into the expert outputs before calling
        this function (``apply_router_weight_on_input=True`` or the caller
        has applied ``mul_`` upstream).
    output:
        Shape ``[M, K]``, same dtype as ``input``.  Written in-place.

    Returns
    -------
    output
        The same tensor passed in, for call-site ergonomics.
    """
    assert input.ndim == 3, f"Expected 3-D input [M, topk, K], got {input.shape}"
    M, topk, K = input.shape
    assert output.shape == (M, K), (
        f"output must be ({M}, {K}), got {output.shape}"
    )

    BLOCK_K = 256
    grid = (M, triton.cdiv(K, BLOCK_K))

    _fused_topk_reduce_kernel[grid](
        input, output,
        M, topk, K,
        input.stride(0), input.stride(1), input.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return output
