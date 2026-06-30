# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused shared-expert-gate sigmoid-mul.

Target chain (``Qwen2MoeMLP.forward``, ``qwen2_moe.py:119-120``), executed inside
the opaque custom op ``vllm::moe_forward_shared`` (so Inductor never sees it):

    if self.expert_gate is not None:
        out = F.sigmoid(self.expert_gate(x)[0]) * out

where ``expert_gate`` is ``ReplicatedLinear(hidden, 1, bias=False)`` (unquantized,
plain ``[1, hidden]`` bf16 ``.weight``). The production chain is three kernels:

    cuBLASLt ``gemv2N_kernel`` (N=1 GEMV)  ->  ATen ``vectorized_elementwise`` sigmoid
    ->  ATen ``elementwise_kernel`` broadcast-mul (UNVECTORIZED iterator)

This module replaces that chain with authored Triton kernels, gated behind
``VLLM_FUSE_EXPERT_GATE`` (default off):

  * decode / small-M (M <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD): ONE kernel
    ``fused_expert_gate_mul_kernel`` -- per-row ``gate = sigmoid(dot(x_row, w))``
    in fp32, then in-place ``out_row *= gate``. Replaces 3 kernels -> 1. At N=1 the
    "GEMV" is a per-row dot; there is no tensor-core tile to preserve, and the
    cuBLASLt kernel being replaced is a single latency-bound CTA.

  * prefill / large-M: keep the library GEMV for ``expert_gate(x)`` (native vendor
    tile preserved), fuse only ``sigmoid + broadcast-mul`` into ONE vectorized
    kernel ``fused_sigmoid_bcast_mul_kernel``. Replaces 2 kernels -> 1 and fixes
    the unvectorized ATen broadcast iterator.

Precision: LOSSLESS. Inputs bf16, output bf16, fp32 accumulation on both sides --
identical to the baseline (cuBLASLt accumulates fp32; ATen sigmoid/MulFunctor
compute in fp32). The fused output is allclose to the baseline, not bit-exact
(it differs only within the bf16-rounding band).

Derived from a standalone CUDA-graph-captured microbenchmark of the fused
sigmoid-mul chain.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Decode kernel reduction block over K. K (=hidden_size, 3072 for Qwen3.5-122B) is
# a multiple of 1024 so masking is unnecessary on the common path; we mask anyway
# to stay correct for any K. num_warps=4 was the best-measured config for this shape.
_DECODE_BLOCK = 1024
_PREFILL_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def fused_expert_gate_mul_kernel(
    out_ptr,
    x_ptr,
    w_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: ``gate = sigmoid(dot(x_row, w)); out_row *= gate``.

    fp32 accumulation; bf16 load/store. ``w`` is the ``[1, K]`` gate weight row.
    Mutates ``out`` in place (safe: ``out`` is a local intermediate of the opaque
    ``moe_forward_shared`` op body, not an op input -- verified dispatch trace).
    """
    row = tl.program_id(0)
    if row >= M:
        return
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        xv = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += xv * wv
    gate = tl.sigmoid(tl.sum(acc, axis=0))
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        ov = tl.load(out_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * K + offs, (ov * gate).to(tl.bfloat16), mask=mask)


@triton.jit
def fused_sigmoid_bcast_mul_kernel(
    out_ptr,
    gate_ptr,
    K: tl.constexpr,
    CBLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """grid = (M * K/BLOCK). ``out[row, cb] *= sigmoid(gate[row])``. Vectorized.

    ``gate`` is the ``[M, 1]`` (or ``[M]``) pre-computed GEMV logits. Mutates
    ``out`` in place.
    """
    pid = tl.program_id(0)
    row = pid // CBLOCKS
    cb = pid % CBLOCKS
    # K % BLOCK == 0 (asserted by caller) and grid is exactly M*CBLOCKS, so
    # offsets never cross a row boundary -- no mask needed.
    g = tl.sigmoid(tl.load(gate_ptr + row).to(tl.float32))
    offs = row * K + cb * BLOCK + tl.arange(0, BLOCK)
    ov = tl.load(out_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, (ov * g).to(tl.bfloat16))


def _decode_supported(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Cheap shape/dtype/contiguity guard for the decode fused path."""
    if not (out.is_cuda and x.is_cuda and weight.is_cuda):
        return False
    if out.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        return False
    if out.dim() != 2 or x.dim() != 2:
        return False
    # weight is [1, K] (ReplicatedLinear(hidden, 1)). Squeeze to [K].
    if weight.numel() != x.shape[1]:
        return False
    K = x.shape[1]
    if out.shape[0] != x.shape[0] or out.shape[1] != K:
        return False
    if not (out.is_contiguous() and x.is_contiguous()):
        return False
    return True


def fused_expert_gate_decode(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Decode/small-M path: one kernel does dot + sigmoid + in-place scale.

    ``out``  : [M, K] bf16 (down_proj output) -- mutated in place and returned.
    ``x``    : [M, K] bf16 (MLP input).
    ``weight``: [1, K] or [K] bf16 (expert_gate.weight).
    """
    M, K = x.shape
    w = weight.reshape(-1)
    fused_expert_gate_mul_kernel[(M,)](
        out, x, w, M=M, K=K, BLOCK=_DECODE_BLOCK, num_warps=_NUM_WARPS
    )
    return out


def fused_sigmoid_bcast_mul(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Prefill/large-M path: vectorized sigmoid(gate) broadcast-mul into ``out``.

    ``out`` : [M, K] bf16 -- mutated in place and returned.
    ``gate``: [M, 1] or [M] -- pre-computed library GEMV logits.
    """
    M, K = out.shape
    assert K % _PREFILL_BLOCK == 0, (
        f"prefill fused path requires K % {_PREFILL_BLOCK} == 0, got K={K}"
    )
    cblocks = K // _PREFILL_BLOCK
    g = gate.reshape(-1)
    fused_sigmoid_bcast_mul_kernel[(M * cblocks,)](
        out, g, K=K, CBLOCKS=cblocks, BLOCK=_PREFILL_BLOCK, num_warps=_NUM_WARPS
    )
    return out
