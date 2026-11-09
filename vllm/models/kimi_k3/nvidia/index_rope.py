# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton RoPE for the Kimi K3.1 DSA indexer.

Applies rope to the leading `rope_dim` dims of each indexer head, using the
leading `rope_dim // 2` frequencies sliced from the *main attention's*
cos_sin_cache (training reuses the leading freqs of the main freqs_cis for
the indexer). Compute is fp32 with a cast back to the input dtype.
dim >= rope_dim is left untouched (in-place).

Two pairing layouts (same frequencies, different dim pairing):
  * neox half-split (interleave=False, default): dim i pairs with dim i+half;
  * GPT-J interleave (interleave=True, GLM native weight layout):
    dim 2i pairs with dim 2i+1, cos/sin broadcast per pair.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def index_rope_kernel(
    q_ptr,
    k_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    H,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_pos,
    stride_cache,
    HALF: tl.constexpr,
    CACHE_HALF: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    INTERLEAVE: tl.constexpr,
):
    # one program = one row: token t, head h; h == H is the MQA k row
    pid = tl.program_id(0)
    t = pid // (H + 1)
    h = pid % (H + 1)

    pos = tl.load(positions_ptr + t * stride_pos)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF
    cos = tl.load(
        cos_sin_cache_ptr + pos * stride_cache + offs, mask=mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * stride_cache + CACHE_HALF + offs, mask=mask, other=0.0
    ).to(tl.float32)

    # Promote t to int64 before computing row bases: prevents t * stride_qt
    # (H*D) from silently wrapping i32 at very large num_tokens (Triton
    # runtime int args are passed as i32).
    t64 = t.to(tl.int64)
    if h < H:
        x_ptr = q_ptr + t64 * stride_qt + h * stride_qh
    else:
        x_ptr = k_ptr + t64 * stride_kt

    # Each lane owns one (x1, x2) pair: load and store of the same memory
    # location stay in the same thread, and program order within a thread
    # guarantees read-before-write, so in-place is race-free.
    if INTERLEAVE:
        # GPT-J interleaved pairing: dim 2i pairs with dim 2i+1, cos/sin
        # broadcast per pair
        p1 = 2 * offs
        p2 = 2 * offs + 1
    else:
        # neox half-split: dim i pairs with dim i+HALF
        p1 = offs
        p2 = HALF + offs
    x1 = tl.load(x_ptr + p1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + p2, mask=mask, other=0.0).to(tl.float32)
    out_dtype = x_ptr.dtype.element_ty
    tl.store(x_ptr + p1, (x1 * cos - x2 * sin).to(out_dtype), mask=mask)
    tl.store(x_ptr + p2, (x2 * cos + x1 * sin).to(out_dtype), mask=mask)


def index_rope(
    index_q: torch.Tensor,  # [T, H, D]
    index_k: torch.Tensor,  # [T, D]
    positions: torch.Tensor,  # [T] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    rope_dim: int,
    interleave: bool,
) -> None:
    """In-place RoPE on the leading rope_dim dims of each indexer head.

    fp32 compute, neox half-split / GPT-J interleaved pairing; dims >=
    rope_dim are untouched, q/k done in a single launch. Replaces the former
    handwritten torch composition (~20 small kernels in eager).
    """
    T, H, D = index_q.shape
    assert index_q.stride(2) == 1 and index_k.stride(1) == 1
    assert index_k.shape[0] == T and index_k.shape[1] == D
    assert cos_sin_cache.stride(1) == 1
    half = rope_dim // 2
    cache_half = cos_sin_cache.shape[1] // 2
    index_rope_kernel[(T * (H + 1),)](
        index_q,
        index_k,
        positions,
        cos_sin_cache,
        H,
        index_q.stride(0),
        index_q.stride(1),
        index_k.stride(0),
        positions.stride(0),
        cos_sin_cache.stride(0),
        HALF=half,
        CACHE_HALF=cache_half,
        BLOCK_HALF=triton.next_power_of_2(half),
        INTERLEAVE=interleave,
        num_warps=1 if half <= 32 else 2,
    )
