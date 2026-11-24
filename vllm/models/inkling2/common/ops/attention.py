# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention kernels shared by every platform: the output gate and
attention sink, the log-scaling tau, and the split-KV decode/verify launch
(`_rel_attention_kernel`, or on Blackwell the tcgen05 kernel of
`nvidia/ops/attention.py`) with its partials combine."""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.platform_utils import num_compute_units

from .attention_sm90 import MODE_GATED, MODE_PARTIAL, rel_attention
from .intrinsics import ex2, lg2, sigmoid
from .pdl import gdc_wait
from .platform import pdl_launch_kwargs, use_sm100_kernels


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    stride_x,
    stride_gate,
    stride_lse_t,
    stride_lse_h,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * stride_x + col, mask, other=0)
    gate = tl.load(Gate + row * stride_gate + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * stride_lse_t + head * stride_lse_h, mask, other=0)
    sink = tl.load(Sink + row * stride_sink + head, mask, other=0).to(tl.float32)
    result = x.to(tl.float32) * sigmoid(gate)
    result = result * sigmoid(lse.to(tl.float32) - sink)
    tl.store(Y + row * H * D + col, result, mask)


def gate_sink(
    output: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """Write bf16 `output * sigmoid(gate) * sigmoid(lse - sink)` per head to
    `result`.

    `gate` and `result` are `(T, H * D)`, `sinks` and `lse` are
    `(T, H)` and `output` is `(T, H, D)`.
    """
    num_tokens, num_heads = sinks.shape
    assert lse.shape == (num_tokens, num_heads)
    assert output.shape[1] == num_heads and output.is_contiguous()
    assert gate.stride(1) == sinks.stride(1) == 1
    assert result.shape == gate.shape and result.is_contiguous()
    width = gate.shape[1]
    _gate_sink_kernel[(num_tokens, triton.cdiv(width, 512))](
        output,
        gate,
        lse,
        sinks,
        result,
        output.stride(0),
        gate.stride(0),
        lse.stride(0),
        lse.stride(1),
        sinks.stride(0),
        num_heads,
        width // num_heads,
        512,
        enable_fp_fusion=False,
        **pdl_launch_kwargs(),
    )


@triton.jit
def _log_scaling_tau_kernel(
    Pos,
    Out,
    n,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tldevice.log(ratio)
    tl.store(Out + i, tau, i < n)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    n = positions.numel()
    _log_scaling_tau_kernel[(triton.cdiv(n, 256),)](
        positions, output, n, n_floor, alpha, 256, enable_fp_fusion=False
    )
    return output


# FA4's key block; the ex2 emulation pattern repeats every block.
_FA4_BLOCK_N = 128
# Query rows (tokens x GQA group) of a CTA.
_VERIFY_BLOCK_M = 64


def plan_decode_attention(
    is_local: bool,
    num_reqs: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
) -> int:
    """Return the split count of `rel_decode_attention` (or
    `block_decode_attention`), or 0 to use FA4.

    Each CTA holds all queries of one request for one KV head. Local layers
    (`rel_extent`: the window) give each split one key block of the
    window, or a few blocks for larger batches (a single split writes the
    gated output without the combine kernel); decode keeps one block per
    split up to twice the work of verify. Full layers give each request and
    KV head `num_sms / 8` splits, fewer once that exceeds 6 CTAs per SM,
    but never less than one CTA per SM. The count depends only on the batch,
    not the KV lengths, so it holds for every replay of a CUDA graph.
    """
    if max_query_len > _VERIFY_BLOCK_M // (num_heads // num_kv_heads):
        return 0
    work = num_reqs * num_kv_heads
    num_sms = num_compute_units()
    if is_local:
        if work <= (16 if max_query_len <= 2 else 8):
            return triton.cdiv(rel_extent - 1, _FA4_BLOCK_N) + 1
        return 5 if work <= 16 else 3 if work <= 32 else 2 if work <= 64 else 1
    return max(num_sms // work, min(num_sms // 8, 6 * num_sms // work))


@triton.jit
def _rel_decode_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    num_tokens,
    num_actual_tokens,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GATED: tl.constexpr,
):
    """Merge the split partials, then apply the output gate and attention sink
    if `GATED`."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    if GATED:
        # The split kernel launches this one after its own PDL wait, so the
        # gate and sink are ready.
        gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
        sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)
    gdc_wait()
    y_ptrs = Y + row * H * D + head * D + offs_d
    if row >= num_actual_tokens:
        tl.store(y_ptrs, tl.zeros([BLOCK_D], tl.float32))
        return
    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = ex2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    # Requests without keys (padding) come out zero.
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    if GATED:
        lse = (lg2(w_sum) + lse_max) * 0.6931471805599453
        o = o * sigmoid(gate)
        o = o * sigmoid(lse - sink)
    tl.store(y_ptrs, o)


def rel_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    is_local: bool,
    num_splits: int,
) -> None:
    """Write the gated bf16 `(T, H * D)` attention output for small batches.

    Split-KV relative attention on `_rel_verify_kernel` (see
    `plan_decode_attention`) reading the sheared relative logits. With more
    than one split, a second launch merges the fp32 partials and applies the
    output gate and attention sink. Tokens from `num_actual_tokens` on are
    zeroed.
    """
    num_tokens, num_heads = q.shape[:2]
    extent_pad = rel_logits.shape[2]
    extent = extent_pad - 2 * _FA4_BLOCK_N
    assert rel_logits.shape[0] >= num_tokens and rel_logits.shape[1] == num_heads
    assert rel_logits.stride(1) == extent_pad and rel_logits.stride(2) == 1
    assert extent % _FA4_BLOCK_N == 0
    assert gate.stride(1) == sinks.stride(1) == 1
    _decode_attention(
        q,
        key_cache,
        value_cache,
        out,
        (rel_logits, gate, sinks),
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        max_query_len,
        num_actual_tokens,
        softmax_scale,
        extent if is_local else 0,
        num_splits,
    )


def block_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block: int,
    window: int,
    num_splits: int,
) -> None:
    """Write the bf16 `(T, H * D)` attention of each `block`-token request
    to `out`: every query sees the request's keys up to its last one, or the
    last `window` of them if `window` (the DSpark draft block attention).

    Runs `_rel_verify_kernel` like `rel_decode_attention`, without the
    relative logits, gate and sink.
    """
    _decode_attention(
        q,
        key_cache,
        value_cache,
        out,
        None,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        block,
        q.shape[0],
        q.shape[2] ** -0.5,
        window,
        num_splits,
    )


def _decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    rel_gate_sinks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
) -> None:
    """Launch `_rel_verify_kernel` (and the combine for several splits);
    `rel_gate_sinks` None is the draft mode, `window` 0 a full layer."""
    num_tokens, num_heads, head_dim = q.shape
    num_blocks, page, num_kv_heads = key_cache.shape[:3]
    group = num_heads // num_kv_heads
    draft = rel_gate_sinks is None
    if rel_gate_sinks is None:
        # Unused placeholders keep the launches typed alike.
        rel_logits = gate = sinks = out
        extent = 0
    else:
        rel_logits, gate, sinks = rel_gate_sinks
        extent = rel_logits.shape[2] - 2 * _FA4_BLOCK_N
    assert max_query_len <= _VERIFY_BLOCK_M // group
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(1) == head_dim and q.stride(2) == 1
    assert out.is_contiguous()
    # A single split writes the (gated) output itself.
    fuse_gate = num_splits == 1
    o_part = q.new_empty(
        (0 if fuse_gate else num_splits, *q.shape), dtype=torch.float32
    )
    lse_part = q.new_empty(
        (0 if fuse_gate else num_splits, num_tokens, num_heads), dtype=torch.float32
    )
    if use_sm100_kernels():
        # Gluon tcgen05 kernel: imported on Blackwell only.
        from vllm.models.inkling2.nvidia.ops.attention import _launch_rel_verify

        _launch_rel_verify(
            q,
            key_cache,
            value_cache,
            out,
            o_part,
            lse_part,
            rel_logits,
            gate,
            sinks,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            softmax_scale,
            extent,
            window,
            num_splits,
            draft,
        )
    else:
        rel_attention(
            q,
            key_cache,
            value_cache,
            out if fuse_gate else o_part,
            None if fuse_gate else lse_part,
            None if draft else rel_logits,
            gate,
            sinks,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max_query_len,
            softmax_scale=softmax_scale,
            window=window,
            num_splits=num_splits,
            mode=MODE_GATED if fuse_gate else MODE_PARTIAL,
        )
    if fuse_gate:
        out[num_actual_tokens:].zero_()
        return
    # Small batches spread the merge over more programs; larger ones merge a
    # whole head per (narrow) program.
    if num_tokens * num_heads >= 512:
        block_d, num_warps = head_dim, 2 if num_splits > 32 else 1
    else:
        block_d, num_warps = 32, 4
    _rel_decode_combine_kernel[(num_tokens, num_heads, head_dim // block_d)](
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        num_tokens,
        num_actual_tokens,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
        GATED=not draft,
        num_warps=num_warps,
        **pdl_launch_kwargs(),
    )
