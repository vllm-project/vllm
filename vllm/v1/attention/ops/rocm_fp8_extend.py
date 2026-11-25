# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 x FP8 causal prefill/extend attention over the paged FP8 e4m3 KV cache on
gfx950, for layers with head size 64 and 8 query heads per KV head, including
sliding-window layers (the window is applied in-kernel). The kernel is
`flydsl_kernels.fp8_extend`.

The softmax probabilities P are formed in FP8 either with exp2 or, in exp-free
mode, as FP8 codes computed directly from the log2-domain score (a piecewise-linear
approximation of 2^x). P is encoded relative to the running row maximum, so keys
far enough below it round to zero; `bias` sets that depth per layer type.
"""

import functools
import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

GQA = 8
HEAD_SIZE = 64
# log2 of P at the running maximum, (full attention, sliding window), for exp2 P
# (False) and exp-free codes (True). exp2 P underflows bias + 10 octaves below the
# maximum, the codes bias + 6.94 octaves below it; log2(448) and 8.75 are each
# encoding's largest value.
_BIAS = {
    False: (0.0, math.log2(448.0)),
    True: (3.0, 8.75),
}


def bias(exp_free: bool, sliding_window: int) -> float:
    """log2 of P at the running maximum for a layer (see `_BIAS`)."""
    return _BIAS[exp_free][1 if sliding_window > 0 else 0]


def supports_layer(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window: int,
    exp_free: bool,
) -> bool:
    """Whether a layer has the shape the kernel handles on gfx950 (the caller
    checks the KV cache dtype and the absence of ALiBi / sinks / soft cap) and the
    kernel variant compiles and runs. Evaluated once per layer; warns once and
    returns False when FlyDSL is unavailable or the variant fails, and the layer
    then keeps the other FP8-query (full attention) or BF16 (sliding window)
    prefill path."""
    from vllm.platforms.rocm import on_gfx950

    if not (on_gfx950() and head_size == HEAD_SIZE and num_heads == GQA * num_kv_heads):
        return False
    try:
        _validate(sliding_window > 0, exp_free)
    except Exception as e:
        logger.warning_once(
            "FlyDSL FP8 extend attention (%s, %s) is unavailable, so its "
            "layers keep the other prefill path: %s",
            "exp-free softmax" if exp_free else "exp2 softmax",
            "sliding window" if sliding_window > 0 else "full attention",
            e,
        )
        return False
    return True


def _fly():
    from vllm.v1.attention.ops.flydsl_kernels import fp8_extend

    return fp8_extend


@functools.cache
def _validate(windowed: bool, exp_free: bool) -> None:
    """Compile and run the kernel variant once on a tiny batch (one request, one KV
    head) and check its output against float32 attention, so a FlyDSL or compiler
    problem shows up when the model is built instead of on the first prefill."""
    device = torch.device("cuda", torch.accelerator.current_device_index())
    gen = torch.Generator(device=device).manual_seed(0)
    num_new, num_cached, page = 16, 32, 16
    seq_len = num_cached + num_new
    pool = torch.randn(
        (seq_len // page, 1, page, 2 * HEAD_SIZE), generator=gen, device=device
    ).to(torch.float8_e4m3fn)
    key_cache = pool[..., :HEAD_SIZE].transpose(1, 2)
    value_cache = pool[..., HEAD_SIZE:].transpose(1, 2)
    query = torch.randn((num_new, GQA, HEAD_SIZE), generator=gen, device=device).to(
        torch.bfloat16
    )
    output = torch.empty_like(query)
    one = torch.ones(1, device=device)
    window = 24 if windowed else 0
    extend(
        query,
        key_cache,
        value_cache,
        output,
        torch.tensor([0, num_new], dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        torch.arange(seq_len // page, dtype=torch.int32, device=device)[None],
        one,
        one,
        one,
        HEAD_SIZE**-0.5,
        window,
        exp_free,
    )
    k = key_cache.reshape(seq_len, HEAD_SIZE).float()
    v = value_cache.reshape(seq_len, HEAD_SIZE).float()
    scores = torch.einsum("thd,kd->thk", query.float(), k) * HEAD_SIZE**-0.5
    pos = torch.arange(seq_len, device=device)
    qpos = num_cached + torch.arange(num_new, device=device)[:, None]
    visible = pos <= qpos
    if window:
        visible &= pos > qpos - window
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
    expected = torch.einsum("thk,kd->thd", scores.softmax(-1), v)
    error = (output.float() - expected).norm() / expected.norm()
    if not error < 0.1:
        raise RuntimeError(f"relative L2 error {error.item():.3g} on a test batch")


def extend(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    sliding_window: int,
    exp_free: bool = False,
) -> None:
    """Causal attention of each request's new tokens (`query`, BF16
    `[tokens, heads, 64]`, rows `cu_seqlens_q`) over its `seq_lens` cached tokens,
    the new ones included; writes `output` (BF16, same shape). `key_cache` /
    `value_cache` are `[pages, page, kv_heads, 64]` FP8 e4m3 views, pages of a
    multiple of 16 tokens; the scales are one-element tensors (`q_scale` is the
    static scale the query is quantized with). `sliding_window` > 0 limits each query
    to its last `sliding_window` keys. `exp_free`: FP8 codes from the log2 domain
    instead of exp2."""
    num_tokens = query.shape[0]
    q = query.view(num_tokens, -1, HEAD_SIZE)
    out = output.view(num_tokens, -1, HEAD_SIZE)
    assert (
        q.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and q.stride(2) == 1
        and q.stride(1) == HEAD_SIZE
        and out.stride(2) == 1
        and out.stride(1) == HEAD_SIZE
        and key_cache.dtype == torch.float8_e4m3fn
        and value_cache.dtype == torch.float8_e4m3fn
        and key_cache.stride() == value_cache.stride()
        and key_cache.stride(3) == 1
        and key_cache.shape[1] % 16 == 0
        and q.shape[1] == GQA * key_cache.shape[2]
        and block_table.stride(1) == 1
        and seq_lens.is_contiguous()
        and cu_seqlens_q.is_contiguous()
        and seq_lens.numel() == block_table.shape[0] == cu_seqlens_q.numel() - 1
    )
    q_scale = q_scale.reshape(1).to(torch.float32)
    window = max(sliding_window, 0)
    _fly().fp8_extend_attention(
        q,
        key_cache,
        value_cache,
        out,
        cu_seqlens_q.to(torch.int32),
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        q_scale,
        k_scale.reshape(1).to(torch.float32),
        v_scale.reshape(1).to(torch.float32),
        softmax_scale,
        window=window,
        bias=bias(exp_free, window),
        exp_free=exp_free,
    )
