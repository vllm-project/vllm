# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and speculative-decode verify attention for `ROCM_AITER_FA` (gfx950) on
GQA 8, head size 64 layers with an FP8 KV cache.

With an FP8 e4m3 KV cache, 16- or 32-token pages and every decode request having the
same number of query tokens, 1 to 8 (plain decode, or 1 to 7 draft tokens; head size
64, GQA 8, no sinks / soft cap / ALiBi), the backend calls `fast_verify` instead of
AITER `paged_attention_v1` / `unified_attention`. It runs a FlyDSL split-KV kernel
(BF16 QK / PV, K / V upcast exactly from FP8):

* full-attention layers: a length-balanced grid, where a fixed one or two workgroups
  per CU share the step's KV chunks evenly (mapped on device from `seq_lens`, once
  per step by the metadata builder), so the kernel tracks total KV bytes instead of
  the longest request while there are well under two requests per CU (each request
  takes at least one workgroup);
* sliding-window layers: vLLM's window (keys with `q_pos - k_pos < sliding_window`
  are visible, the same mask as `unified_attention`'s `1 + window_size[0]`), with
  equal splits per request.

The kernel reads the block table and sequence lengths on device only, launches fixed
grids for a given batch size and allocates its partials with `torch.empty`, so it is
CUDA-graph capturable. Zero-length padded requests (FULL-graph padding) produce
zeros.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import num_compute_units

if TYPE_CHECKING:
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionMetadata

logger = init_logger(__name__)

QUERY_LEN = 8  # longest verify: 7 draft tokens + 1
MIN_QUERY_LEN = 1
GQA = 8
HEAD_SIZE = 64
BLOCK_SIZES = (16, 32)
# KV heads per rank: one 64-lane wave each in a workgroup of at most 1024 threads.
MAX_KV_HEADS = 16


def supports_layer(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    alibi_slopes: torch.Tensor | None,
    sinks: torch.Tensor | None,
    logits_soft_cap: float,
) -> bool:
    """Whether an attention layer has the shape and features the kernels handle.

    Evaluated once per layer. Warns once and returns False when FlyDSL cannot be
    imported.
    """
    if not (
        head_size == HEAD_SIZE
        and num_heads == GQA * num_kv_heads
        and num_kv_heads <= MAX_KV_HEADS
        and alibi_slopes is None
        and sinks is None
        and not logits_soft_cap
    ):
        return False
    try:
        _fly()
    except ImportError as e:
        logger.warning_once(
            "GQA verify attention kernels are disabled, FlyDSL is unavailable (%s); "
            "falling back to the AITER kernels.",
            e,
        )
        return False
    return True


def uniform_query_len(
    num_decodes: int, num_decode_tokens: int, decode_max_query_len: int
) -> bool:
    """Whether every decode request has the same 1 to 8 query tokens (a batch with
    mixed query lengths has fewer than `decode_max_query_len * num_decodes`)."""
    return (
        MIN_QUERY_LEN <= decode_max_query_len <= QUERY_LEN
        and num_decode_tokens == decode_max_query_len * num_decodes
    )


def layout_supported(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> bool:
    """Host-only dtype / page size / stride checks; no device reads.

    A failing check warns once: the backend then uses its AITER kernels for every
    decode step.
    """
    ks = key_cache.stride()
    ok = (
        key_cache.dtype == torch.float8_e4m3fn
        and value_cache.dtype == torch.float8_e4m3fn
        and key_cache.shape[1] in BLOCK_SIZES
        and key_cache.shape[3] == HEAD_SIZE
        and ks == value_cache.stride()
        and ks[3] == 1
        and ks[0] % 16 == 0
        and ks[1] % 16 == 0
        and ks[2] % 16 == 0
        and query.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and query.stride(2) == 1
        and query.stride(1) == HEAD_SIZE
        and output.stride(2) == 1
        and seq_lens.dtype == torch.int32
        and block_table.dtype == torch.int32
    )
    if not ok:
        logger.warning_once(
            "GQA verify attention kernels are inactive: unsupported layout (KV cache "
            "%s %s strides %s, query %s %s, seq_lens %s, block_table %s). They need "
            "FP8 e4m3 pages of 16 or 32 tokens x head size %d; "
            "falling back to the AITER kernels.",
            key_cache.dtype,
            tuple(key_cache.shape),
            tuple(ks),
            query.dtype,
            tuple(query.stride()),
            seq_lens.dtype,
            block_table.dtype,
            HEAD_SIZE,
        )
    return ok


def _fly():
    from vllm.v1.attention.ops.flydsl_kernels import gqa_verify

    return gqa_verify


def _balanced_grid(
    num_decodes: int, kv_heads: int, device: torch.device
) -> tuple[int, int, int] | None:
    return _fly().pick_balanced(num_decodes, kv_heads, num_compute_units(device.index))


def map_buffers(
    max_requests: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent storage for `verify_map` (full CUDA graphs keep reading it)."""
    groups = 2 * num_compute_units(device.index)
    return (
        torch.empty((groups, 4), dtype=torch.int32, device=device),
        torch.empty(max_requests + 1, dtype=torch.int32, device=device),
    )


def verify_map(
    seq_lens: torch.Tensor,
    kv_heads: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The full-attention layers' work map for a step's decode `seq_lens`, shared by
    every layer with `kv_heads` KV heads per rank; None if the batch is too large
    for the balanced grid. `out` takes buffers from `map_buffers`."""
    grid = _balanced_grid(seq_lens.numel(), kv_heads, seq_lens.device)
    if grid is None:
        return None
    return _fly().balance_map(seq_lens, grid[0], out)


def fast_verify(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: "AiterFlashAttentionMetadata",
    num_decodes: int,
    num_decode_tokens: int,
    decode_max_query_len: int,
    softmax_scale: float,
    sliding_window: int,
    layer: torch.nn.Module,
    maps: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> bool:
    """Attention for the decode requests of a step, if the batch fits.

    Runs when every decode request has the same number of query tokens, 1 to 8, the
    cache layout is supported and (full attention) the batch fits the balanced grid;
    returns False, having done nothing, otherwise.

    Args:
        query: `[tokens, heads, 64]` bf16 query; the decode rows come first.
        key_cache: `[num_blocks, 16 or 32, kv_heads, 64]` FP8 view of the K cache.
        value_cache: the same for V.
        output: `[tokens, heads, 64]` bf16, the decode rows are written.
        attn_metadata: the backend's metadata (`seq_lens`, `block_table`).
        num_decodes: number of decode requests.
        num_decode_tokens: number of decode tokens.
        decode_max_query_len: longest decode query.
        softmax_scale: attention scale.
        sliding_window: vLLM's window in tokens, or -1 / 0 for full attention.
        layer: the attention layer, for its `_k_scale` / `_v_scale`.
        maps: the step's `verify_map` for full-attention layers, if prebuilt.

    Returns:
        Whether the kernels ran.

    """
    query_len = decode_max_query_len
    if not uniform_query_len(num_decodes, num_decode_tokens, query_len):
        return False
    grid = None
    if sliding_window <= 0:
        grid = _balanced_grid(num_decodes, key_cache.shape[2], query.device)
        if grid is None:
            return False
    if not layout_supported(
        query,
        key_cache,
        value_cache,
        output,
        attn_metadata.seq_lens,
        attn_metadata.block_table,
    ):
        return False
    fly = _fly()
    args = (
        query[:num_decode_tokens],
        key_cache,
        value_cache,
        output[:num_decode_tokens],
        attn_metadata.seq_lens[:num_decodes],
        attn_metadata.block_table[:num_decodes],
        softmax_scale,
        # FlyDSL needs a 1-D descale; `_k_scale` is 0-dim and `view(1)` shares it.
        layer._k_scale.view(1),
        layer._v_scale,
    )
    if grid is None:
        fly.gqa_verify_window(
            *args,
            query_len=query_len,
            window=sliding_window,
            num_cus=num_compute_units(query.device.index),
        )
    else:
        if maps is None:
            maps = fly.balance_map(args[4], grid[0])
        fly.gqa_verify_balanced(*args, query_len=query_len, grid=grid, maps=maps)
    logger.info_once(
        "GQA verify attention kernels active for %s layers.",
        "full-attention" if grid else "sliding-window",
    )
    return True
