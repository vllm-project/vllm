# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import logging
import math
import os
import pathlib
import tempfile
from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
else:
    _ON_GFX942 = False
    _ON_GFX950 = False


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    kv_cache_value = kv_cache[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        "SHUFFLE",
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            block_id * kv_cache_stride
            + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        "SHUFFLE",
        head_dim,
        block_tile_size,
        head_tile_size,
    )


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full(
            [batch_size, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            assert seq_len <= max_model_len
            num_pages = cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            scale_offset = block_size * dim
            cache_value = (
                cache[..., :scale_offset].view(dtype=fp8_dtype).to(torch.float32)
            )
            cache_scale = (
                cache[..., scale_offset:].view(dtype=torch.float32).contiguous()
            )
            cache_value = cache_value.view(padded_seq_len, dim)
            cache_scale = cache_scale.view(padded_seq_len)
            score = F.linear(cache_value, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    # if rocm_aiter_ops.is_enabled():
    batch_size, next_n, heads, head_dim = q_fp8.shape
    num_blocks, block_size, _, _ = kv_cache_fp8.shape

    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        if _ON_GFX942 or _ON_GFX950:
            deepgemm_fp8_paged_mqa_logits = (
                aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits
            )
            batch_size, next_n, heads, _ = q_fp8.shape
            out_logits = torch.full(
                [batch_size * next_n, max_model_len],
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=256,
                Preshuffle=block_size == 64,
                KVBlockSize=block_size,
                WavePerEU=2,
            )
            return out_logits
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ=heads,
        )
        return out_qk.sum(dim=0)
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        k_fp8, scale = kv
        return fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def _topk_indices_torch(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if row_starts is not None:
        # Match the CUDA top_k_per_row_prefill contract: indices are local to
        # each row's valid [row_start, row_end) range, not columns in the
        # concatenated chunk logits matrix.
        starts = row_starts.to(dtype=torch.int32).view(-1, 1)
        indices = torch.where(indices < 0, indices, indices - starts)
    if k == topk_tokens:
        return indices
    padded = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    padded[:, :k] = indices
    return padded


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


@eager_break_during_capture
def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import _resolve_layer_name

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens
    device = hidden_states.device if k is None else k.device

    # during speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        if _ON_GFX942:
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )
        else:
            indexer_k_quant_and_cache_triton(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=device,
                dtype=torch.uint8,
            )
            if _ON_GFX942:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            else:
                cp_gather_indexer_k_quant_cache_triton(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                )

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            num_rows = logits.shape[0]

            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        num_rows = logits.shape[0]

        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _apply_gptj_inv_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or x.numel() == 0:
        return x
    half_rot = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    dtype = x.dtype
    x = x.to(torch.float32)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half_rot].to(torch.float32)
    sin = cache[:, half_rot : 2 * half_rot].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.dim() - 2) + (half_rot,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rope = x[..., nope_dim:]
    y_even = rope[..., 0::2]
    y_odd = rope[..., 1::2]
    rope_out = torch.stack(
        (y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
        dim=-1,
    ).flatten(-2)
    x = x.clone()
    x[..., nope_dim:] = rope_out
    return x.to(dtype)


def _apply_inv_rope_ref(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if hasattr(rotary_emb, "forward_native"):
        try:
            query, _ = rotary_emb.forward_native(
                positions,
                x.clone(),
                None,
                inverse=True,
            )
            return query
        except TypeError:
            pass
    return _apply_gptj_inv_rope_ref(x, positions, rotary_emb.cos_sin_cache, rope_dim)


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Reference inverse-RoPE + WO_A einsum path used on ROCm."""
    o_ref = _apply_inv_rope_ref(rotary_emb, o, positions, rope_head_dim).to(
        torch.bfloat16
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    hidden_dim = o_ref.shape[-1]
    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank,
            hidden_dim,
        )
        wo_a_weight = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.bfloat16
        )

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)


_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = torch.empty(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    indptr[0] = 0
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            int(indptr[-1].item()), dtype=torch.int32, device=indices.device
        )
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_kv)

        kv = tl.load(
            kv_ptr + slot[:, None] * kv_stride_n + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        kv = tl.where(valid[:, None] & dim_mask[None, :], kv, 0.0)

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _sparse_attn_decode_ragged_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ:
                x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = tl.where(l_i[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


def _rocm_sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert q.is_cuda and kv.is_cuda and indices.is_cuda and indptr.is_cuda

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_ragged_triton",
    )

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 16 if head_dim >= 256 else 32
    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )


def _rocm_sparse_attn_decode_ragged_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        q.is_cuda
        and main_cache.is_cuda
        and main_indices.is_cuda
        and main_indptr.is_cuda
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_decode_ragged_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    block_h = 16
    block_k = 16 if head_dim >= 256 else 32
    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_decode_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        IS_FNUZ=current_platform.is_fp8_fnuz(),
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _rocm_sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    main_lengths: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    main_ragged_indices: torch.Tensor | None = None,
    main_ragged_indptr: torch.Tensor | None = None,
    extra_ragged_indices: torch.Tensor | None = None,
    extra_ragged_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    if main_ragged_indices is None or main_ragged_indptr is None:
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lengths
            if main_lengths is not None
            else (main_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=main_cache.shape[0] * main_cache.shape[1],
        )

    if (
        (extra_ragged_indices is None or extra_ragged_indptr is None)
        and extra_cache is not None
        and extra_indices is not None
    ):
        extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
            extra_indices,
            extra_lengths
            if extra_lengths is not None
            else (extra_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=extra_cache.shape[0] * extra_cache.shape[1],
        )

    return _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
    )


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"ROCm Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_prefill",
    )
    if ragged_indices is not None and ragged_indptr is not None:
        output_chunk = _rocm_sparse_attn_prefill_ragged_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )
    else:
        indices_2d = indices.reshape(indices.shape[0], -1)
        output_chunk = _rocm_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=indices_2d,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            topk_length=topk_length,
        )
    output.copy_(output_chunk.to(output.dtype))


# ============================================================================
# HIP MFMA kernel implementation for sparse-MLA decode.
# ============================================================================

_HIP_SPARSE_MLA_DECODE_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16x8 = __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16;
using fx4    = __attribute__((__vector_size__(4 * sizeof(float))))  float;

static constexpr int NOPE_DIM    = 448;
static constexpr int ROPE_DIM    = 64;
static constexpr int TOKEN_BYTES = 576;
static constexpr int SCALE_BYTES = 8;
static constexpr int HEAD_DIM    = 512;
static constexpr int BLOCK_H     = 16;
static constexpr int BLOCK_K     = 32;
static constexpr int N_TILES     = HEAD_DIM / 16;  // 32
static constexpr int QK_N_TILES  = BLOCK_K / 16;   // 2

__device__ __forceinline__ fx4 mfma_16x16x32_bf16(
    bf16x8 a, bf16x8 b, fx4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ void gather_and_dequant_k_tile(
    int k_start, int k_len, const uint8_t* cache_base,
    int64_t cache_stride0, int num_rows, int block_size,
    const int32_t* idx_base,
    __bf16* k_lds, int8_t* kv_lds, int tid)
{
    const int tok_id = tid >> 3;  // 0..31
    const int chunk  = tid & 7;   // 0..7
    const int col0   = chunk * 64;

    int k_pos = k_start + tok_id;
    bool in_range = (k_pos < k_len);
    int slot = in_range ? idx_base[k_pos] : 0;
    bool valid = in_range && (slot >= 0) && (slot < num_rows);
    int safe_slot = valid ? slot : 0;
    int bi = safe_slot / block_size;
    int pib = safe_slot - bi * block_size;
    const uint8_t* block_ptr = cache_base
                               + (int64_t)bi * cache_stride0;
    const uint8_t* token_ptr = block_ptr + pib * TOKEN_BYTES;

    __bf16* dst_row = &k_lds[tok_id * HEAD_DIM + col0];

    if (!valid) {
        int4 z; z.x = z.y = z.z = z.w = 0;
        int4* d4 = reinterpret_cast<int4*>(dst_row);
        #pragma unroll
        for (int j = 0; j < 8; ++j) d4[j] = z;
    } else if (col0 < NOPE_DIM) {
        const uint8_t* scale_ptr = block_ptr
                                   + block_size * TOKEN_BYTES
                                   + pib * SCALE_BYTES;
        uint8_t scl_u = scale_ptr[chunk];
        union { uint32_t u; float fv; } sb;
        sb.u = ((uint32_t)scl_u) << 23;
        float scl_f = sb.fv;

        const uint32_t* src32 = reinterpret_cast<const uint32_t*>(
            token_ptr + col0);
        #pragma unroll
        for (int u32_i = 0; u32_i < 16; ++u32_i) {
            uint32_t word = src32[u32_i];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint8_t kb = (word >> (b * 8)) & 0xFF;
                uint32_t packed = (uint32_t)kb;
                float f = __builtin_amdgcn_cvt_f32_fp8(packed, 0) * scl_f;
                dst_row[u32_i * 4 + b] = (__bf16)f;
            }
        }
    } else {
        const int4* src4 = reinterpret_cast<const int4*>(
            token_ptr + NOPE_DIM);
        int4* d4 = reinterpret_cast<int4*>(dst_row);
        #pragma unroll
        for (int j = 0; j < 8; ++j) d4[j] = src4[j];
    }

    if (tid < BLOCK_K) {
        int kp = k_start + tid;
        int sl = (kp < k_len) ? idx_base[kp] : -1;
        kv_lds[tid] = (kp < k_len) && (sl >= 0) && (sl < num_rows) ? 1 : 0;
    }
}


constexpr int N_TILES_PER_WAVE = 8;  // 32 N-tiles / 4 waves
__device__ __forceinline__ void process_k_tile(
    const __bf16* q_lds, const __bf16* k_lds, const int8_t* kv_lds,
    __bf16* p_lds, float* scores_lds,
    float* m_state, float* l_state, fx4* acc, float scale,
    int lane, int m_a, int kg, int n_b, int m_d_base, int n_d,
    int wave)
{
    if (wave == 0) {
        fx4 qk[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < HEAD_DIM / 32; ++c) {
            bf16x8 q_reg;
            const __bf16* q_src = &q_lds[m_a * HEAD_DIM + c * 32 + kg * 8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) q_reg[i] = q_src[i];

            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                bf16x8 k_reg;
                const __bf16* k_src = &k_lds[(nt * 16 + n_b) * HEAD_DIM
                                              + c * 32 + kg * 8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) k_reg[i] = k_src[i];
                qk[nt] = mfma_16x16x32_bf16(q_reg, k_reg, qk[nt]);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int k_col = nt * 16 + n_d;
                float s = qk[nt][i] * scale;
                if (!kv_lds[k_col]) s = -3.4028234663852886e38f;
                scores_lds[(m_d_base + i) * BLOCK_K + nt * 16 + n_d] = s;
            }
        }
    }

    __syncthreads();

    fx4 qk_local[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qk_local[0][i] = scores_lds[(m_d_base + i) * BLOCK_K + n_d];
        qk_local[1][i] = scores_lds[(m_d_base + i) * BLOCK_K + 16 + n_d];
    }

    fx4 p[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float row_max = fmaxf(qk_local[0][i], qk_local[1][i]);
        row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 4));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 8));

        float m_new = fmaxf(m_state[i], row_max);
        float alpha = __builtin_amdgcn_exp2f(
            (m_state[i] - m_new) * 1.4426950408889634f);

        float e0 = __builtin_amdgcn_exp2f(
            (qk_local[0][i] - m_new) * 1.4426950408889634f);
        float e1 = __builtin_amdgcn_exp2f(
            (qk_local[1][i] - m_new) * 1.4426950408889634f);

        float row_sum = e0 + e1;
        row_sum += __shfl_xor(row_sum, 1);
        row_sum += __shfl_xor(row_sum, 2);
        row_sum += __shfl_xor(row_sum, 4);
        row_sum += __shfl_xor(row_sum, 8);

        float l_new = l_state[i] * alpha + row_sum;
        p[0][i] = e0;
        p[1][i] = e1;

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WAVE; ++nt) acc[nt][i] *= alpha;

        m_state[i] = m_new;
        l_state[i] = l_new;
    }

    if (wave == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p_lds[(m_d_base + i) * BLOCK_K + n_d]      = (__bf16)p[0][i];
            p_lds[(m_d_base + i) * BLOCK_K + 16 + n_d] = (__bf16)p[1][i];
        }
    }

    __syncthreads();

    bf16x8 p_reg;
    const __bf16* p_src = &p_lds[m_a * BLOCK_K + kg * 8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) p_reg[i] = p_src[i];

    #pragma unroll
    for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
        int n_tile = wave * N_TILES_PER_WAVE + nt_local;
        bf16x8 k_reg;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            k_reg[i] = k_lds[(kg * 8 + i) * HEAD_DIM
                             + n_tile * 16 + n_b];
        }
        acc[nt_local] = mfma_16x16x32_bf16(p_reg, k_reg, acc[nt_local]);
    }
}


__device__ __forceinline__ void load_q(
    const __bf16* q, int64_t q_stride0, int64_t q_stride1,
    int query, int pid_h, int num_heads,
    __bf16* q_lds, int tid)
{
    const int qh  = tid >> 4;          // 0..15
    const int qc0 = (tid & 15) << 5;   // 0,32,...,480
    const int head_global = pid_h * BLOCK_H + qh;
    __bf16* dst = &q_lds[qh * HEAD_DIM + qc0];
    if (head_global < num_heads) {
        const __bf16* src = q + query * q_stride0
                              + head_global * q_stride1 + qc0;
        const int4* s4 = reinterpret_cast<const int4*>(src);
        int4* d4 = reinterpret_cast<int4*>(dst);
        #pragma unroll
        for (int i = 0; i < 4; ++i) d4[i] = s4[i];
    } else {
        int4 z; z.x = z.y = z.z = z.w = 0;
        int4* d4 = reinterpret_cast<int4*>(dst);
        #pragma unroll
        for (int i = 0; i < 4; ++i) d4[i] = z;
    }
}


template <bool HAS_ATTN_SINK, bool HAS_EXTRA>
__global__ __launch_bounds__(256, 2)
void sparse_mla_decode_kernel(
    const __bf16* __restrict__ q,
    const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr,
    const float* __restrict__ attn_sink,
    __bf16* __restrict__ output,
    int64_t q_stride0, int64_t q_stride1,
    int64_t out_stride0, int64_t out_stride1,
    int64_t main_cache_stride0, int64_t extra_cache_stride0,
    int main_num_rows, int extra_num_rows,
    int main_block_size, int extra_block_size,
    float scale, int num_heads)
{
    const int query = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;

    const int m_a       = lane & 15;
    const int kg        = lane >> 4;
    const int n_b       = lane & 15;
    const int m_d_base  = (lane >> 4) * 4;
    const int n_d       = lane & 15;

    __shared__ __bf16 q_lds[BLOCK_H * HEAD_DIM];
    __shared__ __bf16 k_lds[BLOCK_K * HEAD_DIM];
    __shared__ __bf16 p_lds[BLOCK_H * BLOCK_K];
    __shared__ float  scores_lds[BLOCK_H * BLOCK_K];
    __shared__ int8_t kv_lds[BLOCK_K];
    __shared__ char   force_1wg_per_cu[48 * 1024];  // pads LDS to ~96 KB
    (void)force_1wg_per_cu;

    load_q(q, q_stride0, q_stride1, query, pid_h, num_heads, q_lds, tid);

    float m_state[4], l_state[4];
    fx4   acc[N_TILES_PER_WAVE];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_state[i] = -3.4028234663852886e38f;
        l_state[i] = 0.f;
    }
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WAVE; ++i) {
        acc[i] = (fx4){0.f, 0.f, 0.f, 0.f};
    }

    __syncthreads();

    {
        int main_start = main_indptr[query];
        int main_end   = main_indptr[query + 1];
        int main_len   = main_end - main_start;
        for (int k_start = 0; k_start < main_len; k_start += BLOCK_K) {
            gather_and_dequant_k_tile(
                k_start, main_len, main_cache, main_cache_stride0,
                main_num_rows, main_block_size,
                main_indices + main_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    if (HAS_EXTRA) {
        int extra_start = extra_indptr[query];
        int extra_end   = extra_indptr[query + 1];
        int extra_len   = extra_end - extra_start;
        for (int k_start = 0; k_start < extra_len; k_start += BLOCK_K) {
            gather_and_dequant_k_tile(
                k_start, extra_len, extra_cache, extra_cache_stride0,
                extra_num_rows, extra_block_size,
                extra_indices + extra_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int head_local = m_d_base + i;
            int head_global = pid_h * BLOCK_H + head_local;
            if (head_global >= num_heads) continue;

            float m_final = m_state[i];
            float l_final = l_state[i];
            float alpha_final = 1.f;
            if (HAS_ATTN_SINK) {
                float sink_val = attn_sink[head_global];
                m_final = fmaxf(m_state[i], sink_val);
                alpha_final = __builtin_amdgcn_exp2f(
                    (m_state[i] - m_final) * 1.4426950408889634f);
                l_final = l_state[i] * alpha_final + __builtin_amdgcn_exp2f(
                    (sink_val - m_final) * 1.4426950408889634f);
            }
            float denom = fmaxf(l_final, 1.0e-30f);
            bool live = (l_final > 0.f);

            __bf16* out_row = output + query * out_stride0
                                     + head_global * out_stride1;
            #pragma unroll
            for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
                int n_tile = wave * N_TILES_PER_WAVE + nt_local;
                int col = n_tile * 16 + n_d;
                float v = live ? (acc[nt_local][i] * alpha_final) / denom : 0.f;
                out_row[col] = (__bf16)v;
            }
        }
    }
}


template <bool HAS_EXTRA, int SPLIT_K>
__global__ __launch_bounds__(256, 2)
void sparse_mla_decode_partial_kernel(
    const __bf16* __restrict__ q,
    const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr,
    float* __restrict__ scratch_m,
    float* __restrict__ scratch_l,
    __bf16* __restrict__ scratch_acc,
    int64_t q_stride0, int64_t q_stride1,
    int64_t main_cache_stride0, int64_t extra_cache_stride0,
    int main_num_rows, int extra_num_rows,
    int main_block_size, int extra_block_size,
    float scale, int num_heads, int num_head_blocks)
{
    const int query = blockIdx.x;
    const int pid_hs = blockIdx.y;
    const int pid_split = pid_hs / num_head_blocks;
    const int pid_h = pid_hs - pid_split * num_head_blocks;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;

    const int m_a       = lane & 15;
    const int kg        = lane >> 4;
    const int n_b       = lane & 15;
    const int m_d_base  = (lane >> 4) * 4;
    const int n_d       = lane & 15;

    __shared__ __bf16 q_lds[BLOCK_H * HEAD_DIM];
    __shared__ __bf16 k_lds[BLOCK_K * HEAD_DIM];
    __shared__ __bf16 p_lds[BLOCK_H * BLOCK_K];
    __shared__ float  scores_lds[BLOCK_H * BLOCK_K];
    __shared__ int8_t kv_lds[BLOCK_K];
    load_q(q, q_stride0, q_stride1, query, pid_h, num_heads, q_lds, tid);

    float m_state[4], l_state[4];
    fx4   acc[N_TILES_PER_WAVE];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_state[i] = -3.4028234663852886e38f;
        l_state[i] = 0.f;
    }
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WAVE; ++i) {
        acc[i] = (fx4){0.f, 0.f, 0.f, 0.f};
    }

    __syncthreads();

    {
        int main_start = main_indptr[query];
        int main_end   = main_indptr[query + 1];
        int main_len   = main_end - main_start;
        for (int k_start = pid_split * BLOCK_K; k_start < main_len;
             k_start += BLOCK_K * SPLIT_K) {
            gather_and_dequant_k_tile(
                k_start, main_len, main_cache, main_cache_stride0,
                main_num_rows, main_block_size,
                main_indices + main_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    if (HAS_EXTRA) {
        int extra_start = extra_indptr[query];
        int extra_end   = extra_indptr[query + 1];
        int extra_len   = extra_end - extra_start;
        for (int k_start = pid_split * BLOCK_K; k_start < extra_len;
             k_start += BLOCK_K * SPLIT_K) {
            gather_and_dequant_k_tile(
                k_start, extra_len, extra_cache, extra_cache_stride0,
                extra_num_rows, extra_block_size,
                extra_indices + extra_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    const int triple = (query * num_head_blocks + pid_h) * SPLIT_K + pid_split;

    if (wave == 0 && n_d == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = triple * BLOCK_H + m_d_base + i;
            scratch_m[idx] = m_state[i];
            scratch_l[idx] = l_state[i];
        }
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = m_d_base + i;
        #pragma unroll
        for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
            int n_tile = wave * N_TILES_PER_WAVE + nt_local;
            int col = n_tile * 16 + n_d;
            k_lds[row * HEAD_DIM + col] = (__bf16)acc[nt_local][i];
        }
    }
    __syncthreads();

    {
        int my_row  = tid >> 4;
        int my_col0 = (tid & 15) << 5;
        __bf16* dst = scratch_acc + (int64_t)triple * BLOCK_H * HEAD_DIM
                                  + my_row * HEAD_DIM + my_col0;
        const int4* src4 = reinterpret_cast<const int4*>(
            &k_lds[my_row * HEAD_DIM + my_col0]);
        int4* dst4 = reinterpret_cast<int4*>(dst);
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst4[i] = src4[i];
    }
}


template <bool HAS_ATTN_SINK, int SPLIT_K>
__global__ __launch_bounds__(256, 4)
void sparse_mla_decode_reduce_kernel(
    const float* __restrict__ scratch_m,
    const float* __restrict__ scratch_l,
    const __bf16* __restrict__ scratch_acc,
    const float* __restrict__ attn_sink,
    __bf16* __restrict__ output,
    int64_t out_stride0, int64_t out_stride1,
    int num_heads, int num_head_blocks)
{
    const int query = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int tid   = threadIdx.x;

    const int my_row  = tid >> 4;        // 0..15
    const int my_col0 = (tid & 15) << 5; // 0,32,...,480
    const int head_global = pid_h * BLOCK_H + my_row;

    float m_merged = -3.4028234663852886e38f;
    float l_merged = 0.f;
    float acc_merged[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) acc_merged[i] = 0.f;

    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        const int triple = (query * num_head_blocks + pid_h) * SPLIT_K + s;
        float m_s = scratch_m[triple * BLOCK_H + my_row];
        float l_s = scratch_l[triple * BLOCK_H + my_row];

        float m_new = fmaxf(m_merged, m_s);
        float alpha = __builtin_amdgcn_exp2f(
            (m_merged - m_new) * 1.4426950408889634f);
        float beta  = __builtin_amdgcn_exp2f(
            (m_s      - m_new) * 1.4426950408889634f);
        l_merged = l_merged * alpha + l_s * beta;
        m_merged = m_new;

        const __bf16* acc_base = scratch_acc
                               + (int64_t)triple * BLOCK_H * HEAD_DIM
                               + my_row * HEAD_DIM + my_col0;
        const int4* src4 = reinterpret_cast<const int4*>(acc_base);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int4 v = src4[i];
            __bf16 vbf[8];
            *reinterpret_cast<int4*>(vbf) = v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a_s = (float)vbf[j];
                acc_merged[i * 8 + j] = acc_merged[i * 8 + j] * alpha
                                      + a_s * beta;
            }
        }
    }

    if (head_global >= num_heads) return;

    float m_final = m_merged;
    float l_final = l_merged;
    float alpha_final = 1.f;
    if (HAS_ATTN_SINK) {
        float sink_val = attn_sink[head_global];
        m_final = fmaxf(m_merged, sink_val);
        alpha_final = __builtin_amdgcn_exp2f(
            (m_merged - m_final) * 1.4426950408889634f);
        l_final = l_merged * alpha_final + __builtin_amdgcn_exp2f(
            (sink_val - m_final) * 1.4426950408889634f);
    }
    float denom = fmaxf(l_final, 1.0e-30f);
    bool live = (l_final > 0.f);
    float inv_denom = live ? (alpha_final / denom) : 0.f;

    __bf16* out_row = output + query * out_stride0
                             + head_global * out_stride1 + my_col0;
    __bf16 out_buf[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) out_buf[i] = (__bf16)(acc_merged[i] * inv_denom);
    int4* dst4 = reinterpret_cast<int4*>(out_row);
    const int4* sb4 = reinterpret_cast<const int4*>(out_buf);
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst4[i] = sb4[i];
}


void sparse_mla_decode_single(
    torch::Tensor q,
    torch::Tensor main_cache,
    torch::Tensor main_indices,
    torch::Tensor main_indptr,
    torch::Tensor extra_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_indptr,
    c10::optional<torch::Tensor> attn_sink,
    torch::Tensor output,
    int64_t main_block_size,
    int64_t extra_block_size,
    int64_t main_num_rows,
    int64_t extra_num_rows,
    double scale_d,
    bool has_extra)
{
    const int num_queries = q.size(0);
    const int num_heads = q.size(1);
    const int num_head_blocks = (num_heads + BLOCK_H - 1) / BLOCK_H;
    const float scale_f = (float)scale_d;
    const bool has_sink = attn_sink.has_value();

    dim3 grid(num_queries, num_head_blocks);
    dim3 block(256);

    const __bf16* q_ptr = reinterpret_cast<const __bf16*>(q.data_ptr());
    const uint8_t* mc_ptr = reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
    const uint8_t* ec_ptr = reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
    const int32_t* mi_ptr = main_indices.data_ptr<int32_t>();
    const int32_t* mip_ptr = main_indptr.data_ptr<int32_t>();
    const int32_t* ei_ptr = extra_indices.data_ptr<int32_t>();
    const int32_t* eip_ptr = extra_indptr.data_ptr<int32_t>();
    __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    const float* sink_ptr = has_sink
        ? attn_sink.value().data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH(HAS_S, HAS_E) do { \
        sparse_mla_decode_kernel<HAS_S, HAS_E><<<grid, block, 0, stream>>>( \
            q_ptr, mc_ptr, mi_ptr, mip_ptr, \
            ec_ptr, ei_ptr, eip_ptr, sink_ptr, out_ptr, \
            q.stride(0), q.stride(1), \
            output.stride(0), output.stride(1), \
            main_cache.stride(0), extra_cache.stride(0), \
            main_num_rows, extra_num_rows, \
            main_block_size, extra_block_size, \
            scale_f, num_heads); \
    } while (0)

    if (has_sink && has_extra)  LAUNCH(true, true);
    else if (has_sink)          LAUNCH(true, false);
    else if (has_extra)         LAUNCH(false, true);
    else                        LAUNCH(false, false);

    #undef LAUNCH
}


void sparse_mla_decode_split(
    torch::Tensor q,
    torch::Tensor main_cache,
    torch::Tensor main_indices,
    torch::Tensor main_indptr,
    torch::Tensor extra_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_indptr,
    c10::optional<torch::Tensor> attn_sink,
    torch::Tensor output,
    torch::Tensor scratch_m,
    torch::Tensor scratch_l,
    torch::Tensor scratch_acc,
    int64_t main_block_size,
    int64_t extra_block_size,
    int64_t main_num_rows,
    int64_t extra_num_rows,
    double scale_d,
    bool has_extra,
    int64_t split_k)
{
    const int num_queries = q.size(0);
    const int num_heads = q.size(1);
    const int num_head_blocks = (num_heads + BLOCK_H - 1) / BLOCK_H;
    const float scale_f = (float)scale_d;
    const bool has_sink = attn_sink.has_value();

    dim3 grid_p(num_queries, num_head_blocks * (int)split_k);
    dim3 grid_r(num_queries, num_head_blocks);
    dim3 block_p(256);
    dim3 block_r(256);

    const __bf16* q_ptr = reinterpret_cast<const __bf16*>(q.data_ptr());
    const uint8_t* mc_ptr = reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
    const uint8_t* ec_ptr = reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
    const int32_t* mi_ptr = main_indices.data_ptr<int32_t>();
    const int32_t* mip_ptr = main_indptr.data_ptr<int32_t>();
    const int32_t* ei_ptr = extra_indices.data_ptr<int32_t>();
    const int32_t* eip_ptr = extra_indptr.data_ptr<int32_t>();
    __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    float* sm_ptr = scratch_m.data_ptr<float>();
    float* sl_ptr = scratch_l.data_ptr<float>();
    __bf16* sa_ptr = reinterpret_cast<__bf16*>(scratch_acc.data_ptr());
    const float* sink_ptr = has_sink
        ? attn_sink.value().data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH_P(HAS_E, SK) do { \
        sparse_mla_decode_partial_kernel<HAS_E, SK><<<grid_p, block_p, 0, stream>>>( \
            q_ptr, mc_ptr, mi_ptr, mip_ptr, \
            ec_ptr, ei_ptr, eip_ptr, \
            sm_ptr, sl_ptr, sa_ptr, \
            q.stride(0), q.stride(1), \
            main_cache.stride(0), extra_cache.stride(0), \
            main_num_rows, extra_num_rows, \
            main_block_size, extra_block_size, \
            scale_f, num_heads, num_head_blocks); \
    } while (0)

    #define LAUNCH_R(HAS_S, SK) do { \
        sparse_mla_decode_reduce_kernel<HAS_S, SK><<<grid_r, block_r, 0, stream>>>( \
            sm_ptr, sl_ptr, sa_ptr, sink_ptr, out_ptr, \
            output.stride(0), output.stride(1), \
            num_heads, num_head_blocks); \
    } while (0)

    #define DISPATCH_SK(SK) do { \
        if (has_extra) LAUNCH_P(true, SK); \
        else            LAUNCH_P(false, SK); \
        if (has_sink)  LAUNCH_R(true, SK); \
        else            LAUNCH_R(false, SK); \
    } while (0)

    switch ((int)split_k) {
        case  2: DISPATCH_SK(2);  break;
        case  4: DISPATCH_SK(4);  break;
        case  8: DISPATCH_SK(8);  break;
        case 16: DISPATCH_SK(16); break;
        default: TORCH_CHECK(false, "Unsupported SPLIT_K");
    }
    #undef DISPATCH_SK
    #undef LAUNCH_P
    #undef LAUNCH_R
}


TORCH_LIBRARY_FRAGMENT(vllm_sparse_mla_hip, m) {
    m.def("decode_single(Tensor q, Tensor main_cache, Tensor main_indices, "
          "Tensor main_indptr, Tensor extra_cache, Tensor extra_indices, "
          "Tensor extra_indptr, Tensor? attn_sink, Tensor output, "
          "int main_block_size, int extra_block_size, int main_num_rows, "
          "int extra_num_rows, float scale, bool has_extra) -> ()");
    m.def("decode_split(Tensor q, Tensor main_cache, Tensor main_indices, "
          "Tensor main_indptr, Tensor extra_cache, Tensor extra_indices, "
          "Tensor extra_indptr, Tensor? attn_sink, Tensor output, "
          "Tensor scratch_m, Tensor scratch_l, Tensor scratch_acc, "
          "int main_block_size, int extra_block_size, int main_num_rows, "
          "int extra_num_rows, float scale, bool has_extra, int split_k) -> ()");
}
TORCH_LIBRARY_IMPL(vllm_sparse_mla_hip, CUDA, m) {
    m.impl("decode_single", &sparse_mla_decode_single);
    m.impl("decode_split", &sparse_mla_decode_split);
}
"""

logger = logging.getLogger(__name__)

_sparse_mla_hip_module_cache: dict = {}


def _build_sparse_mla_hip_ext():
    if "ext" in _sparse_mla_hip_module_cache:
        return _sparse_mla_hip_module_cache["ext"]
    cache_dir = os.environ.get(
        "VLLM_SPARSE_MLA_HIP_CACHE_DIR",
        str(pathlib.Path(tempfile.gettempdir()) / "vllm_sparse_mla_hip_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    ext = load_inline(
        name="vllm_sparse_mla_hip",
        cpp_sources=[""],
        cuda_sources=[_HIP_SPARSE_MLA_DECODE_SRC],
        functions=[],
        extra_cflags=["-O3", "-DNDEBUG", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx950",
            "-DNDEBUG",
            "-Wno-c++11-narrowing",
            "-Wno-unused-result",
        ],
        with_cuda=True,
        build_directory=cache_dir,
        verbose=False,
    )
    _sparse_mla_hip_module_cache["ext"] = ext
    return ext


def _sparse_mla_hip_as_int32_1d(x):
    if x.dtype != torch.int32:
        x = x.to(torch.int32)
    if not x.is_contiguous():
        x = x.contiguous()
    return x.view(-1)


_NUM_CUS = 256
_MIN_K_PER_SPLIT = int(os.environ.get("SPARSE_MLA_HIP_MIN_K_PER_SPLIT", "32"))
_SPLIT_K_OVERRIDE = os.environ.get("SPARSE_MLA_HIP_SPLIT_K")


def _pick_split_k(num_queries, num_head_blocks, max_total_k):
    if _SPLIT_K_OVERRIDE is not None:
        return int(_SPLIT_K_OVERRIDE)
    base_tiles = max(1, num_queries * num_head_blocks)
    cu_target = max(1, _NUM_CUS // base_tiles)
    k_limit = max(1, max_total_k // _MIN_K_PER_SPLIT)
    target = max(1, min(cu_target, k_limit))
    best = 1
    for b in (1, 2, 4, 8):
        if b <= target:
            best = b
    return best


def _decode_sparse_mla_hip(
    q,
    main_cache,
    main_indices,
    main_indptr,
    scale,
    attn_sink,
    nope_head_dim,
    rope_head_dim,
    extra_cache,
    extra_indices,
    extra_indptr,
    max_main_len,
    max_extra_len,
):
    main_indices = _sparse_mla_hip_as_int32_1d(main_indices)
    main_indptr = _sparse_mla_hip_as_int32_1d(main_indptr)
    num_queries, num_heads, _ = q.shape

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        extra_indices = _sparse_mla_hip_as_int32_1d(extra_indices)
        extra_indptr = _sparse_mla_hip_as_int32_1d(extra_indptr)
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    out = torch.empty_like(q, dtype=torch.bfloat16)
    sink = attn_sink.contiguous() if attn_sink is not None else None
    q_in = q.contiguous() if not q.is_contiguous() else q

    BLOCK_H = 16
    num_head_blocks = (num_heads + BLOCK_H - 1) // BLOCK_H
    total_max_k = max_main_len + (max_extra_len if has_extra else 0)
    split_k = _pick_split_k(num_queries, num_head_blocks, total_max_k)

    _build_sparse_mla_hip_ext()

    if split_k == 1:
        torch.ops.vllm_sparse_mla_hip.decode_single(
            q_in,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            sink,
            out,
            int(main_cache.shape[1]),
            int(extra_cache.shape[1]),
            int(main_cache.shape[0] * main_cache.shape[1]),
            int(extra_cache.shape[0] * extra_cache.shape[1]),
            float(scale),
            bool(has_extra),
        )
    else:
        scratch_m = torch.empty(
            num_queries * num_head_blocks * split_k * BLOCK_H,
            device=q.device,
            dtype=torch.float32,
        )
        scratch_l = torch.empty_like(scratch_m)
        scratch_acc = torch.empty(
            num_queries * num_head_blocks * split_k * BLOCK_H * 512,
            device=q.device,
            dtype=torch.bfloat16,
        )
        torch.ops.vllm_sparse_mla_hip.decode_split(
            q_in,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            sink,
            out,
            scratch_m,
            scratch_l,
            scratch_acc,
            int(main_cache.shape[1]),
            int(extra_cache.shape[1]),
            int(main_cache.shape[0] * main_cache.shape[1]),
            int(extra_cache.shape[0] * extra_cache.shape[1]),
            float(scale),
            bool(has_extra),
            int(split_k),
        )
    return out


def rocm_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "ROCm sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_decode",
    )
    assert nope_head_dim == 448
    assert rope_head_dim == 64

    if swa_ragged_indices is None or swa_ragged_indptr is None:
        main_indices_dense = swa_indices.reshape(swa_indices.shape[0], -1)
        lengths = (
            swa_lens
            if swa_lens is not None
            else ((main_indices_dense >= 0).sum(dim=-1, dtype=torch.int32))
        )
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices_dense,
            lengths,
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
        )
    else:
        main_ragged_indices = swa_ragged_indices
        main_ragged_indptr = swa_ragged_indptr

    has_extra = not swa_only
    extra_cache = None
    extra_ragged_indices = None
    extra_ragged_indptr = None
    if has_extra:
        assert kv_cache is not None
        assert kv_cache.dtype == torch.uint8
        extra_cache = kv_cache
        if topk_ragged_indices is None or topk_ragged_indptr is None:
            assert topk_indices is not None
            ex_dense = topk_indices.reshape(topk_indices.shape[0], -1)
            lengths = (
                topk_lens
                if topk_lens is not None
                else ((ex_dense >= 0).sum(dim=-1, dtype=torch.int32))
            )
            extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
                ex_dense,
                lengths,
                num_rows=kv_cache.shape[0] * kv_cache.shape[1],
            )
        else:
            extra_ragged_indices = topk_ragged_indices
            extra_ragged_indptr = topk_ragged_indptr

    if torch.cuda.is_current_stream_capturing():
        max_main_len = swa_indices.shape[-1]
        max_extra_len = max_main_len if has_extra else 0
    else:
        max_main_len = int(swa_lens.max().item()) if swa_lens is not None else 0
        max_extra_len = 0
        if has_extra and topk_lens is not None:
            max_extra_len = int(topk_lens.max().item())

    attn_out = _decode_sparse_mla_hip(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
        max_main_len=max_main_len,
        max_extra_len=max_extra_len,
    )
    output.copy_(attn_out.to(output.dtype))
