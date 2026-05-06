# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import math
from importlib.util import find_spec

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942
else:
    _ON_GFX942 = False


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
        if _ON_GFX942:
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


def _topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
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
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    device = hidden_states.device if k is None else k.device
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=device, dtype=torch.uint8
    )
    fp8_dtype = current_platform.fp8_dtype()
    _k_fp8 = _flattened_kv[..., :head_dim].view(fp8_dtype).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer_native(
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
            topk_indices.copy_(_topk_indices_torch(logits, topk_tokens))

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

        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        topk_indices.copy_(_topk_indices_torch(logits, topk_tokens)[:num_decode_tokens])

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


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
) -> torch.Tensor:
    return rocm_aiter_sparse_attn_indexer_native(
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
        skip_k_cache_insert=False,
    )


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
_DSV4_SPARSE_NUM_WARPS = 4


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, "
        f"got {head_dim}"
    )
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


def _mask_sparse_indices(
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    num_rows: int,
) -> torch.Tensor:
    indices = indices.reshape(indices.shape[0], -1).clone()
    if topk_length is not None:
        valid_mask = torch.arange(indices.shape[-1], device=indices.device).view(
            1, -1
        ) < topk_length.reshape(-1, 1)
        indices[~valid_mask] = -1
    invalid_mask = (indices < 0) | (indices >= num_rows)
    indices[invalid_mask] = -1
    return indices.contiguous()


@triton.jit
def _load_prefill_row(
    kv_ptr,
    row_idx,
    kv_stride0,
    nope_offsets,
    rope_offsets,
    nope_mask,
    NOPE_DIM: tl.constexpr,
):
    k_nope = tl.load(
        kv_ptr + row_idx * kv_stride0 + nope_offsets,
        mask=nope_mask,
        other=0.0,
    ).to(tl.float32)
    k_rope = tl.load(
        kv_ptr + row_idx * kv_stride0 + NOPE_DIM + rope_offsets,
    ).to(tl.float32)
    k_nope = tl.where(k_nope == k_nope, k_nope, 0.0)
    k_rope = tl.where(k_rope == k_rope, k_rope, 0.0)
    return k_nope, k_rope


@triton.jit
def _load_decode_row(
    cache_ptr,
    row_idx,
    cache_stride0,
    block_size,
    nope_offsets,
    rope_offsets,
    nope_mask,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    block_idx = row_idx // block_size
    pos_in_block = row_idx % block_size

    cache_block_ptr = cache_ptr + block_idx.to(tl.int64) * cache_stride0
    token_data_ptr = cache_block_ptr + pos_in_block * 576
    token_scale_ptr = cache_block_ptr + block_size * 576 + pos_in_block * 8

    x_uint8 = tl.load(token_data_ptr + nope_offsets, mask=nope_mask, other=0)
    # The packed NoPE bytes use the platform's native FP8 encoding.
    if IS_FNUZ:
        x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
    else:
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
    encoded_scales = tl.load(
        token_scale_ptr + nope_offsets // 64,
        mask=nope_mask,
        other=127,
    )
    scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
    k_nope = (x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)).to(tl.float32)
    k_nope = tl.where(nope_mask, k_nope, 0.0)
    k_nope = tl.where(k_nope == k_nope, k_nope, 0.0)

    rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    k_rope = tl.load(rope_ptr + rope_offsets).to(tl.float32)
    k_rope = tl.where(k_rope == k_rope, k_rope, 0.0)
    return k_nope, k_rope


@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    kv_stride0,
    indices_stride0,
    out_stride0,
    out_stride1,
    num_kv,
    topk,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_idx * q_stride1
    q_nope = tl.load(q_row_ptr + nope_offsets, mask=nope_mask, other=0.0).to(
        tl.float32
    )
    q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)

    neg_inf = float("-inf")
    max_score = neg_inf

    for topk_idx in range(topk):
        kv_idx = tl.load(indices_ptr + query_idx * indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < num_kv)
        if valid:
            k_nope, k_rope = _load_prefill_row(
                kv_ptr,
                kv_idx,
                kv_stride0,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
            )
            score = tl.sum(q_nope * k_nope) + tl.sum(q_rope * k_rope)
            score *= scale
            max_score = tl.maximum(max_score, score)

    has_valid = max_score != neg_inf
    max_score_safe = tl.where(has_valid, max_score, 0.0)
    sum_exp = tl.zeros((), dtype=tl.float32)
    acc_nope = tl.zeros((NOPE_BLOCK,), dtype=tl.float32)
    acc_rope = tl.zeros((ROPE_DIM,), dtype=tl.float32)

    for topk_idx in range(topk):
        kv_idx = tl.load(indices_ptr + query_idx * indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < num_kv)
        if valid:
            k_nope, k_rope = _load_prefill_row(
                kv_ptr,
                kv_idx,
                kv_stride0,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
            )
            score = tl.zeros((), dtype=tl.float32)
            for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                chunk_mask = (nope_offsets >= chunk_start) & (
                    nope_offsets < chunk_start + 64
                )
                score += tl.sum(
                    tl.where(chunk_mask, q_nope * k_nope, 0.0),
                    axis=0,
                )
            score += tl.sum(q_rope * k_rope, axis=0)
            score *= scale
            weight = tl.where(has_valid, tl.exp(score - max_score_safe), 0.0)
            sum_exp += weight
            acc_nope += weight * k_nope
            acc_rope += weight * k_rope

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        sink_term = tl.where(has_valid, tl.exp(sink - max_score_safe), 0.0)
        denom = sum_exp + sink_term
    else:
        denom = sum_exp

    inv_denom = tl.where(has_valid, 1.0 / denom, 0.0)
    out_row_ptr = out_ptr + query_idx * out_stride0 + head_idx * out_stride1
    tl.store(
        out_row_ptr + nope_offsets,
        (acc_nope * inv_denom).to(tl.bfloat16),
        mask=nope_mask,
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets,
        (acc_rope * inv_denom).to(tl.bfloat16),
    )


@triton.jit
def _sparse_attn_decode_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    main_indices_stride0,
    extra_indices_stride0,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    main_topk,
    extra_topk,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_idx * q_stride1
    q_nope = tl.load(q_row_ptr + nope_offsets, mask=nope_mask, other=0.0).to(
        tl.float32
    )
    q_rope = tl.load(q_row_ptr + NOPE_DIM + rope_offsets).to(tl.float32)

    neg_inf = float("-inf")
    max_score = neg_inf
    sum_exp = tl.zeros((), dtype=tl.float32)
    acc_nope = tl.zeros((NOPE_BLOCK,), dtype=tl.float32)
    acc_rope = tl.zeros((ROPE_DIM,), dtype=tl.float32)

    for topk_idx in range(main_topk):
        kv_idx = tl.load(main_indices_ptr + query_idx * main_indices_stride0 + topk_idx)
        valid = (kv_idx >= 0) & (kv_idx < main_num_rows)
        if valid:
            k_nope, k_rope = _load_decode_row(
                main_cache_ptr,
                kv_idx,
                main_cache_stride0,
                main_block_size,
                nope_offsets,
                rope_offsets,
                nope_mask,
                NOPE_DIM,
                ROPE_DIM,
                IS_FNUZ,
            )
            score = tl.zeros((), dtype=tl.float32)
            for chunk_start in tl.static_range(0, NOPE_DIM, 64):
                chunk_mask = (nope_offsets >= chunk_start) & (
                    nope_offsets < chunk_start + 64
                )
                score += tl.sum(
                    tl.where(chunk_mask, q_nope * k_nope, 0.0),
                    axis=0,
                )
            score += tl.sum(q_rope * k_rope, axis=0)
            score *= scale

            new_max = tl.maximum(max_score, score)
            old_scale = tl.exp(max_score - new_max)
            new_weight = tl.exp(score - new_max)
            sum_exp = sum_exp * old_scale + new_weight
            acc_nope = acc_nope * old_scale + new_weight * k_nope
            acc_rope = acc_rope * old_scale + new_weight * k_rope
            max_score = new_max

    if HAS_EXTRA:
        for topk_idx in range(extra_topk):
            kv_idx = tl.load(
                extra_indices_ptr + query_idx * extra_indices_stride0 + topk_idx
            )
            valid = (kv_idx >= 0) & (kv_idx < extra_num_rows)
            if valid:
                k_nope, k_rope = _load_decode_row(
                    extra_cache_ptr,
                    kv_idx,
                    extra_cache_stride0,
                    extra_block_size,
                    nope_offsets,
                    rope_offsets,
                    nope_mask,
                    NOPE_DIM,
                    ROPE_DIM,
                    IS_FNUZ,
                )
                score = tl.sum(q_nope * k_nope) + tl.sum(q_rope * k_rope)
                score *= scale

                new_max = tl.maximum(max_score, score)
                old_scale = tl.exp(max_score - new_max)
                new_weight = tl.exp(score - new_max)
                sum_exp = sum_exp * old_scale + new_weight
                acc_nope = acc_nope * old_scale + new_weight * k_nope
                acc_rope = acc_rope * old_scale + new_weight * k_rope
                max_score = new_max

    has_valid = max_score != neg_inf
    max_score_safe = tl.where(has_valid, max_score, 0.0)

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        sink_term = tl.where(has_valid, tl.exp(sink - max_score_safe), 0.0)
        denom = sum_exp + sink_term
    else:
        denom = sum_exp

    inv_denom = tl.where(has_valid, 1.0 / denom, 0.0)
    out_row_ptr = out_ptr + query_idx * out_stride0 + head_idx * out_stride1
    tl.store(
        out_row_ptr + nope_offsets,
        (acc_nope * inv_denom).to(tl.bfloat16),
        mask=nope_mask,
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets,
        (acc_rope * inv_denom).to(tl.bfloat16),
    )


def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 2, f"expected indices=[sq,topk], got {indices.shape}"
    assert q.is_cuda and kv.is_cuda and indices.is_cuda

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_triton",
    )

    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_prefill_kernel[(num_queries, num_heads)](
        q,
        kv,
        indices,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        kv.shape[0],
        indices.shape[-1],
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        num_warps=_DSV4_SPARSE_NUM_WARPS,
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
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 2, (
        f"expected main_indices=[b,topk], got {main_indices.shape}"
    )
    assert q.is_cuda and main_cache.is_cuda and main_indices.is_cuda

    q = q.contiguous()
    main_cache = main_cache.contiguous()
    main_indices = main_indices.contiguous()
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    has_extra = extra_cache is not None and extra_indices is not None
    if has_extra:
        assert extra_cache is not None and extra_indices is not None
        extra_cache = extra_cache.contiguous()
        extra_indices = extra_indices.contiguous()
    else:
        extra_cache = main_cache
        extra_indices = main_indices[:, :1]

    num_queries, num_heads, head_dim = q.shape
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_decode_triton",
    )

    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_decode_kernel[(num_queries, num_heads)](
        q,
        main_cache,
        main_indices,
        extra_cache,
        extra_indices,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        main_indices.stride(0),
        extra_indices.stride(0),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        main_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        IS_FNUZ=current_platform.is_fp8_fnuz(),
        num_warps=_DSV4_SPARSE_NUM_WARPS,
    )
    return out


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
    indices_2d = _mask_sparse_indices(indices, topk_length, kv.shape[0])
    output_chunk = _rocm_sparse_attn_prefill_triton(
        q=q,
        kv=kv.squeeze(1),
        indices=indices_2d,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )
    output.copy_(output_chunk.to(output.dtype))


def rocm_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "ROCm Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_decode",
    )

    main_indices = _mask_sparse_indices(
        swa_indices,
        swa_lens,
        swa_k_cache.shape[0] * swa_k_cache.shape[1],
    )

    extra_cache = None
    extra_indices = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None
        assert kv_cache.dtype == torch.uint8, (
            "ROCm Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        extra_indices = _mask_sparse_indices(
            topk_indices,
            topk_lens,
            kv_cache.shape[0] * kv_cache.shape[1],
        )

    attn_out = _rocm_sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_indices,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
    )
    output.copy_(attn_out.to(output.dtype))
