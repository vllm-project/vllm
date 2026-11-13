# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention side of the MiniMax-M3.1 NVFP4 sparse layers on amd_kernels: the
``qkv`` route (QKV GEMM + norms + RoPE + NVFP4 inserts), the ``attention`` route
(sparse decode / prefill; pure decode steps write o_proj's MXFP8 input) and the
``indexer`` route (the decode top-16, run inside the same layer's sparse decode, and
the prefill top-16). The library owns the kernels, their dtypes and warm-up; this
module passes tensors."""

import functools
import threading
import time
from typing import Any

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8 import rocm_akr
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.minimax_m3.amd import akr_config
from vllm.v1.worker.ubatching import dbo_enabled

logger = init_logger(__name__)


@functools.cache
def kernels():
    """The amd_kernels layer entries, resolved once (the eager attention calls them)."""
    from amd_kernels.attention import nvfp4_sparse_layer_gqa16_d128_p128_top16

    return nvfp4_sparse_layer_gqa16_d128_p128_top16


def qkv_on(attn: torch.nn.Module) -> bool:
    """Whether the layer's QKV step (QKV GEMM + norms + RoPE + NVFP4 inserts) runs
    amd_kernels; the library names why not (logged once)."""
    if not (akr_config.on("qkv") and attn.use_nvfp4_kv):
        return False
    from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
        insert_blocker,
    )

    rope = attn.rotary_emb
    why = insert_blocker(
        attn.num_heads,
        attn.num_kv_heads,
        attn.num_idx_heads,
        rope.rotary_dim,
        head_dim=attn.head_dim,
        index_head_dim=attn.idx_head_dim,
        neox=rope.is_neox_style,
    )
    if why is not None:
        logger.warning_once("amd_kernels qkv stays stock: %s", why)
    return why is None


def qkv_norm_rope_nvfp4_insert(
    attn: torch.nn.Module,
    x,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's QKV + norms + RoPE + NVFP4 inserts -> (q, index-q): the library's
    one call from amd_kernels' codes, else the layer's linear, then the insert."""
    proj = attn.qkv_proj
    ins = (
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.index_q_norm.weight,
        attn.index_k_norm.weight,
        attn.rotary_emb.cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        attn.kv_cache,
        attn.indexer.index_cache.kv_cache,
        attn._k_scale,
        attn._v_scale,
    )
    kw = dict(
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        num_index_heads=attn.num_idx_heads,
        eps=attn.q_norm.variance_epsilon,
        index_q_dtype=attn.indexer.index_cache.query_dtype,
    )
    if isinstance(x, QuantizedActivation) and rocm_akr.is_akr_layer(proj):
        return kernels().mxfp8_gemm_qkv_norm_rope_nvfp4_insert(
            x.data,
            x.scale,
            proj.weight,
            proj.weight_scale,
            x.orig_shape[:-1].numel(),
            x.orig_dtype,
            *ins,
            **kw,
            # one counter set per stream and capture: not in breakable graphs
            one_launch=oproj_mxfp8_mode_ok(),
        )
    out, _ = proj(x)
    return kernels().qkv_norm_rope_nvfp4_insert(out, *ins, **kw)


# ---- the attention's MXFP8 output as o_proj's input


def oproj_input(quant, rows: int, k: int):
    """``oproj_quant``'s buffers, written by the attention, as o_proj's input."""
    codes, scales, tile = quant
    data = codes.view(torch.float8_e4m3fn)
    if not tile:
        data = data[:rows]
    return QuantizedActivation(
        data, scales, torch.bfloat16, torch.Size((rows, k)), kMxfp8Dynamic
    )


def oproj_mxfp8_mode_ok() -> bool:
    """False while a breakable PIECEWISE graph is captured: its segments replay for
    any batch of the size, so the choice must not depend on the dummy batch."""
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
    )


def oproj_quant(attn: torch.nn.Module, num_tokens: int):
    """(codes, scales, tile) for the NVFP4 sparse decode's MXFP8 output on a pure
    decode step; None: bf16 output and o_proj's quantizer."""
    if not (
        akr_config.on("attention")
        and rocm_akr.is_akr_layer(attn.o_proj)
        and oproj_mxfp8_mode_ok()
    ):
        return None
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    md: Any = attn_metadata.get(attn.layer_name)  # the backend's metadata class
    if md is None or md.num_prefills > 0 or md.num_decodes == 0:
        return None
    n, k = attn.o_proj.weight.shape
    return kernels().quant_out_buffers(num_tokens, n, k, attn.o_proj.weight.device)


# ---- indexer + decode attention

_IDX: Any = None  # amd_kernels IndexDecodeTop16 (init_and_warm_up)
# The indexer's decode half, run by the same layer's sparse decode; per thread (DBO).
_pending = threading.local()


def index_decode(
    indexer: torch.nn.Module,  # MiniMaxM3IndexerRocmNvfp4Impl
    iq: torch.Tensor,  # [rows, index heads, 128] e4m3
    index_cache: torch.Tensor,
    d,  # MiniMaxM3IndexerDecodeMetadata
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Record the decode rows' top-16 for the same layer's ``decode``."""
    assert getattr(_pending, "rec", None) is None, "unconsumed index record"
    index_args = (iq, index_cache, d.block_table, d.seq_lens, init_blocks, local_blocks)
    _pending.rec = (indexer, index_args, d.max_seq_len)


@functools.cache
def _prefill_top16():
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import prefill_top16

    return prefill_top16


def index_prefill_blocker(indexer: torch.nn.Module) -> str | None:
    """Why the library's prefill top-16 cannot serve this indexer (None: it can)."""
    return _prefill_top16().index_prefill_top16_blocker(
        num_index_heads=indexer.num_index_heads,
        index_head_dim=indexer.index_head_dim,
        block_size=indexer.block_size,
        topk_blocks=indexer.topk_blocks,
    )


def index_prefill(
    indexer: torch.nn.Module,  # MiniMaxM3IndexerRocmNvfp4Impl
    iq: torch.Tensor,  # [rows, index heads, 128] e4m3
    index_cache: torch.Tensor,
    p,  # the prefill half of MiniMaxM3IndexerMSAMetadata
    num_valid_pages: torch.Tensor,  # [rows] int32
    topk: torch.Tensor,  # [index heads, rows, 16] int32, written
) -> None:
    """The prefill rows' top-16: scorer + top-k in one library call."""
    _prefill_top16().index_prefill_top16(
        iq,
        index_cache,
        p.block_table,
        p.cu_seqlens_q,
        p.seq_lens,
        indexer.init_blocks,
        indexer.local_blocks,
        topk,
        max_query_len=p.max_query_len,
        max_seq_len=p.max_seq_len,
        num_valid_pages=num_valid_pages,
    )


def decode(
    layer: Any,  # MiniMaxM3SparseAttention
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,  # [Hkv, rows, 16] int32
    d,  # MiniMaxM3SparseDecodeMetadata
    sm_scale: float,
    out: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    quant_out=None,
) -> None:
    """The sparse decode: with this layer's indexer recorded, index top-16 +
    attention as one library call; else the attention alone."""
    attn = (q, kv_cache, topk, d.block_table, d.seq_lens, sm_scale, out)
    kw = dict(
        k_scale=k_scale,
        v_scale=v_scale,
        decode_query_len=d.decode_query_len,
        quant_out=quant_out,
    )
    rec = getattr(_pending, "rec", None)
    if rec is None:
        kernels().sparse_decode(*attn, **kw)
        return
    _pending.rec = None
    indexer, index_args, max_seq_len = rec
    assert indexer is layer.indexer.impl, "index record of another layer"
    kernels().index_sparse_decode(
        _IDX,
        *index_args,
        *attn,
        max_seq_len=max_seq_len,
        split=not dbo_enabled(),  # no all-reduce orders the push under DBO
        **kw,
    )


# ---- warm-up


def init_and_warm_up(
    vllm_config, num_heads: int, num_kv_heads: int, num_index_heads: int
) -> None:
    """At model build: the decode indexer (with ``indexer_decode_cp`` a TP
    collective, every rank in order) and every variant the switched paths launch."""
    global _IDX
    if not any(akr_config.on(r) for r in ("qkv", "attention", "indexer")):
        return
    t0 = time.perf_counter()
    rank = group = None
    if akr_config.on("indexer") and vllm_config.attention_config.indexer_decode_cp:
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        rank, group = tp.rank_in_group, tp.cpu_group
    spec = vllm_config.speculative_config
    ql = 1 + (spec.num_speculative_tokens if spec is not None else 0)
    _IDX = kernels().init(
        torch.device("cuda", torch.accelerator.current_device_index()),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        query_len=ql,
        max_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_seq_len=vllm_config.model_config.max_model_len,
        insert=akr_config.on("qkv"),
        attention=akr_config.on("attention"),
        index=akr_config.on("indexer"),
        quant_out=akr_config.on("gemm"),
        rank=rank,
        group=group,
    )
    logger.info(
        "amd_kernels qkv / attention / indexer = %d%d%d (decode rows per request %d, "
        "four-GPU decode indexer: %s); warm-up %.1f s",
        akr_config.on("qkv"),
        akr_config.on("attention"),
        akr_config.on("indexer"),
        ql,
        "off" if _IDX is None else (_IDX.split_off or "on"),
        time.perf_counter() - t0,
    )
