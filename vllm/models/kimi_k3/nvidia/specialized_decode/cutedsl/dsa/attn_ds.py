# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8_ds_mla sparse core of a K3.1 DSA layer (FLASHMLA_SPARSE), as the stock decode:
the BF16 query rows [B, 64 (padded heads), 576], the logical -> physical top-k and
FlashMLA sparse decode over the packed fp8_ds_mla blocks; under DCP, FlashMLA over
this rank's compacted slots."""

import torch

# Top-k entries per query row (the K3.1 indexer's).
TOPK = 2048


def flashmla_core(impl, query, cache, topk, metadata):
    """As the stock decode: the packed blocks as 64-row pages of flat rows, the
    logical top-k -> physical, FlashMLA sparse decode -> [B, 64, 512]."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import _fp8_kv_pages
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache

    pages, stride_rows = _fp8_kv_pages(cache, metadata.block_size)
    impl.record_logical_topk_ready()
    physical = impl._convert_logical_to_physical_topk(
        topk,
        metadata,
        block_stride_rows=stride_rows,
        return_valid_counts=False,
    )
    kernel_metadata = metadata.fp8_extra_metadata
    latent, _ = flash_mla_with_kvcache(
        q=query.unsqueeze(0),
        k_cache=pages.view(torch.uint8).unsqueeze(-2),
        block_table=kernel_metadata.dummy_block_table,
        head_dim_v=512,
        cache_seqlens=kernel_metadata.cache_lens,
        tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
        is_fp8_kvcache=True,
        indices=physical.unsqueeze(0),
        softmax_scale=impl.softmax_scale,
    )
    return latent[0]


def flashmla_dcp_core(impl, query, pages, slots, valid, out, table, cache_lens):
    """DCP: FlashMLA sparse decode of every rank's BF16 query heads (the gathered q
    mailbox) over this rank's compacted ``slots`` (natural-log LSE; a fresh
    scheduler metadata per call follows the ``valid`` lengths) into ``out``, its
    LSE and ``valid`` (DcpValue drops the rows this rank owns none of)."""
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    rows, lse = flash_mla_with_kvcache(
        q=query.unsqueeze(1),
        k_cache=pages.view(torch.uint8).unsqueeze(-2),
        block_table=table,
        head_dim_v=512,
        cache_seqlens=cache_lens,
        tile_scheduler_metadata=get_mla_metadata()[0],
        is_fp8_kvcache=True,
        indices=slots.unsqueeze(1),
        topk_length=valid,
        softmax_scale=impl.softmax_scale,
        out=out,
    )
    return rows, lse, valid
