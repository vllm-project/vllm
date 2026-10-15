# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA as one graph-splitting custom op: index-cache write, region
compression, scoring, top-k and sparse attention over the paged K/V."""

from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.models.step5.common.csa_cache import Step5CSAMetadata
from vllm.models.step5.common.ops.csa import select_regions
from vllm.models.step5.common.ops.csa_paged import (
    compress_regions_paged,
    indexer_scores_paged,
    sparse_attention_paged,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

if TYPE_CHECKING:
    from vllm.models.step5.amd.model import Step5Attention


def _token_requests_and_positions(
    md: Step5CSAMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = md.num_actual_tokens
    device = md.seq_lens.device
    query_lens = (md.query_start_loc[1:] - md.query_start_loc[:-1])[: md.num_reqs]
    req = torch.repeat_interleave(
        torch.arange(md.num_reqs, device=device), query_lens, output_size=n
    )
    pos = (
        (md.seq_lens[: md.num_reqs] - query_lens).index_select(0, req)
        + torch.arange(n, device=device)
        - md.query_start_loc.index_select(0, req)
    )
    return req.to(torch.int32), pos.to(torch.int32)


def _completed_regions(md: Step5CSAMetadata, region_size: int) -> torch.Tensor:
    """``(request, region)`` pairs whose last token was cached this step.

    Built on device from exact ``seq_lens``, padded with ``-1`` rows to a
    host-known upper bound so the host never reads the count.
    """
    q_lens = (md.query_start_loc[1:] - md.query_start_loc[:-1])[: md.num_reqs]
    seq = md.seq_lens[: md.num_reqs]
    first = torch.div(seq - q_lens, region_size, rounding_mode="floor")
    counts = (torch.div(seq, region_size, rounding_mode="floor") - first).clamp_min(0)
    ends = torch.cumsum(counts, 0)
    capacity = md.num_actual_tokens // region_size + md.num_reqs
    slot = torch.arange(capacity, device=seq.device)
    req = torch.searchsorted(ends, slot, right=True)
    live = req < md.num_reqs
    req = req.clamp_max(md.num_reqs - 1)
    region = first[req] + slot - (ends - counts)[req]
    work = torch.stack([torch.where(live, req, -1), torch.where(live, region, -1)], -1)
    return work.to(torch.int32)


def step5_csa_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    del kv_cache_dummy_dep
    layer: Step5Attention = get_forward_context().no_compile_layers[
        _resolve_layer_name(layer_name)
    ]
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        output.zero_()  # profiling run: caches are not bound yet
        return
    indexer = layer.indexer
    assert indexer is not None
    index_md = metadata[indexer.index_cache.prefix]
    summary_md = metadata[indexer.summary_cache.prefix]
    main_md = metadata[layer.attn.layer_name]
    assert isinstance(index_md, Step5CSAMetadata)
    assert isinstance(summary_md, Step5CSAMetadata)
    assert isinstance(main_md, RocmAttentionMetadata)
    n = index_md.num_actual_tokens
    output[n:].zero_()
    if n == 0:
        return

    index_q, index_k, index_z, weights = indexer(hidden_states[:n], positions[:n])
    index_q, weights = layer.local_index_groups(index_q, weights)

    index_pages = indexer.index_cache.pages
    ops.concat_and_cache_mla(
        index_k.to(index_pages.dtype),
        index_z.to(index_pages.dtype),
        index_pages,
        index_md.slot_mapping[:n],
        "auto",
        indexer.index_cache_scale,
    )

    region_size = indexer.region_size
    summary_pages = indexer.summary_cache.pages
    compress_regions_paged(
        index_pages,
        summary_pages,
        index_md.block_table,
        summary_md.block_table,
        _completed_regions(index_md, region_size),
        region_size=region_size,
    )

    req, pos = _token_requests_and_positions(index_md)
    max_regions = index_md.max_seq_len // region_size
    scores = indexer_scores_paged(
        index_q,
        weights,
        summary_pages,
        summary_md.block_table,
        req,
        pos,
        max_regions=max_regions,
        region_size=region_size,
    )
    regions = select_regions(scores, pos, region_size, indexer.topk)
    key_cache, value_cache = layer.attn.kv_cache.transpose(1, 2).split(
        layer.head_dim, dim=-1
    )
    sparse_attention_paged(
        query[:n],
        key_cache,
        value_cache,
        regions,
        main_md.block_table,
        req,
        pos,
        output[:n],
        region_size=region_size,
        scale=layer.scaling,
    )


def step5_csa_attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="step5_csa_attention",
    op_func=step5_csa_attention,
    mutates_args=["output"],
    fake_impl=step5_csa_attention_fake,
)
