# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA as one graph-splitting custom op: index-cache write, region
compression, scoring, top-k and sparse attention over the paged K/V."""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
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

# Peak bytes per (row, group, region) across scoring and select_regions: FP32
# scores, the masked copy and the sort's values, int64 indices and workspace.
_SCORE_BYTES = 48


def _rows_per_chunk(groups: int, max_regions: int) -> int:
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    return max(1, budget // (_SCORE_BYTES * groups * max(max_regions, 1)))


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
        # Profiling run: caches are not bound yet; claim the scoring budget.
        torch.empty(
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=output.device,
        )
        output.zero_()
        return
    indexer = layer.indexer
    assert indexer is not None
    index_md = metadata[indexer.index_cache.prefix]
    summary_md = metadata[indexer.summary_cache.prefix]
    main_md = metadata[layer.attn.layer_name]
    assert isinstance(index_md, Step5CSAMetadata)
    assert isinstance(summary_md, Step5CSAMetadata)
    assert isinstance(main_md, RocmAttentionMetadata)
    req, pos = index_md.token_req, index_md.positions
    work = index_md.completed_regions
    assert req is not None and pos is not None and work is not None
    n = index_md.num_actual_tokens
    output[n:].zero_()
    if n == 0:
        return

    index_q, index_k, index_z, weights = indexer(hidden_states[:n], positions[:n])

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
        work,
        region_size=region_size,
    )

    max_regions = index_md.max_seq_len // region_size
    regions = torch.empty(
        (n, index_q.shape[1], indexer.topk + 1), dtype=torch.int32, device=query.device
    )
    step = _rows_per_chunk(index_q.shape[1], max_regions)
    for start in range(0, n, step):
        rows = slice(start, min(start + step, n))
        scores = indexer_scores_paged(
            index_q[rows],
            weights[rows],
            summary_pages,
            summary_md.block_table,
            req[rows],
            pos[rows],
            max_regions=max_regions,
            region_size=region_size,
        )
        regions[rows] = select_regions(scores, pos[rows], region_size, indexer.topk)
        del scores
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
