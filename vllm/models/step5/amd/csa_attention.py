# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA as one graph-splitting custom op: region summaries, scoring,
top-k and sparse attention over the paged K/V."""

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.models.step5.amd.csa_cache import Step5CSAMetadata
from vllm.models.step5.amd.ops.csa_paged import sparse_attention_paged
from vllm.models.step5.common.ops.sparse_decode_attention import (
    sparse_decode_attention,
    sparse_decode_splits,
)
from vllm.models.step5.common.ops.sparse_indexer import (
    pack_summary_pages,
    select_all_regions,
    select_regions,
)
from vllm.models.step5.common.ops.sparse_indexer_scores import (
    decode_summary_scores,
    paged_mqa_logits_sm90,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

if TYPE_CHECKING:
    from vllm.models.step5.amd.model import Step5Attention


@eager_break_during_capture
def step5_csa_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    gate: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    del kv_cache_dummy_dep
    layer: Step5Attention = get_forward_context().no_compile_layers[
        _resolve_layer_name(layer_name)
    ]
    indexer = layer.indexer
    assert indexer is not None
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        # Profiling run: caches are not bound yet; claim the scoring workspace.
        torch.empty(indexer.workspace_bytes, dtype=torch.uint8, device=output.device)
        output.zero_()
        return
    md = metadata[layer.sparse_summary_cache.layer_name]
    main_md = metadata[layer.attn.layer_name]
    assert isinstance(md, Step5CSAMetadata)
    assert isinstance(main_md, RocmAttentionMetadata)
    n = md.num_actual_tokens
    output[n:].zero_()
    if n == 0:
        return

    cache = cast(Any, layer.attn.impl).summary_cache
    ring = layer.csa_tail_ring
    slots = md.slot_mapping[:n]
    uniform = md.prefill_groups is None
    # Without drafts every decode token closes at most its own region, so the
    # summary update fuses into the indexer kernel.
    fuse = uniform and indexer.num_speculative_tokens == 0
    q8, k, z, w = indexer(
        hidden_states[:n], positions[:n], (cache, ring, slots) if fuse else None
    )
    groups = q8.shape[1]
    rows = n * groups

    kv_cache = layer.attn.kv_cache
    k_scale = v_scale = None
    if kv_cache.dtype == torch.uint8:
        kv_cache = kv_cache.view(current_platform.fp8_dtype())
        k_scale, v_scale = layer.attn._k_scale, layer.attn._v_scale

    if uniform:
        # Decode or MTP verify: each request owns max_query_len consecutive rows.
        query_len = md.max_query_len
        reqs = n // query_len
        table = md.block_table[:reqs]
        if indexer.num_speculative_tokens:
            cache.update_verify(
                k, z, slots, md.query_start_loc[: reqs + 1], md.seq_lens[:reqs], ring
            )
        capacity = min(indexer.max_regions, table.shape[1] * cache.mean.shape[1])
        scores, decode = decode_summary_scores(
            q8,
            w,
            cache.mean,
            table,
            slots,
            md.seq_lens[:reqs],
            capacity,
            num_sms=indexer.num_sms,
            row_groups=groups,
            query_len=query_len,
        )
        regions = select_regions(
            scores,
            decode.current,
            indexer.topk,
            indexer.topk_backend,
            topk_lengths=decode.topk_lengths,
            sort=False,
        )
        sparse_decode_attention(
            query[:n].reshape(rows, -1, layer.head_dim),
            kv_cache,
            regions,
            main_md.block_table,
            decode.positions,
            decode.active,
            output[:n].view(rows, -1, layer.head_dim),
            scale=layer.scaling,
            query_len=query_len,
            splits=sparse_decode_splits(rows),
            k_scale=k_scale,
            v_scale=v_scale,
            gate=None if gate is None else gate[:n],
        )
        return

    assert md.prefill_groups is not None
    cache.update_prefill(k, z, md.prefill_groups, md.positions, ring)
    if md.prefill_indexer is None:
        # Every earlier region fits in top-k: nothing to score.
        own = torch.div(md.positions, indexer.region_size, rounding_mode="floor")
        regions = select_all_regions(
            own.repeat_interleave(groups) if groups > 1 else own, indexer.topk
        ).view(n, groups, -1)
    else:
        regions = torch.empty(
            (n, groups, indexer.topk + 1), dtype=torch.int64, device=query.device
        )
        packed, _ = pack_summary_pages(
            cache.mean, md.block_table[: md.num_reqs], md.num_regions
        )
        start = 0
        for chunk in md.prefill_indexer:
            end = start + chunk.current.numel()
            for g in range(groups):
                scores = paged_mqa_logits_sm90(
                    q=q8[start:end, g].unsqueeze(1),
                    kv_fused=packed,
                    weights=w[start:end, g],
                    seq_lens=chunk.lengths,
                    block_tables=chunk.block_table,
                    max_seq_len=md.num_regions,
                )
                regions[start:end, g] = select_regions(
                    scores,
                    chunk.current,
                    indexer.topk,
                    indexer.topk_backend,
                    topk_lengths=chunk.topk_lengths,
                    sort=False,
                )
            start = end
    key_cache, value_cache = kv_cache.transpose(1, 2).split(layer.head_dim, dim=-1)
    sparse_attention_paged(
        query[:n],
        key_cache,
        value_cache,
        regions,
        main_md.block_table,
        md.token_req,
        md.positions,
        output[:n],
        region_size=indexer.region_size,
        scale=layer.scaling,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if gate is not None:
        output[:n] *= gate[:n].unsqueeze(-1).sigmoid().to(output.dtype)


def step5_csa_attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    gate: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="step5_csa_attention",
    op_func=step5_csa_attention,
    mutates_args=["output"],
    fake_impl=step5_csa_attention_fake,
)
