# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route `indexer`: the decode rows of DeepSeek-V4.1's MXFP4 top-512 indexer on
amd_kernels.

One entry scores the paged MXFP4 keys, selects the top 512 and writes the
sparse attention's ragged slot lists itself, which this layer then publishes as
its `_topk_ragged_cache` (the stock packer is skipped). The dense layers score
the whole context (the candidate source also writes its candidate blocks); the
candidate consumers score only the source's candidates. Prefill rows of the
same step stay on the stock op (`skip_decode`).

A layer stays stock when its longest row, `max_model_len / compress_ratio`
compressed entries (or `indexer_max_ctx`), exceeds the kernel's limit. Per
step: no uniform decode length, sharded decode rows or a row count outside the
gate run the stock op.
"""

from types import SimpleNamespace
from typing import Any

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

logger = init_logger(__name__)

TOPK = 512
MAX_ROWS = 1024


def unsupported_reason(layer, bound: int, page: int, next_n: int) -> str | None:
    indexer = layer.indexer
    if not getattr(indexer.k_cache, "rocm_mxfp4", False) or not indexer.use_fp4_kv:
        return "the indexer cache is not the ROCm MXFP4 layout"
    if bound > akr_config.INDEXER_MAX_ENTRIES:
        return (
            f"rows up to {bound} entries exceed {akr_config.INDEXER_MAX_ENTRIES}; "
            "lower --max-model-len or set amd_kernels.indexer_max_ctx"
        )
    return akr_kernels.indexer_ok(indexer.n_head, indexer.head_dim, page, bound, next_n)


def dispatch(layer, index_q, index_q_scale, index_weights) -> bool:
    """Run this step's decode rows on the kernel where it serves. Returns
    whether the stock indexer op still has rows to do."""
    st = layer.akr.indexer
    if st is None:
        return True
    op = layer.indexer.indexer_op
    op.skip_decode = False
    if index_q_scale is None or not _run_decode(
        layer, st, index_q, index_q_scale, index_weights
    ):
        return True
    op.skip_decode = True
    md: Any = get_forward_context().attn_metadata
    return md[layer.indexer.k_cache.prefix].prefill is not None


def _run_decode(layer, st, q, q_scale, weights) -> bool:
    md: Any = get_forward_context().attn_metadata
    if not isinstance(md, dict):
        return False
    imeta = md[layer.indexer.k_cache.prefix]
    native = imeta.decode_native
    row_ends = imeta.decode_row_lens
    if imeta.decode is None or row_ends is None or native is None:
        return False
    if imeta.decode_shard is not None:
        return False
    rows = row_ends.shape[0]
    next_n = native.next_n
    if (
        not 0 < rows <= MAX_ROWS
        or rows % next_n
        or not akr_config.serves("indexer", rows)
        or not akr_config.indexer_rows_ok(st.bound, rows, st.consumer)
    ):
        return False
    swa = md.get(layer.swa_cache_layer.prefix)
    comp = md.get(layer.compressed_cache_prefix)
    if (
        swa is None
        or comp is None
        or swa.is_valid_token is None
        or swa.num_decode_tokens != rows
    ):
        return False
    ragged = akr_kernels.indexer_ragged(
        st.buffers,
        rows,
        _is_valid_i32(swa, rows),
        comp.block_table[: rows // next_n],
        comp.block_size // layer.compress_ratio,
    )
    if ragged is None:
        return False
    done = akr_kernels.indexer_topk(
        q[:rows],
        q_scale[:rows],
        weights[:rows],
        layer.indexer.k_cache.kv_cache,
        native.block_table,
        row_ends,
        layer.topk_indices_buffer[:rows],
        next_n=next_n,
        max_ctx=st.bound,
        ragged=ragged,
        candidates=None if st.candidates is None else st.candidates[:rows],
        consumer=st.consumer,
        buffers=st.buffers,
    )
    if not done:
        return False
    layer._topk_ragged_cache = {
        layer.compress_ratio: (ragged.slots, ragged.indptr, ragged.lens)
    }
    layer.akr.ragged_fresh = True
    return True


def _is_valid_i32(swa_metadata, rows: int) -> torch.Tensor:
    """One int32 copy of the step's valid-token mask, shared by the layers."""
    held = get_forward_context().additional_kwargs
    key = f"akr_is_valid_i32_{rows}"
    valid = held.get(key)
    if valid is None:
        valid = swa_metadata.is_valid_token[:rows].to(torch.int32)
        held[key] = valid
    return valid


def post_load(model) -> None:
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    cfg = get_current_vllm_config()
    mml = cfg.model_config.max_model_len
    block = cfg.cache_config.block_size
    override = cfg.kernel_config.amd_kernels.indexer_max_ctx
    spec = cfg.speculative_config
    next_n = 1 if spec is None else spec.num_speculative_tokens + 1
    layers: list[Any] = [
        m
        for m in model.modules()
        if isinstance(m, DeepseekV41ROCMAiterMLAAttention)
        and m.akr is not None
        and m.indexer is not None
    ]
    on = []
    for layer in layers:
        bound = akr_config.indexer_bound(mml, layer.compress_ratio, override)
        page = block // layer.compress_ratio
        why = unsupported_reason(layer, bound, page, next_n)
        if why is not None:
            logger.info("amd_kernels indexer: %s stays stock: %s", layer.prefix, why)
            continue
        op = layer.indexer.indexer_op
        candidates = getattr(op, "candidate_blocks", None)
        write = bool(getattr(op, "candidate_write", False))
        layer.akr.indexer = SimpleNamespace(
            bound=bound,
            consumer=candidates is not None and not write,
            candidates=candidates,
            buffers=None,
        )
        on.append(layer)
    if not on:
        return
    buffers = akr_kernels.IndexerBuffers(
        on[0].topk_indices_buffer.device, MAX_ROWS, TOPK
    )
    for layer in on:
        layer.akr.indexer.buffers = buffers
        # allocate the shared workspace before graph capture
        buffers.scores_for(layer.akr.indexer.bound)
    logger.info("amd_kernels indexer on %d of %d index layers", len(on), len(layers))
