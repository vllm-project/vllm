# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 sparse-attention indexer on SM80.

The indexer scores every compressed key of a request against each query
token and keeps the `index_topk` best for sparse MLA. On SM100 its queries
and keys are MXFP4 and DeepGEMM computes the scores; SM80 has neither the
FP4 conversion instructions nor a DeepGEMM build. This module keeps the stock
indexer layer, its MXFP4 key cache, and vLLM's top-k and candidate selection,
and brings its own versions of the pieces that need DeepGEMM:

* `forward_q`, which RoPEs the indexer queries and quantizes them to MXFP4,
  and `_produce_k`, which writes the MXFP4 keys;
* the indexer op `dsv41_ampere_sparse_attn_indexer`: the stock op's SM80
  subset (MXFP4 cache, no context parallelism, no fused MQA+top-k) on the
  SM80 logits kernels of `mqa_logits.py`;
* the index K cache layer, whose metadata builder (`indexer_metadata.py`)
  plans no DeepGEMM schedule.

The op scores prefill tokens chunk by chunk over keys gathered from the cache
(the ragged kernel), and decode tokens one row each straight from the cache
pages (the paged kernel). Both write FP32 logits from column zero, which the
two-level candidate selection and the per-row top-k then read.

A decode step's logits are as wide as `max_model_len` (CUDA graphs fix the
shapes) while its rows are far shorter, and vLLM's shared selection kernels
size their work by that width. With the default top-k backend (`auto`),
the decode tokens therefore select with this package's kernels, which read
only each row's own length: `topk.py` selects the candidate blocks and the
top-k, and the candidate consumers score their candidate keys only (at most
`2048 * 8` per row, however long the context) and select among them. They
select the same sets as the shared kernels over masked dense logits (ties at
the threshold go to the lowest positions) and list them in ascending order.
An explicit backend keeps vLLM's shared candidate and top-k kernels.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    kv_cache_as_quant_view,
)
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    dense_logits_rows,
)
from vllm.v1.worker.workspace import current_workspace_manager

from . import mqa_logits as logits_kernels
from . import rope_table_f32
from .indexer_metadata import AmpereIndexerBackend
from .quantize import indexer_q_rope_quant, store_index_keys
from .row_shard import PrefillRowShard, reserve_exchange_memory
from .topk import decode_topk, workspace_bytes

if TYPE_CHECKING:
    from .all_gather import AmpereAllGather

# MXFP4 keys: 64 packed value bytes and 4 UE8M0 scale bytes per 128 dims.
_VALUE_BYTES, _SCALE_BYTES = 64, 4
# Fewest requests of a decode step whose rows are split over the TP ranks:
# from two on, the scoring a rank saves outweighs the top-k exchange.
ROW_SHARD_MIN_REQUESTS = 2
# The Lamport all-gather of the decode shares, built once by
# init_ampere_all_gather.
_ampere_all_gather: "AmpereAllGather | None" = None


def init_ampere_all_gather(vllm_config: VllmConfig) -> None:
    """Build the Lamport all-gather of the row-sharded decode top-k.
    Collective over the TP group; the model calls it where it builds the SM80
    boundaries' all-reduce, whose conditions (TP 2, 4 or 8 with full peer
    access) it needs."""
    global _ampere_all_gather
    from .all_gather import AmpereAllGather
    from .all_reduce import MAX_ROWS

    group = get_tp_group()
    # int32 indices; a rank's share of at most MAX_ROWS rows.
    _ampere_all_gather = AmpereAllGather(
        group.device_group,
        vllm_config.model_config.hf_config.index_topk * 4,
        MAX_ROWS // group.world_size,
        group.device,
    )


def _scale_bytes(scales: torch.Tensor) -> torch.Tensor:
    """Four UE8M0 bytes packed as one int32 per (token, head) back to uint8
    `[..., 4]`."""
    return scales.contiguous().view(torch.uint8).view(*scales.shape, 4)


def _reserve_profile_memory(
    device: torch.device,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> None:
    """Allocate, once, the op's peak transient memory for the memory profile:
    the key-gather workspace, the larger of a prefill chunk's FP32 logits and
    a decode step's (with its top-k workspace), and the TP row-shard
    exchange."""
    assert head_dim // 2 == _VALUE_BYTES
    current_workspace_manager().get_simultaneous(
        ((total_seq_lens, _VALUE_BYTES), torch.uint8),
        ((total_seq_lens, _SCALE_BYTES), torch.uint8),
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    cfg = get_current_vllm_config_or_none()
    max_decode_tokens = 0
    if cfg is not None:
        max_decode_tokens = min(
            cfg.scheduler_config.max_num_seqs,
            cfg.scheduler_config.max_num_batched_tokens,
        ) * (1 + cfg.num_speculative_tokens)
    decode_row_bytes = round_up(
        max_model_len, 256
    ) * torch.float32.itemsize + workspace_bytes(
        max_model_len, topk_indices_buffer.shape[1]
    )
    logits_bytes = max(
        envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
        max_decode_tokens * decode_row_bytes,
    )
    _ = torch.empty(logits_bytes, dtype=torch.uint8, device=device)
    reserve_exchange_memory(topk_indices_buffer, device)


def _select(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    topk_backend: str,
) -> None:
    """Two-level selection (V4.1): the candidate source indexer publishes its
    rows' best key blocks; the later indexers mask their scores to them."""
    if candidate_blocks is None:
        return
    if candidate_write:
        select_candidate_blocks(
            logits,
            None,
            row_ends,
            candidate_blocks.shape[1],
            candidate_block_size,
            candidate_blocks,
            score_dtype=(
                torch.float32 if topk_backend in ("auto", "persistent") else None
            ),
        )
    else:
        apply_candidate_mask(
            logits, None, row_ends, candidate_blocks, candidate_block_size
        )


def _prefill_topk(
    metadata: DeepseekV32IndexerMetadata,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    topk_backend: str,
) -> None:
    """Top-k of the prefill tokens, chunk by chunk over gathered keys."""
    prefill = metadata.prefill
    assert prefill is not None
    num_decode_tokens = metadata.num_decode_tokens
    # Sized for this step's largest chunk, at the workspace head.
    max_keys = max(c.max_local_total_seq_lens for c in prefill.chunks)
    k_values_all, k_scales_all = current_workspace_manager().get_simultaneous(
        ((max_keys, _VALUE_BYTES), torch.uint8),
        ((max_keys, _SCALE_BYTES), torch.uint8),
    )
    # The rows are split over the TP ranks, candidate rows included: the
    # candidate consumers read their source's index K cache, so they narrow
    # by the source's metadata, a rank's consumers read only the candidate
    # blocks its own source wrote, and only the top-k indices are exchanged.
    row_shard = PrefillRowShard.from_metadata(prefill, num_decode_tokens)
    for chunk in prefill.chunks:
        k_values = k_values_all[: chunk.max_local_total_seq_lens]
        k_scales = k_scales_all[: chunk.max_local_total_seq_lens]
        if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
            assert chunk.block_table is not None
            assert chunk.local_cu_seq_lens is not None
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache, k_values, k_scales, chunk.block_table, chunk.local_cu_seq_lens
            )
        row_start, row_end = chunk.token_start, chunk.token_end
        row_ks, row_ke = chunk.cu_seqlen_ks, chunk.cu_seqlen_ke
        if row_shard is not None:
            narrowed = row_shard.narrow(chunk)
            if narrowed is None:
                continue
            row_start, row_end, row_ks, row_ke = narrowed
        topk = topk_indices_buffer[row_start:row_end, :topk_tokens]
        if chunk.local_total_seq_lens == 0 or row_end == row_start:
            topk.fill_(-1)
            continue
        step = dense_logits_rows(chunk.max_seqlen_k)
        for sub in range(0, row_end - row_start, step):
            rows = slice(sub, min(sub + step, row_end - row_start))
            tokens = slice(row_start + rows.start, row_start + rows.stop)
            # Row m's scores start at column zero.
            logits = logits_kernels.mqa_logits(
                q_values[tokens],
                q_scales[tokens],
                k_values,
                k_scales,
                weights[tokens].contiguous(),
                row_ks[rows],
                row_ke[rows],
                chunk.max_seqlen_k,
            )
            row_ends = row_ke[rows] - row_ks[rows]
            _select(
                logits,
                row_ends,
                candidate_blocks[tokens] if candidate_blocks is not None else None,
                candidate_block_size,
                candidate_write,
                topk_backend,
            )
            ops.top_k_per_row_prefill(
                logits,
                torch.zeros_like(row_ends),
                row_ends,
                topk[rows],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            del logits
    if row_shard is not None:
        row_shard.exchange_topk(topk_indices_buffer, topk_tokens)


def _decode_topk(
    metadata: DeepseekV32IndexerMetadata,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    topk_backend: str,
) -> None:
    """Top-k of the decode tokens, one row per token straight from the cache
    pages.

    In a pure-decode step of at least `ROW_SHARD_MIN_REQUESTS` requests,
    each TP rank scores and selects only its share of the rows, whole
    requests when the decodes are uniform (`_decode_row_shard`), and a TP
    all-gather of the int32 top-k rows rebuilds `topk_indices_buffer`
    (`_gather_topk_shares`). The candidate consumers read their source's
    index K cache and so take its share: a rank's consumers read only the
    candidate blocks its own source wrote, and the candidate blocks are not
    exchanged."""
    decode = metadata.decode
    assert decode is not None
    rows = metadata.num_decode_tokens
    # The decode tokens lead the step's rows.
    q_values, q_scales, weights = q_values[:rows], q_scales[:rows], weights[:rows]
    tp_group = get_tp_group()
    shard = None
    if topk_backend == "auto" and metadata.num_prefills == 0:
        shard = _decode_row_shard(
            rows, metadata.num_decodes, tp_group.world_size, tp_group.rank_in_group
        )
    if shard is None:
        _decode_rows_topk(
            decode,
            kv_cache,
            q_values,
            q_scales,
            weights,
            topk_tokens,
            max_model_len,
            metadata.max_seq_len,
            topk_indices_buffer[:rows, :topk_tokens],
            candidate_blocks,
            candidate_block_size,
            candidate_write,
            topk_backend,
            0,
            rows,
            metadata.num_decodes,
        )
        return
    start, end, share_rows = shard
    _gather_topk_shares(
        topk_indices_buffer,
        topk_tokens,
        rows,
        share_rows,
        end - start,
        lambda out: _decode_rows_topk(
            decode,
            kv_cache,
            q_values[start:end],
            q_scales[start:end],
            weights[start:end],
            topk_tokens,
            max_model_len,
            metadata.max_seq_len,
            out,
            candidate_blocks,
            candidate_block_size,
            candidate_write,
            topk_backend,
            start,
            end,
            (end - start) * metadata.num_decodes // rows,
        ),
    )


def _gather_topk_shares(
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    rows: int,
    share_rows: int,
    filled: int,
    score: Callable[[torch.Tensor], None],
) -> None:
    """Score this rank's share of the decode rows and gather every rank's
    share, in rank order, into the first `rows` rows of
    `topk_indices_buffer`.

    `score` writes the share's first `filled` top-k rows into the tensor
    it is given; the rest of the share is -1. With SM80's Lamport all-gather
    (`ampere/all_gather.py`) the shares are scored into and gathered in
    place in the buffer's first `world * share_rows` rows (past `rows`
    only -1 padding lands); otherwise into a private share, gathered with
    the TP group's custom all-gather and copied.
    """
    tp_group = get_tp_group()
    shares = topk_indices_buffer[: tp_group.world_size * share_rows, :topk_tokens]
    first = tp_group.rank_in_group * share_rows
    local = shares[first : first + share_rows]
    gather = _ampere_all_gather
    if gather is not None and gather.supports(local, shares):
        if filled < share_rows:
            local[filled:] = -1
        if filled:
            score(local[:filled])
        gather(local, shares)
        return
    # Equal shares for the all-gather; a short or empty share is -1 padded.
    local = torch.full(
        (share_rows, topk_tokens),
        -1,
        dtype=torch.int32,
        device=topk_indices_buffer.device,
    )
    if filled:
        score(local[:filled])
    # The custom all-gather takes floating-point rows and copies the int32
    # bits unchanged.
    gathered = sp_all_gather(local.view(torch.float32)).view(torch.int32)
    topk_indices_buffer[:rows, :topk_tokens] = gathered[:rows]


def _decode_row_shard(
    rows: int, num_requests: int, world_size: int, rank: int
) -> tuple[int, int, int] | None:
    """Rows `[start, end)` that `rank` scores in a pure-decode step and
    the rows of a full share, or None when the step stays replicated.

    Each rank takes `ceil(num_requests / world_size)` whole requests in
    order, so the last ranks may take fewer or none. The split reads only
    host state that every rank shares.
    """
    if world_size < 2 or num_requests < ROW_SHARD_MIN_REQUESTS or rows % num_requests:
        return None
    per_request = rows // num_requests
    share = -(-num_requests // world_size)
    first = min(rank * share, num_requests)
    last = min(first + share, num_requests)
    return first * per_request, last * per_request, share * per_request


def _decode_rows_topk(
    decode: DeepSeekV32IndexerDecodeMetadata,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    max_seq_len: int,
    topk_out: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    topk_backend: str,
    start: int,
    end: int,
    num_requests: int,
) -> None:
    """Top-k of decode rows `[start, end)`, whole requests, into
    `topk_out`."""
    rows = end - start
    lens = decode.seq_lens[start:end]
    row_lens = lens.reshape(-1)
    block_table = decode.block_table[start:end]
    candidates = candidate_blocks[start:end] if candidate_blocks is not None else None
    own = topk_backend == "auto"
    if own and candidates is not None and not candidate_write:
        # A candidate consumer: the source listed each row's blocks in
        # ascending order (decode_topk), so it scores and selects only those.
        logits, compact_lens = logits_kernels.candidate_mqa_logits(
            q_values,
            q_scales,
            kv_cache,
            weights.contiguous(),
            row_lens,
            block_table,
            1,
            candidates,
            candidate_block_size,
        )
        decode_topk(
            logits,
            compact_lens,
            topk_out,
            1,
            candidates,
            candidate_block_size,
            row_lens,
        )
        return
    # A uniform step's tokens of one request share their keys: score them in
    # one CTA (their rows are consecutive and their block tables equal).
    group = decode.write_max_decode_len
    if not (
        decode.decode_is_uniform
        and group > 1
        and rows == num_requests * group
        and group * q_values.shape[1] <= logits_kernels.THREADS
    ):
        group = 1
    logits = logits_kernels.paged_mqa_logits(
        q_values,
        q_scales,
        kv_cache,
        weights.contiguous(),
        row_lens,
        block_table,
        1,
        max_model_len,
        group=group,
    )
    if not own:
        _select(
            logits,
            row_lens,
            candidates,
            candidate_block_size,
            candidate_write,
            topk_backend,
        )
        get_indexer_topk(topk_backend)(
            logits, lens, 1, topk_out, topk_tokens, max_seq_len
        )
        return
    if candidates is not None and candidate_write:
        decode_topk(logits, row_lens, candidates, candidate_block_size)
    decode_topk(logits, row_lens, topk_out)


@eager_break_during_capture
def dsv41_ampere_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    dense_mha_metadata_layer_name: LayerNameType,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    """The indexer top-k of every token of the step into
    `topk_indices_buffer` (request-local compressed positions, -1 padded)."""
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        # Profile and dummy runs: account for the peak transient memory.
        _reserve_profile_memory(
            hidden_states.device,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
        return topk_indices_buffer
    metadata = attn_metadata[_resolve_layer_name(k_cache_prefix)]
    assert isinstance(metadata, DeepseekV32IndexerMetadata)
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        # As the stock op: a prefill-only step whose main MLA runs dense MHA
        # reads no top-k, so leave the buffer untouched.
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            if (
                getattr(getattr(mla_metadata, "prefill", None), "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                return topk_indices_buffer
    # Decode rows are written whole by the default backend's top-k.
    first_unwritten = metadata.num_decode_tokens if topk_backend == "auto" else 0
    topk_indices_buffer[first_unwritten : hidden_states.shape[0]] = -1
    q_scales = _scale_bytes(q_scale)
    selection = (candidate_blocks, candidate_block_size, candidate_write, topk_backend)
    if metadata.num_prefills > 0:
        _prefill_topk(
            metadata,
            kv_cache,
            q_values,
            q_scales,
            weights,
            topk_tokens,
            topk_indices_buffer,
            *selection,
        )
    if metadata.num_decodes > 0:
        _decode_topk(
            metadata,
            kv_cache_as_quant_view(kv_cache, head_dim, True),
            q_values,
            q_scales,
            weights,
            topk_tokens,
            max_model_len,
            topk_indices_buffer,
            *selection,
        )
    return topk_indices_buffer


def _dsv41_ampere_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    dense_mha_metadata_layer_name: LayerNameType,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="dsv41_ampere_sparse_attn_indexer",
    op_func=dsv41_ampere_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "candidate_blocks"],
    fake_impl=_dsv41_ampere_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class AmpereIndexerCache(DeepseekV4IndexerCache):
    """The index K cache, with the SM80 metadata builder."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AmpereIndexerBackend


class AmpereSparseAttnIndexer(SparseAttnIndexer):
    """The stock indexer layer on the SM80 indexer op."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                "The DeepSeek-V4.1 SM80 indexer supports no context parallelism."
            )
        assert self.skip_k_cache_insert, "AmpereIndexer writes its own keys"

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k indices of the step's tokens; `k` was already written."""
        assert isinstance(q_quant, tuple), "SM80 indexer queries are MXFP4"
        q_values, q_scale = q_quant
        return torch.ops.vllm.dsv41_ampere_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            topk_backend=self.topk_backend,
        )


class AmpereIndexer(DeepseekV4Indexer):
    """The DeepSeek-V4.1 indexer with SM80 query quantization and logits."""

    attn_cls: ClassVar[type[SparseAttnIndexer]] = AmpereSparseAttnIndexer

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.use_fp4_kv or not isinstance(
            self.indexer_op, AmpereSparseAttnIndexer
        ):
            raise ValueError(
                "The SM80 indexer scores the MXFP4 key cache with dense logits; "
                "leave attention_config.indexer_sparse_logits off."
            )

    def _produce_k(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        """MXFP4 index keys of the group-boundary latents into the key cache:
        `k_norm(wk(latent))`, RoPE'd at the group's first position."""
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or latent is None:
            return
        slot_mapping = cast(Any, attn_metadata[self.k_cache.prefix]).slot_mapping
        k_pre, _ = self.wk(latent)
        store_index_keys(
            k_pre,
            positions,
            slot_mapping,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            rope_table_f32(rotary_emb),
            self.compress_ratio,
            self.k_cache.kv_cache,
        )

    def forward_q(
        self,
        qr: torch.Tensor | QuantizedActivation,
        qr_scale: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """MXFP4 indexer queries: values `[T, H, 64]`, scales as one int32
        per (token, head), and the FP32 head weights."""
        q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_head, self.head_dim)
        values, scales, weights = indexer_q_rope_quant(
            positions,
            q,
            rope_table_f32(rotary_emb),
            indexer_weights,
            self.softmax_scale * self.n_head**-0.5,
        )
        return values, scales.view(torch.int32).squeeze(-1), weights
