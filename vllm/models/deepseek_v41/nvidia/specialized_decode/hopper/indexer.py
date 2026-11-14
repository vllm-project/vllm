# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 sparse-attention indexer on SM90, with length-bounded
candidate selection.

DeepSeek-V4.1 selects a decode token's compressed keys in two levels. The
candidate source indexer (layer 20) keeps each row's best 2048 blocks of 8
compressed positions, and the indexers after it (layers 24, 28, 32 and 36,
the candidate consumers) keep their top 512 among those blocks only. vLLM's
indexer op sizes this work by the logits' width, `max_model_len`, which CUDA
graphs fix: the source ranks the blocks of the whole width with
`torch.topk`, whatever the row's length. A consumer runs like any other
indexer: DeepGEMM scores every key of the row, a mask kernel sets the keys
outside the candidate blocks to `-inf`, and the top-k reads the whole row
again. A row of 256K compressed positions so costs a consumer about 16 times
the candidates' work, three times over.

This layer keeps vLLM's op (`sparse_attn_indexer`) for every call except the
two-level calls on a step of only decode rows, and narrows the op to its rank's
prefill rows when a step splits them over the TP ranks
(`indexer_metadata.py`):

* the source (`_candidate_source`) runs vLLM's decode path, DeepGEMM's
  logits and the configured top-k, but selects its candidate blocks with the
  specialized decode's `select_candidate_blocks_cooperative`, which ranks
  only the blocks below each row's length;
* a consumer (`_compact_consumer`) runs the SM80 package's CuTe DSL kernels
  on SM90's index cache as it is (pages of 128-byte E4M3 keys, then their FP32
  scales). `candidate_mqa_logits` scores only the row's candidate keys,
  into compact FP32 rows (column `c` is key `8 * candidates[c // 8] +
  c % 8`, `-inf` past the row's length), so a row costs at most
  `2048 * 8` keys however long its context; `decode_topk` keeps the 512
  best columns of each compact row and writes their key positions (all of a
  row's keys when it has at most 512);
* on a split step, the other indexers (`_row_sharded`) run the stock op
  over this rank's share of the prefill rows and gather the ranks' top-k
  rows back (`../ampere/row_shard.py`); the candidate rows of the
  two-level selection stay replicated.

The source's candidate set is `torch.topk`'s with the newest block pinned,
except at the selection boundary: blocks tied there, and a boundary bin of
the cooperative kernel's FP16 radix more crowded than it keeps, are selected
approximately, so a row may hold fewer than 2048 candidates. Its rows are
unordered and -1 padded anywhere, but a row of at most 2048 blocks lists
block `i` in slot `i`, so every valid block lies within the first
`min(2048, ceil(len / 8))` slots that a compact consumer scores; vLLM's
mask reads all slots. A consumer's key scores
`scale * sum_h w_h * relu(q_h . k)` as DeepGEMM computes it (up to the FP32
summation order), and its top-k is the same set as the masked dense row's, up
to ties at the threshold. Like vLLM's op, the call is a break of a breakable
CUDA graph capture, so its Python chooses the path at every step; a FULL graph
replays the path of its capture, a decode step.
"""

import dataclasses

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    as_fp32_logits,
    kv_cache_as_quant_view,
    sparse_attn_indexer,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV41IndexerBackend,
)

from ..ampere.mqa_logits import candidate_mqa_logits
from ..ampere.row_shard import PrefillRowShard, reserve_exchange_memory
from ..ampere.topk import decode_topk
from ..attention.candidate_topk import (
    select_candidate_blocks_cooperative,
    supports_cooperative_candidates,
)
from .indexer_metadata import HopperIndexerBackend


@eager_break_during_capture
def dsv41_hopper_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    pcp_shard_decode_requests: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    """The stock `sparse_attn_indexer` with the same arguments, whose
    two-level calls on steps of only decode rows bound their candidate work
    by each row's length (`_candidate_source`, `_compact_consumer`), and
    whose other calls score only this rank's prefill rows on a split step."""
    attn_metadata = get_forward_context().attn_metadata
    layer_name = _resolve_layer_name(k_cache_prefix)
    meta = attn_metadata.get(layer_name) if isinstance(attn_metadata, dict) else None
    if not isinstance(attn_metadata, dict):
        # The memory profile's dummy run, which never reaches the exchange.
        reserve_exchange_memory(topk_indices_buffer, hidden_states.device)
    if (
        candidate_blocks is not None
        and q_scale is None
        and not use_fp4_cache
        and skip_k_cache_insert
        and not use_pcp
        and dcp_world_size == 1
        and topk_backend == "auto"
        and not mqa_logits_use_mx_scales()
        and isinstance(meta, DeepseekV32IndexerMetadata)
    ):
        if candidate_write:
            done = _candidate_source(
                meta,
                hidden_states.shape[0],
                kv_cache,
                q_quant,
                weights,
                topk_tokens,
                head_dim,
                max_model_len,
                topk_indices_buffer,
                candidate_blocks,
                candidate_block_size,
                topk_backend,
            )
        else:
            done = _compact_consumer(
                meta,
                hidden_states.shape[0],
                kv_cache,
                q_quant,
                weights,
                topk_tokens,
                topk_indices_buffer,
                candidate_blocks,
                candidate_block_size,
            )
        if done:
            return topk_indices_buffer
    row_shard = None
    if (
        candidate_blocks is None
        and isinstance(meta, DeepseekV32IndexerMetadata)
        and meta.prefill is not None
    ):
        row_shard = PrefillRowShard.from_metadata(meta.prefill, meta.num_decode_tokens)
    if row_shard is not None:
        # The stock op reads this layer's metadata from the forward context:
        # give it this rank's rows for the call.
        assert isinstance(attn_metadata, dict) and meta is not None
        assert meta.prefill is not None
        attn_metadata[layer_name] = dataclasses.replace(
            meta, prefill=row_shard.narrow_prefill(meta.prefill)
        )
    try:
        sparse_attn_indexer(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
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
            use_pcp,
            pcp_shard_decode_requests,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
            dcp_rank,
            dcp_world_size,
            cp_kv_cache_interleave_size,
            skip_topk_buffer_clear,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
            topk_backend=topk_backend,
        )
    finally:
        if row_shard is not None:
            attn_metadata[layer_name] = meta
    if row_shard is not None:
        row_shard.exchange_topk(topk_indices_buffer, topk_tokens)
    return topk_indices_buffer


def _candidate_source(
    meta: DeepseekV32IndexerMetadata,
    num_tokens: int,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    topk_backend: str,
) -> bool:
    """The source's top-k and candidate blocks of a step of only decode rows,
    as vLLM's op computes them but for the length-bounded block selection;
    False (nothing written) for any other step."""
    decode = meta.decode
    if meta.num_prefills > 0 or meta.num_decodes == 0 or decode is None:
        return False
    rows = meta.num_decode_tokens
    batch_size = decode.decode_lens.shape[0]
    seq_lens = decode.seq_lens[:batch_size]
    candidates = candidate_blocks[:rows]
    # As _compact_consumer, and the selection's row, width and k limits.
    if (
        decode.requires_padding
        or rows % batch_size
        or seq_lens.numel() != rows
        or not supports_cooperative_candidates(
            rows,
            triton.cdiv(max_model_len, candidate_block_size),
            candidates.shape[1],
        )
    ):
        return False
    next_n = rows // batch_size
    # vLLM's op clears the step's rows before its top-k scatters into them.
    topk_indices_buffer[:num_tokens] = -1
    q = q_fp8[:rows].reshape(batch_size, next_n, *q_fp8.shape[1:])
    q_scale, decode_weights = mqa_logits_mx_inputs(
        q, None, weights[:rows], torch.float32
    )
    logits = fp8_fp4_paged_mqa_logits(
        (q, q_scale),
        kv_cache_as_quant_view(kv_cache, head_dim, False),
        decode_weights,
        seq_lens,
        decode.block_table,
        decode.schedule_metadata,
        max_model_len=max_model_len,
        indices=decode.indices,
    )
    # The rows' lengths, in compressed positions (logits columns).
    row_lens = seq_lens.reshape(-1)
    # DeepGEMM's SM90 logits are FP32 already; this is then a no-op.
    logits = as_fp32_logits(logits, row_lens)
    select_candidate_blocks_cooperative(
        logits, row_lens, candidate_block_size, candidates
    )
    get_indexer_topk(topk_backend)(
        logits,
        seq_lens,
        next_n,
        topk_indices_buffer[:rows, :topk_tokens],
        topk_tokens,
        meta.max_seq_len,
    )
    return True


def _compact_consumer(
    meta: DeepseekV32IndexerMetadata,
    num_tokens: int,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
) -> bool:
    """A consumer's top-k of a step of only decode rows, from its candidate
    keys; False (nothing written) for any other step."""
    decode = meta.decode
    if meta.num_prefills > 0 or meta.num_decodes == 0 or decode is None:
        return False
    rows = meta.num_decode_tokens
    batch_size = decode.decode_lens.shape[0]
    seq_lens = decode.seq_lens[:batch_size]
    # As vLLM's op: a padded decode layout (short prefills counted as
    # decodes) goes there. Otherwise the rows are the requests' next_n
    # tokens, each with its own length (native MTP or flattened rows).
    if decode.requires_padding or rows % batch_size or seq_lens.numel() != rows:
        return False
    next_n = rows // batch_size
    # The rows' lengths, in compressed positions.
    row_lens = seq_lens.reshape(-1)
    # The decode rows' top-k columns are written whole below.
    topk_indices_buffer[rows:num_tokens] = -1
    candidates = candidate_blocks[:rows]
    logits, compact_lens = candidate_mqa_logits(
        q_fp8[:rows],
        None,
        kv_cache.view(torch.uint8),
        weights[:rows].contiguous(),
        row_lens,
        decode.block_table,
        next_n,
        candidates,
        candidate_block_size,
    )
    decode_topk(
        logits,
        compact_lens,
        topk_indices_buffer[:rows, :topk_tokens],
        1,
        candidates,
        candidate_block_size,
        row_lens,
    )
    return True


def _dsv41_hopper_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    pcp_shard_decode_requests: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    """The op's output buffer, for tracing."""
    return topk_indices_buffer


direct_register_custom_op(
    op_name="dsv41_hopper_sparse_attn_indexer",
    op_func=dsv41_hopper_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "candidate_blocks"],
    fake_impl=_dsv41_hopper_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class HopperIndexerCache(DeepseekV4IndexerCache):
    """The index K cache, with the SM90 metadata builder where it would
    take the stock dense-logits one."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        backend = super().get_attn_backend()
        if backend is DeepseekV41IndexerBackend:
            return HopperIndexerBackend
        return backend


class HopperSparseAttnIndexer(SparseAttnIndexer):
    """vLLM's indexer layer on `dsv41_hopper_sparse_attn_indexer`."""

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k indices of the step's tokens, as vLLM's layer."""
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.dsv41_hopper_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            self.pcp_shard_decode_requests,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self._parallel_config.cp_kv_cache_interleave_size,
            False,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            topk_backend=self.topk_backend,
        )


class HopperIndexer(DeepseekV4Indexer):
    """The DeepSeek-V4.1 indexer whose candidate source and consumers bound
    their candidate work by each row's length on decode steps."""

    attn_cls = HopperSparseAttnIndexer
