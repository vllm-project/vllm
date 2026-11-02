# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlashInfer sparse-MLA decode of a specialized decode step.

It is the decode branch of ``DeepseekV4FlashInferMLAAttention.forward_mqa``
with two changes, neither of which issues a collective:

- Shared sparse-index tables (``SparseIndexTables``). Compressed layers that
  share an index source build equal tables in a step, so the first of them
  builds the table and the others reuse it; the first layer of such a run also
  builds, on a side stream, the tables its later layers look up in other SWA
  KV-cache groups.
- Persistent multi-CTA KV counters. Without a caller-owned counter buffer,
  ``trtllm_batch_decode_sparse_mla_dsv4`` allocates and zeroes a fresh one
  before every call, a small fill kernel ahead of each decode launch. The
  kernel returns every counter to zero when it finishes, so a buffer zeroed
  once stays valid for launches that do not overlap, as in the FlashInfer MLA
  backend.

Batches whose clusters fit the GPU at once (at width 640, up to 8 requests with
DSpark on a GB300) run the CuTe DSL kernel of ``sparse_mla_cute`` instead of
TRTLLM-gen: it spreads a token's columns over one CTA per 128 instead of at most
3.
"""

import inspect
from functools import cache
from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

from ...flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    _get_flashinfer_dsv4_workspace,
)
from .sparse_mla_cute import (
    SWA_COLUMNS,
    sparse_mla_decode,
    supports_sparse_mla_cute,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# (sparse_indices, sparse_topk_lens) of one step, SWA group and key.
SparseIndexTable = tuple[torch.Tensor, torch.Tensor]

_kv_counters_by_device: dict[torch.device, torch.Tensor] = {}


@cache
def _takes_kv_counters() -> bool:
    """Whether the installed FlashInfer's sparse-MLA decode takes caller-owned
    KV counters (``multi_ctas_kv_counter_buffer``, since 0.7.1); logs once when
    it does not."""
    try:
        import flashinfer.utils
        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4
    except ImportError:
        return False
    if "multi_ctas_kv_counter_buffer" in inspect.signature(
        trtllm_batch_decode_sparse_mla_dsv4
    ).parameters and hasattr(
        flashinfer.utils, "get_trtllm_gen_multi_ctas_kv_counter_bytes"
    ):
        return True
    logger.warning(
        "DeepSeek-V4.1 specialized decode: the installed FlashInfer's sparse-MLA "
        "decode takes no KV counter buffer; each decode allocates its own."
    )
    return False


def kv_counter_buffer(max_rows: int) -> torch.Tensor | None:
    """Zeroed multi-CTA KV counters on the current device for decode launches
    of up to ``max_rows`` rows (requests x heads), or None when FlashInfer's
    decode takes no counter buffer.

    Allocated when the executor is built, outside any CUDA graph capture. Every
    layer shares the buffer, as it shares the workspace: their decode launches
    run one after another on one stream. Decode launches that overlap on one
    device would race on it.
    """
    if not _takes_kv_counters():
        return None
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    device = torch.device("cuda", torch.accelerator.current_device_index())
    nbytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        max_rows, 1, get_device_sm_count(device)
    )
    counters = _kv_counters_by_device.get(device)
    if counters is None or counters.nbytes < nbytes:
        counters = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        _kv_counters_by_device[device] = counters
    return counters


def trtllm_decode_sparse_mla_dsv4(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    cum_seq_lens_q: torch.Tensor | None,
    max_q_len: int,
    kv_counters: torch.Tensor | None,
) -> None:
    """``trtllm_batch_decode_sparse_mla_dsv4`` for ``sparse_indices`` that are
    KV storage offsets, on ``kv_counters`` (from ``kv_counter_buffer``) when
    they are large enough for the launch."""
    counter_kwargs: dict[str, torch.Tensor] = {}
    if kv_counters is not None:
        from flashinfer.utils import (
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )

        if kv_counters.nbytes >= get_trtllm_gen_multi_ctas_kv_counter_bytes(
            seq_lens.size(0), query.size(-2), get_device_sm_count(query.device)
        ):
            counter_kwargs["multi_ctas_kv_counter_buffer"] = kv_counters
    flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=swa_kv_cache,
        workspace_buffer=workspace_buffer,
        sparse_indices=sparse_indices,
        sparse_indices_are_storage_offsets=True,
        compressed_kv_cache=compressed_kv_cache,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        **counter_kwargs,
    )


def sparse_index_key(attn: DeepseekV4FlashInferMLAAttention) -> str:
    """Layers with equal keys build equal sparse-index tables in a step: a
    compressed layer's table depends on its compressed cache, compress ratio
    and top-k indices, which only its index source writes."""
    if attn.compress_ratio == 0:
        return "swa_only"
    return (
        f"{attn.compressed_cache_prefix}:{attn.index_source_layer_id}:"
        f"{attn.compress_ratio}"
    )


class SparseIndexTables:
    """The sparse-index tables of one decode step, shared by equal keys.

    Each SWA KV-cache group has its own metadata, and the layers of a run of
    equal keys interleave the groups: each group would build each key's table
    on the critical path. A table depends only on the step's metadata, the
    group's SWA indices and block table and, for compressed layers, the index
    source's top-k indices; not on any layer's KV write. So the first layer of
    a run builds its own group's table through the stock
    ``_build_sparse_index_metadata`` and then forks, onto ``stream``, the
    builds that later layers of the run look up in other groups:

    * ``lookup`` returns a layer's table, building and forking on a miss and
      making the current stream wait for a table built on ``stream``.
    * ``join`` makes the current stream wait for everything ``stream`` ran.
      Index sources call it before their indexer overwrites the top-k indices,
      and the executor at the end of its step, so a CUDA graph capture ends
      joined.

    Events are reused across forks: a wait on an event that a later fork
    re-recorded waits for more work on the same in-order stream, never less.
    """

    def __init__(self, attns: list[DeepseekV4FlashInferMLAAttention]) -> None:
        self.keys = {attn: sparse_index_key(attn) for attn in attns}
        runs: list[list[DeepseekV4FlashInferMLAAttention]] = []
        previous: str | None = None
        for attn in attns:
            key = self.keys[attn]
            if key == previous:
                runs[-1].append(attn)
            else:
                runs.append([attn])
            previous = key
        self.followers = {
            attn: tuple(run[i + 1 :]) for run in runs for i, attn in enumerate(run)
        }
        self.stream = torch.cuda.Stream()
        self._events = [
            torch.cuda.Event() for _ in range(max(map(len, runs), default=1) - 1)
        ]
        self._tables: dict[tuple[int, str], SparseIndexTable] = {}
        self._pending: dict[tuple[int, str], torch.cuda.Event] = {}
        self._forked = False

    def reset(self) -> None:
        """Forget the previous step's tables."""
        self._tables.clear()
        self._pending.clear()
        self._forked = False

    def lookup(
        self,
        attn: DeepseekV4FlashInferMLAAttention,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``attn._build_sparse_index_metadata``, on the step's shared tables."""
        key = self.keys[attn]
        table = self._tables.get((id(swa_metadata), key))
        if table is None:
            compressed_kv_cache, seq_lens, indices, lens = (
                attn._build_sparse_index_metadata(
                    kv_cache, swa_k_cache, swa_metadata, attn_metadata, swa_only
                )
            )
            self._store(swa_metadata, key, (indices, lens))
            self._fork(attn, key, kv_cache, swa_metadata, attn_metadata, swa_only)
            return compressed_kv_cache, seq_lens, indices, lens
        event = self._pending.get((id(swa_metadata), key))
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        assert swa_metadata.seq_lens is not None
        num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
        compressed_kv_cache = swa_k_cache if swa_only else kv_cache
        assert compressed_kv_cache is not None
        return compressed_kv_cache, swa_metadata.seq_lens[:num_reqs], *table

    def join(self) -> None:
        """Make the current stream wait for every build since the last join."""
        if self._forked:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._forked = False
            self._pending.clear()

    def _store(
        self,
        swa_metadata: "DeepseekSparseSWAMetadata",
        key: str,
        table: SparseIndexTable,
    ) -> None:
        group = id(swa_metadata)
        if key != "swa_only":
            # Runs of a key are consecutive, so a new key ends the group's
            # previous one; drop its table to bound memory.
            for old in [k for k in self._tables if k[0] == group and k[1] != key]:
                if old[1] != "swa_only":
                    del self._tables[old]
        self._tables[(group, key)] = table

    def _fork(
        self,
        attn: DeepseekV4FlashInferMLAAttention,
        key: str,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
    ) -> None:
        """Build, on ``stream``, the tables of the other KV-cache groups whose
        later layers of the run look ``key`` up: for each group, the build its
        first such layer would run on a miss. The key fixes the compressed
        cache and its metadata, so these are ``attn``'s."""
        by_layer = get_forward_context().attn_metadata
        assert isinstance(by_layer, dict)
        builds: list[
            tuple[DeepseekV4FlashInferMLAAttention, DeepseekSparseSWAMetadata]
        ] = []
        seen = {id(swa_metadata)}
        for follower in self.followers[attn]:
            metadata = cast(
                "DeepseekSparseSWAMetadata | None",
                by_layer.get(follower.swa_cache_layer.prefix),
            )
            if metadata is None or id(metadata) in seen:
                continue
            seen.add(id(metadata))
            if (id(metadata), key) not in self._tables:
                builds.append((follower, metadata))
        if not builds:
            return
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        self._forked = True
        try:
            with torch.cuda.stream(self.stream):
                for event, (follower, metadata) in zip(self._events, builds):
                    *_, indices, lens = follower._build_sparse_index_metadata(
                        kv_cache,
                        follower.swa_cache_layer.kv_cache,
                        metadata,
                        attn_metadata,
                        swa_only,
                    )
                    event.record(self.stream)
                    indices.record_stream(main)
                    lens.record_stream(main)
                    self._store(metadata, key, (indices, lens))
                    self._pending[(id(metadata), key)] = event
        except BaseException:
            # Rejoin so that a capture reports this error, not unjoined work.
            self.join()
            raise


class SparseMlaDecode:
    """``forward_mqa`` of a pure-decode step on the shared tables and the
    persistent KV counters."""

    def __init__(self, tables: SparseIndexTables, max_rows: int) -> None:
        self.tables = tables
        self.kv_counters = kv_counter_buffer(max_rows)

    def run(
        self, attn: DeepseekV4FlashInferMLAAttention, q: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Write ``attn``'s sparse MLA of ``q`` into ``out``."""
        assert attn.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        by_layer = get_forward_context().attn_metadata
        assert isinstance(by_layer, dict)
        attn_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            by_layer.get(attn.compressed_cache_prefix)
            if attn.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", by_layer[attn.swa_cache_layer.prefix]
        )
        assert swa_metadata.num_prefills == 0
        num_decodes = swa_metadata.num_decodes
        num_tokens = swa_metadata.num_decode_tokens
        if num_tokens == 0:
            return
        swa_only = attn.compress_ratio == 0
        swa_k_cache = attn.swa_cache_layer.kv_cache
        compressed_kv_cache, seq_lens, sparse_indices, sparse_topk_lens = (
            self.tables.lookup(
                attn,
                None if swa_only else attn._compressed_kv_cache(),
                swa_k_cache,
                swa_metadata,
                attn_metadata,
                swa_only,
            )
        )

        # CUDA graph execution can pad q/out past the scheduled token count.
        query = q[:num_tokens]
        output = out[:num_tokens]
        bmm1_scale: float | torch.Tensor = attn.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if attn.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = attn._flashinfer_fp8_bmm1_scale
            bmm2_scale = attn._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()
        # TRTLLM-gen requires h_q in {64, 128}: zero-pad the query heads.
        padded_heads = output.shape[1]
        if query.shape[1] < padded_heads:
            padded_query = query.new_zeros(
                (query.shape[0], padded_heads, query.shape[2])
            )
            padded_query[:, : query.shape[1], :] = query
            query = padded_query

        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc is not None
        decode_cu: torch.Tensor | None = query_start_loc[: num_decodes + 1]
        decode_seq_lens = seq_lens[:num_decodes]
        decode_topk_lens = sparse_topk_lens[:num_tokens]
        max_decode_query_len = swa_metadata.max_decode_query_len
        if swa_metadata.decode_swa_width > attn.window_size:
            # DSpark's non-causal window, as in the stock decode.
            assert swa_only
            assert swa_metadata.flashinfer_decode_topk_lens is not None
            assert swa_metadata.flashinfer_decode_seq_lens is not None
            decode_topk_lens = swa_metadata.flashinfer_decode_topk_lens
            decode_seq_lens = swa_metadata.flashinfer_decode_seq_lens
            query = query.unsqueeze(1)
            output = output.unsqueeze(1)
            decode_cu = None
            max_decode_query_len = 1
        decode_indices = sparse_indices[:num_tokens]
        # The CuTe kernel reads the first SWA_COLUMNS columns from the SWA pool.
        if (
            attn.window_size == SWA_COLUMNS
            and swa_metadata.decode_swa_width == SWA_COLUMNS
            and supports_sparse_mla_cute(
                query,
                swa_k_cache,
                compressed_kv_cache,
                decode_indices,
                decode_topk_lens,
                output,
                bmm1_scale,
                bmm2_scale,
                attn.attn_sink,
            )
        ):
            sparse_mla_decode(
                query,
                swa_k_cache,
                compressed_kv_cache,
                decode_indices,
                decode_topk_lens,
                output,
                cast(float, bmm1_scale),
                cast(float, bmm2_scale),
                attn.attn_sink,
            )
            return
        trtllm_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=swa_k_cache,
            compressed_kv_cache=compressed_kv_cache,
            workspace_buffer=_get_flashinfer_dsv4_workspace(q.device),
            sparse_indices=decode_indices,
            sparse_topk_lens=decode_topk_lens,
            seq_lens=decode_seq_lens,
            out=output,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=attn.attn_sink,
            cum_seq_lens_q=decode_cu,
            max_q_len=max_decode_query_len,
            kv_counters=self.kv_counters,
        )
