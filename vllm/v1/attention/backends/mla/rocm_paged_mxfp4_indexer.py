# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 indexer backend for the ROCm paged MXFP4 path.

Extends the dense indexer metadata with what aiter's paged MXFP4 MQA-logits
kernel needs to read the cache in place: each prefill chunk split into its
requests (or packed through query_start_loc when their rows differ), the decode
rows as next_n-row sequences on uniform steps, and, with
``AttentionConfig.indexer_sparse_logits``, the per-step state the candidate
consumers share. Selected for every V4.1 indexer cache when
``indexer_kv_dtype="mxfp4"`` on ROCm.
"""

import bisect
from dataclasses import dataclass, field, fields

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV41IndexerBackend,
)
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    MAX_LOGITS_BYTES,
    build_rocm_mxfp4_decode_schedule,
    rocm_mxfp4_consumer_rows,
    rocm_mxfp4_decode_schedule_words,
    rocm_paged_mxfp4_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

logger = init_logger(__name__)

# The consumers walk the pool instead of the context once the context is this
# many pools long. Not measured end to end yet: at speculative decode the
# gather loses to the dense walk until the pool is about a 3x cut (32 heads),
# and at prefill it issues 1.15-1.19x the dense walk's loads per key.
_GATHER_MIN_CUT_DECODE = 3.0
_GATHER_MIN_CUT_PREFILL = 1.5


@dataclass
class RocmMxfp4PrefillPlan:
    """One prefill chunk as the requests the kernel launches on. A request's
    rows are one sequence's query chunk, which is the kernel's next_n."""

    requests: list[tuple[int, int, int]]
    """(first row, end row, request) per request with rows in the chunk; rows
    count from ``chunk.token_start``, requests index ``chunk.block_table``."""
    launches: list[tuple[int, int, int, int]]
    """(first row, end row, first request, requests) per dense launch. The
    kernel's sequences all have next_n rows, so a launch takes a run of
    consecutive requests with equal query rows."""
    width: int
    """Logits columns: an upper bound on the chunk's compressed contexts."""
    row_ends: torch.Tensor
    """[rows] int32 exclusive compressed key bound of each row."""
    context_lens: torch.Tensor
    """[chunk.num_reqs] int32 compressed context of each request."""
    first_request: int
    """The step's index of the chunk's first request."""
    block_ends: torch.Tensor | None = None
    """[rows] int32 candidate blocks each row sees, for the source's pool."""
    use_gather: bool = False
    query_start_loc: torch.Tensor | None = None
    """[chunk.num_reqs + 1] int32 row offsets of the chunk's requests when they
    launch packed, as one varlen launch; None launches per run instead."""


@dataclass
class RocmMxfp4GatherLaunch:
    """Query rows [token_start, token_end) of one request, which the candidate
    consumers score against the pool in one launch."""

    token_start: int
    token_end: int
    block_table: torch.Tensor
    """[1, max_blocks] int32, the request's block table row."""
    context_len: torch.Tensor
    """[1] int32 compressed context of the request."""
    row_ends: torch.Tensor
    """[rows] int32 exclusive compressed key bound of each row."""


@dataclass
class RocmMxfp4NativeDecode:
    """A decode step whose requests all have next_n query rows, as the
    kernel's sequences: a workgroup then walks a KV tile once for up to
    next_n rows instead of once per row."""

    next_n: int
    context_lens: torch.Tensor
    """[requests] int32 compressed context of each request."""
    block_table: torch.Tensor
    """[requests, max_blocks] int32, each request's first flattened row."""


def native_decode(
    row_lens: torch.Tensor,
    row_block_table: torch.Tensor,
    query_lens: list[int],
    next_n: int,
    context_lens: torch.Tensor,
) -> RocmMxfp4NativeDecode | None:
    """The step as next_n-row sequences, or None when a request has fewer rows.

    ``row_lens`` and ``row_block_table`` are the flattened rows' bounds and
    block table, ``query_lens`` the decode requests' query lengths. Trailing
    empty requests are cudagraph padding: they keep their next_n rows, with no
    context. Only the step's shape decides, so a FULL graph captured on a
    uniform batch replays the same launch on a padded one.
    """
    num_reqs = len(query_lens)
    num_full = next((i for i, n in enumerate(query_lens) if n != next_n), num_reqs)
    if (
        next_n <= 1
        or row_lens.shape[0] != num_reqs * next_n
        or any(query_lens[num_full:])
    ):
        return None
    return RocmMxfp4NativeDecode(
        next_n, context_lens[:num_reqs], row_block_table[::next_n]
    )


@dataclass
class DeepseekV41RocmMxfp4IndexerMetadata(DeepseekV32IndexerMetadata):
    prefill_plans: list[RocmMxfp4PrefillPlan] = field(default_factory=list)
    """Parallel to ``prefill.chunks``."""
    decode_row_lens: torch.Tensor | None = None
    """[rows] int32 compressed context of each decode query row."""
    decode_block_ends: torch.Tensor | None = None
    decode_native: RocmMxfp4NativeDecode | None = None
    """The same rows as next_n-row sequences, for the dense launches."""
    decode_schedule: torch.Tensor | None = None
    """The dense decode launches' work schedule, built once for the step."""
    decode_use_gather: bool = False
    decode_gather: tuple[dict, torch.Tensor] | None = None
    """The pool the first consumer resolved for the decode rows."""
    gather_launches: list[RocmMxfp4GatherLaunch] = field(default_factory=list)
    """The prefill rows of the chunks that gather, as the consumers launch
    them."""


def plan_prefill_chunks(
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata],
    query_start_loc: list[int],
    seq_lens_cpu: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int,
    candidate_block_size: int,
    min_gather_width: float | None,
    query_start_loc_device: torch.Tensor | None = None,
) -> list[RocmMxfp4PrefillPlan]:
    """Split each chunk into its requests.

    ``query_start_loc`` and ``seq_lens_cpu`` are the step's host copies (the
    latter an upper bound), ``context_lens`` the device compressed lengths.
    A chunk covers whole requests, or a query slice of one request. Given the
    device ``query_start_loc``, a chunk whose requests do not all share one
    query length launches once, packed, instead of once per run.
    """
    block = candidate_block_size
    plans = []
    for chunk in chunks:
        t0, t1 = chunk.token_start, chunk.token_end
        first = bisect.bisect_right(query_start_loc, t0) - 1
        requests = []
        launches: list[tuple[int, int, int, int]] = []
        for req in range(chunk.num_reqs):
            lo = max(query_start_loc[first + req], t0) - t0
            hi = min(query_start_loc[first + req + 1], t1) - t0
            if hi <= lo:
                continue
            requests.append((lo, hi, req))
            if launches:
                run_lo, run_hi, run_req, seqs = launches[-1]
                if run_req + seqs == req and run_hi - run_lo == seqs * (hi - lo):
                    launches[-1] = (run_lo, hi, run_req, seqs + 1)
                    continue
            launches.append((lo, hi, req, 1))
        reqs = slice(first, first + chunk.num_reqs)
        packed = None
        if query_start_loc_device is not None and len(launches) > 1:
            packed = torch.clamp(query_start_loc_device[first : reqs.stop + 1], t0, t1)
            packed -= t0
        width = int(seq_lens_cpu[reqs].max()) // compress_ratio
        row_ends = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
        plans.append(
            RocmMxfp4PrefillPlan(
                requests=requests,
                launches=launches,
                width=width,
                row_ends=row_ends,
                context_lens=context_lens[reqs],
                first_request=first,
                block_ends=(row_ends + block - 1) // block if block else None,
                use_gather=min_gather_width is not None and width >= min_gather_width,
                query_start_loc=packed,
            )
        )
    return plans


def split_prefill_chunks(
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """(request slice, query slice) chunks, as the base chunker returns them.

    Nothing gathers K here, so the requests' contexts do not add up: a chunk's
    logits are [rows, widest compressed context] fp32, at most
    ``max_logits_bytes``. A request too wide to launch whole is cut on its
    query rows.
    """
    budget = min(max_logits_bytes, MAX_LOGITS_BYTES)

    def max_rows(width: int) -> int:
        return max(1, budget // (4 * max(width, 1)))

    chunks: list[tuple[slice, slice]] = []
    end = 0
    while end < len(context_lens):
        start, rows, width = end, 0, 0
        while end < len(context_lens):
            q = int(query_lens[end])
            w = max(width, int(context_lens[end]))
            if rows and rows + q > max_rows(w):
                break
            rows, width = rows + q, w
            end += 1
        step = max_rows(width)
        reqs = slice(start + request_offset, end + request_offset)
        chunks.extend(
            (reqs, slice(lo, min(lo + step, rows))) for lo in range(0, rows, step)
        )
    return chunks


def plan_gather_launches(
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata],
    plans: list[RocmMxfp4PrefillPlan],
    max_rows: int,
) -> list[RocmMxfp4GatherLaunch]:
    """The candidate consumers' launches over the chunks that gather.

    A consumer's logits are [rows, pool] however long the context, so the
    dense split, sized for the context, would launch it far more often than
    its memory needs. Each request's rows go back together across the chunks
    that cut them, then out again at most ``max_rows`` a launch.
    """
    # Per request: (first token, [(plan, first row, end row)]), and the chunk
    # and request index its block table and context come from.
    runs: list[tuple[int, list, DeepseekV32IndexerPrefillChunkMetadata, int]] = []
    last: tuple[int, int] | None = None
    for chunk, plan in zip(chunks, plans):
        if not plan.use_gather:
            last = None
            continue
        for lo, hi, req in plan.requests:
            request, t_lo = plan.first_request + req, chunk.token_start + lo
            if last != (request, t_lo):
                runs.append((t_lo, [], chunk, req))
            runs[-1][1].append((plan, lo, hi))
            last = (request, chunk.token_start + hi)
    launches = []
    for token_start, pieces, chunk, req in runs:
        ends = [plan.row_ends[lo:hi] for plan, lo, hi in pieces]
        row_ends = ends[0] if len(ends) == 1 else torch.cat(ends)
        block_table = chunk.block_table[req : req + 1]
        context_len = pieces[0][0].context_lens[req : req + 1]
        for lo in range(0, row_ends.shape[0], max_rows):
            hi = min(lo + max_rows, row_ends.shape[0])
            launches.append(
                RocmMxfp4GatherLaunch(
                    token_start + lo,
                    token_start + hi,
                    block_table,
                    context_len,
                    row_ends[lo:hi],
                )
            )
    return launches


class DeepseekV41RocmMxfp4IndexerBackend(DeepseekV41IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V41_ROCM_MXFP4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # 128 is preferred (see DeepseekV4ROCMAiterMLASparseBackend); 64 keeps
        # an explicit --block-size 64 working.
        return [64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV41RocmMxfp4IndexerMetadataBuilder"]:
        return DeepseekV41RocmMxfp4IndexerMetadataBuilder


class DeepseekV41RocmMxfp4IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Dense indexer metadata plus the in-place paged launches' layout.

    Requirements are checked here, at engine start.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        # Decode rows are flattened into persistent buffers, and whether a
        # step launches them as next_n-row sequences depends on its shape
        # alone (see native_decode).
        return AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)
        name = "the ROCm MXFP4 indexer"
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                f"{name} does not support decode/prefill context parallel."
            )
        hf_config = vllm_config.model_config.hf_text_config
        num_heads, head_dim = hf_config.index_n_heads, hf_config.index_head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.page_entries = kv_cache_spec.block_size // self.compress_ratio
        layout = rocm_paged_mxfp4_cache_layout(num_heads, head_dim, self.page_entries)
        # The consumers' gather and ragged steps take a row per query token.
        # Uniform steps also launch the dense layers on next_n-row sequences,
        # see native_decode.
        self.use_flattening = True

        self.candidate_block_size = getattr(hf_config, "candidate_block_size", 0)
        self.num_candidate_cols = 0
        if vllm_config.attention_config.indexer_sparse_logits:
            prefix = "attention_config.indexer_sparse_logits"
            block = self.candidate_block_size
            topk_blocks = getattr(hf_config, "candidate_topk_blocks", 0)
            if block <= 0 or topk_blocks <= 0:
                raise ValueError(
                    f"{prefix} requires a two-level (candidate block) indexer "
                    "model such as DeepSeek-V4.1."
                )
            if topk_blocks & (topk_blocks - 1):
                raise ValueError(
                    f"{prefix} on ROCm resolves the candidate pool with a sort "
                    f"that needs a power-of-two block count, not {topk_blocks}."
                )
            if self.page_entries % block or block > layout.n_per_tile:
                raise ValueError(
                    f"{prefix} on ROCm needs {block}-entry candidate blocks to "
                    f"tile a {self.page_entries}-entry indexer page and fit in "
                    f"one {layout.n_per_tile}-entry shuffle group."
                )
            self.num_candidate_cols = topk_blocks * block
            self.gather_rows = rocm_mxfp4_consumer_rows(self.num_candidate_cols)
            logger.info_once(
                "DeepSeek V4.1 indexer: candidate consumers score the %d "
                "candidate positions with aiter's paged MXFP4 gather once the "
                "context is %.1fx (decode) / %.1fx (prefill) as long.",
                self.num_candidate_cols,
                _GATHER_MIN_CUT_DECODE,
                _GATHER_MIN_CUT_PREFILL,
            )

        int32 = dict(dtype=torch.int32, device=device)
        self.decode_block_ends_buffer = torch.zeros(
            self.arange_buffer.shape[0], **int32
        )
        self.context_lens_buffer = torch.zeros(self.arange_buffer.shape[0], **int32)
        spec_config = vllm_config.speculative_config
        # Adaptive verification replays a FULL graph on drafts reallocated
        # across requests, so a step's shape does not fix its query lengths.
        adaptive = spec_config is not None and spec_config.enable_adaptive_verification
        self.native_next_n = 1 if adaptive else self.num_speculative_tokens + 1
        # The indexer's logits width, which the decode schedule sizes slices by.
        max_model_len = vllm_config.model_config.max_model_len
        self.logits_width = max_model_len // self.compress_ratio
        # Every dense layer of this group reads the same rows through the same
        # page geometry, so the step builds their schedule once; the scheduler
        # launch costs about half a decode launch.
        self.decode_schedule_buffer = torch.empty(
            rocm_mxfp4_decode_schedule_words(
                num_heads, head_dim, self.page_entries, self.native_next_n
            ),
            **int32,
        )

    def _split_indexer_prefill_chunks(  # type: ignore[override]
        self,
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        workspace_size: int,
        max_logits_bytes: int,
        request_offset: int = 0,
    ) -> list[tuple[slice, slice]]:
        return split_prefill_chunks(
            compressed_seq_lens_cpu,
            prefill_query_lens_cpu,
            max_logits_bytes,
            request_offset,
        )

    def _gather_pays(self, context: int, cut: float) -> bool:
        return self.num_candidate_cols > 0 and context >= cut * self.num_candidate_cols

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV41RocmMxfp4IndexerMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = DeepseekV41RocmMxfp4IndexerMetadata(
            **{f.name: getattr(base, f.name) for f in fields(base)}
        )
        cm = common_attn_metadata
        context_lens = self.context_lens_buffer[: cm.num_reqs]
        torch.floor_divide(
            cm.seq_lens[: cm.num_reqs], self.compress_ratio, out=context_lens
        )
        if base.decode is not None:
            lengths = base.decode.seq_lens.view(-1)
            metadata.decode_row_lens = lengths
            block = self.candidate_block_size
            if block:
                ends = self.decode_block_ends_buffer[: lengths.shape[0]]
                torch.add(lengths, block - 1, out=ends)
                ends.floor_divide_(block)
                metadata.decode_block_ends = ends
            query_start_loc = cm.query_start_loc_cpu[: base.num_decodes + 1]
            metadata.decode_native = native_decode(
                lengths,
                base.decode.block_table,
                torch.diff(query_start_loc).tolist(),
                self.native_next_n,
                context_lens,
            )
            metadata.decode_schedule = build_rocm_mxfp4_decode_schedule(
                lengths,
                self.num_heads,
                self.head_dim,
                self.page_entries,
                self.decode_schedule_buffer,
                self.logits_width,
                metadata.decode_native,
            )
            metadata.decode_use_gather = self._gather_pays(
                base.max_seq_len // self.compress_ratio, _GATHER_MIN_CUT_DECODE
            )
        if base.prefill is not None:
            assert cm.seq_lens_cpu_upper_bound is not None
            metadata.prefill_plans = plan_prefill_chunks(
                base.prefill.chunks,
                cm.query_start_loc_cpu.tolist(),
                cm.seq_lens_cpu_upper_bound,
                context_lens,
                self.compress_ratio,
                self.candidate_block_size,
                self.num_candidate_cols * _GATHER_MIN_CUT_PREFILL
                if self.num_candidate_cols
                else None,
                cm.query_start_loc if envs.VLLM_ROCM_MXFP4_INDEXER_VARLEN else None,
            )
            if self.num_candidate_cols:
                metadata.gather_launches = plan_gather_launches(
                    base.prefill.chunks, metadata.prefill_plans, self.gather_rows
                )
        return metadata
