# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV hierarchical Quest + KIVI retrieval pipeline."""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar
import torch

from vllm.logger import init_logger
from vllm.v1.attention.ops.zoomkv import stage_timer as _zt
from vllm.v1.attention.ops.zoomkv.kernels import (
    chunk_density_scores,
    density_score_physical,
    dense_mask_from_topk,
    direct_physical_retrieval_available,
    float_topk_3d_varlen,
    float_topk_values_3d,
    float_topk_values_3d_varlen,
    get_quest_ops,
    kivi_physical,
    quest_parent_score_physical,
    quest_sub_score_physical,
)

from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk
from vllm.v1.attention.ops.zoomkv.retrieval_metadata_triton import (
    build_actual_num_chunks,
    build_stage_budgets,
)
from vllm.v1.attention.ops.zoomkv.state import ZoomKVBlockSummary

_RETRIEVE_DETAIL_TIMER = (os.environ.get("VLLM_ZOOMKV_RETRIEVE_STAGE_TIMER", "0") == "1")
logger = init_logger(__name__)


class _NoopStage:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


_NOOP_STAGE = _NoopStage()


def _retrieve_stage(name: str) -> _zt.Stage | _NoopStage:
    if _RETRIEVE_DETAIL_TIMER:
        return _zt.Stage(f"retrieve.{name}")
    return _NOOP_STAGE


@dataclass
class ZoomKVRuntimeConfig:
    sink_size: int = 64
    local_size: int = 256
    max_model_len: int = 131072
    final_topk: int = 100
    quest_chunk: int = 16
    quest_large_chunk: int = 256
    quest_large_ratio: float = 0.5
    quest_small_ratio: float = 0.3
    dense_ratio: float = 0.4
    dense_topk: int = 8
    sparse_topk: int = 4
    max_small_candidates: int = 1024
    full_attention_threshold: int = 2000
    dense_fallback: bool = False
    strict_kernels: bool = False
    enable_offload: bool = False

    @property
    def hq_factor(self) -> int:
        return max(1, self.quest_large_chunk // self.quest_chunk)


@dataclass(frozen=True)
class ZoomKVRetrievalResult:
    topk: torch.Tensor
    context_fully_valid: bool
    used_direct_physical: bool


@dataclass(frozen=True)
class _CachedBatchMeta:
    """Layer-/step-shared retrieval geometry (seq_lens → chunk widths)."""

    key: tuple[Any, ...]
    actual_num_chunks: torch.Tensor
    bucket: int
    start_b: int
    sink_local_full: bool


@dataclass(frozen=True)
class _StageBudgets:
    parent_lengths: torch.Tensor
    large_ks: torch.Tensor
    sub_lengths: torch.Tensor
    small_ks: torch.Tensor
    dense_ks: torch.Tensor
    final_ks: torch.Tensor
    max_large: int
    max_small: int
    max_dense: int


@dataclass
class _RetrievalCudaGraph:
    graph: torch.cuda.CUDAGraph
    batch_size: int
    raw_q: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    topk: torch.Tensor
    actual_num_chunks: torch.Tensor


def _topk_3d(scores: torch.Tensor, k: int, strict: bool = False) -> torch.Tensor:
    from vllm.v1.attention.ops.zoomkv.kernels import float_topk_3d
    return float_topk_3d(scores, k, strict=strict)


def prepare_retrieval_query(
    query: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Average Q heads within each GQA group -> [bs, num_kv_heads, head_dim].

    Direct physical kernels require a dense contiguous [B, H, D] layout.
    """
    if query.dim() != 3:
        raise ValueError(f"Unexpected query shape {tuple(query.shape)}")
    bs, hq, d = query.shape
    if hq == num_kv_heads:
        return query.contiguous()
    if hq % num_kv_heads != 0:
        raise ValueError(f"Hq={hq} not divisible by Hkv={num_kv_heads}")
    return query.reshape(bs, num_kv_heads, -1, d).mean(dim=2).contiguous()



class ZoomKVRetriever:
    # Shared across per-layer retriever instances: metadata depends only on
    # seq_lens / block-table capacity, not on layer weights or Q.
    _batch_meta_cache: ClassVar[_CachedBatchMeta | None] = None

    def __init__(self, cfg: ZoomKVRuntimeConfig) -> None:
        self.cfg = cfg
        self.quest = get_quest_ops(prefer_triton=True, strict=cfg.strict_kernels)
        # Scratch-buffer cache to align with the reference implementation:
        # score/index tensors depend only on (n_chunks, n_large, kv_heads), which stay constant across all layers of a decode step (and only change when a new block completes ~every block_size tokens).  Reusing
        # them removes per-layer/per-step allocations in the retrieve hot path.
        self._scratch: dict[tuple[Any, ...], torch.Tensor] = {}
        self._mixed_cudagraphs: dict[tuple[Any, ...], _RetrievalCudaGraph] = {}
        self._mixed_cudagraph_disabled = False
        # Set by retrieve_* when sink+local+topk are guaranteed fully valid,
        # so the attention backend can skip host-side ``_ctx_valid.all()`` sync.
        self.last_context_fully_valid: bool = False
        self._last_topk_fully_filled: bool = False

    @classmethod
    def clear_batch_meta_cache(cls) -> None:
        cls._batch_meta_cache = None

    @staticmethod
    def _actual_num_chunks_host(
        host_lens: torch.Tensor,
        *,
        sink_size: int,
        local_size: int,
        block_size: int,
        start_block: int,
        max_chunks: int,
    ) -> list[int]:
        """CPU mirror of ``build_actual_num_chunks`` (no Triton / no D2H)."""
        out: list[int] = []
        for seq_len in host_lens.tolist():
            local_start = max(sink_size, int(seq_len) - local_size)
            actual = local_start // block_size - start_block
            out.append(max(0, min(actual, max_chunks)))
        return out

    def _scratch_buf(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        fill: float | None = None,
    ) -> torch.Tensor:
        cache_key = (key, tuple(shape), dtype, device)
        buf = self._scratch.get(cache_key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._scratch[cache_key] = buf
        if fill is not None:
            buf.fill_(fill)
        return buf

    @staticmethod
    def _cudagraph_batch_bucket(batch: int) -> int:
        """Smallest power-of-two request bucket covering ``batch``."""
        return 1 << max(0, batch - 1).bit_length()

    def _capture_mixed_retrieval_graph(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        chunk_bucket: int,
    ) -> _RetrievalCudaGraph:
        """Lazily capture the fixed-shape physical retrieval pipeline."""
        cfg = self.cfg
        batch = self._cudagraph_batch_bucket(raw_q.shape[0])
        device = raw_q.device
        start_b = cfg.sink_size // block_summary.block_size
        local_blocks = (
            cfg.local_size + block_summary.block_size - 1
        ) // block_summary.block_size
        available_chunks = max(
            0, block_table.shape[1] - start_b - local_blocks
        )
        max_chunks = min(chunk_bucket, available_chunks)

        static_q = torch.zeros(
            (batch, raw_q.shape[1], raw_q.shape[2]),
            dtype=raw_q.dtype,
            device=device,
        )
        static_bt = torch.full(
            (batch, block_table.shape[1]),
            -1,
            dtype=block_table.dtype,
            device=device,
        )
        static_seq = torch.zeros(
            (batch,), dtype=seq_lens.dtype, device=device
        )
        static_topk = torch.full(
            (batch, raw_q.shape[1], cfg.final_topk),
            -1,
            dtype=torch.int64,
            device=device,
        )
        actual_num_chunks = self._scratch_buf(
            "cudagraph_actual_num_chunks",
            (batch,),
            torch.int32,
            device,
        )
        physical_ids = static_bt[:, start_b:]

        def run() -> None:
            build_actual_num_chunks(
                static_seq,
                actual_num_chunks,
                sink_size=cfg.sink_size,
                local_size=cfg.local_size,
                block_size=block_summary.block_size,
                start_block=start_b,
                max_chunks=max_chunks,
            )
            self._retrieve_topk_physical(
                static_q,
                block_summary,
                physical_ids,
                chunk_bucket,
                start_b * block_summary.block_size,
                actual_num_chunks=actual_num_chunks,
                topk_out=static_topk,
            )

        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            # Populate every lazy scratch/output allocation before capture.
            run()
            run()
        current_stream.wait_stream(capture_stream)
        current_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            run()
        current_stream.wait_stream(capture_stream)

        logger.info_once(
            "Captured ZoomKV mixed retrieval CUDA graph: "
            "batch_bucket=%d, chunk_bucket=%d",
            batch,
            chunk_bucket,
        )
        return _RetrievalCudaGraph(
            graph=graph,
            batch_size=batch,
            raw_q=static_q,
            block_table=static_bt,
            seq_lens=static_seq,
            topk=static_topk,
            actual_num_chunks=actual_num_chunks,
        )

    def _retrieve_mixed_cudagraph(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        chunk_bucket: int,
    ) -> torch.Tensor | None:
        """Replay retrieval-only CUDA graph, falling back safely on capture errors."""
        if self._mixed_cudagraph_disabled or not raw_q.is_cuda:
            return None
        batch = raw_q.shape[0]
        batch_bucket = self._cudagraph_batch_bucket(batch)
        key = (
            raw_q.device,
            raw_q.dtype,
            batch_bucket,
            raw_q.shape[1],
            raw_q.shape[2],
            block_table.shape[1],
            chunk_bucket,
            block_summary.chunk_min.data_ptr(),
        )
        state = self._mixed_cudagraphs.get(key)
        if state is None:
            try:
                state = self._capture_mixed_retrieval_graph(
                    raw_q,
                    block_summary,
                    block_table,
                    seq_lens,
                    chunk_bucket,
                )
            except Exception:
                self._mixed_cudagraph_disabled = True
                logger.exception(
                    "ZoomKV mixed retrieval CUDA graph capture failed; "
                    "falling back to eager retrieval"
                )
                return None
            self._mixed_cudagraphs[key] = state

        state.raw_q.zero_()
        state.raw_q[:batch].copy_(raw_q)
        state.block_table.fill_(-1)
        state.block_table[:batch].copy_(block_table[:batch])
        state.seq_lens.zero_()
        state.seq_lens[:batch].copy_(seq_lens[:batch])
        state.topk.fill_(-1)
        state.graph.replay()
        self._last_topk_fully_filled = True
        return state.topk[:batch]




    @staticmethod
    def _chunk_bucket(n_chunks: int) -> int:
        """Round a retrieval width to a stable power-of-two bucket."""
        if n_chunks <= 0:
            return 0
        return max(16, 1 << (int(n_chunks) - 1).bit_length())


    def _topk_out(
        self,
        out: torch.Tensor | None,
        batch: int,
        kv_heads: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if out is None:
            return None
        expected = (batch, kv_heads, self.cfg.final_topk)
        if (tuple(out.shape) != expected or out.dtype != torch.int64 or out.device != device or not out.is_contiguous()):
            raise ValueError(
                "topk_out must be contiguous int64 "
                f"{expected} on {device}, got {tuple(out.shape)} "
                f"{out.dtype} on {out.device}"
            )
        return out


    def _empty_batch_topk_result(
        self,
        batch: int,
        kv_heads: int,
        device: torch.device,
        topk_out: torch.Tensor | None,
    ) -> ZoomKVRetrievalResult:
        topk = topk_out
        if topk is None:
            topk = torch.empty((batch, kv_heads, self.cfg.final_topk),dtype=torch.int64,device=device)
        topk.fill_(-1)
        return ZoomKVRetrievalResult(topk=topk,context_fully_valid=False,used_direct_physical=False        )


    @staticmethod
    def _require_direct_physical(
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
    ) -> None:
        if not ZoomKVRetriever._can_use_direct_physical(
            raw_q, block_summary, physical_ids
        ):
            raise RuntimeError(
                "ZoomKV sparse retrieval requires the direct physical CUDA "
                "path (BF16, D=128/256, int32 physical_ids, vllm._zoomkv_C)."
            )


    def should_use_dense(self, seq_len: int) -> bool:
        del seq_len  # Decode always uses sparse unless dense_fallback is set.
        return bool(self.cfg.dense_fallback)



    def retrieve_topk_from_block_summaries(
        self,
        raw_q: torch.Tensor,
        packed: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        centroid: torch.Tensor,
        valid: torch.Tensor,
        seq_len: int,
        block_size: int,
        start_b: int,
    ) -> torch.Tensor:
        """Run Quest+KIVI on pre-gathered CPU-slot or physical summaries."""
        cfg = self.cfg
        batch = raw_q.shape[0]
        n_chunks = packed.shape[2]
        if n_chunks <= 0:
            return torch.full((batch, raw_q.shape[1], cfg.final_topk),-1,dtype=torch.int64,device=raw_q.device)
        factor = cfg.hq_factor
        parent_min, parent_max, parent_valid = self._parent_minmax_from_children(cmin, cmax, valid, factor)
        chunk_idx = self._hierarchical_quest(raw_q, cmin, cmax, parent_min, parent_max, parent_valid, n_chunks, factor)
        topk_local = self._cds_select(chunk_idx, packed, cmin, cmax, centroid, raw_q, block_size)
        ret_token_offset = start_b * block_size
        return torch.where(topk_local >= 0,topk_local + ret_token_offset,torch.full_like(topk_local, -1))





    def retrieval_block_range(self, seq_len: int, block_size: int) -> tuple[int, int]:
        """Return [start_block, end_block) of retrieval-zone child chunks."""
        sink_blocks = self.cfg.sink_size // block_size
        local_tokens = min(self.cfg.local_size, max(0, seq_len - self.cfg.sink_size))
        local_start = max(self.cfg.sink_size, seq_len - local_tokens)
        # Only fully completed child chunks in the retrieval zone.
        ret_start = self.cfg.sink_size
        ret_end = (local_start // block_size) * block_size
        if ret_end <= ret_start:
            return sink_blocks, sink_blocks
        start_b = ret_start // block_size
        end_b = ret_end // block_size
        return start_b, end_b



    def retrieve_topk_tokens(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_block_ids: torch.Tensor,
        seq_len: int,
        cache_key: tuple | None = None,
    ) -> torch.Tensor:
        """Run Quest + KIVI and return logical token indices in full-seq coords.

        Production sparse decode uses the batched direct path; this entry
        point remains for the offload serial loop and tests.
        """
        del cache_key
        cfg = self.cfg
        block_size = block_summary.block_size
        start_b, end_b = self.retrieval_block_range(seq_len, block_size)
        n_ret = end_b - start_b
        if n_ret <= 0 or physical_block_ids.numel() == 0:
            return torch.full((raw_q.shape[0], raw_q.shape[1], cfg.final_topk),-1,dtype=torch.int64,device=raw_q.device)

        ids = physical_block_ids[:n_ret]
        bucket = self._chunk_bucket(n_ret)
        if bucket <= 0:
            return torch.full((raw_q.shape[0], raw_q.shape[1], cfg.final_topk),-1,dtype=torch.int64,device=raw_q.device)
        direct_ids = self._scratch_buf("physical_ids",(raw_q.shape[0], bucket),torch.int32,raw_q.device,fill=-1)
        direct_ids[:, :n_ret].copy_(ids.reshape(raw_q.shape[0], n_ret).to(torch.int32))
        self._require_direct_physical(raw_q, block_summary, direct_ids)
        return self._retrieve_topk_physical(raw_q,block_summary,direct_ids,bucket,start_b * block_size,actual_num_chunks=torch.full((raw_q.shape[0],),n_ret,dtype=torch.int32,device=raw_q.device))




    def retrieve_topk_tokens_batch(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        topk_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper returning only Top-K logical token ids."""
        return self.retrieve_topk_tokens_batch_result(raw_q,block_summary,block_table,seq_lens,summaries_guaranteed_valid=False,topk_out=topk_out).topk





    def retrieve_topk_tokens_batch_result(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        summaries_guaranteed_valid: bool,
        topk_out: torch.Tensor | None = None,
        assume_context_fully_valid: bool = False,
        chunk_bucket: int | None = None,
        seq_lens_host: torch.Tensor | None = None,
        use_cudagraph: bool = False,
    ) -> ZoomKVRetrievalResult:
        """Batched Quest + KIVI over a decode batch of requests.
        Args:
            raw_q: [B, kv_heads, D]
            block_table: [B, max_blocks] physical block ids
            seq_lens: [B] full sequence lengths (CPU or GPU)
        Returns:
            Explicit Top-K/direct-path validity result. The caller may use
            unmasked FlashAttention only when ``context_fully_valid`` is true.
        """
        cfg = self.cfg
        batch = raw_q.shape[0]
        block_size = block_summary.block_size
        device = raw_q.device
        topk_out = self._topk_out(topk_out, batch, raw_q.shape[1], device)
        self.last_context_fully_valid = False
        self._last_topk_fully_filled = False
        if use_cudagraph and chunk_bucket is not None:
            graph_topk = self._retrieve_mixed_cudagraph(
                raw_q,
                block_summary,
                block_table,
                seq_lens,
                int(chunk_bucket),
            )
            if graph_topk is not None:
                host_lens = seq_lens_host
                sink_local_full = (
                    assume_context_fully_valid
                    or host_lens is not None
                    and all(
                        int(sl) >= cfg.sink_size + cfg.local_size
                        for sl in host_lens.tolist()
                    )
                )
                self.last_context_fully_valid = (
                    summaries_guaranteed_valid
                    and sink_local_full
                    and self._last_topk_fully_filled
                )
                return ZoomKVRetrievalResult(
                    topk=graph_topk,
                    context_fully_valid=self.last_context_fully_valid,
                    used_direct_physical=True,
                )
        with _retrieve_stage("metadata"):
            start_b = cfg.sink_size // block_size
            host_lens = seq_lens_host
            if host_lens is None and not seq_lens.is_cuda:
                host_lens = seq_lens
            # The block table also contains the fixed local-attention tail.
            # Exclude it from the static retrieval upper bound so a uniform
            # batch lands in the tight bucket without inspecting GPU values.
            local_blocks = (cfg.local_size + block_size - 1) // block_size
            available_chunks = max(0, block_table.shape[1] - start_b - local_blocks)
            # CPU metadata can cheaply choose the tight batch bucket. CUDA
            # metadata deliberately avoids .item()/tolist and uses the static
            # block-table upper bound, which is graph-capture friendly.
            if chunk_bucket is not None:
                max_chunks = min(int(chunk_bucket), available_chunks)
            elif host_lens is not None:
                max_seq_len = int(host_lens.max().item()) if batch else 0
                _, max_end_b = self.retrieval_block_range(max_seq_len, block_size)
                max_chunks = max(0, max_end_b - start_b)
            else:
                max_chunks = available_chunks
            bucket = (
                int(chunk_bucket)
                if chunk_bucket is not None
                else self._chunk_bucket(max_chunks)
            )
            if bucket > self._chunk_bucket(available_chunks):
                raise ValueError(
                    f"chunk_bucket={bucket} exceeds block-table capacity "
                    f"{available_chunks}"
                )
            clamp_chunks = min(available_chunks, bucket)
            # Geometry depends only on seq_lens + capacity, not on layer / Q.
            # Cache across the 36 layer calls in a decode step, and across the
            # 15/16 steps where no new retrieval block enters the zone.
            if host_lens is not None:
                host_lens_list = host_lens.tolist()
                actuals = [
                    max(
                        0,
                        min(
                            max(cfg.sink_size, int(sl) - cfg.local_size)
                            // block_size
                            - start_b,
                            clamp_chunks,
                        ),
                    )
                    for sl in host_lens_list
                ]
                sink_local_full = assume_context_fully_valid or all(
                    int(sl) >= cfg.sink_size + cfg.local_size
                    for sl in host_lens_list
                )
                meta_key = (
                    device.type,
                    device.index,
                    batch,
                    start_b,
                    available_chunks,
                    bucket,
                    tuple(actuals),
                    sink_local_full,
                )
                cached = ZoomKVRetriever._batch_meta_cache
                if (
                    cached is not None
                    and cached.key == meta_key
                    and cached.actual_num_chunks.device == device
                    and cached.actual_num_chunks.shape[0] >= batch
                ):
                    actual_num_chunks = cached.actual_num_chunks
                else:
                    actual_num_chunks = self._scratch_buf(
                        "actual_num_chunks", (batch,), torch.int32, device
                    )
                    actual_num_chunks.copy_(
                        torch.tensor(actuals, dtype=torch.int32),
                        non_blocking=True,
                    )
                    ZoomKVRetriever._batch_meta_cache = _CachedBatchMeta(
                        key=meta_key,
                        actual_num_chunks=actual_num_chunks,
                        bucket=bucket,
                        start_b=start_b,
                        sink_local_full=sink_local_full,
                    )
            else:
                seq_lens_dev = (
                    seq_lens
                    if seq_lens.is_cuda
                    else torch.as_tensor(seq_lens, dtype=torch.int32, device=device)
                )
                actual_num_chunks = self._scratch_buf(
                    "actual_num_chunks", (batch,), torch.int32, device
                )
                build_actual_num_chunks(
                    seq_lens_dev,
                    actual_num_chunks,
                    sink_size=cfg.sink_size,
                    local_size=cfg.local_size,
                    block_size=block_size,
                    start_block=start_b,
                    max_chunks=clamp_chunks,
                )
                # Graph capture sets assume_context_fully_valid to avoid a
                # device sync on this scalar gate.
                sink_local_full = assume_context_fully_valid or bool(
                    torch.all(seq_lens_dev >= cfg.sink_size + cfg.local_size)
                )

        if bucket <= 0:
            return self._empty_batch_topk_result(
                batch, raw_q.shape[1], device, topk_out
            )

        with _retrieve_stage("physical_ids"):
            bt = block_table.to(device=device)
            physical_ids = bt[:batch, start_b:]

        self._require_direct_physical(raw_q, block_summary, physical_ids)
        topk = self._retrieve_topk_physical(raw_q,block_summary,physical_ids,bucket,start_b * block_size,actual_num_chunks=actual_num_chunks,topk_out=topk_out)
        self.last_context_fully_valid = summaries_guaranteed_valid and sink_local_full and self._last_topk_fully_filled

        return ZoomKVRetrievalResult(topk=topk,context_fully_valid=self.last_context_fully_valid,used_direct_physical=True)




    @staticmethod
    def _can_use_direct_physical(
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
    ) -> bool:
        """Keep direct kernels behind an exact-layout compatibility gate."""
        return (
            direct_physical_retrieval_available()
            and raw_q.is_cuda
            and raw_q.is_contiguous()
            and raw_q.dtype == torch.bfloat16
            and raw_q.shape[-1] in (128, 256)
            and physical_ids.is_cuda
            and physical_ids.device == raw_q.device
            and physical_ids.dtype == torch.int32
            and physical_ids.stride(-1) == 1
            and block_summary.block_size == 16
            and block_summary.dtype == torch.bfloat16
            and block_summary.chunk_min.device == raw_q.device
        )




    def _build_stage_budgets(
        self,
        actual_num_chunks: torch.Tensor,
        *,
        n_chunks: int,
        factor: int,
        kv_heads: int,
    ) -> _StageBudgets:
        cfg = self.cfg
        batch = actual_num_chunks.shape[0]
        n_large = n_chunks // factor
        max_large = min(
            n_large,
            max(1, int(math.ceil(n_large * cfg.quest_large_ratio))),
        )
        max_small = min(
            max_large * factor,
            cfg.max_small_candidates,
            max(
                1,
                int(math.ceil(max_large * factor * cfg.quest_small_ratio)),
            ),
        )
        max_dense = min(max_small, max(1, int(max_small * cfg.dense_ratio)))
        shape = (batch, kv_heads)
        outputs = [
            self._scratch_buf(name, shape, torch.int32, actual_num_chunks.device)
            for name in (
                "parent_lengths",
                "large_ks",
                "sub_lengths",
                "small_ks",
                "dense_ks",
                "final_ks",
            )
        ]
        dense_topk = max(1, min(cfg.dense_topk, cfg.quest_chunk))
        sparse_topk = max(1, min(cfg.sparse_topk, cfg.quest_chunk))
        build_stage_budgets(
            actual_num_chunks,
            *outputs,
            factor=factor,
            large_ratio=cfg.quest_large_ratio,
            small_ratio=cfg.quest_small_ratio,
            dense_ratio=cfg.dense_ratio,
            max_large=max_large,
            max_small=max_small,
            dense_topk=dense_topk,
            sparse_topk=sparse_topk,
            final_topk=cfg.final_topk,
        )
        return _StageBudgets(*outputs, max_large, max_small, max_dense)


    def _retrieve_topk_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        token_offset: int,
        *,
        actual_num_chunks: torch.Tensor | None = None,
        topk_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quest+CDS+KIVI directly over global physical summary pools."""
        cfg = self.cfg
        if actual_num_chunks is None:
            actual_num_chunks = torch.full((raw_q.shape[0],),n_chunks,dtype=torch.int32,device=raw_q.device)
        factor = cfg.hq_factor
        budgets = self._build_stage_budgets(
            actual_num_chunks,
            n_chunks=n_chunks,
            factor=factor,
            kv_heads=raw_q.shape[1],
        )
        chunk_idx = self._hierarchical_quest_physical(
            raw_q,
            block_summary,
            physical_ids,
            n_chunks,
            factor,
            actual_num_chunks,
            budgets,
        )
        return self._cds_select_physical(
            chunk_idx,
            raw_q,
            block_summary,
            physical_ids,
            n_chunks,
            token_offset,
            actual_num_chunks,
            budgets,
            topk_out,
        )




    @staticmethod
    def _parent_minmax_from_children(
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        valid: torch.Tensor,
        factor: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate gathered child summaries into parent min/max tensors."""
        tmp = ZoomKVBlockSummary.__new__(ZoomKVBlockSummary)
        tmp.num_kv_heads = cmin.shape[1]
        tmp.head_dim = cmin.shape[-1]
        tmp.blocks_per_parent = factor
        return ZoomKVBlockSummary.build_parent_minmax(tmp, torch.empty(0), cmin, cmax, valid)




    def _hierarchical_quest_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        factor: int,
        actual_num_chunks: torch.Tensor,
        budgets: _StageBudgets,
    ) -> torch.Tensor:
        batch, kv_heads = raw_q.shape[:2]
        n_large = n_chunks // factor
        nk_large = budgets.max_large
        large_scores = self._scratch_buf("large_scores",(batch, kv_heads, n_large),torch.float32,raw_q.device)

        with _retrieve_stage("quest_parent_score_direct"):
            quest_parent_score_physical(raw_q,physical_ids,block_summary.chunk_min,block_summary.chunk_max,block_summary.valid,large_scores,n_chunks,factor,actual_num_chunks,block_summary.parent_min,block_summary.parent_max,block_summary.parent_valid,block_summary.parent_first_child)

        with _retrieve_stage("quest_large_topk"):
            large_idx = float_topk_3d_varlen(
                large_scores,
                budgets.parent_lengths,
                budgets.large_ks,
                nk_large,
                strict=self.cfg.strict_kernels,
            )
        sub_scores = self._scratch_buf("sub_scores",(batch, kv_heads, nk_large * factor),torch.float32,raw_q.device)

        with _retrieve_stage("quest_sub_score_direct"):
            quest_sub_score_physical(raw_q,physical_ids,block_summary.chunk_min,block_summary.chunk_max,block_summary.valid,large_idx,sub_scores,nk_large,factor,n_chunks,actual_num_chunks)
        nk_small = budgets.max_small

        with _retrieve_stage("quest_sub_topk"):
            sub_pos = float_topk_3d_varlen(
                sub_scores,
                budgets.sub_lengths,
                budgets.small_ks,
                nk_small,
                strict=self.cfg.strict_kernels,
            )
        chunk_idx = self._scratch_buf("chunk_idx",(batch, kv_heads, nk_small),torch.int64,raw_q.device)

        with _retrieve_stage("quest_map_back"):
            self.quest.quest_map_back(large_idx, sub_pos, chunk_idx, factor, n_chunks)
        return chunk_idx




    def _cds_select_physical(
        self,
        chunk_idx: torch.Tensor,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        token_offset: int,
        actual_num_chunks: torch.Tensor,
        budgets: _StageBudgets,
        topk_out: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch, kv_heads = raw_q.shape[:2]
        nk = budgets.max_small
        density = self._scratch_buf("density_scores", (batch, kv_heads, nk), torch.float32, raw_q.device)

        with _retrieve_stage("cds_density_direct"):
            density_score_physical(chunk_idx,physical_ids,block_summary.centroid,block_summary.valid,raw_q,density,n_chunks,actual_num_chunks)
        n_dense = budgets.max_dense
        with _retrieve_stage("cds_density_topk"):
            dense_pos = float_topk_3d_varlen(
                density,
                budgets.small_ks,
                budgets.dense_ks,
                n_dense,
                strict=cfg.strict_kernels,
            )

        dense_mask = self._scratch_buf("dense_mask",(batch, kv_heads, nk),torch.bool,raw_q.device)
        dense_mask_from_topk(dense_pos,nk,out=dense_mask,strict=cfg.strict_kernels)

        dense_topk = max(1, min(cfg.dense_topk, block_summary.block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_summary.block_size))
        output_slots = max(dense_topk, sparse_topk)
        out_width = nk * output_slots

        out_scores = self._scratch_buf("kivi_scores",(batch, kv_heads, out_width),torch.float32,raw_q.device)
        out_indices = self._scratch_buf("kivi_indices",(batch, kv_heads, out_width),torch.int64,raw_q.device)

        with _retrieve_stage("kivi_qk_direct"):
            kivi_physical(chunk_idx,dense_mask,physical_ids,block_summary.packed,block_summary.chunk_min,block_summary.chunk_max,block_summary.valid,raw_q,dense_topk,sparse_topk,token_offset,out_scores,out_indices,actual_num_chunks)
        actual_topk = min(cfg.final_topk, out_width)
        final_lengths = self._scratch_buf(
            "final_lengths",
            (batch, kv_heads),
            torch.int32,
            raw_q.device,
            fill=out_width,
        )

        with _retrieve_stage("final_topk_direct"):
            selected = float_topk_values_3d_varlen(
                out_scores,
                out_indices,
                final_lengths,
                budgets.final_ks,
                actual_topk,
                strict=cfg.strict_kernels,
                out=topk_out if actual_topk == cfg.final_topk else None,
            )

        if actual_topk == cfg.final_topk:
            # Sparse routing's minimum-context gate guarantees enough runtime
            # candidates without introducing a device-to-host sync here.
            self._last_topk_fully_filled = True
            return selected
        self._last_topk_fully_filled = False
        padded = self._scratch_buf("selected_direct",(batch, kv_heads, cfg.final_topk),torch.int64,raw_q.device,fill=-1)
        padded[..., :actual_topk].copy_(selected)
        return padded




    def _hierarchical_quest(
        self,
        raw_q: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        parent_min: torch.Tensor,
        parent_max: torch.Tensor,
        parent_valid: torch.Tensor,
        n_chunks: int,
        factor: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch = raw_q.shape[0]
        n_large = parent_min.shape[2]
        nk_large = max(1, int(math.ceil(n_large * cfg.quest_large_ratio)))
        nk_large = min(nk_large, n_large)
        large_scores = self._scratch_buf("large_scores",(batch, raw_q.shape[1], n_large),torch.float32,raw_q.device)

        with _retrieve_stage("quest_large_score"):
            self.quest.quest_chunk_score(raw_q, parent_min, parent_max, large_scores, n_large, parent_valid)

        with _retrieve_stage("quest_large_topk"):
            large_idx = _topk_3d(large_scores, nk_large, strict=cfg.strict_kernels)

        sub_scores = self._scratch_buf("sub_scores",(batch, raw_q.shape[1], nk_large * factor),torch.float32,raw_q.device)

        with _retrieve_stage("quest_sub_score"):
            self.quest.quest_sub_chunk_score(raw_q, cmin, cmax, large_idx, sub_scores, nk_large, factor)
        nk_small = max(1, int(math.ceil(nk_large * factor * cfg.quest_small_ratio)))
        nk_small = min(nk_small, nk_large * factor, cfg.max_small_candidates)

        with _retrieve_stage("quest_sub_topk"):
            sub_pos = _topk_3d(sub_scores, nk_small, strict=cfg.strict_kernels)
        chunk_idx = self._scratch_buf("chunk_idx",(batch, raw_q.shape[1], nk_small),torch.int64,raw_q.device)

        with _retrieve_stage("quest_map_back"):
            self.quest.quest_map_back(large_idx, sub_pos, chunk_idx, factor, n_chunks)
        return chunk_idx






    def _cds_select(
        self,
        chunk_idx: torch.Tensor,
        packed: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        centroid: torch.Tensor,
        raw_q: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch = raw_q.shape[0]
        kv_heads = raw_q.shape[1]
        nk = chunk_idx.shape[2]
        # Density via centroid @ q
        density = self._scratch_buf("density_scores",(batch, kv_heads, nk),torch.float32,raw_q.device)
        density = chunk_density_scores(chunk_idx,centroid,raw_q,out=density,strict=cfg.strict_kernels)
        n_dense = max(1, int(nk * cfg.dense_ratio))
        n_dense = min(n_dense, nk)
        with _retrieve_stage("cds_density_topk"):
            dense_pos = _topk_3d(density, n_dense, strict=cfg.strict_kernels)

        dense_mask = self._scratch_buf("dense_mask",(batch, kv_heads, nk),torch.bool,raw_q.device)
        dense_mask = dense_mask_from_topk(dense_pos,nk,out=dense_mask,strict=cfg.strict_kernels)
        dense_topk = max(1, min(cfg.dense_topk, block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_size))
        output_slots = max(dense_topk, sparse_topk)
        out_width = nk * output_slots


        out_scores = self._scratch_buf("kivi_scores",(batch, kv_heads, out_width),torch.float32,raw_q.device)
        out_indices = self._scratch_buf("kivi_indices",(batch, kv_heads, out_width),torch.int64,raw_q.device)

        with _retrieve_stage("kivi_qk"):
            out_scores, out_indices = partial_chunk_kivi_qk(chunk_idx,dense_mask,packed,cmin,cmax,raw_q.to(cmin.dtype),group_size=block_size,dense_topk=dense_topk,sparse_topk=sparse_topk,out_scores=out_scores,out_indices=out_indices,strict=cfg.strict_kernels)
        actual_topk = min(cfg.final_topk, out_scores.shape[-1])
        with _retrieve_stage("final_topk"):
            selected = float_topk_values_3d(
                out_scores,
                out_indices,
                actual_topk,
                strict=cfg.strict_kernels,
            )


        if actual_topk == cfg.final_topk:
            return selected

        padded = self._scratch_buf("selected",(batch, kv_heads, cfg.final_topk),torch.int64,raw_q.device,fill=-1)
        padded[..., :actual_topk].copy_(selected)
        return padded
