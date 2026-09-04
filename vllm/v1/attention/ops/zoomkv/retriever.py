# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-layer chunk-mean + KIVI retrieval pipeline for ZoomKV."""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, ClassVar
import torch

from vllm.logger import init_logger
from vllm.v1.attention.ops.zoomkv import stage_timer as _zt
from vllm.v1.attention.ops.zoomkv.kernels import (
    centroid_score_physical,
    density_score_physical,
    direct_physical_retrieval_available,
    float_topk_3d_varlen,
    float_topk_values_3d,
    float_topk_values_3d_varlen,
    kivi_physical,
    try_load_zoomkv_c,
)

from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk
from vllm.v1.attention.ops.zoomkv.retrieval_metadata_triton import (
    build_actual_num_chunks,
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
    chunk_size: int = 16
    chunk_candidates: int = 200
    dense_chunks: int = 60
    dense_topk: int = 8
    sparse_topk: int = 4
    full_attention_threshold: int = 2000
    dense_fallback: bool = False
    strict_kernels: bool = False
    enable_offload: bool = False
    offload_unit_tokens: int = 64
    @property
    def sparse_chunks(self) -> int:
        return max(0, int(self.chunk_candidates) - int(self.dense_chunks))

    @property
    def kivi_width(self) -> int:
        return (
            int(self.dense_chunks) * int(self.dense_topk)
            + int(self.sparse_chunks) * int(self.sparse_topk)
        )


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
        state.raw_q[:batch].copy_(raw_q, non_blocking=True)
        state.block_table.fill_(-1)
        state.block_table[:batch].copy_(block_table[:batch], non_blocking=True)
        state.seq_lens.zero_()
        state.seq_lens[:batch].copy_(seq_lens[:batch], non_blocking=True)
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
        return bool(
            self.cfg.dense_fallback
            or seq_len < self.cfg.full_attention_threshold
        )



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
        """Run chunk-mean + KIVI on pre-gathered reference summaries."""
        cfg = self.cfg
        batch = raw_q.shape[0]
        n_chunks = packed.shape[2]
        if n_chunks <= 0:
            return torch.full((batch, raw_q.shape[1], cfg.final_topk),-1,dtype=torch.int64,device=raw_q.device)
        topk_local = self._select_gathered(
            raw_q, packed, cmin, cmax, centroid, valid, block_size, n_chunks
        )
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
        """Run chunk-mean + KIVI and return full-sequence token indices.

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
        """Batched chunk-mean + KIVI retrieval for decode requests.
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
            if block_table.device == device:
                physical_ids = block_table[:batch, start_b:]
            else:
                physical_ids = block_table.to(device=device)[:batch, start_b:]

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
        """Chunk-mean Top-K + KIVI over global physical summary pools."""
        if actual_num_chunks is None:
            actual_num_chunks = torch.full(
                (raw_q.shape[0],),
                n_chunks,
                dtype=torch.int32,
                device=raw_q.device,
            )
        return self._retrieve_physical(
            raw_q,
            block_summary,
            physical_ids,
            n_chunks,
            token_offset,
            actual_num_chunks,
            topk_out,
        )

    def _budgets(
        self,
        actual_num_chunks: torch.Tensor,
        *,
        n_chunks: int,
        kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Fixed 200/60/140 budgets for chunk-mean retrieval."""
        cfg = self.cfg
        batch = actual_num_chunks.shape[0]
        device = actual_num_chunks.device
        n_cand = min(int(cfg.chunk_candidates), int(n_chunks))
        n_dense = min(int(cfg.dense_chunks), n_cand)
        n_sparse = n_cand - n_dense
        out_width = n_dense * int(cfg.dense_topk) + n_sparse * int(cfg.sparse_topk)
        shape = (batch, kv_heads)
        chunk_lengths = self._scratch_buf(
            "single_chunk_lengths", shape, torch.int32, device
        )
        chunk_ks = self._scratch_buf("single_chunk_ks", shape, torch.int32, device)
        final_ks = self._scratch_buf("single_final_ks", shape, torch.int32, device)
        # Broadcast per-request actual chunk counts to [B, H].
        actual = actual_num_chunks.to(dtype=torch.int32).view(batch, 1).expand(shape)
        chunk_lengths.copy_(actual)
        chunk_ks.copy_(torch.minimum(actual, actual.new_full((), n_cand)))
        final_ks.fill_(int(cfg.final_topk))
        return chunk_lengths, chunk_ks, final_ks, n_cand, n_dense, out_width

    def _score_all_chunk_means_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        actual_num_chunks: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        mod = try_load_zoomkv_c()
        if mod is not None and hasattr(mod, "centroid_score_physical"):
            centroid_score_physical(
                physical_ids,
                block_summary.centroid,
                block_summary.valid,
                raw_q,
                scores,
                n_chunks,
                actual_num_chunks,
            )
            return
        # Fallback for older extensions: score identity chunk ids.
        batch, kv_heads = raw_q.shape[:2]
        identity = self._scratch_buf(
            "identity_chunks",
            (batch, kv_heads, n_chunks),
            torch.int64,
            raw_q.device,
        )
        base = torch.arange(n_chunks, device=raw_q.device, dtype=torch.int64)
        identity.copy_(base.view(1, 1, n_chunks).expand(batch, kv_heads, n_chunks))
        density_score_physical(
            identity,
            physical_ids,
            block_summary.centroid,
            block_summary.valid,
            raw_q,
            scores,
            n_chunks,
            actual_num_chunks,
        )

    def _retrieve_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        token_offset: int,
        actual_num_chunks: torch.Tensor,
        topk_out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-score → Top-200 → dense/sparse KIVI → Top-100."""
        cfg = self.cfg
        batch, kv_heads = raw_q.shape[:2]
        (
            chunk_lengths,
            chunk_ks,
            final_ks,
            n_cand,
            n_dense,
            out_width,
        ) = self._budgets(
            actual_num_chunks, n_chunks=n_chunks, kv_heads=kv_heads
        )

        mean_scores = self._scratch_buf(
            "single_mean_scores",
            (batch, kv_heads, n_chunks),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("mean_score"):
            self._score_all_chunk_means_physical(
                raw_q,
                block_summary,
                physical_ids,
                n_chunks,
                actual_num_chunks,
                mean_scores,
            )

        with _retrieve_stage("chunk_topk"):
            chunk_idx = float_topk_3d_varlen(
                mean_scores,
                chunk_lengths,
                chunk_ks,
                n_cand,
                strict=cfg.strict_kernels,
            )

        dense_mask = self._scratch_buf(
            "single_dense_mask", (batch, kv_heads, n_cand), torch.bool, raw_q.device
        )
        dense_mask.fill_(False)
        dense_mask[..., :n_dense] = True

        dense_topk = max(1, min(cfg.dense_topk, block_summary.block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_summary.block_size))
        # Compact KIVI indexes outputs as tightly packed [B,H,out_width].
        # Older wheels without compact still need the padded width.
        mod = try_load_zoomkv_c()
        supports_compact = mod is not None and hasattr(mod, "centroid_score_physical")
        padded_w = n_cand * max(dense_topk, sparse_topk)
        buf_w = padded_w if not supports_compact else out_width
        out_scores = self._scratch_buf(
            "single_kivi_scores", (batch, kv_heads, buf_w), torch.float32, raw_q.device
        )
        out_indices = self._scratch_buf(
            "single_kivi_indices", (batch, kv_heads, buf_w), torch.int64, raw_q.device
        )
        out_scores.fill_(float("-inf"))
        out_indices.fill_(-1)
        with _retrieve_stage("kivi"):
            kivi_physical(
                chunk_idx,
                dense_mask,
                physical_ids,
                block_summary.packed,
                block_summary.chunk_min,
                block_summary.chunk_max,
                block_summary.valid,
                raw_q,
                dense_topk,
                sparse_topk,
                token_offset,
                out_scores,
                out_indices,
                actual_num_chunks,
                compact=True,
                n_dense=n_dense,
            )

        final_lengths = self._scratch_buf(
            "single_final_lengths",
            (batch, kv_heads),
            torch.int32,
            raw_q.device,
            fill=out_width,
        )
        actual_topk = min(cfg.final_topk, out_width)
        with _retrieve_stage("final_topk"):
            selected = float_topk_values_3d_varlen(
                out_scores,
                out_indices,
                final_lengths,
                final_ks,
                actual_topk,
                strict=cfg.strict_kernels,
                out=topk_out if actual_topk == cfg.final_topk else None,
            )
        if actual_topk == cfg.final_topk:
            self._last_topk_fully_filled = True
            return selected
        self._last_topk_fully_filled = False
        padded = self._scratch_buf(
            "single_selected",
            (batch, kv_heads, cfg.final_topk),
            torch.int64,
            raw_q.device,
            fill=-1,
        )
        padded[..., :actual_topk].copy_(selected)
        return padded
    def _select_gathered(
        self,
        raw_q: torch.Tensor,
        packed: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        centroid: torch.Tensor,
        valid: torch.Tensor,
        block_size: int,
        n_chunks: int,
    ) -> torch.Tensor:
        """Gathered/offload path for single-chunk retrieval."""
        cfg = self.cfg
        batch, kv_heads = raw_q.shape[:2]
        n_cand = min(int(cfg.chunk_candidates), n_chunks)
        n_dense = min(int(cfg.dense_chunks), n_cand)
        # valid: [B, H, N]; mean score over all chunks.
        q = raw_q.to(dtype=centroid.dtype).unsqueeze(2)
        scores = (centroid * q).sum(dim=-1).float()
        scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
        chunk_idx = _topk_3d(scores, n_cand, strict=cfg.strict_kernels)
        dense_mask = self._scratch_buf(
            "single_dense_mask_g", (batch, kv_heads, n_cand), torch.bool, raw_q.device
        )
        dense_mask.fill_(False)
        dense_mask[..., :n_dense] = True
        dense_topk = max(1, min(cfg.dense_topk, block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_size))
        out_width = n_dense * dense_topk + (n_cand - n_dense) * sparse_topk
        # Use padded KIVI then compact-pack for the gathered CUDA/ref path.
        padded_slots = max(dense_topk, sparse_topk)
        padded_w = n_cand * padded_slots
        out_scores = self._scratch_buf(
            "single_kivi_scores_g", (batch, kv_heads, max(padded_w, out_width)), torch.float32, raw_q.device
        )
        out_indices = self._scratch_buf(
            "single_kivi_indices_g", (batch, kv_heads, max(padded_w, out_width)), torch.int64, raw_q.device
        )
        out_scores, out_indices = partial_chunk_kivi_qk(
            chunk_idx,
            dense_mask,
            packed,
            cmin,
            cmax,
            raw_q.to(cmin.dtype),
            group_size=block_size,
            dense_topk=dense_topk,
            sparse_topk=sparse_topk,
            out_scores=out_scores,
            out_indices=out_indices,
            strict=cfg.strict_kernels,
        )
        from vllm.v1.attention.ops.zoomkv.kernels import _pack_padded_kivi_to_compact

        _pack_padded_kivi_to_compact(
            out_scores,
            out_indices,
            nk=n_cand,
            n_dense=n_dense,
            dense_topk=dense_topk,
            sparse_topk=sparse_topk,
        )
        actual_topk = min(cfg.final_topk, out_width)
        selected = float_topk_values_3d(
            out_scores[..., :out_width],
            out_indices[..., :out_width],
            actual_topk,
            strict=cfg.strict_kernels,
        )
        if actual_topk == cfg.final_topk:
            return selected
        padded = self._scratch_buf(
            "single_selected_g",
            (batch, kv_heads, cfg.final_topk),
            torch.int64,
            raw_q.device,
            fill=-1,
        )
        padded[..., :actual_topk].copy_(selected)
        return padded
