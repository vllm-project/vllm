# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident suffix decoding proposer.

Wraps the SuffixGPU library (device-tensor drafter with a local
per-request matcher and a cross-request global suffix index) behind the
same device-state contract as NgramProposerGPU: the previous step's
sampled ids are scattered into a resident token buffer and drafting
runs entirely on device, so the proposer composes with async
scheduling (no host sync on the draft path).

The draft path runs eagerly (fused Triton kernels) by default at first
call; when ``suffix_gpu_use_cuda_graph`` is enabled the whole
update+propose chain is captured into per-batch-bucket CUDA graphs
(powers of two up to max_num_seqs) on shared staging buffers, and each
step replays the smallest bucket covering the batch, falling back to
eager if capture fails.
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


class SuffixProposerGPU:
    """Device-state suffix-decoding drafter (async-scheduling safe)."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"

        # Lazy import so vLLM works without the SuffixGPU package.
        from suffix_gpu.proposer import SuffixGPUDrafter

        self.k = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.use_cuda_graph = config.suffix_gpu_use_cuda_graph
        self.ingest_chunk = config.suffix_gpu_ingest_chunk
        enable_global = config.suffix_decoding_max_cached_requests != 0

        self.drafter = SuffixGPUDrafter(
            k=self.k,
            device=device,
            max_pattern_len=config.suffix_decoding_max_tree_depth,
            min_match_len=1,
            max_occurrences=config.suffix_gpu_max_occurrences,
            enable_global=enable_global,
            global_capacity=config.suffix_gpu_global_capacity,
            delta_capacity=config.suffix_gpu_delta_capacity,
            rebuild_stream=torch.cuda.Stream(device) if device.type == "cuda" else None,
            max_spec_factor=config.suffix_decoding_max_spec_factor,
            max_spec_offset=0.0,
            min_token_prob=config.suffix_decoding_min_token_prob,
        )

        # CUDA-graph state (captured at engine warmup, or lazily on the
        # first propose call after eager warmup has JIT-compiled the
        # Triton kernels). One graph per batch bucket so small batches
        # do not replay max-batch kernels; all buckets share the
        # max-batch staging buffers below.
        self._graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]
        ] = {}
        self._graph_buckets: list[int] = []
        self._graph_failed = False
        self._g_num_tokens: torch.Tensor | None = None
        self._g_sampled: torch.Tensor | None = None
        self._g_counts: torch.Tensor | None = None
        self._g_token_ids: torch.Tensor | None = None

        self._warmed_up = False

        # Global-index ingestion runs on a side stream so its delta
        # copies stay off the step critical path; sync_pending_ingest()
        # orders later default-stream work after the pending reads.
        self._ingest_stream: torch.cuda.Stream | None = None
        self._ingest_event: torch.cuda.Event | None = None
        self._ingest_pending = False
        if device.type == "cuda" and self.drafter.global_index is not None:
            self._ingest_stream = torch.cuda.Stream(device)
            self._ingest_event = torch.cuda.Event()

    def update_token_ids_ngram(
        self,
        sampled_token_ids: torch.Tensor | list[list[int]],
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse NgramProposerGPU's device-side bookkeeping helper verbatim
        (it only depends on self.device)."""
        return NgramProposerGPU.update_token_ids_ngram(
            self,  # type: ignore[arg-type]
            sampled_token_ids,
            gpu_input_batch,
            token_ids_gpu,
            num_tokens_no_spec,
            discard_request_mask,
        )

    def _ingest_async(
        self,
        keys: list,
        rows: list[torch.Tensor],
        lengths: list[int],
        final: bool = False,
    ) -> None:
        if self._ingest_stream is None:
            self.drafter.ingest_active(
                keys, rows, lengths, final=final, chunk=self.ingest_chunk
            )
            return
        # The event is created together with the stream.
        assert self._ingest_event is not None
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._ingest_stream):
            # Token rows are written on the default stream.
            self._ingest_stream.wait_stream(default_stream)
            self.drafter.ingest_active(
                keys, rows, lengths, final=final, chunk=self.ingest_chunk
            )
            self._ingest_event.record()
        self._ingest_pending = True

    def sync_pending_ingest(self) -> None:
        """Order later default-stream work after pending ingest reads.

        Must run before rewriting ingested token rows (row reuse after
        request finish) and before querying the global index (graph
        replay / eager propose).
        """
        if self._ingest_pending:
            torch.cuda.current_stream().wait_event(self._ingest_event)
            self._ingest_pending = False

    def _warmup(self, token_ids_gpu: torch.Tensor) -> None:
        """JIT-compile the Triton kernels on dummy max-shape data."""
        b = self.max_num_seqs
        s = token_ids_gpu.shape[1]
        buf = torch.zeros(b, s, dtype=torch.int32, device=self.device)
        counts = torch.randint(
            1, max(2, min(64, s // 2)), (b,), dtype=torch.int32, device=self.device
        )
        sampled = torch.full((b, self.k + 1), -1, dtype=torch.int32, device=self.device)
        sampled[:, 0] = 1
        for _ in range(3):
            self.drafter.propose_with_update(
                counts, buf, sampled, max_model_len=self.max_model_len
            )
        torch.cuda.synchronize(self.device)
        self._warmed_up = True

    def capture_draft_graph(self, token_ids_gpu: torch.Tensor) -> None:
        """Warm up (Triton JIT) and capture the draft graphs.

        Called by the runner during engine warmup (capture_model) so the
        first serving step doesn't pay JIT + capture latency; propose()
        also calls it lazily as a fallback (e.g. enforce_eager engines,
        where capture_model never runs). Warmup runs even when the CUDA
        graph is disabled — only the capture itself is gated.
        """
        if self.device.type != "cuda":
            self._graph_failed = True
            return
        if not self._warmed_up:
            self._warmup(token_ids_gpu)
        if not self.use_cuda_graph or self._graphs or self._graph_failed:
            return
        full = self._full_alias(token_ids_gpu)
        if full is None:
            self._graph_failed = True
            return
        try:
            # Always capture at the maximum sampled width
            # (num_spec_tokens + 1); narrower per-step inputs are
            # left-aligned into the staging buffer.
            self._capture_buckets(full, self.k + 1)
        except Exception:
            logger.exception(
                "suffix_gpu: CUDA graph capture failed; falling back to eager kernels."
            )
            self._graph_failed = True
            self._graphs = {}
            self._graph_buckets = []

    @staticmethod
    def _bucket_sizes(max_batch: int) -> list[int]:
        sizes = []
        b = 1
        while b < max_batch:
            sizes.append(b)
            b *= 2
        sizes.append(max_batch)
        return sizes

    def _capture_buckets(self, token_ids_gpu: torch.Tensor, sampled_width: int) -> None:
        """Capture update+propose per batch bucket on shared buffers.

        token_ids_gpu is the runner's persistent buffer, so the graphs
        bind its storage directly; per-step inputs are staged into the
        shared fixed buffers before replay.
        """
        b_max = self.max_num_seqs
        self._g_num_tokens = torch.zeros(b_max, dtype=torch.int32, device=self.device)
        self._g_sampled = torch.full(
            (b_max, sampled_width), -1, dtype=torch.int32, device=self.device
        )
        self._g_counts = torch.zeros(b_max, dtype=torch.int64, device=self.device)
        self._g_token_ids = token_ids_gpu
        buckets = self._bucket_sizes(b_max)
        # Eagerly warm every bucket shape first: Triton JIT inside a
        # capture would invalidate it.
        for b in buckets:
            self.drafter.propose_with_update(
                self._g_num_tokens[:b],
                self._g_token_ids[:b],
                self._g_sampled[:b],
                self._g_counts[:b],
                max_model_len=self.max_model_len,
            )
        torch.cuda.synchronize(self.device)
        for b in buckets:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                draft, nv, _ = self.drafter.propose_with_update(
                    self._g_num_tokens[:b],
                    self._g_token_ids[:b],
                    self._g_sampled[:b],
                    self._g_counts[:b],
                    max_model_len=self.max_model_len,
                )
            self._graphs[b] = (graph, draft, nv)
        self._graph_buckets = buckets
        logger.info_once(
            "suffix_gpu: draft path captured into CUDA "
            "graphs (buckets=%s, sampled_width=%d)",
            str(buckets),
            sampled_width,
        )

    def propose(
        self,
        num_speculative_tokens: int,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        valid_sampled_token_ids_gpu: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft tokens for the batch; mirrors NgramProposerGPU.propose.

        Scatters the sampled ids into token_ids_gpu, then matches and
        drafts on device. Returns (draft_tokens [B, k] int32,
        num_valid_draft_tokens [B] int32).
        """
        assert num_speculative_tokens == self.k
        assert token_ids_gpu.device == self.device

        # Host-side upkeep: swap in finished background SA rebuilds.
        self.drafter.poll()
        # Global-index queries below must see pending side-stream ingest.
        self.sync_pending_ingest()

        if not self._warmed_up:
            self._warmup(token_ids_gpu)

        bs = num_tokens_no_spec.shape[0]
        width = valid_sampled_token_ids_gpu.shape[1]

        use_graph = (
            self.use_cuda_graph
            and not self._graph_failed
            and bs <= self.max_num_seqs
            and width <= self.k + 1
        )
        if use_graph and not self._graphs:
            # Fallback for engines that never ran capture_model.
            self.capture_draft_graph(token_ids_gpu)
        if (
            self._graphs
            and use_graph
            and self._g_token_ids is not None
            and token_ids_gpu.data_ptr() == self._g_token_ids.data_ptr()
            and token_ids_gpu.stride(0) == self._g_token_ids.stride(0)
        ):
            # Staging buffers are allocated together with the graphs.
            assert self._g_num_tokens is not None
            assert self._g_sampled is not None
            assert self._g_counts is not None
            b = next(s for s in self._graph_buckets if s >= bs)
            graph, g_draft, g_nv = self._graphs[b]
            self._g_num_tokens[:bs].copy_(num_tokens_no_spec)
            self._g_num_tokens[bs:b].zero_()
            self._g_sampled[:b].fill_(-1)
            self._g_sampled[:bs, :width].copy_(valid_sampled_token_ids_gpu)
            self._g_counts[:bs].copy_(valid_sampled_tokens_count)
            self._g_counts[bs:b].zero_()
            graph.replay()
            return g_draft[:bs], g_nv[:bs]

        draft, num_valid, _ = self.drafter.propose_with_update(
            num_tokens_no_spec,
            token_ids_gpu,
            valid_sampled_token_ids_gpu,
            valid_sampled_tokens_count,
            max_model_len=self.max_model_len,
        )
        return draft, num_valid

    def _full_alias(self, view: torch.Tensor) -> torch.Tensor | None:
        """Max-batch alias of the persistent buffer backing `view`.

        The runner passes token_ids_gpu_tensor[:batch_size], a fresh
        slice each step; the graph must bind the full buffer so any
        later batch size replays correctly.
        """
        s = view.shape[1]
        stride0 = view.stride(0)
        rows = self.max_num_seqs
        needed = (rows - 1) * stride0 + s
        cap = view.untyped_storage().size() // view.element_size()
        if view.stride(1) != 1 or view.storage_offset() != 0 or cap < needed:
            logger.warning_once(
                "suffix_gpu: token buffer layout unsuitable for CUDA "
                "graph capture; using eager kernels."
            )
            return None
        return torch.as_strided(view, (rows, s), (stride0, 1))

    # ------------------------------------------------------------------
    # global-memory ingestion (host-side, off the draft path)
    # ------------------------------------------------------------------
    def ingest_active_requests(
        self, input_batch: InputBatch, token_ids_gpu: torch.Tensor
    ) -> None:
        """Chunked incremental ingestion of in-flight responses."""
        if self.drafter.global_index is None:
            return
        keys: list[str] = []
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        num_tokens = input_batch.num_tokens_no_spec
        num_prompt = input_batch.num_prompt_tokens
        for req_id, idx in input_batch.req_id_to_index.items():
            resp_len = int(num_tokens[idx]) - int(num_prompt[idx])
            if resp_len < self.ingest_chunk:
                continue
            start = int(num_prompt[idx])
            keys.append(req_id)
            rows.append(token_ids_gpu[idx, start : start + resp_len])
            lengths.append(resp_len)
        if keys:
            self._ingest_async(keys, rows, lengths)

    def on_requests_finished(
        self, finished_req_ids, input_batch: InputBatch, token_ids_gpu: torch.Tensor
    ) -> None:
        """Final-flush finished requests before their rows are reused."""
        if self.drafter.global_index is None:
            return
        keys: list[str] = []
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        num_tokens = input_batch.num_tokens_no_spec
        num_prompt = input_batch.num_prompt_tokens
        for req_id in finished_req_ids:
            idx = input_batch.req_id_to_index.get(req_id)
            if idx is None:
                self.drafter._ingested.pop(req_id, None)
                continue
            resp_len = int(num_tokens[idx]) - int(num_prompt[idx])
            start = int(num_prompt[idx])
            keys.append(req_id)
            rows.append(token_ids_gpu[idx, start : start + max(resp_len, 0)])
            lengths.append(max(resp_len, 0))
        if keys:
            self._ingest_async(keys, rows, lengths, final=True)

    def load_model(self, *args, **kwargs) -> None:
        pass

    def dummy_run(self, num_tokens: int = 1) -> None:
        pass
