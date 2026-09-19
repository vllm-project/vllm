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

from bisect import bisect_left

import torch

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


class _SuffixCudagraphDispatcher:
    """Dispatch request batches to power-of-two CUDA graph buckets."""

    def __init__(self, max_num_seqs: int):
        buckets = []
        size = 1
        while size < max_num_seqs:
            buckets.append(size)
            size *= 2
        buckets.append(max_num_seqs)
        self.buckets = tuple(buckets)
        self.capture_descriptors = tuple(
            BatchDescriptor(num_tokens=size, num_reqs=size)
            for size in reversed(self.buckets)
        )

    def dispatch(self, num_reqs: int) -> BatchDescriptor | None:
        idx = bisect_left(self.buckets, num_reqs)
        if idx == len(self.buckets):
            return None
        size = self.buckets[idx]
        return BatchDescriptor(num_tokens=size, num_reqs=size)


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
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.device = device
        self.vllm_config = vllm_config
        self.use_cuda_graph = bool(
            config.suffix_gpu_use_cuda_graph
            and device.type == "cuda"
            and not vllm_config.model_config.enforce_eager
            and not config.enforce_eager
            and vllm_config.compilation_config.cudagraph_mode
            != CUDAGraphMode.NONE
        )
        if config.suffix_gpu_use_cuda_graph and not self.use_cuda_graph:
            logger.info_once(
                "suffix_gpu: CUDA graphs follow the target model graph mode; "
                "using eager kernels."
            )
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
            num_backoff=config.suffix_gpu_num_backoff,
            coordinated_rebuild=self.tp_size > 1,
        )

        self._tp_rebuild_send = torch.empty(6, dtype=torch.int64, device="cpu")
        self._tp_rebuild_recv = torch.empty(
            self.tp_size * 6, dtype=torch.int64, device="cpu"
        )

        # Persistent inputs are staged before replay. Graphs are owned by the
        # standard wrapper and share the decoder graph pool.
        self._graph_dispatcher = _SuffixCudagraphDispatcher(self.max_num_seqs)
        self._graph_failed = False
        self._g_num_tokens = torch.zeros(
            self.max_num_seqs, dtype=torch.int32, device=self.device
        )
        self._g_sampled = torch.full(
            (self.max_num_seqs, self.k + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._g_counts = torch.zeros(
            self.max_num_seqs, dtype=torch.int64, device=self.device
        )
        self._g_token_ids: torch.Tensor | None = None
        self._graph_runner = (
            CUDAGraphWrapper(
                self._graphable_propose,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
            )
            if self.use_cuda_graph
            else None
        )

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

    def _poll_rebuild(self) -> None:
        state = self.drafter.rebuild_state()
        if state is None or state.pending_epoch is None:
            return
        if self.tp_size == 1:
            self.drafter.poll()
            return

        signature = state.snapshot_signature
        if signature is None:
            raise RuntimeError("suffix_gpu pending rebuild has no signature")
        self._tp_rebuild_send.copy_(
            torch.tensor(
                [
                    state.active_epoch,
                    state.pending_epoch,
                    int(state.ready),
                    *signature,
                ],
                dtype=torch.int64,
            )
        )
        torch.distributed.all_gather_into_tensor(
            self._tp_rebuild_recv,
            self._tp_rebuild_send,
            group=get_tp_group().cpu_group,
        )
        rank_states = self._tp_rebuild_recv.view(self.tp_size, 6).tolist()
        expected = [rank_states[0][i] for i in (0, 1, 3, 4, 5)]
        for rank, rank_state in enumerate(rank_states[1:], start=1):
            actual = [rank_state[i] for i in (0, 1, 3, 4, 5)]
            if actual != expected:
                raise RuntimeError(
                    "suffix_gpu TP rebuild state mismatch: "
                    f"rank0={expected}, rank{rank}={actual}"
                )
        if all(rank_state[2] for rank_state in rank_states):
            self.drafter.commit_rebuild(state.pending_epoch)

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
        """JIT-compile the Triton kernels for every graph bucket."""
        b = self.max_num_seqs
        s = token_ids_gpu.shape[1]
        buf = torch.zeros(b, s, dtype=torch.int32, device=self.device)
        counts = torch.randint(
            1, max(2, min(64, s // 2)), (b,), dtype=torch.int32, device=self.device
        )
        sampled = torch.full((b, self.k + 1), -1, dtype=torch.int32, device=self.device)
        sampled[:, 0] = 1
        for bucket in self._graph_dispatcher.buckets:
            self.drafter.propose_with_update(
                counts[:bucket],
                buf[:bucket],
                sampled[:bucket],
                max_model_len=self.max_model_len,
            )
        torch.accelerator.synchronize(self.device)
        self._warmed_up = True

    def _clear_draft_graphs(self) -> None:
        if self._graph_runner is not None:
            self._graph_runner.clear_graphs()
        self._g_token_ids = None
        torch.accelerator.empty_cache()

    def _graphable_propose(
        self,
        num_tokens: torch.Tensor,
        token_ids: torch.Tensor,
        sampled: torch.Tensor,
        counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.drafter.propose_with_update(
            num_tokens,
            token_ids,
            sampled,
            counts,
            max_model_len=self.max_model_len,
        )

    def _graphs_captured(self) -> bool:
        if self._graph_runner is None:
            return False
        entries = self._graph_runner.concrete_cudagraph_entries
        return all(
            desc in entries and entries[desc].cudagraph is not None
            for desc in self._graph_dispatcher.capture_descriptors
        )

    def _capture_buckets(self, token_ids_gpu: torch.Tensor) -> None:
        assert self._graph_runner is not None
        self._g_token_ids = token_ids_gpu
        for desc in self._graph_dispatcher.capture_descriptors:
            bucket = desc.num_tokens
            with set_forward_context(
                None,
                self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=desc,
            ):
                outputs = self._graph_runner(
                    self._g_num_tokens[:bucket],
                    self._g_token_ids[:bucket],
                    self._g_sampled[:bucket],
                    self._g_counts[:bucket],
                )
            del outputs
        logger.info_once(
            "suffix_gpu: draft path captured by CUDAGraphWrapper (buckets=%s)",
            str(list(self._graph_dispatcher.buckets)),
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
        self._poll_rebuild()
        # Global-index queries below must see pending side-stream ingest.
        self.sync_pending_ingest()

        if not self._warmed_up:
            self._warmup(token_ids_gpu)

        bs = num_tokens_no_spec.shape[0]
        width = valid_sampled_token_ids_gpu.shape[1]

        use_graph = bool(
            self.use_cuda_graph
            and not self._graph_failed
            and bs <= self.max_num_seqs
            and width <= self.k + 1
        )
        if (
            use_graph
            and self._graphs_captured()
            and self._g_token_ids is not None
            and token_ids_gpu.data_ptr() == self._g_token_ids.data_ptr()
            and token_ids_gpu.stride(0) == self._g_token_ids.stride(0)
        ):
            desc = self._graph_dispatcher.dispatch(bs)
            assert desc is not None
            bucket = desc.num_tokens
            self._g_num_tokens[:bs].copy_(num_tokens_no_spec)
            self._g_num_tokens[bs:bucket].zero_()
            self._g_sampled[:bucket].fill_(-1)
            self._g_sampled[:bs, :width].copy_(valid_sampled_token_ids_gpu)
            self._g_counts[:bs].copy_(valid_sampled_tokens_count)
            self._g_counts[bs:bucket].zero_()
            assert self._graph_runner is not None
            with set_forward_context(
                None,
                self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=desc,
            ):
                draft, num_valid, _ = self._graph_runner(
                    self._g_num_tokens[:bucket],
                    self._g_token_ids[:bucket],
                    self._g_sampled[:bucket],
                    self._g_counts[:bucket],
                )
            return draft[:bs], num_valid[:bs]

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
        request_items = input_batch.req_id_to_index.items()
        if self.tp_size > 1:
            request_items = sorted(request_items)
        for req_id, idx in request_items:
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
        if self.tp_size > 1:
            finished_req_ids = sorted(finished_req_ids)
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

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        token_ids_gpu: torch.Tensor,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
    ) -> None:
        if not self._warmed_up:
            self._warmup(token_ids_gpu)
        if (
            not use_cudagraphs
            or not is_graph_capturing
            or not self.use_cuda_graph
            or self._graph_failed
            or self._graphs_captured()
        ):
            return
        full = self._full_alias(token_ids_gpu)
        if full is None:
            self._graph_failed = True
            return
        try:
            self._capture_buckets(full)
        except Exception:
            logger.exception(
                "suffix_gpu: CUDA graph capture failed; falling back to eager kernels."
            )
            self._graph_failed = True
            self._clear_draft_graphs()
