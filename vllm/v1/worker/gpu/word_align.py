# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper word-timestamp capture for the V2 GPU model runner.

Cross-attention Q (and the encoder K at prefill) are recorded into per-request
buffers before the fused attention kernel, then DTW-aligned once a request
finishes. Attaching this by composition rather than in the runner follows
``kv_connector.py``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# A slot holds one request's decoder Q and encoder K, and a request past the pool
# gets no timestamps, so the pool targets one slot per ``max_num_seqs``. Per-slot
# cost swings an order of magnitude with the model -- (max_source + max_target) *
# alignment_layers * d_model * itemsize -- so a fixed byte budget is right for one
# checkpoint and wrong for the next.
#
# The fraction is a ceiling on that target, not the sizing rule. ``init()`` runs
# inside ``load_model()``, before ``_initialize_kv_caches()``, so every byte
# reserved here comes out of the KV cache; without a ceiling a large
# ``max_num_seqs`` on a many-alignment-layer model can drive the KV cache
# negative. When the ceiling binds, the pool cannot cover ``max_num_seqs`` and
# ``_log_pool_size`` warns.
CAPTURE_MEMORY_FRACTION = 0.25


class WordAlignCapturer:
    """Routes cross-attention capture rows to per-request slots (opt-in)."""

    def __init__(self) -> None:
        self.enabled = False
        # req_id -> decoder positions decoded so far, for cropping the DTW.
        self.npos: dict[str, int] = {}
        # req_id -> capture slot, held for the request's lifetime.
        self.slot_of: dict[str, int] = {}
        self._free: list[int] = []
        # Requests that asked for word timestamps. The server flag only arms the
        # machinery: without this set every request would pay the readout and
        # occupy a slot that a request actually wanting words could not get.
        self.wants: set[str] = set()
        self._parked = False
        # req_id -> the last decoder position it can reach, so a request that
        # stops on its token budget instead of on eos is still read out.
        self._final_npos: dict[str, int] = {}
        # Requests already reported as unable to get a slot, so a decode loop
        # does not re-log one line per step.
        self._denied: set[str] = set()
        self.num_denied = 0

    @staticmethod
    def _slot_bytes(gen_config, hf_config, itemsize: int, max_frames: int) -> int:
        """Bytes one capture slot needs: decoder Q plus encoder K, per layer.

        Only layers that carry an alignment head are captured, and a layer counts
        once no matter how many of its heads are used (turbo: 6 heads, 2 layers).
        """
        n_layers = len({h[0] for h in gen_config.alignment_heads})
        return (
            (max_frames + int(hf_config.max_target_positions))
            * n_layers
            * int(hf_config.d_model)
            * itemsize
        )

    @staticmethod
    def _capture_budget_bytes(runner, memory_info: Callable | None = None) -> int:
        """Bytes the capture pool may take: a share of what is still unspent.

        Measured rather than assumed. The allowance is
        ``total * gpu_memory_utilization``, and the weights are already resident
        by the time this runs, so ``allowance - already_used`` is what KV cache,
        activations and capture buffers still have to share.
        """
        if memory_info is None:
            memory_info = torch.accelerator.get_memory_info
        try:
            free, total = memory_info(runner.device)
        except Exception:
            # Not fatal: an unreadable budget just falls back to a single slot.
            logger.warning(
                "Could not read device memory to size the word-timestamp capture "
                "pool; falling back to the minimum pool.",
                exc_info=True,
            )
            return 0
        unspent = total * runner.cache_config.gpu_memory_utilization - (total - free)
        return max(0, int(unspent * CAPTURE_MEMORY_FRACTION))

    @staticmethod
    def _pool_size(max_num_reqs: int, budget_bytes: int, slot_bytes: int) -> int:
        """One slot per concurrent request, clipped to the budget if it must be.

        ``max_num_reqs`` is the target, not a ceiling to grow into: fewer slots
        than that means some concurrent requests get no timestamps. ``init()`` also
        allocates one extra "scratch" slot that unmapped requests park their
        capture rows on, so it comes out of the same budget.
        """
        if slot_bytes <= 0:
            return 1
        return max(1, min(max_num_reqs, budget_bytes // slot_bytes - 1))

    @staticmethod
    def _log_pool_size(
        num_slots: int, max_num_reqs: int, slot_bytes: int, budget_bytes: int
    ) -> None:
        """Report the pool, at WARNING if it cannot cover ``max_num_seqs``.

        Past the pool a request returns ``words: null``, which a client cannot
        tell apart from silence, so a partially covered pool is said at startup
        rather than left to show up as missing data under load.
        """
        if num_slots >= max_num_reqs:
            logger.info(
                "Whisper word-timestamp cross-attention capture enabled: %d slot(s) "
                "x %.1f MiB = %.2f GiB, full coverage for max_num_seqs=%d.",
                num_slots,
                slot_bytes / 1024**2,
                (num_slots + 1) * slot_bytes / 1024**3,
                max_num_reqs,
            )
            return
        logger.warning(
            "Whisper word-timestamp capture pool holds %d slot(s) but max_num_seqs "
            "is %d, so under full load only ~%.0f%% of requests get word timestamps "
            "and the rest return words=None. Full coverage needs %.2f GiB (%d x "
            "%.1f MiB) but only %.2f GiB is available. Raise "
            "--gpu-memory-utilization or lower --max-num-seqs to widen coverage.",
            num_slots,
            max_num_reqs,
            100.0 * num_slots / max_num_reqs,
            (max_num_reqs + 1) * slot_bytes / 1024**3,
            max_num_reqs + 1,
            slot_bytes / 1024**2,
            budget_bytes / 1024**3,
        )

    def init(self, runner: "GPUModelRunner") -> None:
        """Allocate capture buffers and turn on capture in the model.

        Must run before graph capture so the compiled decoder includes the
        capture op.
        """
        if not runner.model_config.enable_word_timestamps:
            return
        model = runner.model
        if not getattr(model, "supports_word_timestamp", False) or not hasattr(
            model, "enable_word_align"
        ):
            return
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(runner.model_config.model)
        hf_config = runner.model_config.hf_config
        max_frames = int(hf_config.max_source_positions)
        slot_bytes = self._slot_bytes(
            gen_config, hf_config, runner.model_config.dtype.itemsize, max_frames
        )
        budget = self._capture_budget_bytes(runner)
        num_slots = self._pool_size(runner.max_num_reqs, budget, slot_bytes)
        model.enable_word_align(
            # alignment_heads is a Whisper-specific dynamic field on the HF
            # generation config, not declared on GenerationConfig.
            alignment_heads=gen_config.alignment_heads,  # type: ignore[attr-defined]
            eos_token_id=gen_config.eos_token_id,
            median_filter_width=getattr(hf_config, "median_filter_width", 7),
            device=runner.device,
            dtype=runner.model_config.dtype,
            max_slots=num_slots + 1,
            positions=runner.input_buffers.positions,
            max_k_frames=num_slots * max_frames,
        )
        self.model = model
        self.qidx, self.kidx = model.word_align_index_tensors()
        self.positions = runner.input_buffers.positions
        self.scratch = num_slots
        self.max_frames = max_frames
        self.max_tgt = int(hf_config.max_target_positions)
        self._qlimit = (num_slots + 1) * self.max_tgt - 1
        self._free = list(range(num_slots))
        # Token row indices, reused every step to map tokens onto batch entries.
        self._arange = torch.arange(
            runner.max_num_tokens, device=runner.device, dtype=torch.int32
        )
        self._frames = np.arange(max_frames, dtype=np.int64)
        # Warmup and graph capture run before any real batch: park them.
        self.qidx.fill_(self.scratch * self.max_tgt)
        self.enabled = True
        self._log_pool_size(num_slots, runner.max_num_reqs, slot_bytes, budget)

    def add_request(self, req_id: str, sampling_params, prompt_len: int = 0) -> None:
        """Record whether this request asked for word timestamps.

        Set by the serving layer via ``extra_args["word_timestamps"]``; requests
        without it are never captured and never read out.
        """
        if not self.enabled or sampling_params is None:
            return
        extra = getattr(sampling_params, "extra_args", None)
        if not (extra and extra.get("word_timestamps")):
            return
        self.wants.add(req_id)
        # The readout runs on the last decode step, and eos is not the only way
        # to reach it: a request stopping on its token budget must be read out
        # too. Capped at the capture buffer's position limit, past which nothing
        # was captured to align.
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if max_tokens:
            self._final_npos[req_id] = (
                min(prompt_len + int(max_tokens), self.max_tgt) - 1
            )

    def before_forward(self, input_batch: InputBatch) -> None:
        """Point this step's capture rows at each request's own slot."""
        if not self.enabled:
            return
        if not self.wants:
            # Park every capture row on the scratch slot once and skip the
            # per-step index build, so an armed server costs nothing while no
            # request is asking for words.
            if not self._parked:
                self.qidx.fill_(self.scratch * self.max_tgt)
                self.kidx.fill_(self.scratch * self.max_frames)
                self._parked = True
            return
        self._parked = False
        for req_id in input_batch.req_ids:
            if req_id in self.wants and req_id not in self.slot_of:
                if self._free:
                    self.slot_of[req_id] = self._free.pop()
                else:
                    self._report_pool_exhausted(req_id)
        # Requests left unmapped by an exhausted pool land on the scratch slot,
        # so they get no timestamps instead of another request's capture.
        slot_np = np.fromiter(
            (self.slot_of.get(r, self.scratch) for r in input_batch.req_ids),
            dtype=np.int64,
            count=input_batch.num_reqs,
        )
        # Rows are addressed as slot * max_tgt + position, so scaling the slot
        # here leaves just the position to add once tokens are laid out.
        slot_gpu = torch.from_numpy(slot_np * self.max_tgt).to(
            self.qidx.device, non_blocking=True
        )
        num_tokens = input_batch.num_tokens
        batch_idx = torch.searchsorted(
            input_batch.query_start_loc[1 : input_batch.num_reqs + 1],
            self._arange[:num_tokens],
            right=True,
        )
        torch.index_select(slot_gpu, 0, batch_idx, out=self.qidx[:num_tokens])
        pad_end = input_batch.num_tokens_after_padding
        if pad_end > num_tokens:
            self.qidx[num_tokens:pad_end].fill_(self.scratch * self.max_tgt)
        self.qidx[:pad_end].add_(self.positions[:pad_end]).clamp_(max=self._qlimit)

        num_scheduled = input_batch.num_scheduled_tokens
        num_computed = input_batch.num_computed_tokens_np
        for i, req_id in enumerate(input_batch.req_ids):
            self.npos[req_id] = int(num_computed[i]) + int(num_scheduled[i])

        # Only requests on their first step contribute encoder K, one full
        # window each. An oversized batch is the profiling run, which the
        # capture op skips as well, so leave the buffer alone.
        prefills = np.flatnonzero(num_computed == 0)
        num_k = prefills.size * self.max_frames
        if num_k and num_k <= self.kidx.shape[0]:
            rows = slot_np[prefills, None] * self.max_frames + self._frames
            self.kidx[:num_k].copy_(torch.from_numpy(rows.ravel()), non_blocking=True)

    def _report_pool_exhausted(self, req_id: str) -> None:
        """Report that this request will get no word timestamps.

        ``words=None`` is indistinguishable from silence to a client, so an
        exhausted pool is otherwise invisible. Logged on a power-of-two schedule
        to bound the lines however long the overload lasts.
        """
        if req_id in self._denied:
            return
        self._denied.add(req_id)
        self.num_denied += 1
        if self.num_denied & (self.num_denied - 1) == 0:
            logger.warning(
                "Word-timestamp capture pool exhausted (%d slot(s)): %d request(s) "
                "so far have returned no word timestamps. The pool is sized from "
                "the memory allowance, so raise --gpu-memory-utilization or lower "
                "--max-num-seqs to widen coverage.",
                self.scratch,
                self.num_denied,
            )

    def make_readout_fn(self) -> Callable | None:
        """Bind this step's slots for the deferred readout."""
        if not self.enabled or not self.wants:
            return None
        slots = {r: s for r, s in self.slot_of.items() if r in self.wants}
        npos = dict(self.npos)
        final_npos = {r: n for r, n in self._final_npos.items() if r in self.wants}

        def _fn(req_ids, token_ids):
            return self.model.compute_word_align(
                req_ids, token_ids, slots, npos, final_npos
            )

        return _fn

    def remove_request(self, req_id: str) -> None:
        if not self.enabled:
            return
        self.npos.pop(req_id, None)
        self.wants.discard(req_id)
        self._final_npos.pop(req_id, None)
        self._denied.discard(req_id)
        slot = self.slot_of.pop(req_id, None)
        if slot is not None:
            self._free.append(slot)
