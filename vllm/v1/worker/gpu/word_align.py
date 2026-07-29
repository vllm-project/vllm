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

# A slot holds one request's decoder Q and encoder K, and requests past the pool
# silently get no timestamps. Sizing the pool by memory rather than by a fixed
# count keeps that coverage as wide as the budget allows: turbo (2 alignment
# layers) then covers a full batch where large-v3 (10) could not.
MAX_CAPTURE_BYTES = 2 * 1024**3


class WordAlignCapturer:
    """Routes cross-attention capture rows to per-request slots (opt-in)."""

    def __init__(self) -> None:
        self.enabled = False
        # req_id -> decoder positions decoded so far, for cropping the DTW.
        self.npos: dict[str, int] = {}
        # req_id -> capture slot, held for the request's lifetime.
        self.slot_of: dict[str, int] = {}
        self._free: list[int] = []

    @staticmethod
    def _pool_size(runner, gen_config, hf_config, max_frames: int) -> int:
        """Largest slot count whose capture buffers fit in the budget."""
        n_layers = len({h[0] for h in gen_config.alignment_heads})
        per_slot = (
            (max_frames + int(hf_config.max_target_positions))
            * n_layers
            * int(hf_config.d_model)
            * runner.model_config.dtype.itemsize
        )
        return max(1, min(runner.max_num_reqs, MAX_CAPTURE_BYTES // per_slot))

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
        num_slots = self._pool_size(runner, gen_config, hf_config, max_frames)
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
        logger.info("Whisper word-timestamp cross-attention capture enabled")

    def before_forward(self, input_batch: InputBatch) -> None:
        """Point this step's capture rows at each request's own slot."""
        if not self.enabled:
            return
        for req_id in input_batch.req_ids:
            if req_id not in self.slot_of and self._free:
                self.slot_of[req_id] = self._free.pop()
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

    def make_readout_fn(self) -> Callable | None:
        """Bind this step's slots for the deferred readout."""
        if not self.enabled:
            return None
        slots = dict(self.slot_of)
        npos = dict(self.npos)
        return lambda req_ids, token_ids: self.model.compute_word_align(
            req_ids, token_ids, slots, npos
        )

    def remove_request(self, req_id: str) -> None:
        if not self.enabled:
            return
        self.npos.pop(req_id, None)
        slot = self.slot_of.pop(req_id, None)
        if slot is not None:
            self._free.append(slot)
