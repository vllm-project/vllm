# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper word-timestamp capture for the V2 GPU model runner.

Cross-attention Q (and the encoder K at prefill) are recorded into per-request
buffers before the fused attention kernel, then DTW-aligned once a request
finishes. Keeping the routing here rather than in the runner follows the
composition pattern used by ``kv_connector.py``.

The capture slot for a request is simply its ``RequestState`` index, which is
stable for the request's lifetime, so no separate slot pool is needed. One
extra slot past ``max_num_reqs`` absorbs cudagraph padding and dummy runs.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


class WordAlignCapturer:
    """Routes cross-attention capture rows to per-request slots (opt-in)."""

    def __init__(self) -> None:
        self.enabled = False
        # req_id -> decoder positions decoded so far, for cropping the DTW.
        self.npos: dict[str, int] = {}

    def init(self, runner: "GPUModelRunner") -> None:
        """Allocate capture buffers and turn on capture in the model.

        Must run before graph capture so the compiled decoder includes the
        capture op. No-op unless requested and the model supports it.
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
        # The request-state index doubles as the capture slot; one extra slot
        # absorbs cudagraph-padding and dummy-run rows.
        num_slots = runner.max_num_reqs
        max_frames = int(hf_config.max_source_positions)
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
        self.qslot, self.kslot, self.kpos = model.word_align_index_tensors()
        self.scratch = num_slots
        self.max_frames = max_frames
        # Park every row on the scratch slot so warmup and cudagraph capture,
        # which run before any real batch, cannot write into a request's slot.
        self.qslot.fill_(self.scratch)
        self.enabled = True
        logger.info("Whisper word-timestamp cross-attention capture enabled")

    def before_forward(self, input_batch: InputBatch) -> None:
        """Point this step's capture rows at each request's own slot."""
        if not self.enabled:
            return
        num_scheduled = input_batch.num_scheduled_tokens
        # Token rows: repeat each request's slot for the tokens it scheduled.
        # idx_mapping_np is batch_idx -> req_state_idx, which is the slot.
        token_slot = np.repeat(input_batch.idx_mapping_np, num_scheduled)
        num_tokens = token_slot.shape[0]
        self.qslot.fill_(self.scratch)
        self.qslot[:num_tokens].copy_(
            torch.from_numpy(token_slot.astype(np.int64)), non_blocking=True
        )

        num_computed = input_batch.num_computed_tokens_np
        for i, req_id in enumerate(input_batch.req_ids):
            self.npos[req_id] = int(num_computed[i]) + int(num_scheduled[i])

        # Encoder rows: only requests on their first step contribute K, and
        # Whisper pads every clip to a full encoder window, so each such
        # request contributes exactly max_frames rows in batch order.
        prefills = np.flatnonzero(num_computed == 0)
        if prefills.size:
            frames = self.max_frames
            num_k = prefills.size * frames
            self.kslot[:num_k].copy_(
                torch.from_numpy(
                    np.repeat(input_batch.idx_mapping_np[prefills], frames).astype(
                        np.int64
                    )
                ),
                non_blocking=True,
            )
            self.kpos[:num_k].copy_(
                torch.from_numpy(
                    np.tile(np.arange(frames, dtype=np.int64), prefills.size)
                ),
                non_blocking=True,
            )

    def make_readout_fn(self, input_batch: InputBatch) -> Any:
        """Bind this step's slots so the deferred readout aligns each finishing
        request against the slot it actually wrote to."""
        if not self.enabled:
            return None
        slots = {
            req_id: int(slot)
            for req_id, slot in zip(input_batch.req_ids, input_batch.idx_mapping_np)
        }
        npos = dict(self.npos)
        return lambda req_ids, token_ids: self.model.compute_word_align(
            req_ids, token_ids, slots, npos
        )

    def remove_request(self, req_id: str) -> None:
        if self.enabled:
            self.npos.pop(req_id, None)
