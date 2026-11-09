# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only batch planning and bounded observability for MoE specialization."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import MAX_DECODE_ROWS, MAX_ROUTED_ROWS

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from .moe import FusedMoeDecode


@dataclass(frozen=True)
class DecodeStep:
    num_reqs: int
    actual_rows: int
    padded_rows: int
    reason: str | None


class DecodeRuntime:
    """Shared planning state; never owns another weight or cache representation."""

    def __init__(self):
        self.executors: list[FusedMoeDecode] = []
        self.step: DecodeStep | None = None
        self.steps: Counter[str] = Counter()
        self.history: deque[tuple[tuple[str, ...], DecodeStep]] | None = None

    def prepare_step(self, batch: "InputBatch") -> DecodeStep:
        reason = None
        if batch.has_prefill:
            reason = "prefill_or_mixed"
        elif batch.num_draft_tokens or not (batch.num_scheduled_tokens == 1).all():
            reason = "not_ordinary_decode"
        elif (
            not 0
            < batch.num_reqs
            == batch.num_tokens
            <= batch.num_tokens_after_padding
            <= MAX_DECODE_ROWS
        ):
            reason = "token_rows"
        elif any(
            batch.num_tokens_after_padding * e.top_k > MAX_ROUTED_ROWS
            for e in self.executors
        ):
            reason = "routed_rows"
        self.step = DecodeStep(
            batch.num_reqs, batch.num_tokens, batch.num_tokens_after_padding, reason
        )
        self.steps[reason or "eligible"] += 1
        if self.history is not None:
            self.history.append((tuple(batch.req_ids), self.step))
        return self.step

    def prepare_capture(self, num_reqs: int, num_tokens: int) -> None:
        reason = (
            None if 0 < num_reqs == num_tokens <= MAX_DECODE_ROWS else "capture_shape"
        )
        self.step = DecodeStep(num_reqs, num_tokens, num_tokens, reason)

    def snapshot(self) -> dict:
        return {
            "layers": len(self.executors),
            "planned_steps": dict(self.steps),
            "persistent_scratch_bytes": sum(e.scratch_bytes for e in self.executors),
            "specialized_weight_bytes": (
                0 if all(e.bindings_match() for e in self.executors) else None
            ),
            "native_launches_including_capture": sum(
                e.native_launches for e in self.executors
            ),
            "bindings_shared": all(e.bindings_match() for e in self.executors),
        }
