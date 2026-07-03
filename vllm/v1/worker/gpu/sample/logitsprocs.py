# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessors

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(eq=False)
class RequestRecord:
    req_id: str
    params: SamplingParams
    prompt_token_ids: list[int] | None
    # Live list of generated tokens, shared with the logits processors
    # through BatchUpdate.added. Extended in place as sampled tokens
    # arrive on the CPU.
    output_token_ids: list[int]


@dataclass
class PendingSampledTokens:
    # Row -> request slot and record at the time of sampling.
    slots: np.ndarray
    records: list["RequestRecord | None"]
    # Backed by the AsyncOutput D2H copies; valid once copy_event completes.
    sampled_token_ids: np.ndarray  # [num_rows, 1]
    num_sampled: np.ndarray  # [num_rows]
    copy_event: torch.Event


class CustomLogitsprocs:
    """Adapts custom logits processors to the V2 sampler.

    The LogitsProcessor interface assumes a contiguous persistent batch
    updated through BatchUpdate add/remove/move events, with a live CPU
    list of each request's output tokens. The V2 model runner instead
    keys request state by stable slot indices and keeps output tokens on
    the GPU. This class bridges the two:

    * Each step, the logits-row -> request assignment (idx_mapping) is
      diffed against the previous step's and translated into BatchUpdate
      added/removed events. A request whose row changed is re-added at
      its new row, which the interface permits ("added requests may
      replace existing requests with the same index"). Unlike the V1
      runner, no moved events are emitted: per-request state is rebuilt
      from the request's params, prompt and (live) output token ids,
      which is lossless for logits processors that derive their state
      from those inputs (the documented pattern).
    * Sampled tokens are appended to the per-request CPU token lists
      from the async D2H copy of the previous step's sampler output,
      right before the logits processors read them (mirroring the V1
      runner's update_async_output_token_ids).
    """

    def __init__(self, max_num_reqs: int, logitsprocs: LogitsProcessors):
        self.logitsprocs = logitsprocs
        self.records: list[RequestRecord | None] = [None] * max_num_reqs
        # Row -> record assignment from the last update_state call.
        self.prev_rows: list[RequestRecord | None] = []
        self.pending: PendingSampledTokens | None = None
        self.num_active = 0

    def add_request(
        self,
        req_idx: int,
        req_id: str,
        prompt_len: int,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None,
        prefill_token_ids: list[int],
    ) -> None:
        # prefill_token_ids holds the prompt plus any output tokens already
        # generated (e.g. when resuming after preemption).
        self.records[req_idx] = RequestRecord(
            req_id=req_id,
            params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=list(prefill_token_ids[prompt_len:]),
        )

    def remove_request(self, req_idx: int) -> None:
        self.records[req_idx] = None

    def register_step_output(
        self,
        input_batch: "InputBatch",
        sampled_token_ids: np.ndarray,
        num_sampled: np.ndarray,
        copy_event: torch.Event,
    ) -> None:
        """Record this step's sampled-token D2H copy for the next step.

        Called after sampling, so prev_rows holds the row -> record
        assignment the tokens were sampled with.
        """
        self.pending = PendingSampledTokens(
            # Copy: idx_mapping_np may be rebuilt before the drain.
            slots=input_batch.idx_mapping_np.copy(),
            records=self.prev_rows,
            sampled_token_ids=sampled_token_ids,
            num_sampled=num_sampled,
            copy_event=copy_event,
        )

    def _drain_pending(self) -> None:
        pending = self.pending
        if pending is None:
            return
        self.pending = None
        pending.copy_event.synchronize()
        num_sampled = pending.num_sampled.tolist()
        sampled_token_ids = pending.sampled_token_ids.tolist()
        for row, record in enumerate(pending.records):
            if record is None:
                continue
            num_tokens = num_sampled[row]
            if num_tokens <= 0:
                # No token sampled (e.g. partial prefill).
                continue
            if self.records[pending.slots[row]] is not record:
                # The slot was reassigned since the tokens were sampled
                # (request finished/preempted, or streaming input update).
                continue
            record.output_token_ids.extend(sampled_token_ids[row][:num_tokens])

    def _make_batch_update(
        self, cur_rows: list["RequestRecord | None"]
    ) -> BatchUpdate | None:
        prev_rows = self.prev_rows
        added = [
            (row, record.params, record.prompt_token_ids, record.output_token_ids)
            for row, record in enumerate(cur_rows)
            if record is not None
            and (row >= len(prev_rows) or prev_rows[row] is not record)
        ]
        removed = [
            row
            for row, record in enumerate(prev_rows)
            if record is not None and (row >= len(cur_rows) or cur_rows[row] is None)
        ]
        if not added and not removed:
            return None
        # Descending order, as guaranteed by BatchUpdateBuilder.
        removed.sort(reverse=True)
        return BatchUpdate(
            batch_size=len(cur_rows),
            removed=removed,
            added=added,
            moved=(),
        )

    def update_state(self, input_batch: "InputBatch") -> None:
        assert input_batch.num_draft_tokens == 0, (
            "Custom logits processors do not support speculative decoding."
        )
        self._drain_pending()
        cur_rows: list[RequestRecord | None] = [
            self.records[slot] for slot in input_batch.idx_mapping_np
        ]
        batch_update = self._make_batch_update(cur_rows)
        self.prev_rows = cur_rows
        self.num_active = sum(record is not None for record in cur_rows)
        for logitproc in self.logitsprocs.all:
            logitproc.update_state(batch_update)

    def apply_non_argmax_invariant(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_active == 0:
            # No tracked requests (e.g. dummy sampler run).
            return logits
        for logitproc in self.logitsprocs.non_argmax_invariant:
            logits = logitproc.apply(logits)
        return logits

    def apply_argmax_invariant(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_active == 0:
            return logits
        for logitproc in self.logitsprocs.argmax_invariant:
            logits = logitproc.apply(logits)
        return logits
