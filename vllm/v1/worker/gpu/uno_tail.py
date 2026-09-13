# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact Uno finish detection using the existing sampled-output copy."""

from dataclasses import dataclass, replace

from vllm.v1.core.sched.output import GrammarOutput, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput


@dataclass
class _RequestProgress:
    output_len: int
    output_cap: int
    stop_ids: frozenset[int]
    finished: bool = False


class UnoTailState:
    def __init__(self, max_model_len: int):
        self.max_model_len = max_model_len
        self.requests: dict[str, _RequestProgress] = {}

    def add_request(self, request: NewRequestData) -> None:
        params = request.sampling_params
        assert params is not None and params.max_tokens is not None
        assert request.prefill_token_ids is not None
        # eos_token_id is already None when ignore_eos is set. Do not use
        # all_stop_token_ids: that set also includes EOS for min-token masking.
        stop_ids = set(params.stop_token_ids or ())
        if params.eos_token_id is not None:
            stop_ids.add(params.eos_token_id)
        self.requests[request.req_id] = _RequestProgress(
            len(request.prefill_token_ids) - request.prompt_len,
            min(params.max_tokens, self.max_model_len - request.prompt_len),
            frozenset(stop_ids),
        )

    def remove_request(self, req_id: str) -> None:
        self.requests.pop(req_id, None)

    def observe(self, output: AsyncOutput) -> frozenset[str]:
        # The copy was issued before postprocess; wait only for that handoff,
        # never for a proposal or a later main-stream operation.
        output.copy_event.synchronize()
        finished = set()
        for row, req_id in enumerate(output.model_runner_output.req_ids):
            progress = self.requests[req_id]
            count = int(output.num_sampled_tokens_np[row])
            for token in output.sampled_token_ids[row, :count]:
                if progress.finished:
                    break
                progress.output_len += 1
                progress.finished = (
                    int(token) in progress.stop_ids
                    or progress.output_len >= progress.output_cap
                )
            if progress.finished:
                finished.add(req_id)
        return frozenset(finished)

    def prepare_followup(
        self, output: SchedulerOutput
    ) -> tuple[SchedulerOutput, dict[str, int]]:
        finished = {
            req_id
            for req_id in output.num_scheduled_tokens
            if self.requests[req_id].finished
        }
        if not finished:
            return output, {}
        # Preserve scheduler accounting/KV ownership, changing only the worker
        # snapshot. Its output length rolls back the original placeholders.
        original_drafts = {
            req_id: len(tokens)
            for req_id, tokens in output.scheduled_spec_decode_tokens.items()
        }
        scheduled = dict(output.num_scheduled_tokens)
        drafts = dict(output.scheduled_spec_decode_tokens)
        for req_id in finished:
            scheduled[req_id] -= len(drafts.pop(req_id, ()))
        zero_rows = output.zero_next_draft_req_ids | finished
        return replace(
            output,
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens=drafts,
            zero_next_draft_req_ids=zero_rows,
            skip_speculator_proposal=zero_rows.issuperset(scheduled),
        ), original_drafts


def trim_finished_grammar(
    grammar: GrammarOutput | None,
    original_drafts: dict[str, int],
    current_drafts: dict[str, int],
) -> GrammarOutput | None:
    if grammar is None or not original_drafts:
        return grammar
    rows = []
    offset = 0
    for req_id in grammar.structured_output_request_ids:
        original_width = original_drafts.get(req_id, 0) + 1
        current_width = current_drafts.get(req_id, 0) + 1
        rows.extend(range(offset, offset + current_width))
        offset += original_width
    assert offset == len(grammar.grammar_bitmask)
    return GrammarOutput(
        grammar.structured_output_request_ids, grammar.grammar_bitmask[rows]
    )
