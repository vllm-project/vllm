# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recognize fixed-width verifier batches containing reject-only prompt tails."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.model_runner import BatchReqState


def is_padded_prompt_tail_batch(
    output: "SchedulerOutput",
    state: "BatchReqState | None",
    *,
    decode_query_len: int,
    num_speculative_tokens: int,
    supported: bool,
) -> bool:
    """Validate padded prompt tails without changing any prefill state.

    Shape alone is insufficient: every row must carry the complete verifier
    layout, and every prefilling row must be exactly one cached prompt-tail
    token followed by reject-only drafts. The caller checks execution support;
    the graph manager still decides whether a matching graph is available.
    """
    if (
        not supported
        or state is None
        or not state.has_prefill
        or not state.req_ids
        or num_speculative_tokens <= 0
        or decode_query_len != 1 + num_speculative_tokens
        or len(state.req_ids) != len(output.num_scheduled_tokens)
        or state.num_tokens != len(state.req_ids) * decode_query_len
    ):
        return False

    for i, req_id in enumerate(state.req_ids):
        if output.num_scheduled_tokens[req_id] != decode_query_len:
            return False
        drafts = output.scheduled_spec_decode_tokens.get(req_id, ())
        if len(drafts) != num_speculative_tokens:
            return False
        if state.is_prefilling_np[i]:
            computed = int(state.num_computed_prefill_tokens_np[i])
            if computed <= 0 or int(state.prefill_len_np[i]) - computed != 1:
                return False
            if any(token != -1 for token in drafts):
                return False

    return True
