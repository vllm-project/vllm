# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-engine beam groups.

A beam group is k sibling requests that the scheduler admits and decodes
together. This is the scheduling primitive from the native beam-search RFC
(https://github.com/vllm-project/vllm/issues/61880). Selection, KV fork, and
the public ``LLM.beam_search`` path are separate steps.
"""

from dataclasses import dataclass

from vllm.v1.request import Request


@dataclass
class BeamSearchGroup:
    """k slot requests that must be scheduled all-or-nothing."""

    group_id: str
    requests: list[Request]

    def __post_init__(self) -> None:
        if len(self.requests) < 2:
            raise ValueError(
                f"Beam group {self.group_id} needs at least 2 slots, "
                f"got {len(self.requests)}"
            )
        prompt_len = self.requests[0].num_prompt_tokens
        max_tokens = self.requests[0].max_tokens
        for request in self.requests:
            if request.num_prompt_tokens != prompt_len:
                raise ValueError(
                    f"Beam group {self.group_id} slots have different prompt lengths"
                )
            if request.max_tokens != max_tokens:
                raise ValueError(
                    f"Beam group {self.group_id} slots have different max_tokens"
                )
            if request.use_structured_output:
                raise ValueError(
                    "Beam groups do not support structured outputs; "
                    "use the beam search wrapper"
                )
            if request.has_encoder_inputs:
                raise ValueError("Beam groups do not support multimodal encoder inputs")

    @property
    def beam_width(self) -> int:
        return len(self.requests)


def equal_chunk_size(
    requests: list[Request],
    *,
    token_budget: int,
    input_budget: int,
    draft_slots: int,
    max_model_len: int,
    num_sampled_tokens: int,
) -> int:
    """Tokens to schedule for every slot this step.

    Returns 0 when the slots cannot move together: one slot has nothing left
    to compute, or the step budget cannot give every slot at least one token.
    """
    width = len(requests)
    if width == 0 or token_budget < width:
        return 0

    per_request: list[int] = []
    for request in requests:
        num_new = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        room = max_model_len - request.num_computed_tokens - num_sampled_tokens
        num_new = min(num_new, room)
        if num_new <= 0:
            return 0
        per_request.append(num_new)

    input_room = input_budget - width * draft_slots
    if input_room < width:
        return 0
    per_slot = min(per_request)
    per_slot = min(per_slot, token_budget // width, input_room // width)
    return per_slot if per_slot >= 1 else 0
