# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pattern-triggered forced sequence LogitsProcessor for Harmony format.
Used for GPT-OSS models to enforce tool_choice='required'.
"""

from enum import Enum, auto
from typing import NamedTuple

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)


class ForcingState(Enum):
    NORMAL = auto()
    FORCING = auto()


class RequestState(NamedTuple):
    """Per-request state for pattern-triggered forced sequence."""

    state: ForcingState
    forcing_pos: int
    output_ids: list[int]
    trigger_pattern: list[int]
    forced_sequence: list[int]


class PatternForcedSequenceLogitsProcessor(LogitsProcessor):
    """
    Detects trigger pattern and forces a token sequence.

    Enabled via SamplingParams.extra_args["harmony_tool_required"] which can be:
    - A dict with "trigger_pattern" and "forced_sequence" keys (list of token IDs)
    - Example:
        extra_args["harmony_tool_required"] = {
            "trigger_pattern": [200007, 200006, 173781, 200005],
            "forced_sequence": [12606, 815, 316, 28],
        }
    """

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        del is_pin_memory  # unused, interface requirement
        self.req_states: dict[int, RequestState] = {}
        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=device
        )

    def is_argmax_invariant(self) -> bool:
        """By forcing specific tokens, this processor changes the outcome
        of the argmax operation in greedy sampling."""
        return False

    def needs_output_token_ids(self) -> bool:
        return True

    @staticmethod
    def add_request(
        params: SamplingParams,
        _: list[int] | None,
        output_tok_ids: list[int],
    ) -> RequestState | None:
        if not params.extra_args:
            return None

        harmony_config = params.extra_args.get("harmony_tool_required")
        if not harmony_config:
            return None

        # Parse dict format with trigger_pattern and forced_sequence
        if not isinstance(harmony_config, dict):
            return None

        trigger_pattern = harmony_config.get("trigger_pattern")
        forced_sequence = harmony_config.get("forced_sequence")

        if not trigger_pattern or not forced_sequence:
            return None

        return RequestState(
            state=ForcingState.NORMAL,
            forcing_pos=0,
            output_ids=output_tok_ids,
            trigger_pattern=trigger_pattern,
            forced_sequence=forced_sequence,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.req_states, batch_update, self.add_request)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        for index, req_state in list(self.req_states.items()):
            state = req_state.state
            pos = req_state.forcing_pos
            output_ids = req_state.output_ids
            trigger_pattern = req_state.trigger_pattern
            forced_sequence = req_state.forced_sequence

            real_tokens = [t for t in output_ids if t != -1]

            if state == ForcingState.NORMAL and len(real_tokens) >= len(
                trigger_pattern
            ):
                tail = real_tokens[-len(trigger_pattern) :]
                if tail == trigger_pattern:
                    req_state = req_state._replace(
                        state=ForcingState.FORCING,
                        forcing_pos=0,
                    )
                    self.req_states[index] = req_state
                    state = ForcingState.FORCING
                    pos = 0

            if state == ForcingState.FORCING:
                if pos < len(forced_sequence):
                    allowed = forced_sequence[pos]
                    original = logits[index, allowed].clone()
                    logits[index] = self.neg_inf_tensor
                    logits[index, allowed] = original
                    self.req_states[index] = req_state._replace(
                        forcing_pos=pos + 1,
                    )
                else:
                    self.req_states[index] = req_state._replace(
                        state=ForcingState.NORMAL,
                        forcing_pos=0,
                    )

        return logits
