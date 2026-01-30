# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pattern-triggered forced sequence LogitsProcessor for Harmony format.
Used for GPT-OSS models to enforce tool_choice='required'.
"""

from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ForcingState(Enum):
    NORMAL = auto()
    FORCING = auto()


# State tuple: (state, forcing_pos, output_ids, trigger_pattern, forced_sequence)
RequestState = tuple[ForcingState, int, list[int], list[int], list[int]]


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

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        # index -> (state, forcing_pos, output_ids, trigger_pattern, forced_sequence)
        self.req_states: dict[int, RequestState] = {}
        self.neg_inf = torch.tensor(-float("inf"), dtype=torch.float32, device=device)

    def is_argmax_invariant(self) -> bool:
        return False

    def needs_output_token_ids(self) -> bool:
        return True

    def _add_request(
        self,
        params: SamplingParams,
        _prompt_tok_ids: list[int] | None,
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

        return (
            ForcingState.NORMAL,
            0,
            output_tok_ids,
            trigger_pattern,
            forced_sequence,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.req_states, batch_update, self._add_request)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        for index, (
            state,
            pos,
            output_ids,
            trigger_pattern,
            forced_sequence,
        ) in list(self.req_states.items()):
            real_tokens = [t for t in output_ids if t != -1]

            if state == ForcingState.NORMAL and len(real_tokens) >= len(
                trigger_pattern
            ):
                tail = real_tokens[-len(trigger_pattern) :]
                if tail == trigger_pattern:
                    self.req_states[index] = (
                        ForcingState.FORCING,
                        0,
                        output_ids,
                        trigger_pattern,
                        forced_sequence,
                    )
                    state = ForcingState.FORCING
                    pos = 0

            if state == ForcingState.FORCING:
                if pos < len(forced_sequence):
                    allowed = forced_sequence[pos]
                    original = logits[index, allowed].clone()
                    logits[index] = self.neg_inf
                    logits[index, allowed] = original
                    self.req_states[index] = (
                        state,
                        pos + 1,
                        output_ids,
                        trigger_pattern,
                        forced_sequence,
                    )
                else:
                    self.req_states[index] = (
                        ForcingState.NORMAL,
                        0,
                        output_ids,
                        trigger_pattern,
                        forced_sequence,
                    )

        return logits
