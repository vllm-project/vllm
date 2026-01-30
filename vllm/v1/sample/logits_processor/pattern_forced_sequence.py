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
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Hardcoded token IDs for Harmony format (GPT-OSS)
# <|end|><|start|>assistant<|channel|>
TRIGGER_PATTERN = [200007, 200006, 173781, 200005]
# commentary to=
FORCED_SEQUENCE = [12606, 815, 316, 28]


class ForcingState(Enum):
    NORMAL = auto()
    FORCING = auto()


class PatternForcedSequenceLogitsProcessor(LogitsProcessor):
    """
    Detects trigger pattern and forces a token sequence.
    Enabled via SamplingParams.extra_args["harmony_tool_required"] = True.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        # index -> (state, forcing_pos, output_ids, output_ids_id)
        # output_ids_id is id() of the list for debugging reference issues
        self.req_states: dict[int, tuple[ForcingState, int, list[int], int]] = {}
        self.neg_inf = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
        self.apply_call_count = 0
        logger.info(
            "[FORCE] Initialized: trigger=%s, forced=%s",
            TRIGGER_PATTERN,
            FORCED_SEQUENCE,
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def needs_output_token_ids(self) -> bool:
        return True

    def _add_request(
        self,
        params: SamplingParams,
        _prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> tuple[ForcingState, int, list[int], int] | None:
        if not params.extra_args:
            return None
        if not params.extra_args.get("harmony_tool_required"):
            return None
        list_id = id(output_tok_ids)
        logger.info(
            "[FORCE] ADD: list_id=%d, initial_len=%d, initial_content=%s",
            list_id,
            len(output_tok_ids),
            output_tok_ids[:10] if output_tok_ids else [],
        )
        return (ForcingState.NORMAL, 0, output_tok_ids, list_id)

    def update_state(self, batch_update: BatchUpdate | None):
        old_count = len(self.req_states)
        process_dict_updates(self.req_states, batch_update, self._add_request)
        new_count = len(self.req_states)
        if old_count != new_count:
            logger.info(
                "[FORCE] UPDATE_STATE: tracked %d -> %d requests",
                old_count,
                new_count,
            )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        self.apply_call_count += 1

        if not self.req_states:
            return logits

        for index, (state, pos, output_ids, list_id) in list(self.req_states.items()):
            # Check list reference consistency
            current_list_id = id(output_ids)
            raw_len = len(output_ids)
            real_tokens = [t for t in output_ids if t != -1]
            real_len = len(real_tokens)

            # Log state every call for debugging
            last6_raw = output_ids[-6:] if raw_len >= 6 else output_ids
            last6_real = real_tokens[-6:] if real_len >= 6 else real_tokens
            logger.info(
                "[FORCE] APPLY #%d idx=%d state=%s pos=%d | "
                "raw_len=%d real_len=%d | last6_raw=%s last6_real=%s | "
                "list_id_match=%s",
                self.apply_call_count,
                index,
                state.name,
                pos,
                raw_len,
                real_len,
                last6_raw,
                last6_real,
                list_id == current_list_id,
            )

            if state == ForcingState.NORMAL and real_len >= len(TRIGGER_PATTERN):
                tail = real_tokens[-len(TRIGGER_PATTERN) :]
                if tail == TRIGGER_PATTERN:
                    logger.info("[FORCE] TRIGGER MATCH! tail=%s -> FORCING", tail)
                    self.req_states[index] = (
                        ForcingState.FORCING,
                        0,
                        output_ids,
                        list_id,
                    )
                    state = ForcingState.FORCING
                    pos = 0

            if state == ForcingState.FORCING:
                if pos < len(FORCED_SEQUENCE):
                    allowed = FORCED_SEQUENCE[pos]
                    logger.info(
                        "[FORCE] MASK: only token %d allowed (pos=%d/%d)",
                        allowed,
                        pos,
                        len(FORCED_SEQUENCE),
                    )
                    original = logits[index, allowed].clone()
                    logits[index] = self.neg_inf
                    logits[index, allowed] = original
                    self.req_states[index] = (state, pos + 1, output_ids, list_id)
                else:
                    logger.info("[FORCE] COMPLETE -> NORMAL")
                    self.req_states[index] = (
                        ForcingState.NORMAL,
                        0,
                        output_ids,
                        list_id,
                    )

        return logits
