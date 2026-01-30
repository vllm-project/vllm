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
        # index -> (state, forcing_pos, output_ids)
        self.req_states: dict[int, tuple[ForcingState, int, list[int]]] = {}
        self.neg_inf = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
        logger.info("[PIPELINE] PatternForcedSequenceLogitsProcessor initialized")

    def is_argmax_invariant(self) -> bool:
        return False

    def _add_request(
        self,
        params: SamplingParams,
        _prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> tuple[ForcingState, int, list[int]] | None:
        logger.info(
            "[PIPELINE] _add_request called: extra_args=%s",
            params.extra_args,
        )
        if not params.extra_args:
            logger.info("[PIPELINE] _add_request: no extra_args, skipping")
            return None
        if not params.extra_args.get("harmony_tool_required"):
            logger.info(
                "[PIPELINE] _add_request: harmony_tool_required not set, skipping"
            )
            return None
        logger.info("[PIPELINE] _add_request: tracking request for tool forcing")
        return (ForcingState.NORMAL, 0, output_tok_ids)

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.req_states, batch_update, self._add_request)

        for index, (state, pos, output_ids) in list(self.req_states.items()):
            if state == ForcingState.NORMAL:
                if (
                    len(output_ids) >= len(TRIGGER_PATTERN)
                    and output_ids[-len(TRIGGER_PATTERN) :] == TRIGGER_PATTERN
                ):
                    logger.info(
                        "[PIPELINE] Trigger pattern detected! output_ids[-4:]=%s, "
                        "transitioning to FORCING",
                        output_ids[-len(TRIGGER_PATTERN) :],
                    )
                    self.req_states[index] = (ForcingState.FORCING, 0, output_ids)
            elif state == ForcingState.FORCING and pos >= len(FORCED_SEQUENCE):
                logger.info("[PIPELINE] Forced sequence complete, returning to NORMAL")
                self.req_states[index] = (ForcingState.NORMAL, 0, output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, (state, pos, output_ids) in list(self.req_states.items()):
            if state == ForcingState.FORCING and pos < len(FORCED_SEQUENCE):
                allowed = FORCED_SEQUENCE[pos]
                logger.info(
                    "[PIPELINE] FORCING: masking logits, allowing only token %d "
                    "(pos=%d/%d)",
                    allowed,
                    pos,
                    len(FORCED_SEQUENCE),
                )
                original = logits[index, allowed].clone()
                logits[index] = self.neg_inf
                logits[index, allowed] = original
                self.req_states[index] = (state, pos + 1, output_ids)
        return logits
