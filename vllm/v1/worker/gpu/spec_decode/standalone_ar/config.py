# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    CheckpointSpeculatorConfig,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DraftModelSpeculatorConfig(CheckpointSpeculatorConfig):
    """A speculator that drafts autoregressively with its own draft model."""

    base_method = "draft_model"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator.StandaloneARSpeculator"
    )
    draft_has_own_vocab = True
    # Rejects recognized drafters (e.g. MTP heads).
    validates_checkpoint = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        # Any draft model checkpoint.
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        if self.spec_config.parallel_drafting:
            # PARD does not shift the existing input, so all K query positions
            # require additional slots.
            return self.num_speculative_tokens
        # The autoregressive draft-model input retains one unsliced token.
        return 1
