# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, ClassVar

from vllm.v1.worker.gpu.spec_decode.eagle.config import wrap_eagle_hf_config
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    CheckpointSpeculatorConfig,
    TargetEmbedSharing,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DFlashSpeculatorConfig(CheckpointSpeculatorConfig):
    base_method = "dflash"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator"
    )
    uses_target_hidden_states = True
    uses_separate_draft_pass = True
    target_embed_sharing: ClassVar[TargetEmbedSharing] = "always"
    supports_watermarking = True
    supports_parallel_drafting = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        return "dflash" in spec.draft_model_config.model.lower()

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        assert cls.base_method is not None
        wrap_eagle_hf_config(spec, cls.base_method)
        spec.parallel_drafting = True

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        # Fill-in drafting uses a bonus query plus one query per draft token.
        return self.num_speculative_tokens + 1

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        return 0
