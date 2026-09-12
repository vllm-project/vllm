# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig


class DFlashSpeculatorConfig(SpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator"
    )
    uses_target_hidden_states = True
    uses_separate_draft_pass = True
    target_embed_sharing = "always"
    supports_watermarking = True
    supports_parallel_drafting = True

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
