# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import ClassVar

from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    TargetEmbedSharing,
)


class EagleSpeculatorConfig(SpeculatorConfig):
    speculator_class = "vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator"
    uses_target_hidden_states = True
    target_embed_sharing: ClassVar[TargetEmbedSharing] = "always"
    supports_watermarking = True
    supports_draft_lm_head_quantization = True

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens


class Eagle3SpeculatorConfig(EagleSpeculatorConfig):
    @property
    def uses_aux_hidden_states(self) -> bool:
        hf_config = self.spec_config.draft_model_config.hf_config
        eagle_config = getattr(hf_config, "eagle_config", None) or {}
        return eagle_config.get("use_aux_hidden_state", True)
