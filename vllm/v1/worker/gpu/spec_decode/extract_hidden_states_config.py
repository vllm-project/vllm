# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import TYPE_CHECKING

from vllm.transformers_utils.configs.extract_hidden_states import (
    ExtractHiddenStatesConfig,
)
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class ExtractHiddenStatesSpeculatorConfig(SpeculatorConfig):
    base_method = "extract_hidden_states"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.extract_hidden_states."
        "ExtractHiddenStatesSpeculator"
    )
    proposes_draft_tokens = False

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        return "extract_hidden_states"

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        # ExtractHiddenStatesModel is instantiated manually in load_model()
        # We just need to store the target model config for KV cache shape info
        spec.model = "extract_hidden_states"
        spec.prompt_lookup_max = 0
        spec.prompt_lookup_min = 0

        draft_model_config = spec.draft_model_config
        if hasattr(draft_model_config, "hf_config"):
            hf_config = draft_model_config.hf_config.to_dict()
        elif isinstance(draft_model_config, dict) and "hf_config" in draft_model_config:
            hf_config = draft_model_config["hf_config"]
        else:
            hf_config = {}

        spec.draft_model_config = copy.copy(spec.target_model_config)
        spec.draft_model_config.hf_config = ExtractHiddenStatesConfig(
            spec.draft_model_config.hf_config, **hf_config
        )
        spec.update_arch_()
        spec.draft_parallel_config = spec.target_parallel_config

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True
