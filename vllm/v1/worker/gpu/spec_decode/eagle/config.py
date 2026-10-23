# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import ClassVar

from vllm.config import SpeculativeConfig
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    CheckpointSpeculatorConfig,
    TargetEmbedSharing,
)


def wrap_eagle_hf_config(spec: SpeculativeConfig, method: str) -> None:
    """Wrap the draft hf_config in an EAGLEConfig for the built-in `method`,
    unless it is one already."""
    draft = spec.draft_model_config
    if isinstance(draft.hf_config, (EAGLEConfig, SpeculatorsConfig)):
        return
    draft.hf_config = EAGLEConfig(draft.hf_config, method=method, model_type="eagle")
    spec.update_arch_()


class EagleSpeculatorConfig(CheckpointSpeculatorConfig):
    base_method = "eagle"
    speculator_class = "vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator"
    supported_options = frozenset({"draft_lm_head_quantization"})
    uses_target_hidden_states = True
    target_embed_sharing: ClassVar[TargetEmbedSharing] = "always"
    supports_watermarking = True

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        # e.g. yuhuili/EAGLE-LLaMA3-Instruct-8B
        return "eagle-" in spec.draft_model_config.model.lower()

    @classmethod
    def adapt_draft_model_config(cls, spec: SpeculativeConfig) -> None:
        assert cls.base_method is not None
        wrap_eagle_hf_config(spec, cls.base_method)

    @classmethod
    def apply_options(cls, spec: SpeculativeConfig) -> None:
        draft = spec.draft_model_config
        if (
            spec.draft_lm_head_quantization is not None
            and draft is not None
            and draft.head_dtype != draft.dtype
        ):
            raise ValueError(
                "draft_lm_head_quantization needs the lm_head to run in the "
                f"model dtype, got head_dtype={draft.head_dtype}."
            )

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens


class Eagle3SpeculatorConfig(EagleSpeculatorConfig):
    base_method = "eagle3"

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        # e.g. yuhuili/EAGLE3-LLaMA3.1-Instruct-8B, AngelSlim/Qwen3-8B_eagle3
        return "eagle3" in spec.draft_model_config.model.lower()

    @property
    def uses_aux_hidden_states(self) -> bool:
        hf_config = self.spec_config.draft_model_config.hf_config
        eagle_config = getattr(hf_config, "eagle_config", None) or {}
        return eagle_config.get("use_aux_hidden_state", True)
