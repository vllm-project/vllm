# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import SpeculativeConfig
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.transformers_utils.repo_utils import get_bundled_checkpoint
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)

EAGLE3_SUBFOLDER = "eagle3"
"""Subfolder in which a target checkpoint can ship its EAGLE-3 draft."""


def wrap_eagle_hf_config(spec: SpeculativeConfig, method: str) -> None:
    """Wrap the draft hf_config in an EAGLEConfig for the built-in `method`,
    unless it is one already."""
    draft = spec.draft_model_config
    if isinstance(draft.hf_config, (EAGLEConfig, SpeculatorsConfig)):
        return
    draft.hf_config = EAGLEConfig(draft.hf_config, method=method, model_type="eagle")
    spec.update_arch_()


def check_draft_lm_head_quantization(spec: SpeculativeConfig) -> None:
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


class EagleSpeculatorConfig(DraftModelSpeculatorConfig):
    base_method = "eagle"
    speculator_class = "vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator"
    supported_options = frozenset({"draft_lm_head_quantization"})
    uses_target_hidden_states = True
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
        check_draft_lm_head_quantization(spec)

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens


class Eagle3SpeculatorConfig(EagleSpeculatorConfig):
    base_method = "eagle3"

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        # e.g. yuhuili/EAGLE3-LLaMA3.1-Instruct-8B, AngelSlim/Qwen3-8B_eagle3
        return "eagle3" in spec.draft_model_config.model.lower()

    @classmethod
    def default_model(cls, spec: SpeculativeConfig) -> str | None:
        """A target checkpoint can ship its draft in an `eagle3/` subfolder."""
        target = spec.target_model_config
        if target is not None:
            load_config = spec.draft_load_config
            draft = get_bundled_checkpoint(
                target.model,
                EAGLE3_SUBFOLDER,
                revision=target.revision,
                cache_dir=load_config.download_dir if load_config else None,
            )
            if draft is not None:
                return draft
        return super().default_model(spec)

    @property
    def uses_aux_hidden_states(self) -> bool:
        hf_config = self.spec_config.draft_model_config.hf_config
        eagle_config = getattr(hf_config, "eagle_config", None) or {}
        return eagle_config.get("use_aux_hidden_state", True)
