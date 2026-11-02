# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.eagle.config import wrap_eagle_hf_config
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DFlashSpeculatorConfig(DraftModelSpeculatorConfig):
    base_method = "dflash"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator"
    )
    uses_target_hidden_states = True
    supports_watermarking = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        draft = spec.draft_model_config
        model_name = (draft.model or "").lower()
        if "dflash" in model_name or "dspine" in model_name:
            return True
        if "DSpineDraftModel" in (draft.architectures or []):
            return True
        # Imported lazily: vllm.config.speculative imports this package.
        from vllm.config.speculative import _get_nested_config_value

        return bool(
            _get_nested_config_value(
                draft.hf_config,
                "dflash_config",
                "selected_prefix_code_bridge",
            )
        )

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        assert cls.base_method is not None
        wrap_eagle_hf_config(spec, cls.base_method)
        spec.parallel_drafting = True
        # Imported lazily: vllm.config.speculative imports this package.
        from vllm.config.speculative import _apply_dspine_draft_config

        _apply_dspine_draft_config(spec)

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        # Imported lazily: vllm.config.speculative imports this package.
        from vllm.config.speculative import _resolve_dspine_num_speculative_tokens

        _resolve_dspine_num_speculative_tokens(spec)
        super().resolve_num_speculative_tokens(spec)

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        # Fill-in drafting uses a bonus query plus one query per draft token.
        return self.num_speculative_tokens + 1

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int:
        # The draft pass has num_lookahead_tokens queries for every request.
        return max_num_batched_tokens // self.num_lookahead_tokens
