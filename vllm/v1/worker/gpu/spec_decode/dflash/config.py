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
    parallel_drafting = True
    supports_watermarking = True
    # Stores its context K/V from the target's hidden states, without drafting.
    supports_prefill_context_kv_only = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        return "dflash" in spec.draft_model_config.model.lower()

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        assert cls.base_method is not None
        wrap_eagle_hf_config(spec, cls.base_method)

    @classmethod
    def default_draft_decode_context_parallel_size(
        cls, spec: "SpeculativeConfig"
    ) -> int:
        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_sliding

        draft = spec.draft_model_config
        # Only full-attention MLA drafts shard their cache with the target:
        # windowed layers cannot run under DCP, and non-MLA drafts stay replicated.
        if (
            draft.use_mla
            and draft.get_sliding_window() is None
            and not dflash_has_any_sliding(draft.hf_config)
        ):
            return super().default_draft_decode_context_parallel_size(spec)
        return 1

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        # Fill-in drafting uses a bonus query plus one query per draft token.
        num_drafts = self.num_scheduled_spec_tokens
        return num_drafts + 1 if num_drafts else 0

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int | None:
        # The draft pass has num_lookahead_tokens queries for every request.
        num_queries = self.num_lookahead_tokens
        return max_num_batched_tokens // num_queries if num_queries else None
