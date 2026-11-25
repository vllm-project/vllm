# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.eagle.config import wrap_eagle_hf_config
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeculativeConfig


class DFlashSpeculatorConfig(DraftModelSpeculatorConfig):
    base_method = "dflash"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator"
    )
    uses_target_hidden_states = True
    parallel_drafting = True
    supports_watermarking = True
    # Its context K/V is projected from the target's hidden states.
    projects_context_kv = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        return "dflash" in spec.draft_model_config.model.lower()

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        assert cls.base_method is not None
        wrap_eagle_hf_config(spec, cls.base_method)

    @classmethod
    def supports_draft_eplb(cls, draft_model_config: "ModelConfig") -> bool:
        return draft_model_config.is_moe

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
    def sample_from_anchor(self) -> bool:
        """Whether the anchor query also predicts a draft token.

        False (the default) is the fill-in layout: the anchor is the bonus token
        and one mask query per draft token follows it. True is the anchor-first
        layout, where every query predicts the next token. Read from
        `dflash_config`, falling back to the top level of the HF config, where
        some checkpoints set it as DSpark does. False without a draft model config.

        Raises:
            ValueError: If `dflash_config` and the top level set different values.

        """
        draft_model_config = self.spec_config.draft_model_config
        if draft_model_config is None:
            return False
        hf_config = draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        nested = dflash_config.get("sample_from_anchor")
        top_level = getattr(hf_config, "sample_from_anchor", None)
        if (
            nested is not None
            and top_level is not None
            and bool(nested) != bool(top_level)
        ):
            raise ValueError(
                f"Conflicting sample_from_anchor: dflash_config has {nested!r} "
                f"but the top-level config has {top_level!r}."
            )
        return bool(nested if nested is not None else top_level)

    @property
    def num_query_tokens(self) -> int:
        """Query tokens per request in one draft forward pass."""
        return self.num_speculative_tokens + (0 if self.sample_from_anchor else 1)

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_query_tokens if self.num_scheduled_spec_tokens else 0

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int | None:
        # The draft pass has num_lookahead_tokens queries for every request.
        num_queries = self.num_lookahead_tokens
        return max_num_batched_tokens // num_queries if num_queries else None
