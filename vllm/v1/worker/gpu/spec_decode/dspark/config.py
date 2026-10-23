# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

from vllm.config.speculative import (
    _QWEN3_OMNI_DSPARK_ARCHITECTURE,
    _validate_qwen3_omni_dspark,
)
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, SpeculativeConfig


class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    base_method = "dspark"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator"
    )
    supported_options = frozenset({"dspark_draft_topk"})
    uses_separate_workspace_lane = True
    # DSpark predicts acceptance with its own confidence head.
    adaptive_verification_uses_draft_logits = False
    supports_pcp = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        # e.g. deepseek-ai/dspark_qwen3_8b_block7
        return "dspark" in spec.draft_model_config.model.lower()

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        # DeepSeek DSpark can ship the weights inside the target checkpoint
        target = spec.target_model_config
        if target is None:
            raise ValueError("target_model_config must be present for dspark")
        if not spec.quantization:
            spec.quantization = target.quantization
        return target.model

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        draft = spec.draft_model_config
        target = spec.target_model_config
        if draft.speculator_config_class is None:
            # Not a self-contained DSpark draft: DeepSeek-V4(.1) DSpark reuses
            # the target's config, and its weights ship in the target checkpoint.
            is_v41 = target.hf_config.model_type == "deepseek_v41"
            draft.hf_config.model_type = "deepseek_v41" if is_v41 else "deepseek_v4"
            draft.hf_config.architectures = [
                "DSparkV41DraftModel" if is_v41 else "DSparkDraftModel"
            ]
            draft.quantization = target.quantization
            spec.update_arch_()
        spec.parallel_drafting = True

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        if spec.num_speculative_tokens is None:
            # DSpark's parallel width is independent of MTP stages.
            hf_config = spec.draft_model_config.hf_config
            block_size = getattr(hf_config, "block_size", None)
            if block_size is None:
                block_size = getattr(hf_config, "dspark_block_size", None)
            spec.num_speculative_tokens = block_size

    @classmethod
    def validate_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        draft = spec.draft_model_config
        hf_config = draft.hf_config
        dspark_draft_topk = spec.dspark_draft_topk
        if dspark_draft_topk is None:
            dspark_draft_topk = getattr(hf_config, "dspark_draft_topk", None)
        if dspark_draft_topk is not None:
            draft_vocab_size = (
                getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
            )
            if not 1 <= dspark_draft_topk <= draft_vocab_size:
                raise ValueError(
                    "dspark_draft_topk must be between 1 and the "
                    f"draft vocabulary size ({draft_vocab_size})"
                )
            if (
                "Qwen3DSparkModel" not in draft.architectures
                and _QWEN3_OMNI_DSPARK_ARCHITECTURE not in draft.architectures
            ):
                raise ValueError(
                    "dspark_draft_topk is only supported by "
                    "Qwen3DSparkModel and Qwen3OmniDSparkModel"
                )
            hf_config.dspark_draft_topk = dspark_draft_topk

        assert spec.target_model_config is not None
        _validate_qwen3_omni_dspark(
            spec.target_model_config, draft, spec.num_speculative_tokens
        )

    @property
    def num_lookahead_tokens(self) -> int:
        hf_config = self.spec_config.draft_model_config.hf_config
        if getattr(hf_config, "sample_from_anchor", True):
            # Anchor-sampling DSpark drafts a block of num_speculative_tokens
            # query tokens in which the anchor itself is the first prediction
            # position (no separate bonus query).
            return self.num_speculative_tokens
        return super().num_lookahead_tokens

    def hash_factors(self) -> list[Any]:
        # Adaptive verification changes the DSpark draft graph.
        return [self.spec_config.enable_adaptive_verification]

    def pcp_unsupported_reason(self, parallel_config: "ParallelConfig") -> str | None:
        dcp_size = parallel_config.decode_context_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        if (
            dcp_size not in (1, pcp_size)
            and self.spec_config.draft_model_config.use_mla
        ):
            return (
                "MRV2 PCP DSpark requires DCP=1 or DCP=PCP; got "
                f"DCP={dcp_size}, PCP={pcp_size}."
            )
        return super().pcp_unsupported_reason(parallel_config)
