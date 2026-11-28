# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DNeoSpeculatorConfig(DFlashSpeculatorConfig):
    """DNeo (arXiv:2611.21925): a DFlash-family block drafter with a MoFA
    feature encoder, prefix-only attention and a sequential causal head."""

    base_method = "dneo"
    speculator_class = "vllm.v1.worker.gpu.spec_decode.dneo.speculator.DNeoSpeculator"
    # The causal head's correction is not part of the draft logits cache yet.
    supports_watermarking = False

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        return "dneo" in spec.draft_model_config.model.lower()

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        hf_config = spec.draft_model_config.hf_config
        # the anchor is the bonus token: block_size - 1 drafts
        max_drafts = int(hf_config.block_size) - (0 if cls._anchor_first(spec) else 1)
        if spec.num_speculative_tokens is None:
            spec.num_speculative_tokens = max_drafts
        elif spec.num_speculative_tokens > max_drafts:
            raise ValueError(
                f"DNeo drafts at most {max_drafts} tokens per block, got "
                f"num_speculative_tokens={spec.num_speculative_tokens}"
            )

    @staticmethod
    def _anchor_first(spec: "SpeculativeConfig") -> bool:
        dflash_config = getattr(spec.draft_model_config.hf_config, "dflash_config", {})
        return bool((dflash_config or {}).get("sample_from_anchor", False))

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        # Prefix-only attention needs the context from the KV cache, not ROCM_ATTN's.
        if current_platform.is_rocm():
            spec.attention_backend = AttentionBackendEnum.TRITON_ATTN
