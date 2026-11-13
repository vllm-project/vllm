# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeculativeConfig

logger = init_logger(__name__)


def _get_nested_config_value(config: Any, section: str, name: str) -> Any:
    nested_config = getattr(config, section, None)
    if isinstance(nested_config, Mapping):
        return nested_config.get(name)
    return getattr(nested_config, name, None)


def _draft_carries_dspine_bridge(draft_model_config: "ModelConfig") -> bool:
    """Whether the draft checkpoint carries a DSpine code bridge."""
    hf_config = draft_model_config.hf_config
    architectures = list(draft_model_config.architectures or [])
    architectures.extend(getattr(hf_config, "architectures", None) or [])
    return "DSpineDraftModel" in architectures or bool(
        _get_nested_config_value(
            hf_config,
            "dflash_config",
            "selected_prefix_code_bridge",
        )
    )


class DSpineSpeculatorConfig(DFlash2SpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dspine.speculator.DSpineSpeculator"
    )
    # Draft-side watermarking doesn't cover candidate-head drafting yet.
    supports_watermarking = False

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        draft = spec.draft_model_config
        model_name = (draft.model or "").lower()
        return "dspine" in model_name or _draft_carries_dspine_bridge(draft)

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        super().adapt_draft_model_config(spec)
        draft = spec.draft_model_config
        if draft is None:
            return
        carries_dspine_bridge = _draft_carries_dspine_bridge(draft)
        if getattr(spec, "_requested_dspine", False) and not carries_dspine_bridge:
            raise ValueError(
                "Speculative method 'dspine' requires a DSpine draft "
                "(dflash_config.selected_prefix_code_bridge is missing)."
            )
        if not carries_dspine_bridge:
            return
        hf_config = draft.hf_config
        hf_config.architectures = ["DSpineDraftModel"]
        spec.update_arch_()
        if (
            _get_nested_config_value(
                hf_config,
                "dflash_config",
                "draft_block_causal",
            )
            and getattr(hf_config, "is_causal", None) is None
        ):
            hf_config.is_causal = True
        if spec.kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
            raise ValueError(
                f"DSpine does not support kv_cache_dtype={spec.kv_cache_dtype!r}."
            )

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        draft = spec.draft_model_config
        if draft is not None and _draft_carries_dspine_bridge(draft):
            hf_config = draft.hf_config
            dspine_block_size = getattr(hf_config, "block_size", None)
            if dspine_block_size is None:
                dspine_block_size = _get_nested_config_value(
                    hf_config,
                    "dflash_config",
                    "block_size",
                )
            if dspine_block_size is not None:
                if spec.num_speculative_tokens is None:
                    spec.num_speculative_tokens = int(dspine_block_size) - 1
                elif int(spec.num_speculative_tokens) + 1 != int(dspine_block_size):
                    logger.warning(
                        "DSpine checkpoint was trained with block_size=%d "
                        "(recommended num_speculative_tokens=%d), got "
                        "num_speculative_tokens=%d; acceptance length may differ.",
                        int(dspine_block_size),
                        int(dspine_block_size) - 1,
                        spec.num_speculative_tokens,
                    )
        super().resolve_num_speculative_tokens(spec)

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        if spec.draft_tensor_parallel_size != 1:
            raise ValueError(
                "DSpine speculative decoding supports tensor_parallel_size == 1 only."
            )
        super().apply_options(spec)
