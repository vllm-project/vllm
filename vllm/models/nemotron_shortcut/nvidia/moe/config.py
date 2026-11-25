# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE config of Nemotron Shortcut's clamped ReLU² experts."""

from dataclasses import dataclass

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


@dataclass
class NemotronShortcutMoEConfig(FusedMoEConfig):
    """`FusedMoEConfig` plus the soft clamp of the experts' ReLU²."""

    # c in relu2(c * tanh(x / c)); None is the plain ReLU².
    activation_clamp_value: float | None = None


def ordered_mxfp8_finalize() -> bool:
    """Whether to order just the MXFP8 expert reduction on a hybrid model."""
    config = get_current_vllm_config_or_none()
    extra = config.additional_config if config is not None else None
    if not isinstance(extra, dict):
        return False
    ordered = extra.get("nemotron_mxfp8_ordered_finalize", False)
    if not isinstance(ordered, bool):
        raise ValueError("nemotron_mxfp8_ordered_finalize must be a boolean")
    return ordered


def activation_clamp(moe_config: FusedMoEConfig) -> float | None:
    """The clamp of a Nemotron Shortcut MoE config, refusing any other config."""
    if not isinstance(moe_config, NemotronShortcutMoEConfig):
        raise TypeError(
            "Nemotron Shortcut experts need a NemotronShortcutMoEConfig, got "
            f"{type(moe_config).__name__}"
        )
    return moe_config.activation_clamp_value
