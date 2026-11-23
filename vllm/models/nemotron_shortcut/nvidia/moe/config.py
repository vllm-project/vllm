# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE config of Nemotron Shortcut's clamped ReLU² experts."""

from dataclasses import dataclass

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


@dataclass
class NemotronShortcutMoEConfig(FusedMoEConfig):
    """`FusedMoEConfig` plus the soft clamp of the experts' ReLU²."""

    # c in relu2(c * tanh(x / c)); None is the plain ReLU².
    activation_clamp_value: float | None = None


def activation_clamp(moe_config: FusedMoEConfig) -> float | None:
    """The clamp of a Nemotron Shortcut MoE config, refusing any other config."""
    if not isinstance(moe_config, NemotronShortcutMoEConfig):
        raise TypeError(
            "Nemotron Shortcut experts need a NemotronShortcutMoEConfig, got "
            f"{type(moe_config).__name__}"
        )
    return moe_config.activation_clamp_value
