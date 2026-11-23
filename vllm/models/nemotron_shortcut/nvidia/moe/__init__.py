# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Shortcut's MoE: latent routed experts with a soft-clamped ReLU²."""

from vllm.models.nemotron_shortcut.nvidia.moe.config import NemotronShortcutMoEConfig
from vllm.models.nemotron_shortcut.nvidia.moe.experts import (
    NemotronShortcutRoutedExperts,
)
from vllm.models.nemotron_shortcut.nvidia.moe.layer import (
    NemotronShortcutMoE,
    NemotronShortcutMoEOutput,
)
from vllm.models.nemotron_shortcut.nvidia.moe.runner import NemotronShortcutMoERunner

__all__ = [
    "NemotronShortcutMoE",
    "NemotronShortcutMoEConfig",
    "NemotronShortcutMoEOutput",
    "NemotronShortcutMoERunner",
    "NemotronShortcutRoutedExperts",
]
