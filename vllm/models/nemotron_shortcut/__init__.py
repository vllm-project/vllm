# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Shortcut model package.

Only an NVIDIA implementation exists: the GDP mixers and SDSA attention run on
FlashInfer kernels.
"""

from .nvidia.model import NemotronShortcutForCausalLM
from .nvidia.mtp import NemotronShortcutMTP
from .nvidia.vision import NemotronShortcutForConditionalGeneration

__all__ = [
    "NemotronShortcutForCausalLM",
    "NemotronShortcutForConditionalGeneration",
    "NemotronShortcutMTP",
]
