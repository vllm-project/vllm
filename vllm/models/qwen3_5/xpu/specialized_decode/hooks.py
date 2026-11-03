# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build specialized MoE executors after quantization and before graph capture."""

import torch


@torch.no_grad()
def initialize_specialized_decode(model: torch.nn.Module) -> None:
    from ..model import Qwen3_5SparseMoeBlock
    from .moe import FusedMoeDecode

    for module in model.modules():
        if isinstance(module, Qwen3_5SparseMoeBlock):
            module._decode = FusedMoeDecode(module)
