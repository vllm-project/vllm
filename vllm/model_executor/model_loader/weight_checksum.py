# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-parameter weight digests and reset, for verifying weight updates."""

import hashlib

import torch
import torch.nn as nn


def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            param.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, param in model.named_parameters()
    }


def zero_weights(model: nn.Module) -> None:
    for param in model.parameters():
        param.data.zero_()
