# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor weight digests and reset, for verifying weight updates."""

import hashlib

import torch
import torch.nn as nn

# Derived rotary caches some models register as persistent; loading never
# restores them, so they are left out.
_DERIVED_BUFFERS = (
    "cos_cached",
    "sin_cached",
    "cos_sin_cache",
    "inv_freq",
    "freqs_cis",
)


def _checksum_targets(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in model.state_dict(keep_vars=True).items()
        if not any(pattern in name for pattern in _DERIVED_BUFFERS)
    }


def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            tensor.detach().cpu().reshape(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, tensor in _checksum_targets(model).items()
    }


def zero_weights(model: nn.Module) -> None:
    for tensor in _checksum_targets(model).values():
        tensor.data.zero_()
