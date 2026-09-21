# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from fnmatch import filter as fnmatch_filter

import torch
from torch import nn


def save_sleep_parameters(
    model: nn.Module, patterns: list[str]
) -> dict[str, torch.Tensor]:
    """Back up selected device parameters before level-2 sleep."""
    if not patterns:
        return {}
    parameters = dict(model.named_parameters())
    names: set[str] = set()
    for pattern in patterns:
        matches = fnmatch_filter(parameters, pattern)
        if not matches:
            raise ValueError(f"No parameter matches sleep retention: {pattern}")
        names.update(matches)
    return {
        name: param.detach().to("cpu")
        for name, param in parameters.items()
        if name in names and not param.is_cpu
    }


@torch.no_grad()
def restore_sleep_parameters(model: nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """Restore parameters in place and release their CPU backups."""
    for name, value in saved.items():
        model.get_parameter(name).copy_(value)
    saved.clear()
