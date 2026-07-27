# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-specific runtime adapters for checked-in generated kernels."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import TYPE_CHECKING

from vllm.kernels.helion_generated.manifests import GENERATED_KERNEL_MANIFESTS

if TYPE_CHECKING:
    import torch

_OP_MODULES = tuple(GENERATED_KERNEL_MANIFESTS)


def import_all_ops() -> tuple[ModuleType, ...]:
    return tuple(
        importlib.import_module(f"{__name__}.{module_name}")
        for module_name in _OP_MODULES
    )


def warm_up_all_ops(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    token_counts = tuple(token_counts)
    for module in import_all_ops():
        module.warmup(token_counts, device)
