# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gates zentorch CPU linear dispatch on platform/op availability."""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

__all__ = ["has_zentorch_op"]


def has_zentorch_op(op_names: list[str]) -> bool:
    """Return ``True`` when running on Zen CPU with all named ops registered."""
    if not op_names:
        raise ValueError("has_zentorch_op requires at least one op name")
    if not current_platform.is_zen_cpu():
        return False
    ns = getattr(torch.ops, "zentorch", None)
    if ns is None:
        return False
    return all(hasattr(ns, op_name) for op_name in op_names)
