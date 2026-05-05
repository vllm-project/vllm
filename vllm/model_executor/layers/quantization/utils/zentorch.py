# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized gate for routing CPU quantized linears through zentorch.

Every caller (TorchAO DA8W8 fast path, compressed-tensors
W8A8 dynamic-symmetric path, compressed-tensors W4A16 GPTQ path) MUST gate
through :func:`has_zentorch_op`.

Implementation notes
--------------------
``hasattr(torch.ops, "zentorch")`` is **not** a safe presence check:
``torch.ops.__getattr__`` lazily creates a (possibly empty) ``_OpNamespace``
for any attribute name and caches it, so the attribute is always present.
We therefore look up the named op(s) on the namespace directly.

The return value is intentionally **not** cached at module scope. Tests
monkeypatch ``current_platform.is_zen_cpu`` and (occasionally) the op
namespace; a module-level cache would freeze those decisions for the
remainder of the process.
"""
from __future__ import annotations

import torch

from vllm.platforms import current_platform

__all__ = ["has_zentorch_op"]


def has_zentorch_op(*op_names: str) -> bool:
    """Return ``True`` iff the running platform is an AMD Zen CPU **and**
    every named ``torch.ops.zentorch.*`` op is registered.

    Examples
    --------
    >>> has_zentorch_op("zentorch_dynamic_qlinear")
    >>> has_zentorch_op("zentorch_woq_repack_weight", "zentorch_woq_linear")
    """
    if not op_names:
        raise ValueError("has_zentorch_op requires at least one op name")
    if not current_platform.is_zen_cpu():
        return False
    ns = getattr(torch.ops, "zentorch", None)
    if ns is None:
        return False
    return all(hasattr(ns, name) for name in op_names)