# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""No-op tracing/compile decorators for the single-node attention port.

ATOM's ``atom.utils.decorators`` wraps functions in ``torch.compile`` /
``graph_marker`` machinery (which depends back on ``atom.config`` and the
graph-marker subsystem). For a single-node port that machinery is unnecessary —
the wrapped functions run eagerly — so ``mark_trace`` / ``support_torch_compile``
here are transparent pass-throughs supporting both ``@deco`` and ``@deco(...)``
call forms, matching the original signatures.
"""

from typing import Callable, Optional


def mark_trace(
    func: Optional[Callable] = None,
    *,
    torch_compile: bool = True,
    prefix: Optional[str] = None,
):
    """Transparent stand-in for ATOM's ``mark_trace`` (eager pass-through)."""

    def _decorate(target: Callable) -> Callable:
        return target

    if func is not None:
        return _decorate(func)
    return _decorate


def support_torch_compile(
    cls=None,
    **_: object,
):
    """Transparent stand-in for ATOM's ``support_torch_compile`` class decorator."""

    def _decorate(target):
        return target

    if cls is not None:
        return _decorate(cls)
    return _decorate
