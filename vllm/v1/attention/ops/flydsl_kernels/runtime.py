# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal FlyDSL compilation and dispatch helper."""

from threading import Lock
from weakref import WeakKeyDictionary

import flydsl.compiler as flyc
import torch

_COMPILE_LOCK = Lock()
_COMPILED: WeakKeyDictionary = WeakKeyDictionary()


def run_compiled(executable, *args) -> None:
    """Compile on first use and dispatch the cached function thereafter."""
    with _COMPILE_LOCK:
        compiled = _COMPILED.get(executable)
        if compiled is None:
            compiled = flyc.compile(executable, *args)
            _COMPILED[executable] = compiled
    compiled(*args)


def get_compiled_static_tensors(executable, *args):
    """Return a compiled callable with tensor layouts fixed by ``args``."""
    with _COMPILE_LOCK:
        compiled = _COMPILED.get(executable)
        if compiled is None:
            compile_args = tuple(
                flyc.from_torch_tensor(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )
            compiled = flyc.compile(executable, *compile_args)
            _COMPILED[executable] = compiled
    return compiled
