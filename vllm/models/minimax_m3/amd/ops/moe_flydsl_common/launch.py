# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE launch helpers."""

import flydsl.compiler as flyc
import torch


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    """The flat byte view the kernels take their tensors as."""
    return t.view(torch.uint8).view(-1)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction (the shim aiter ships in ``ops/flydsl/kernels``)."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


_launches: dict[str, object] = {}


def _get(compile_fn, **kw):
    """One launch object per distinct kernel. The kernels depend on ``n_tokens``
    only through a few buckets, so launches built for different ``n_tokens`` are
    shared by kernel name (the compiled code hangs off the launch object)."""
    launch = compile_fn(**kw)
    return _launches.setdefault(launch.kernel_name, launch)
