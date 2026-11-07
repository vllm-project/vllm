# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Relaunch a Triton/Gluon JIT kernel without re-binding its arguments.

The MiniMax-M3 attention runs in the layer's eager cudagraph break, so every
kernel launch there pays ``JITFunction.run``'s host-side argument binding,
specialization and cache-key build for each of the 60 layers per
step. A launch whose tensors are the same persistent buffers (same pointers,
shapes and strides) and whose scalars are unchanged binds identically, so the
compiled kernel and its bound argument list can be replayed directly. The
constexpr `kwargs` are assumed fixed for a given kernel, grid and arguments.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import triton

_CALLS: dict[tuple, tuple[Callable[[], None], Any]] = {}


class _Ptr:
    """A device address standing in for a tensor in a replayed launch, so the
    cache never keeps the tensor (or its memory) alive. The cache key pins the
    address, shape and strides, so a replay sees the same buffer layout."""

    __slots__ = ("ptr",)

    def __init__(self, ptr: int):
        self.ptr = ptr

    def data_ptr(self) -> int:
        return self.ptr


# Bounded: keys include buffer addresses, so a workload whose per-step buffers
# move would otherwise grow the cache without limit.
_MAX_ENTRIES = 4096


def launch_cached(
    kernel: Any,
    grid: Sequence[int],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> None:
    """``kernel[grid](*args, **kwargs)``, replaying the resolved launch when
    the same buffers and scalars were launched before on this stream."""
    if torch.cuda.is_current_stream_capturing():
        kernel[tuple(grid)](*args, **kwargs)
        return
    stream = torch.cuda.current_stream().cuda_stream
    key = (id(kernel), tuple(grid), stream, *kwargs.values()) + tuple(
        (a.data_ptr(), a.dtype, a.shape, a.stride())
        if isinstance(a, torch.Tensor)
        else a
        for a in args
    )
    hit = _CALLS.get(key)
    if hit is not None:
        hit[0](hit[1])
        return
    compiled = kernel[tuple(grid)](*args, **kwargs)
    if len(_CALLS) >= _MAX_ENTRIES:
        _CALLS.clear()
    try:
        device = triton.runtime.driver.active.get_current_device()
        binder = kernel.device_caches[device][4]
        bound_args, _, _ = binder(*args, **kwargs)
        values = tuple(
            _Ptr(v.data_ptr()) if isinstance(v, torch.Tensor) else v
            for v in bound_args.values()
        )
        g = tuple(grid) + (1,) * (3 - len(grid))
        knobs = triton.knobs.runtime
        meta = compiled.launch_metadata(g, stream, *bound_args.values())
        run, fn, pm = compiled.run, compiled.function, compiled.packed_metadata
        enter, exit_ = knobs.launch_enter_hook, knobs.launch_exit_hook

        def replay(vals):
            run(g[0], g[1], g[2], stream, fn, pm, meta, enter, exit_, *vals)

        _CALLS[key] = (replay, values)
    except Exception:  # noqa: BLE001 - other Triton internals: stay on the JIT path
        # Do not cache a closure over ``args``: it would keep the tensors
        # (and their memory) alive. The next call simply takes the JIT path.
        pass
