# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only helpers for TileLang JIT warmup."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TileLangWarmupTensor:
    """Minimal tensor-like object accepted by TileLang compile().

    TileLang builds its cache key and TIR from tensor dtype, shape and stride.
    This object deliberately has no storage, so compile-only warmup does not
    allocate GPU memory and does not launch the kernel.
    """

    dtype: torch.dtype
    shape: tuple[int, ...] = (1,)
    strides: tuple[int, ...] | None = field(default=None)

    def stride(self) -> tuple[int, ...]:
        if self.strides is not None:
            return self.strides

        strides: list[int] = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        return tuple(reversed(strides))


def make_tilelang_warmup_tensor(
    dtype: torch.dtype,
    *shape: int,
    strides: tuple[int, ...] | None = None,
) -> TileLangWarmupTensor:
    return TileLangWarmupTensor(dtype=dtype, shape=tuple(shape), strides=strides)


def _tilelang_call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    call_kwargs = dict(kwargs)
    tune_params = call_kwargs.pop("__tune_params", {})
    if isinstance(tune_params, dict):
        call_kwargs.update(tune_params)
    return call_kwargs


def compile_tilelang(jit_impl: Any, *args: Any, **kwargs: Any) -> None:
    """Compile one TileLang specialization and populate its call cache.

    TileLang's ``compile()`` materializes the kernel without launching it.
    We also store the compiled kernel in ``_kernel_cache`` using the same
    parsed key as ``__call__`` so runtime does not report a cache miss for an
    already materialized specialization.
    """

    compiled = jit_impl.compile(*args, **kwargs)
    func = getattr(jit_impl, "func", None)
    parse_args = getattr(func, "parse_args", None)
    cache = getattr(jit_impl, "_kernel_cache", None)
    if not callable(parse_args) or not isinstance(cache, MutableMapping):
        return

    key, _ = parse_args(*args, **_tilelang_call_kwargs(kwargs))
    cache[key] = compiled
