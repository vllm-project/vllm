# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only helpers for TileLang JIT warmup."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar, cast

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel

CompileKeyT = TypeVar("CompileKeyT")
# (kernel_args, launch_args), optionally followed by keywords and a result.
TileLangLaunchSpec: TypeAlias = (
    tuple[tuple[Any, ...], tuple[Any, ...]]
    | tuple[tuple[Any, ...], tuple[Any, ...], Mapping[str, Any] | None]
    | tuple[tuple[Any, ...], tuple[Any, ...], Mapping[str, Any] | None, Any]
)


@contextmanager
def _quiet_tilelang_warmup_logs() -> Iterator[None]:
    loggers = [
        logging.getLogger("tilelang.cache.kernel_cache"),
        logging.getLogger("tilelang.jit.kernel"),
    ]
    previous_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level in zip(loggers, previous_levels):
            logger.setLevel(level)


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

    def new_empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: Any = None,
    ) -> TileLangWarmupTensor:
        return TileLangWarmupTensor(dtype or self.dtype, shape)


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
    with _quiet_tilelang_warmup_logs():
        compiled = jit_impl.compile(*args, **kwargs)
    func = getattr(jit_impl, "func", None)
    parse_args = getattr(func, "parse_args", None)
    cache = getattr(jit_impl, "_kernel_cache", None)
    if not callable(parse_args) or not isinstance(cache, MutableMapping):
        return

    key, _ = parse_args(*args, **_tilelang_call_kwargs(kwargs))
    cache[key] = compiled


def _compile_tilelang_batch(
    jit_impl: Any,
    launches: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    parse_args = getattr(getattr(jit_impl, "func", None), "parse_args", None)
    cache = getattr(jit_impl, "_kernel_cache", None)
    if not callable(getattr(jit_impl, "par_compile", None)) or not (
        callable(parse_args) and isinstance(cache, MutableMapping)
    ):
        for args, kwargs in launches:
            compile_tilelang(jit_impl, *args, **kwargs)
        return

    missing: dict[Any, dict[str, Any]] = {}
    for args, kwargs in launches:
        call_kwargs = _tilelang_call_kwargs(kwargs)
        jit_impl.initialize_jit_mode(*args, **call_kwargs)
        key, _ = parse_args(*args, **call_kwargs)
        if key not in cache and key not in missing:
            missing[key] = dict(jit_impl.signature.bind(*args, **call_kwargs).arguments)

    if not missing:
        return
    if len(missing) == 1:
        compile_tilelang(jit_impl, **next(iter(missing.values())))
        return

    with _quiet_tilelang_warmup_logs():
        compiled = jit_impl.par_compile(
            list(missing.values()), num_workers=min(4, len(missing))
        )
    cache.update(zip(missing, compiled, strict=True))


class VllmTileLangJitKernel(VllmJitKernel[CompileKeyT], Generic[CompileKeyT]):
    """TileLang owner whose runtime launch specification is reused for warmup."""

    kernel: ClassVar[Any]
    _warming_key: CompileKeyT | None = None
    _warmup_launches: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] | None = None

    @abstractmethod
    def warmup_inputs(self, compile_key: CompileKeyT) -> dict[str, Any]:
        """Return runtime-shaped inputs that reproduce one compile key."""
        raise NotImplementedError

    def compile(self, compile_key: CompileKeyT) -> None:
        self._warming_key = compile_key
        try:
            cast(Callable[..., Any], self)(**self.warmup_inputs(compile_key))
        finally:
            self._warming_key = None

    def compile_many(self, compile_keys: Iterable[CompileKeyT]) -> None:
        launches: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self._warmup_launches = launches
        try:
            super().compile_many(dict.fromkeys(compile_keys))
        finally:
            self._warmup_launches = None

        groups: dict[int, tuple[Any, list[tuple[tuple[Any, ...], dict[str, Any]]]]] = {}
        for jit_impl, args, kwargs in launches:
            _, batch = groups.setdefault(id(jit_impl), (jit_impl, []))
            batch.append((args, kwargs))
        for jit_impl, batch in groups.values():
            _compile_tilelang_batch(jit_impl, batch)

    def launch(
        self,
        jit_impl: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        call_kwargs = dict(kwargs or {})
        if self._warming_key is not None:
            if self._warmup_launches is not None:
                self._warmup_launches.append((jit_impl, args, call_kwargs))
                return None
            return compile_tilelang(jit_impl, *args, **call_kwargs)
        return jit_impl(*args, **call_kwargs)


def kernel_launcher(
    call_fn: Callable[..., TileLangLaunchSpec],
) -> Callable[..., Any]:
    """Launch TileLang from declarative kernel and runtime argument tuples."""

    @wraps(call_fn)
    def wrapper(
        self: VllmTileLangJitKernel[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        launch_spec = call_fn(self, *args, **kwargs)
        kernel_args, launch_args = launch_spec[:2]
        if self._warming_key is not None and kernel_args not in (
            (),
            (self._warming_key,),
        ):
            raise RuntimeError(
                "TileLang warmup inputs produced a different compile key: "
                f"expected {self._warming_key!r}, got kernel arguments "
                f"{kernel_args!r}"
            )
        launch_kwargs = launch_spec[2] if len(launch_spec) > 2 else None
        output = launch_spec[3] if len(launch_spec) > 3 else None
        result = self.launch(self.kernel(*kernel_args), launch_args, launch_kwargs)
        return output if len(launch_spec) > 3 else result

    return wrapper
