# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Call-site routing for eager Helion quant ops."""

from __future__ import annotations

import functools
import importlib
import sys
import types
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.deep_gemm import get_tma_aligned_size

logger = init_logger(__name__)
_is_compiling = torch.compiler.is_compiling

_PTG_OP_NAME = "per_token_group_fp8_quant"
_ROUTABLE_EAGER_OPS = frozenset({_PTG_OP_NAME})


@functools.cache
def _use_helion_kernels() -> bool:
    return envs.VLLM_USE_HELION_KERNELS


@functools.cache
def _helion_available(op_name: str) -> bool:
    try:
        importlib.import_module(f"vllm.kernels.helion.ops.{op_name}")
        from vllm.kernels.helion.register import _HOP_AVAILABLE

        ready = (not _HOP_AVAILABLE) and hasattr(torch.ops.vllm_helion, op_name)
    except Exception:
        ready = False
    if not ready:
        logger.warning_once(
            "VLLM_USE_HELION_KERNELS is set but Helion kernels are not available "
            "for '%s'; falling back to the native kernel. Install the `helion` "
            "package to enable them.",
            op_name,
        )
    return ready


@functools.cache
def _eager_kernel(op_name: str) -> Callable[..., Any]:
    from vllm.kernels.helion import get_kernel_by_name

    kernel = get_kernel_by_name(op_name)
    if kernel is None:
        raise RuntimeError(f"Helion kernel '{op_name}' is not registered")
    return kernel.eager_callable()


@functools.cache
def _checked_eager_kernel(op_name: str) -> Callable[..., Any] | None:
    if _helion_available(op_name):
        return _eager_kernel(op_name)
    return None


def use_helion_per_token_group_fp8_quant() -> bool:
    return _use_helion_kernels() and not _is_compiling()


def _ptg_dimensions(input: torch.Tensor) -> tuple[int, int] | None:
    if input.__class__ is not torch.Tensor:
        return None
    shape = input.shape
    if len(shape) != 2:
        return None
    return shape[0], shape[1]


def _ptg_output_s_stride(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    scale_transposed: bool,
    tma_aligned: bool,
) -> tuple[int, int]:
    groups_per_row = hidden_size // group_size
    if scale_transposed:
        tma_aligned_m = (
            get_tma_aligned_size(num_tokens, 4) if tma_aligned else num_tokens
        )
        return (1, tma_aligned_m)
    return (groups_per_row, 1)


def _ptg_fast_cache_key(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    scale_transposed: bool,
    tma_aligned: bool,
    dimensions: tuple[int, int] | None = None,
) -> tuple[Any, ...] | None:
    if dimensions is None:
        dimensions = _ptg_dimensions(input)
        if dimensions is None:
            return None
    num_tokens, hidden_size = dimensions
    if group_size <= 0 or hidden_size % group_size != 0:
        return None
    groups_per_row = hidden_size // group_size
    if input.stride() != (hidden_size, 1) or output_q.stride() != (hidden_size, 1):
        return None
    if output_s.shape != (num_tokens, groups_per_row):
        return None
    expected_s_stride = _ptg_output_s_stride(
        num_tokens, hidden_size, group_size, scale_transposed, tma_aligned
    )
    if output_s.stride() != expected_s_stride:
        return None
    if (input.data_ptr() | output_q.data_ptr() | output_s.data_ptr()) & 15:
        return None
    return (
        num_tokens,
        hidden_size,
        input.dtype,
        input.device,
        output_q.dtype,
        output_s.dtype,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        scale_transposed,
        tma_aligned,
    )


def _ptg_cache_key(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    scale_transposed: bool,
    tma_aligned: bool,
    dimensions: tuple[int, int] | None = None,
) -> tuple[Any, ...] | None:
    fast_key = _ptg_fast_cache_key(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        scale_transposed,
        tma_aligned,
        dimensions,
    )
    if fast_key is None:
        return None
    return (*fast_key[:6], output_s.stride(), *fast_key[6:])


def _ptg_fast_key_from_args(*args: object) -> tuple[Any, ...] | None:
    return _ptg_fast_cache_key(*args)  # type: ignore[arg-type]


try:
    from helion.runtime import FusedCallsiteRouter

    _PTG_ROUTER = FusedCallsiteRouter()
except Exception:
    _PTG_ROUTER = None

# Compatibility aliases used by the launch-overhead benchmark. The owning
# implementation lives in Helion's FusedCallsiteRouter.
_PTG_LAUNCH_CACHE: dict[Any, Any] = (
    {} if _PTG_ROUTER is None else _PTG_ROUTER.launch_cache
)
_PTG_FAST_LAUNCH_CACHE: dict[Any, Any] = (
    {} if _PTG_ROUTER is None else _PTG_ROUTER.fast_launch_cache
)
_PTG_CPP_LAUNCH_CACHE: dict[Any, Any] = (
    {} if _PTG_ROUTER is None else _PTG_ROUTER.cpp_launch_cache
)
_PTG_FAST_LAST: tuple[Any, ...] | None = None
_PTG_CPP_LAST: tuple[Any, ...] | None = None


def _remember_last_alias(
    recent: list[tuple[Any, ...]],
    entry: tuple[Any, ...] | None,
) -> None:
    if entry is None:
        recent.clear()
        return
    if entry not in recent:
        recent.insert(0, entry)


class _RoutingModule(types.ModuleType):
    def __getattribute__(self, name: str) -> object:
        if name in {"_PTG_FAST_LAST", "_PTG_CPP_LAST"}:
            router = types.ModuleType.__getattribute__(self, "_PTG_ROUTER")
            if router is not None:
                if name == "_PTG_FAST_LAST":
                    return router.fast_last
                return router.cpp_last
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        router = self.__dict__.get("_PTG_ROUTER")
        if router is None:
            return
        if name == "_PTG_FAST_LAST":
            router.fast_last = value
            _remember_last_alias(router.fast_recent, value)
        elif name == "_PTG_CPP_LAST":
            router.cpp_last = value
            _remember_last_alias(router.cpp_recent, value)


sys.modules[__name__].__class__ = _RoutingModule


def try_launch_per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    scale_transposed: bool,
    tma_aligned: bool,
    dimensions: tuple[int, int] | None = None,
) -> bool:
    """Fast Helion route for vLLM's 2D per-token-group FP8 quant call site."""
    if (
        _PTG_ROUTER is None
        or not use_helion_per_token_group_fp8_quant()
        or not output_q.is_contiguous()
    ):
        return False

    full_args = (
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        scale_transposed,
        tma_aligned,
    )
    dimensions = dimensions if dimensions is not None else _ptg_dimensions(input)
    if dimensions is None:
        return False

    fast_key = _ptg_fast_cache_key(*full_args, dimensions=dimensions)
    if _PTG_ROUTER.try_recent_cpp(full_args):
        return True

    cache_key = _ptg_cache_key(*full_args, dimensions=dimensions)
    kernel = _checked_eager_kernel(_PTG_OP_NAME)
    if kernel is None:
        return False

    return _PTG_ROUTER.try_launch(
        kernel,
        full_args,
        cache_key=cache_key,
        fast_key=fast_key,
        fast_key_fn=_ptg_fast_key_from_args,
        tensor_args=(input, output_q, output_s),
    )


def route_quant(op_name: str, fn: Callable[..., Any], *args: object) -> Any:
    if (
        op_name in _ROUTABLE_EAGER_OPS
        and _use_helion_kernels()
        and not _is_compiling()
        and (kernel := _checked_eager_kernel(op_name)) is not None
    ):
        return kernel(*args)
    return fn(*args)
