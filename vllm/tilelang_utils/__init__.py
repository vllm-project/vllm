# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if TYPE_CHECKING or current_platform.is_cuda():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]


def _ensure_tilelang_imported() -> None:
    """Bind the `tilelang` and `T` module globals, importing them if needed.

    On ROCm, this runs on the first kernel call instead of at import time.

    Raises:
        ImportError: If TileLang is not installed.
    """
    global T, tilelang

    if tilelang is not None:
        return
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang as tilelang_module
    import tilelang.language as tilelang_language

    tilelang = tilelang_module
    T = tilelang_language


@cache
def _get_pass_configs() -> dict[Any, Any]:
    _ensure_tilelang_imported()
    pass_configs: dict[Any, Any] = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if current_platform.is_cuda():
        pass_configs[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10
    return pass_configs


def tilelang_jit(kernel_function: Callable[..., Any]) -> Callable[..., Any]:
    """Apply `tilelang.jit`, deferring until first call on ROCm.

    ROCm defers JIT decoration so importing the caller's module does not
    require TileLang immediately. CUDA keeps the eager decoration behavior.

    The kernel body parsed by TileLang references `T` as an unqualified
    global, so on the deferred ROCm path this rebinds `T`/`tilelang` in the
    decorated function's own module globals once they become available.
    """
    if not current_platform.is_rocm():
        _ensure_tilelang_imported()
        return tilelang.jit(pass_configs=_get_pass_configs())(kernel_function)

    compiled_kernel: Callable[..., Any] | None = None

    @functools.wraps(kernel_function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled_kernel
        if compiled_kernel is None:
            _ensure_tilelang_imported()
            kernel_function.__globals__["tilelang"] = tilelang
            kernel_function.__globals__["T"] = T
            compiled_kernel = tilelang.jit(pass_configs=_get_pass_configs())(
                kernel_function
            )
        return compiled_kernel(*args, **kwargs)

    return wrapper
