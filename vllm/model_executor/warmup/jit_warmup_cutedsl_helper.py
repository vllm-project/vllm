# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

DEFAULT_CUTEDSL_COMPILE_OPTIONS = "--enable-tvm-ffi"


def cutedsl_fake_stream(*, use_tvm_ffi_env_stream: bool = True) -> Any:
    import cutlass.cute as cute

    return cute.runtime.make_fake_stream(
        use_tvm_ffi_env_stream=use_tvm_ffi_env_stream
    )


def compile_cutedsl(
    entry: Callable[..., Any],
    *args: Any,
    options: str = DEFAULT_CUTEDSL_COMPILE_OPTIONS,
    use_tvm_ffi_env_stream: bool = True,
) -> Any:
    import cutlass.cute as cute

    return cute.compile(
        entry,
        *args,
        cutedsl_fake_stream(use_tvm_ffi_env_stream=use_tvm_ffi_env_stream),
        options=options,
    )
