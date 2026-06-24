# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CuTe DSL compile helpers.

These wrap the fake-tensor / fake-stream construction used by each kernel's
``functools.cache``-decorated ``_compile_*`` helper, following the convention
in ``vllm/v1/attention/ops/deepseek_v4_ops``.
"""

import cutlass.cute as cute


def _fake(dtype, shape, stride, *, align):
    """Build a fake (compile-time only) CuTe tensor for ``cute.compile``."""
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align
    )


def _fake_stream():
    # The compiled executor sources its launch stream from the TVM-FFI
    # environment (set by ``options="--enable-tvm-ffi"``), so callers do not
    # pass a stream at runtime.
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
