# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the FlyDSL internals the JIT monitor hooks."""

import pytest

pytestmark = pytest.mark.cpu_test

jit_function = pytest.importorskip("flydsl.compiler.jit_function")

_ASM = """  gpu.module @kernels {
    gpu.func @gdn_h_s4096(%arg0: memref<f32>) kernel {
  func.func @launch(%arg0: memref<f32>) {
"""


def test_kernel_name_inference_still_selects_gpu_func_symbols():
    """The monitor names the kernel, not the ``launch`` wrapper, from this."""
    assert jit_function._infer_kernel_names_from_asm(_ASM) == ["gdn_h_s4096"]
