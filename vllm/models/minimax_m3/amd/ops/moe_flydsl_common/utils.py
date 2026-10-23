# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""MiniMax-M3 FlyDSL MoE utils helpers."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm

# launch.compile_hints: preload kernel arguments into SGPRs (the first 56 bytes)
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _tensor_addr(t):
    """Tensor as a device address: a tensor kernel argument ends the kernarg preload."""
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _lds_atomic_add_i32(ptr, value):
    """Workgroup-scope LDS i32 atomic add; returns the old value."""
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )
