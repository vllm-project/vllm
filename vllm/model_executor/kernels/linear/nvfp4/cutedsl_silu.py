# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe SiLU and multiply for the shared-weight NVFP4 decode path."""

from typing import Any

import torch
from flashinfer.utils import get_compute_capability, get_device_index

_COMPILED: dict[tuple, Any] = {}


def _compile(m, k, block, vector, enable_pdl):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Uint32
    from flashinfer.cute_dsl import fp4_common
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    @cute.jit
    def silu_float(value):
        denominator = Float32(1.0) + cute.math.exp(-value, fastmath=False)
        result = Float32(0.0)
        if value < Float32(-80.0):
            result = value / denominator
        else:
            result = value * cute.arch.rcp_approx(denominator)
        return result

    class TiledSilu:
        @cute.jit
        def __call__(self, gate_up, out, stream: cuda.CUstream):
            self.kernel(gate_up, out).launch(
                grid=(cute.ceil_div(k // 2, block * vector), m, 1),
                block=(block, 1, 1),
                stream=stream,
                use_pdl=enable_pdl,
            )

        @cute.kernel
        def kernel(self, gate_up: cute.Tensor, out: cute.Tensor):
            tid, _, _ = cute.arch.thread_idx()
            column_block, row, _ = cute.arch.block_idx()
            pair = (column_block * block + tid) * vector
            if cutlass.const_expr(enable_pdl):
                cute.arch.griddepcontrol_wait()
            if pair < k // 2:
                for index in cutlass.range_constexpr(vector):
                    packed_gate = Uint32(gate_up[row, pair + index])
                    packed_up = Uint32(gate_up[row, k // 2 + pair + index])
                    lo, hi = fp4_common.bfloat2_to_float2_scaled(
                        packed_gate, Float32(1.0)
                    )
                    activated = fp4_common.cvt_f32x2_to_bfloat2(
                        silu_float(lo), silu_float(hi)
                    )
                    out[row, pair + index] = fp4_common.bfloat2_mul(
                        activated, packed_up
                    )
            if cutlass.const_expr(enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

    operands = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (m, k), stride_order=(1, 0), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (m, k // 2), stride_order=(1, 0), assumed_align=16
        ),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return build_and_load_cute_dsl_kernel(
        "native_tiled_silu_sm12x",
        f"m{m}_k{k}_b{block}_v{vector}_p{int(enable_pdl)}",
        lambda: cute.compile(TiledSilu(), *operands, options="--enable-tvm-ffi"),
        extra_key_files=(__file__, fp4_common.__file__),
    )


def run(gate_up, *, block=64, vector=1, enable_pdl=True, out=None):
    if (
        not gate_up.is_cuda
        or gate_up.ndim != 2
        or gate_up.dtype != torch.bfloat16
        or not gate_up.is_contiguous()
        or gate_up.data_ptr() % 16
    ):
        raise ValueError("Tiled SiLU requires contiguous aligned BF16 CUDA input")
    if get_compute_capability(gate_up.device) not in ((12, 0), (12, 1)):
        raise ValueError("Tiled SiLU requires SM12x")
    m, width = gate_up.shape
    if not 1 <= m <= 16 or width < 16 or width % 16:
        raise ValueError("Tiled SiLU requires M in 1..16 and K divisible by 8")
    if block not in (32, 64, 128, 256) or vector not in (1, 2, 4):
        raise ValueError("Unsupported tiled SiLU launch configuration")
    k = width // 2
    if out is None:
        out = torch.empty((m, k), device=gate_up.device, dtype=gate_up.dtype)
    if (
        out.device != gate_up.device
        or out.dtype != gate_up.dtype
        or out.shape != (m, k)
        or not out.is_contiguous()
        or out.data_ptr() % 16
        or out.untyped_storage().data_ptr() == gate_up.untyped_storage().data_ptr()
    ):
        raise ValueError("Tiled SiLU requires an aligned, separate BF16 output")
    key = get_device_index(gate_up.device), m, k, block, vector, enable_pdl
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm tiled SiLU before CUDA graph capture")
        with torch.accelerator.device_index(gate_up.device.index):
            compiled = _compile(m, k, block, vector, enable_pdl)
        _COMPILED[key] = compiled
    compiled(gate_up.view(torch.int32), out.view(torch.int32))
    return out
