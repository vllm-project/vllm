# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device intrinsics of the Inkling2 Triton kernels: the CUDA PTX the
kernels were tuned with, and portable Triton on every other backend.

`IS_CUDA` is a module-level `tl.constexpr`, so only one branch of each helper
is compiled: on CUDA the helpers emit exactly the inline PTX the kernels used
before; elsewhere (e.g. XPU) the portable branch, whose numerics differ from
CUDA's approximations:

- `rsqrt`: `div_rn(1, sqrt_rn(x))`, at most 1 ulp from the exact value
  (CUDA: non-FTZ `rsqrt.approx.f32`, up to 2 ulp).
- `ex2` / `lg2`: `tl.exp2` / `tl.log2` (CUDA: FTZ `ex2` / `lg2.approx`).
- `sigmoid`: `1 / (1 + exp2(-x * log2(e)))`, as `_rel_attention_kernel`
  computes it on every platform (CUDA: non-FTZ `ex2` / `rcp.approx`).
- `sigmoid_ftz`: the same expression (CUDA: the FTZ variant of the experts'
  SiTU); subnormals are kept.
- `tanh_approx`: libdevice `tanh` (CUDA: `tanh.approx.f32`, about 2^-11
  relative error, so a softcapped value may round to the neighbouring bf16).
- `keep_separate`: identity. On CUDA an opaque move keeps a K group's
  partial sum from being folded into the next MMA's accumulator. Only a
  group of a single 128-wide tile (K % 1024 in [1, 128]) can be folded: a
  rounding-order change of the K-group sum, not hit by Inkling 2's K.
- `prefetch_l2`: no-op (CUDA: `prefetch.global.L2`).

PDL (`gdc_wait` / `gdc_launch_dependents`) is in `pdl.py`.
"""

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton

IS_CUDA = tl.constexpr(current_platform.is_cuda())

LOG2E = tl.constexpr(1.4426950408889634)
EX2_ASM = tl.constexpr("ex2.approx.ftz.f32 $0, $1;")
LG2_ASM = tl.constexpr("lg2.approx.ftz.f32 $0, $1;")


@triton.jit
def rsqrt(x):
    if IS_CUDA:
        # Non-FTZ rsqrt.approx, as emitted by CuTe's cute.math.rsqrt.
        y = tl.inline_asm_elementwise(
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = tl.div_rn(1.0, tl.sqrt_rn(x))
    return y


@triton.jit
def ex2(x):
    if IS_CUDA:
        y = tl.inline_asm_elementwise(
            EX2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
        )
    else:
        y = tl.exp2(x)
    return y


@triton.jit
def lg2(x):
    if IS_CUDA:
        y = tl.inline_asm_elementwise(
            LG2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
        )
    else:
        y = tl.log2(x)
    return y


@triton.jit
def sigmoid(x):
    if IS_CUDA:
        exp = tl.inline_asm_elementwise(
            "ex2.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[-x * 1.4426950408889634],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        y = tl.inline_asm_elementwise(
            "rcp.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[exp + 1.0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = 1.0 / (1.0 + tl.exp2(-x * LOG2E))
    return y


@triton.jit
def sigmoid_ftz(x):
    if IS_CUDA:
        y = tl.inline_asm_elementwise(
            "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
            "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
            "rcp.approx.ftz.f32 $0, t; }",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = 1.0 / (1.0 + tl.exp2(-x * LOG2E))
    return y


@triton.jit
def tanh_approx(x):
    if IS_CUDA:
        y = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = tldevice.tanh(x)
    return y


@triton.jit
def keep_separate(x):
    """`x` (fp32), as a value the compiler cannot fold into an MMA."""
    if IS_CUDA:
        y = tl.inline_asm_elementwise(
            "mov.b32 $0, $1;",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = x
    return y


@triton.jit
def prefetch_l2(ptrs, mask):
    """Pull the lines of `ptrs` (where `mask`) into L2."""
    if IS_CUDA:
        tl.inline_asm_elementwise(
            "{ .reg .pred p; setp.ne.b32 p, $2, 0; "
            "@p prefetch.global.L2 [$1]; mov.b32 $0, 0; }",
            "=r,l,r",
            [ptrs, mask.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
