# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Triton before 3.9 converts fp32 to e4m3 directly only from SM90. On SM89 it
# truncates to fp16 first (`cvt.rz.f16.f32`) and rounds that to nearest even,
# so an input just past an e4m3 tie lands on the tie and goes the wrong way.
_ROUND_TO_ODD = tl.constexpr(
    current_platform.is_cuda()
    and not current_platform.has_device_capability(90)
    and version.parse(triton.__version__) < version.parse("3.9.0")
)


@triton.jit
def to_fp8(x, dtype):
    """Cast `x` to the fp8 type `dtype`, rounding to nearest even."""
    if _ROUND_TO_ODD and dtype == tl.float8e4nv and x.dtype == tl.float32:
        # Round to odd in fp16: the sticky bit keeps what the truncation
        # dropped, and fp16 has enough bits to make the e4m3 rounding exact.
        truncated = x.to(tl.float16, fp_downcast_rounding="rtz")
        inexact = truncated.to(tl.float32) != x
        bits = truncated.to(tl.int16, bitcast=True) | inexact.to(tl.int16)
        return bits.to(tl.float16, bitcast=True).to(dtype)
    return x.to(dtype)
