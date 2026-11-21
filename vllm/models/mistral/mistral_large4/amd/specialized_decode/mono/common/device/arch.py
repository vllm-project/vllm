# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GPU architecture the mono kernels are compiled for.

FlyDSL compiles for the architecture in the ARCH environment variable, and
otherwise for the GPU it finds. The kernels were written for gfx950 (MI355X).
gfx942 (MI300X, MI325X) lacks several instructions they use: the scaled
f8f6f4 MFMA, the 16x16x32 bf16 MFMA, v_permlane16_swap / v_permlane32_swap and
the scaled FP8 / FP4 conversions. Its FP8 format is FNUZ (bias 8, no negative
zero, 0x80 is NaN, largest finite 240) instead of OCP e4m3 (448). The helpers
that differ test `GFX942` and take another code path there.
"""

from flydsl.compiler.backends import current_target
from flydsl.runtime.device import SMEM_CAPACITY_MAP


def target_arch() -> str:
    """The architecture FlyDSL compiles for."""
    return current_target().arch.lower().split(":")[0]


GFX942 = target_arch().startswith("gfx942")
# the LDS one workgroup can allocate
LDS_BYTES = SMEM_CAPACITY_MAP[target_arch()]
