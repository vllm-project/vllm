# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GPU architecture the mono kernels are compiled for.

FlyDSL compiles for the architecture in the ARCH environment variable, and
otherwise for the GPU it finds. The kernels were written for gfx950 (MI355X).
gfx942 (MI300X, MI325X) has 64 KB of LDS a CU instead of 160 KB, and lacks the
scaled f8f6f4 MFMA, the 16x16x32 bf16 MFMA, v_permlane16_swap /
v_permlane32_swap and the FP4 conversions. The helpers that differ test
``GFX942`` and take another code path there; the gfx950 code is unchanged.

Only kernel modules import this one: the runtime (``plan`` included) must import
without FlyDSL, as the CPU tests do.
"""

import os

from flydsl.runtime.device import get_rocm_arch


def target_arch() -> str:
    """The architecture FlyDSL compiles for, as FlyDSL's ROCm backend picks it."""
    arch = os.environ.get("ARCH") or get_rocm_arch()
    return arch.lower().split(":")[0]


GFX942 = target_arch().startswith("gfx942")
LDS_BYTES = (64 if GFX942 else 160) * 1024  # a CU's LDS: one CTA's whole
