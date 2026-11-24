# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PDL (griddepcontrol) calls of the Triton kernels, compiled out without
`use_pdl` (below SM90, and on XPU): there, launches are stream ordered and
need no wait."""

from vllm.triton_utils import tl, triton

from .platform import use_pdl

_PDL = tl.constexpr(use_pdl())


@triton.jit
def gdc_wait():
    if _PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if _PDL:
        tl.extra.cuda.gdc_launch_dependents()
