# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PDL (griddepcontrol) calls of the Triton kernels, compiled out below SM90
(`use_pdl`): there, launches are stream ordered and need no wait."""

from vllm.triton_utils import tl, triton

from . import use_pdl

_PDL = tl.constexpr(use_pdl())


@triton.jit
def gdc_wait():
    if _PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if _PDL:
        tl.extra.cuda.gdc_launch_dependents()
