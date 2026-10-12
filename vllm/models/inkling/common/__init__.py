# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling implementation shared by ROCm and XPU.

Mostly Triton kernels plus vLLM layers, so it is not tied to one vendor.
Two exceptions: the gfx950 Gluon kernels in ``inkling.amd`` are dispatched
from ``ops.fa4_rel_attention`` behind an architecture guard, and the fused
multimodal tower kernels are imported from ``inkling.nvidia.ops.mm_towers``.
"""
