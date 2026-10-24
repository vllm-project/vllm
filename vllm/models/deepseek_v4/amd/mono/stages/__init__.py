# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stages the mono layer adapts from ROCm/ATOM: the mHC seams (``seam``),
the MoE (``moe``, ``router``) and the GEMV / LDS helpers they share (``gemv``);
the in-kernel all-reduce of both (``allreduce``)."""
