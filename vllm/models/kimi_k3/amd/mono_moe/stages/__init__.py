# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono MoE launch's stages: ``route`` (top-k, sort), ``shared`` (the bf16
shared expert) and ``gemm1`` (AITER's a4w4 stage-1 tile, write-through out).
gemm2 is AITER's a4w4 stage-2 tile as its dispatcher emits it."""
