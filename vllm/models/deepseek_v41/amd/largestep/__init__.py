# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 decode steps of 43 to 256 rows under `--monokernel` (MI300X at TP8,
MI325X at TP4): the MoE block as a grouped GEMM over groups of 16 picks of
one expert, so each expert's weights are read once per 16 picks. The matrix
steps reuse the mono kernels' gfx942 FP4-to-FNUZ step (`mono.common.gfx942`).
"""
