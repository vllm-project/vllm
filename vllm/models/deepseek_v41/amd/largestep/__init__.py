# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 decode steps of 43 to 256 rows under `--monokernel` on MI300X and
MI325X at TP4 and TP8: the MoE block as a grouped GEMM over groups of 16 picks of
one expert, so each expert's weights are read once per 16 picks. The matrix
steps reuse the mono kernels' gfx942 FP4-to-FNUZ step (`mono.common.gfx942`).
"""
