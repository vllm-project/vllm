// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Instantiates the fused MoE int4 skinny GEMM kernel for N_in == 3 only; see
// skinny_gemms_int4.h.
#include "skinny_gemms_int4.cuh"
VLLM_SKINNY_INT4_MOE_INSTANTIATE(3)
