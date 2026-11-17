// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Instantiates the int4 skinny GEMM kernels for N_in == 5 only; see
// skinny_gemms_int4.h.
#include "skinny_gemms_int4.cuh"
VLLM_SKINNY_INT4_INSTANTIATE(5)
