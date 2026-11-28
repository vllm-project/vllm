// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Instantiates the int8 skinny GEMM kernels for N_in == 2 only; see
// skinny_gemms_int8.h.
#include "skinny_gemms_int8.cuh"
VLLM_SKINNY_INT8_INSTANTIATE(2)
