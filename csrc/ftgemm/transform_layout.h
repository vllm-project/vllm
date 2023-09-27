/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "int8_utils.cuh"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void invokeRowMajorToCOL32(int8_t *dst, const int8_t *src, const int m,
                           const int n, cudaStream_t stream);
void invokeCOL32ToRowMajor(int8_t *dst, const int8_t *src, const int m,
                           const int n, cudaStream_t stream);
void invokeRowMajorToAmpere(int8_t *dst, const int8_t *src, const int m,
                            const int n, cudaStream_t stream);
void invokeRowMajorToTuring(int8_t *dst, const int8_t *src, const int m,
                            const int n, cudaStream_t stream);