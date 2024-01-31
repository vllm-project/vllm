/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstddef>
#include <stdint.h>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

enum class QuantType
{
    INT8_WEIGHT_ONLY,
    PACKED_INT4_WEIGHT_ONLY
};
int get_bits_in_quant_type(QuantType quant_type);

// Shapes here can be 2 or 3D. 2-D shapes are [num_rows, num_cols]
// 3-D shapes are [num_experts, num_rows, num_cols]
void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type, const int64_t arch_version);

void subbyte_transpose(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type);

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type);

void preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, const int8_t* row_major_quantized_weight,
    const std::vector<size_t>& shape, QuantType quant_type);

template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, const WeightType* input_weight_ptr,
    const std::vector<size_t>& shape, QuantType quant_type);

// This is exposed so that we can write tests that use the processed weights for CUTLASS but the unprocessed weight
// to implement a simple reference implementation.
template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight,
    ComputeType* scale_ptr, const WeightType* input_weight_ptr, const std::vector<size_t>& shape, QuantType quant_type);

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
