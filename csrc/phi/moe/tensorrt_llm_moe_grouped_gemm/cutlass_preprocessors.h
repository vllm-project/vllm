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
    W8_A16,
    W4_A16,
    W4_AFP8
};

constexpr int get_weight_quant_bits(QuantType quant_type)
{
    switch (quant_type)
    {
    case QuantType::W8_A16: return 8;
    case QuantType::W4_A16: return 4;
    case QuantType::W4_AFP8: return 4;
    default: TLLM_CHECK_WITH_INFO(false, "Invalid quant_type"); return -1;
    }
}

// Shapes here can be 2 or 3D. 2-D shapes are [num_rows, num_cols]
// 3-D shapes are [num_experts, num_rows, num_cols]
void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type, const int64_t arch_version);

void subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type);

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type);

void preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t const* row_major_quantized_weight,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave = false);

template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, WeightType const* input_weight_ptr,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave);

// This is exposed so that we can write tests that use the processed weights for CUTLASS but the unprocessed weight
// to implement a simple reference implementation.
template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight,
    ComputeType* scale_ptr, WeightType const* input_weight_ptr, std::vector<size_t> const& shape, QuantType quant_type,
    bool force_interleave);

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
