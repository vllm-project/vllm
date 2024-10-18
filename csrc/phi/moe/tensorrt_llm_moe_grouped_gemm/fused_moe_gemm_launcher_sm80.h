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

namespace tensorrt_llm::kernels::cutlass_kernels
{
template <typename ElementType_, typename CutlassWeightType_, int MaxTileM_, int TileN_, int TileK_, int Stages_,
    typename EpilogueTag>
void sm80_generic_fused_moe_gemm_kernelLauncher(ElementType_ const* A, CutlassWeightType_ const* B,
    ElementType_ const* biases, ElementType_* C, int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n,
    int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
}
