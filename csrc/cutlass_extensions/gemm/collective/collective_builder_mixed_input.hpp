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

/////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass_extensions/gemm/collective/collective_mma_array_mixed_input.hpp"

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ArchTag, class OpClass, class ElementA, class GmemLayoutA,
          int AlignmentA, class ElementB, class GmemLayoutB, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType, class Enable = void>
struct CollectiveBuilderMixedInput {
  static_assert(sizeof(ElementA) == 0,
                "Could not build a collective for given parameters.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl"
/////////////////////////////////////////////////////////////////////////////////////////////////
