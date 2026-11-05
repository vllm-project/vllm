// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The FP8 candidate producer and recovery kernels (fp8_indexer.cuh) for 32 and
// 64 index heads, launched by fused_mqa_topk.cu. A translation unit of its own
// so that the -disable-new-nvvm-remat workaround of CMakeLists.txt (nvcc <
// 13.5) applies to these kernels only.

#include "fp8_indexer.cuh"

namespace fused_mqa_topk {

Fp8Kernel fp8_kernel(int64_t heads, bool recovery) {
  using namespace device::fp8_indexer;
  if (heads == 32) {
    return recovery ? fp8_recovery_kernel<Fp8Impl<32, true>>
                    : fp8_indexer_kernel<Fp8Impl<32, false>>;
  }
  return recovery ? fp8_recovery_kernel<Fp8Impl<64, true>>
                  : fp8_indexer_kernel<Fp8Impl<64, false>>;
}

}  // namespace fused_mqa_topk
