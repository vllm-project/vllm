/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu
 * which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/619709fc33bd5dc268f19d6a741fe7ed51c0f8f5/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
 *
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "core/registration.h"

#include <cstdlib>
#include <mutex>

namespace {

inline int getSMVersion() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}

inline bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (getSMVersion() >= 90) {
      char const* env = std::getenv("TRTLLM_ENABLE_PDL");
      enablePDL = env && env[0] == '1' && env[1] == '\0';
    }
  });
  return enablePDL;
}

}  // namespace

using bf16_t = __nv_bfloat16;

__device__ void hmma_16_8_16_f32acc_bf16ab(float (&d_reg)[4],
                                           const bf16_t (&a_reg)[8],
                                           const bf16_t (&b_reg)[4],
                                           float const (&c_reg)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a0 = *reinterpret_cast<uint32_t const*>(a_reg + 0);
  uint32_t a1 = *reinterpret_cast<uint32_t const*>(a_reg + 2);
  uint32_t a2 = *reinterpret_cast<uint32_t const*>(a_reg + 4);
  uint32_t a3 = *reinterpret_cast<uint32_t const*>(a_reg + 6);
  uint32_t b0 = *reinterpret_cast<uint32_t const*>(b_reg + 0);
  uint32_t b1 = *reinterpret_cast<uint32_t const*>(b_reg + 2);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d_reg[0]), "=f"(d_reg[1]), "=f"(d_reg[2]), "=f"(d_reg[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d_reg[0]),
        "f"(d_reg[1]), "f"(d_reg[2]), "f"(d_reg[3]));
#endif
}

extern "C" {
__device__ uint32_t __nvvm_get_smem_pointer(void*);
}

__device__ void ldgsts_128(void const* gPtr, void* sPtr, uint32_t pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (pred) {
    uint32_t smemPtrAsUint32 = __nvvm_get_smem_pointer(sPtr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                     smemPtrAsUint32),
                 "l"(gPtr), "n"(16));
  }
#endif
}

__device__ void ldsm_x4(void* smem_ptr, uint32_t* reg_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
      : "r"(__nvvm_get_smem_pointer(smem_ptr)));
#endif
}

template <class Type>
__device__ int apply_swizzle_343_on_elem_row_col(int row_idx_, int col_idx_) {
  uint32_t row_idx = *reinterpret_cast<uint32_t*>(&row_idx_);
  uint32_t col_idx = *reinterpret_cast<uint32_t*>(&col_idx_);
  row_idx = row_idx % 8;
  row_idx = row_idx * (16 / sizeof(Type));
  col_idx = col_idx ^ row_idx;
  return *reinterpret_cast<int*>(&col_idx);
}

__device__ void initialize_barrier(
    uint64_t* smem_barrier,  // 64 bits user-manged barrier in smem
    int thread_count =
        1)  // Thread count expected to arrive/wait on this barrier
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
               "r"(thread_count));
#endif
}

// Barrier wait
__device__ void wait_barrier(
    uint64_t* smem_barrier,  // 64 bits user-manged barrier in smem
    int phase_bit)           // Current phase bit the barrier waiting to flip
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra DONE;\n"
      "bra                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
#endif
}

__device__ bool try_wait_barrier(uint64_t* smem_ptr, int phase_bit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t wait_complete;
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(wait_complete)
      : "r"(smem_int_ptr), "r"(phase_bit));
  return static_cast<bool>(wait_complete);
#endif
  return false;
}

// Barrier arrive
__device__ void arrive_barrier(
    uint64_t* smem_barrier)  // 64 bits user-manged barrier in smem
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile(
      "{\n"
      ".reg .b64 state; \n"
      "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
      "}\n" ::"r"(smem_int_ptr));
#endif
}

__device__ void ldgsts_arrive(uint64_t* smem_barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"
               :
               : "r"(smem_int_ptr));
#endif
}

template <int gemm_k, int tile_m, int tile_k, int stage_cnt>
struct GmemLoaderA {
  static constexpr int elem_bytes = 2;
  static constexpr int vec_bytes = 16;
  static constexpr int vec_elems = vec_bytes / elem_bytes;
  static constexpr int thread_cnt = 64;
  static_assert((tile_m * tile_k) % (vec_elems * thread_cnt) == 0);
  static constexpr int a_inst_cnt_per_iter =
      (tile_m * tile_k) / (vec_elems * thread_cnt);
  static_assert(gemm_k % tile_k == 0);
  static constexpr int k_iter_cnt = gemm_k / tile_k;

  // Extra params to keep the order of k reduction...
  static constexpr int mma_warp_cnt = 4;
  static constexpr int per_mma_warp_k = tile_k / mma_warp_cnt;
  static constexpr int k_each_chunk = gemm_k / mma_warp_cnt;

 private:
  __device__ int k_project(int tile_k_idx) {
    return (tile_k_idx / per_mma_warp_k * k_each_chunk) +
           (tile_k_idx % per_mma_warp_k);
  }

 public:
  __device__ GmemLoaderA(bf16_t const* gmem_a_local_, bf16_t* smem_a_,
                         uint64_t* smem_barrier_)
      : gmem_a(gmem_a_local_),
        smem_a(smem_a_),
        smem_barrier(smem_barrier_),
        local_tid(threadIdx.x % thread_cnt) {}

  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // swizzle, that's what we want.
  #pragma unroll
    for (int i = 0; i < a_inst_cnt_per_iter; i++) {
      int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
      int m_idx = linear_idx / tile_k;
      int k_idx = linear_idx % tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(m_idx, k_idx);
      a_smem_offsets[i] = m_idx * tile_k + k_idx;
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      if (need_wait) {
        wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
      }
      int next_stage_idx = stage_idx + 1;
      int next_phase_bit =
          next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
      if (loop_idx != k_iter_cnt - 1) {
        need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2,
                                      next_phase_bit);
      }

  #pragma unroll
      for (int i = 0; i < a_inst_cnt_per_iter; i++) {
        int smem_offset = a_smem_offsets[i];
        bf16_t* smem_ptr_this_iter =
            smem_a + stage_idx * tile_m * tile_k + smem_offset;
        int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
        int m_idx = linear_idx / tile_k;
        int k_idx = linear_idx % tile_k;
        int gmem_offset = m_idx * gemm_k + k_project(k_idx);
        bf16_t const* gmem_ptr_this_iter = gmem_a + gmem_offset;
        ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, true);
      }
      ldgsts_arrive(smem_barrier + stage_idx * 2);

      stage_idx = next_stage_idx;
      phase_bit = next_phase_bit;
      gmem_a += per_mma_warp_k;
    }
#endif
  }

  bf16_t const* gmem_a;
  bf16_t* smem_a;
  uint64_t* smem_barrier;
  int local_tid;
  int stage_idx = 0;
  int phase_bit = 1;
  bool need_wait = true;

  // per smem_stage, store with swizzle information
  int a_smem_offsets[a_inst_cnt_per_iter];
};

template <int gemm_k, int tile_n, int tile_k, int stage_cnt>
struct GmemLoaderB {
  static constexpr int elem_bytes = 2;
  static constexpr int vec_bytes = 16;
  static constexpr int vec_elems = vec_bytes / elem_bytes;
  static constexpr int thread_cnt = 64;
  static_assert((tile_n * tile_k) % (vec_elems * thread_cnt) == 0);
  static constexpr int b_inst_cnt_per_iter =
      (tile_n * tile_k) / (vec_elems * thread_cnt);
  static_assert(gemm_k % tile_k == 0);
  static constexpr int k_iter_cnt = gemm_k / tile_k;

  // Extra params to keep the order of k reduction...
  static constexpr int mma_warp_cnt = 4;
  static constexpr int per_mma_warp_k = tile_k / mma_warp_cnt;
  static constexpr int k_each_chunk = gemm_k / mma_warp_cnt;

 private:
  __device__ int k_project(int tile_k_idx) {
    return (tile_k_idx / per_mma_warp_k * k_each_chunk) +
           (tile_k_idx % per_mma_warp_k);
  }

 public:
  __device__ GmemLoaderB(bf16_t const* gmem_b_local_, bf16_t* smem_b_,
                         uint64_t* smem_barrier_, int gemm_n_)
      : gmem_b(gmem_b_local_),
        smem_b(smem_b_),
        smem_barrier(smem_barrier_),
        gemm_n(gemm_n_),
        local_tid(threadIdx.x % thread_cnt) {}

  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // swizzle, that's what we want.
  #pragma unroll
    for (int i = 0; i < b_inst_cnt_per_iter; i++) {
      int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
      int n_idx = linear_idx / tile_k;
      int k_idx = linear_idx % tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(n_idx, k_idx);
      b_smem_offsets[i] = n_idx * tile_k + k_idx;
      preds[i] = n_idx < gemm_n;
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;");
  #pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      if (need_wait) {
        wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
      }
      int next_stage_idx = stage_idx + 1;
      int next_phase_bit =
          next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
      if (loop_idx != k_iter_cnt - 1) {
        need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2,
                                      next_phase_bit);
      }
  #pragma unroll
      for (int i = 0; i < b_inst_cnt_per_iter; i++) {
        int smem_offset = b_smem_offsets[i];
        bf16_t* smem_ptr_this_iter =
            smem_b + stage_idx * tile_n * tile_k + smem_offset;
        int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
        int n_idx = linear_idx / tile_k;
        int k_idx = linear_idx % tile_k;
        int gmem_offset = n_idx * gemm_k + k_project(k_idx);
        bf16_t const* gmem_ptr_this_iter = gmem_b + gmem_offset;
        ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, preds[i]);
      }
      ldgsts_arrive(smem_barrier + stage_idx * 2);

      stage_idx = next_stage_idx;
      phase_bit = next_phase_bit;
      gmem_b += per_mma_warp_k;
    }
#endif
  }

  bf16_t const* gmem_b;
  bf16_t* smem_b;
  uint64_t* smem_barrier;
  int gemm_n;
  int local_tid;
  int stage_idx = 0;
  int phase_bit = 1;
  bool need_wait = true;

  // per smem_stage, store with swizzle information
  int b_smem_offsets[b_inst_cnt_per_iter];
  uint32_t preds[b_inst_cnt_per_iter];
};

template <int gemm_m, int gemm_k, int tile_m, int tile_n, int tile_k,
          int stage_cnt>
struct MmaComputer {
  static constexpr int elem_bytes = 2;
  static constexpr int thread_cnt = 128;
  static_assert(gemm_k % tile_k == 0);
  static_assert(tile_k % (thread_cnt / 32) == 0);
  static constexpr int per_warp_tile_k = tile_k / (thread_cnt / 32);
  static constexpr int k_iter_cnt = gemm_k / tile_k;
  static constexpr int k_phase_cnt = per_warp_tile_k / 16;
  static constexpr int m_iter_cnt = (tile_m + 15) / 16;
  static constexpr int n_iter_cnt =
      (tile_n + 7) /
      8;  // Possible to have non-1 n_iter_cnt for ab_swap m16 case.
  static_assert(m_iter_cnt == 1);
  static_assert(n_iter_cnt == 1 || n_iter_cnt == 2);

  __device__ MmaComputer(bf16_t* gmem_c_local_, bf16_t* smem_a_,
                         bf16_t* smem_b_, uint64_t* smem_barrier_,
                         int warp_idx_, int gemm_n_)
      : gmem_c(gmem_c_local_),
        smem_a(smem_a_),
        smem_b(smem_b_),
        smem_barrier(smem_barrier_),
        warp_idx(warp_idx_ - (thread_cnt / 32)),
        gemm_n(gemm_n_) {}

 private:
  __device__ constexpr int internal_b_atom_func(int tid) {
    if constexpr (tile_n < 8) {
      return (tid % tile_n) + ((tid % 8) / tile_n * 0) + tid / 8 * 8 * tile_n;
    } else {
      return (tid % 8) + ((tid % 32) / 8 * (tile_n * 8));
    }
  }

 public:
  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #pragma unroll
    for (int i = 0; i < k_phase_cnt; i++) {
      int linear_idx = (lane_idx % 16) + (lane_idx / 16) * 128 + i * 256;
      int m_idx = linear_idx % tile_m;
      int k_idx = linear_idx / tile_m + warp_k_offset_in_tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(m_idx, k_idx);
      a_smem_offsets[0][i] = m_idx * tile_k + k_idx;
    }
  #pragma unroll
    for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
  #pragma unroll
      for (int i = 0; i < k_phase_cnt; i += 2) {  // Special i+=2 for B.
        int linear_idx =
            internal_b_atom_func(lane_idx) + i * tile_n * 16 + n_iter_idx * 8;
        int n_idx = linear_idx % tile_n;
        int k_idx = linear_idx / tile_n + warp_k_offset_in_tile_k;
        k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(n_idx, k_idx);
        b_smem_offsets[n_iter_idx][i] = n_idx * tile_k + k_idx;
      }
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      wait_barrier(smem_barrier + 0 + stage_idx * 2, phase_bit);

  #pragma unroll
      for (int i = 0; i < k_phase_cnt; i++) {
        int smem_offset = a_smem_offsets[0][i];
        bf16_t* smem_ptr_this_iter =
            smem_a + stage_idx * tile_m * tile_k + smem_offset;
        ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(a_reg[0][i]));
      }

  #pragma unroll
      for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
  #pragma unroll
        for (int i = 0; i < k_phase_cnt; i += 2) {
          int smem_offset = b_smem_offsets[n_iter_idx][i];
          bf16_t* smem_ptr_this_iter =
              smem_b + stage_idx * tile_n * tile_k + smem_offset;
          ldsm_x4(smem_ptr_this_iter,
                  reinterpret_cast<uint32_t*>(b_reg[n_iter_idx][i]));
        }
      }

  #pragma unroll
      for (int k_iter_idx = 0; k_iter_idx < k_phase_cnt; k_iter_idx++) {
  #pragma unroll
        for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
          hmma_16_8_16_f32acc_bf16ab(
              acc_reg[0][n_iter_idx], a_reg[0][k_iter_idx],
              b_reg[n_iter_idx][k_iter_idx], acc_reg[0][n_iter_idx]);
        }
      }
      ::arrive_barrier(smem_barrier + 1 + stage_idx * 2);
      stage_idx += 1;
      phase_bit = stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      stage_idx = stage_idx == stage_cnt ? 0 : stage_idx;
    }
#endif
  }

  __device__ void epi() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));
    // reorganize the acc_reg
    constexpr int thread_m = 2;
    constexpr int thread_n = 2 * n_iter_cnt;
    constexpr int cta_mma_n = n_iter_cnt * 8;
    float acc_reg_reorg[thread_m][thread_n];

    for (int i = 0; i < thread_m; i++) {
      for (int j = 0; j < thread_n; j++) {
        acc_reg_reorg[i][j] = acc_reg[0][j / 2][(j % 2) + (i * 2)];
      }
    }

    // 4 x cosize(smem_c_layout)
    float* smem_c = reinterpret_cast<float*>(smem_a);
    // coord -> index
    auto smem_c_index_func = [&](int m_idx, int n_idx) {
      int group_rows = 32 / cta_mma_n;
      int group_cnt = 2;
      return (m_idx % group_rows * cta_mma_n) +
             (m_idx / group_rows * (32 + group_cnt)) + n_idx;
    };
    constexpr int cosize_smem_c = ((tile_m * cta_mma_n) / 32) * (32 + 2);

  // This should be optimized to STS.64 but can not be STS.128 due to the bank
  // index.
  #pragma unroll
    for (int m_idx_thread = 0; m_idx_thread < thread_m; m_idx_thread++) {
  #pragma unroll
      for (int n_idx_thread = 0; n_idx_thread < thread_n; n_idx_thread++) {
        int m_idx = (lane_idx / 4) + m_idx_thread * 8;
        int n_idx =
            ((lane_idx % 4) * 2) + (n_idx_thread % 2) + (n_idx_thread / 2) * 8;
        smem_c[cosize_smem_c * warp_idx + smem_c_index_func(m_idx, n_idx)] =
            acc_reg_reorg[m_idx_thread][n_idx_thread];
      }
    }
    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));

    if (warp_idx == 0) {
      constexpr int final_acc_reg_cnt = (tile_m * tile_n + 31) / 32;
      float acc_final[final_acc_reg_cnt]{};

  #pragma unroll
      for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++) {
        int linear_idx = reg_idx * 32 + lane_idx;
        int m_idx = linear_idx % tile_m;
        int n_idx = linear_idx / tile_m;
        acc_final[reg_idx] +=
            smem_c[smem_c_index_func(m_idx, n_idx) + 0 * cosize_smem_c] +
            smem_c[smem_c_index_func(m_idx, n_idx) + 1 * cosize_smem_c] +
            smem_c[smem_c_index_func(m_idx, n_idx) + 2 * cosize_smem_c] +
            smem_c[smem_c_index_func(m_idx, n_idx) + 3 * cosize_smem_c];
      }

  #pragma unroll
      for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++) {
        int linear_idx = reg_idx * 32 + lane_idx;
        int m_idx = linear_idx % tile_m;
        int n_idx = linear_idx / tile_m;
        if (m_idx < tile_m && n_idx < gemm_n) {
          gmem_c[n_idx * gemm_m + m_idx] = acc_final[reg_idx];
        }
      }
    }
#endif
  }

  bf16_t* gmem_c;
  bf16_t* smem_a;
  bf16_t* smem_b;
  uint64_t* smem_barrier;
  int warp_idx;
  int gemm_n;
  int stage_idx = 0;
  int phase_bit = 0;
  int lane_idx = threadIdx.x % 32;
  int warp_k_offset_in_tile_k = warp_idx * per_warp_tile_k;

  int a_smem_offsets[m_iter_cnt][k_phase_cnt];
  int b_smem_offsets[n_iter_cnt][k_phase_cnt];

  bf16_t a_reg[m_iter_cnt][k_phase_cnt][8];
  bf16_t b_reg[n_iter_cnt][k_phase_cnt][4];
  float acc_reg[m_iter_cnt][n_iter_cnt][4]{};
};

// AB swapped, kernel is k-major, k-major, m-major
template <int batch_size, int gemm_m, int gemm_k, int tile_m, int tile_n,
          int tile_k, int stage_cnt>
__global__ __launch_bounds__(256, 1) void fused_a_gemm_kernel(
    bf16_t* output, bf16_t const* mat_a, bf16_t const* mat_b, int gemm_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int load_thread_cnt = 128;
  constexpr int compute_thread_cnt = 128;
  constexpr int thread_cnt = load_thread_cnt + compute_thread_cnt;
  (void)thread_cnt;
  static_assert(gemm_m % 16 == 0);
  static_assert(gemm_k % tile_k == 0);
  static_assert(gemm_m % tile_m == 0);
  static_assert(
      tile_k == 128 || tile_k == 256 || tile_k == 512 ||
      tile_k == 1024);  // tile_k must be larger than 64 since 4 warp splitK.
  static_assert(tile_m == 16);
  constexpr int g2s_vec_bytes = 16;
  constexpr int a_elem_bytes = 2;
  constexpr int b_elem_bytes = 2;
  static_assert((tile_m * a_elem_bytes + tile_n * b_elem_bytes) * tile_k *
                    stage_cnt <=
                225 * 1024);
  static_assert((tile_m * tile_k * a_elem_bytes) %
                    (load_thread_cnt * g2s_vec_bytes) ==
                0);
  static_assert((tile_n * tile_k * b_elem_bytes) %
                    (load_thread_cnt * g2s_vec_bytes) ==
                0);

  extern __shared__ char smem[];
  uint64_t* smem_barrier = reinterpret_cast<uint64_t*>(
      smem);  // producer,consumer; producer,consumer; ...
  bf16_t* smem_a = reinterpret_cast<bf16_t*>(smem + (stage_cnt * 8 * 2 + 1024) /
                                                        1024 * 1024);
  bf16_t* smem_b = smem_a + tile_m * tile_k * stage_cnt;

  int cta_m_idx = tile_m * blockIdx.x;
  int cta_n_idx = tile_n * blockIdx.y;
  bf16_t const* gmem_a_local = mat_a + cta_m_idx * gemm_k;
  bf16_t const* gmem_b_local = mat_b + cta_n_idx * gemm_k;
  bf16_t* gmem_c_local = output + cta_n_idx * gemm_m + cta_m_idx;

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

  if (warp_idx == 4) {
    for (int i = 0; i < stage_cnt; i++) {
      initialize_barrier(smem_barrier + i * 2 + 0,
                         load_thread_cnt);  // producer
      initialize_barrier(smem_barrier + i * 2 + 1,
                         compute_thread_cnt);  // consumer
    }
  }
  __syncthreads();

  if (warp_idx < 2) {
    GmemLoaderA<gemm_k, tile_m, tile_k, stage_cnt> a_loader(
        gmem_a_local, smem_a, smem_barrier);
    a_loader.prepare();
    a_loader.issue_mainloop();
  } else if (warp_idx < 4) {
    GmemLoaderB<gemm_k, tile_n, tile_k, stage_cnt> b_loader(
        gmem_b_local, smem_b, smem_barrier, gemm_n);
    b_loader.prepare();
    b_loader.issue_mainloop();
  } else {
    MmaComputer<gemm_m, gemm_k, tile_m, tile_n, tile_k, stage_cnt> mma_computer(
        gmem_c_local, smem_a, smem_b, smem_barrier, warp_idx, gemm_n);
    mma_computer.prepare();
    mma_computer.issue_mainloop();
    mma_computer.epi();
  }
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, int kHdIn, int kHdOut, int kTileN>
void invokeFusedAGemm(T* output, T const* mat_a, T const* mat_b, int num_tokens,
                      cudaStream_t const stream) {
  constexpr int gemm_m = kHdOut;  // 2112
  int const gemm_n = num_tokens;  // 1-16
  constexpr int gemm_k = kHdIn;   // 7168
  constexpr int batch_size = 1;
  std::swap(mat_a, mat_b);
  constexpr int tile_m = 16;
  constexpr int tile_n = kTileN;                        // 8 or 16
  constexpr int tile_k = std::max(256, 1024 / tile_n);  // 256
  constexpr int max_stage_cnt =
      1024 * 192 / ((tile_m + tile_n) * tile_k * sizeof(bf16_t));
  constexpr int k_iter_cnt = gemm_k / tile_k;
  constexpr int stage_cnt =
      k_iter_cnt > max_stage_cnt ? max_stage_cnt : k_iter_cnt;
  int cta_m_cnt = gemm_m / tile_m;
  int cta_n_cnt = (gemm_n + tile_n - 1) / tile_n;
  constexpr int barrier_bytes = (stage_cnt * 16 + 1023) / 1024 * 1024;
  constexpr int smem_bytes =
      ((tile_m * 2 + tile_n * 2) * tile_k * stage_cnt + barrier_bytes + 1023) /
      1024 * 1024;

  dim3 grid(cta_m_cnt, cta_n_cnt, 1);
  dim3 block_size(256);
  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block_size;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  if (smem_bytes >= (48 * 1024)) {
    cudaFuncSetAttribute(fused_a_gemm_kernel<batch_size, gemm_m, gemm_k, tile_m,
                                             tile_n, tile_k, stage_cnt>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
  }
  cudaLaunchKernelEx(&config,
                     fused_a_gemm_kernel<batch_size, gemm_m, gemm_k, tile_m,
                                         tile_n, tile_k, stage_cnt>,
                     output, mat_a, mat_b, gemm_n);
}

template void invokeFusedAGemm<__nv_bfloat16, 7168, 2112, 8>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, int num_tokens,
    cudaStream_t);

template void invokeFusedAGemm<__nv_bfloat16, 7168, 2112, 16>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, int num_tokens,
    cudaStream_t);

void dsv3_fused_a_gemm(torch::Tensor& output, torch::Tensor const& mat_a,
                       torch::Tensor const& mat_b) {
  TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && output.dim() == 2);
  int const num_tokens = mat_a.size(0);
  int const hd_in = mat_a.size(1);
  int const hd_out = mat_b.size(1);

  constexpr int kHdIn = 7168;
  constexpr int kHdOut = 2112;
  TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
              "required 1 <= mat_a.shape[0] <= 16")
  TORCH_CHECK(hd_in == kHdIn, "required mat_a.shape[1] == 7168")
  TORCH_CHECK(hd_out == kHdOut, "required mat_b.shape[1] == 2112")
  TORCH_CHECK(output.size(0) == num_tokens,
              "required output.shape[0] == mat_a.shape[0]")
  TORCH_CHECK(output.size(1) == hd_out,
              "required output.shape[1] == mat_b.shape[1]")

  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(output.stride(1) == 1, "output must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be a column major tensor");

  TORCH_CHECK(mat_a.scalar_type() == torch::kBFloat16 &&
                  mat_b.scalar_type() == torch::kBFloat16,
              "Only BFloat16 input dtype is supported")
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "Only BFloat16 output dtype is supported")

  TORCH_CHECK(getSMVersion() >= 90, "required CUDA ARCH >= SM_90");

  auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());
  if (num_tokens <= 8) {
    invokeFusedAGemm<__nv_bfloat16, kHdIn, kHdOut, 8>(
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), num_tokens,
        stream);
  } else {
    invokeFusedAGemm<__nv_bfloat16, kHdIn, kHdOut, 16>(
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), num_tokens,
        stream);
  }
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("dsv3_fused_a_gemm", &dsv3_fused_a_gemm);
}
