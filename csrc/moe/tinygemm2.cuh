/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc7/cpp/tensorrt_llm/kernels/tinygemm2/tinygemm2_kernel.cuh
 * Copyright (c) 2025, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_bf16.h"
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "cuda_pipeline.h"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cuda_runtime.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace ptx = cuda::ptx;

#define gpuErrChk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, char const* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      throw std::runtime_error(cudaGetErrorString(code));
    }
  }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ uint64_t gclock64() {
  unsigned long long int rv;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(rv));
  return rv;
}

__device__ void ldmatrix(__nv_bfloat16 rv[2], uint32_t smem_ptr) {
  int dst;
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(dst)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = dst;
}

__device__ void ldmatrix2(__nv_bfloat16 rv[4], uint32_t smem_ptr) {
  int x, y;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(x), "=r"(y)
               : "r"(smem_ptr));

  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = x;
  rvi[1] = y;
}

__device__ void ldmatrix4(__nv_bfloat16 rv[8], uint32_t smem_ptr) {
  int x, y, z, w;
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
      : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = x;
  rvi[1] = y;
  rvi[2] = z;
  rvi[3] = w;
}

__device__ void HMMA_1688(float d[4], __nv_bfloat16 a[4], __nv_bfloat16 b[2],
                          float c[4]) {
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
  float const* C = reinterpret_cast<float const*>(&c[0]);
  float* D = reinterpret_cast<float*>(&d[0]);

  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]),
        "f"(C[3]));
}

__device__ void HMMA_16816(float d[4], __nv_bfloat16 a[8], __nv_bfloat16 b[4],
                           float c[4]) {
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
  float const* C = reinterpret_cast<float const*>(&c[0]);
  float* D = reinterpret_cast<float*>(&d[0]);

  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ void bar_wait(uint32_t bar_ptr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar_ptr),
      "r"(phase));
}

__device__ bool bar_try_wait(uint32_t bar_ptr, int phase) {
  uint32_t success;
  #ifdef INTERNAL
  asm volatile(".pragma \"set knob DontInsertYield\";\n" : : : "memory");
  #endif
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(success)
      : "r"(bar_ptr), "r"(phase));
  return success;
}

__device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}
#endif

struct Profile {
  uint64_t start;
  uint64_t weight_load_start;
  uint64_t act_load_start;
  uint64_t compute_start;
  uint64_t complete;
};

template <int WARP_TILE_M, int TILE_M, int TILE_N, int TILE_K, int STAGES,
          int STAGE_UNROLL, bool PROFILE>
__global__ __launch_bounds__(384, 1) void tinygemm_kernel(
    __nv_bfloat16* output, __nv_bfloat16* weights, __nv_bfloat16* activations,
    __nv_bfloat16* bias, int M, int N, int K,
    const __grid_constant__ CUtensorMap weight_map,
    const __grid_constant__ CUtensorMap activation_map,
    Profile* profile = nullptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

  if (PROFILE && threadIdx.x == 0 && blockIdx.y == 0)
    profile[blockIdx.x].start = gclock64();

  extern __shared__ __align__(128) char smem[];

  __nv_bfloat16* sh_weights = (__nv_bfloat16*)&smem[0];
  __nv_bfloat16* sh_activations =
      (__nv_bfloat16*)&smem[STAGES * STAGE_UNROLL * TILE_M * TILE_K *
                            sizeof(__nv_bfloat16)];

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar_wt_ready[STAGES];
  __shared__ barrier bar_act_ready[STAGES];
  __shared__ barrier bar_data_consumed[STAGES];

  __shared__ float4 reduction_buffer[128];

  __shared__ nv_bfloat16 sh_bias[TILE_M];

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      init(&bar_wt_ready[i], 1);
      init(&bar_act_ready[i], 1);
      init(&bar_data_consumed[i], 32);
    }
    ptx::fence_proxy_async(ptx::space_shared);
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&weight_map))
                 : "memory");
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&activation_map))
                 : "memory");
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int phase = 0;

  int mib = blockIdx.x * TILE_M;
  int ni = blockIdx.y * TILE_N;

  float accum[4];
  for (int i = 0; i < 4; i++) accum[i] = 0.f;

  int const K_LOOPS_DMA =
      (K + 4 * TILE_K * STAGE_UNROLL - 1) / (4 * (TILE_K * STAGE_UNROLL));
  int const K_LOOPS_COMPUTE = K_LOOPS_DMA;

  // Data loading thread
  if (warp_id >= 4 && elect_one_sync()) {
    int stage = warp_id % 4;

    bool weight_warp = warp_id < 8;
    if (!weight_warp) {
      cudaGridDependencySynchronize();
      cudaTriggerProgrammaticLaunchCompletion();
    }

    for (int ki = 0; ki < K_LOOPS_DMA; ki++) {
      int k = (ki * 4 + (warp_id % 4)) * TILE_K * STAGE_UNROLL;

      uint64_t desc_ptr_wt = reinterpret_cast<uint64_t>(&weight_map);
      uint64_t desc_ptr_act = reinterpret_cast<uint64_t>(&activation_map);

      uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
      uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);
      int bytes_wt = TILE_M * TILE_K * sizeof(__nv_bfloat16);
      int bytes_act = TILE_N * TILE_K * sizeof(__nv_bfloat16);

      bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);

      if (weight_warp)
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :
                     : "r"(bar_ptr_wt), "r"(STAGE_UNROLL * bytes_wt));
      if (!weight_warp)
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :
                     : "r"(bar_ptr_act), "r"(STAGE_UNROLL * bytes_act));

      if (PROFILE && blockIdx.y == 0 && ki == 0 && weight_warp)
        profile[blockIdx.x].weight_load_start = gclock64();
      if (PROFILE && blockIdx.y == 0 && ki == 0 && !weight_warp)
        profile[blockIdx.x].act_load_start = gclock64();

      for (int i = 0; i < STAGE_UNROLL; i++) {
        uint32_t smem_ptr_wt = __cvta_generic_to_shared(
            &sh_weights[(stage * STAGE_UNROLL + i) * TILE_M * TILE_K]);
        uint32_t crd0 = k + i * TILE_K;
        uint32_t crd1 = mib;
        if (weight_warp)
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
              "tx::bytes [%0], [%1, {%3,%4}], "
              "[%2];"
              :
              : "r"(smem_ptr_wt), "l"(desc_ptr_wt), "r"(bar_ptr_wt), "r"(crd0),
                "r"(crd1)
              : "memory");

        uint32_t smem_ptr_act = __cvta_generic_to_shared(
            &sh_activations[(stage * STAGE_UNROLL + i) * TILE_N * TILE_K]);
        crd0 = k + i * TILE_K;
        crd1 = ni;
        if (!weight_warp)
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
              "tx::bytes [%0], [%1, {%3,%4}], "
              "[%2];"
              :
              : "r"(smem_ptr_act), "l"(desc_ptr_act), "r"(bar_ptr_act),
                "r"(crd0), "r"(crd1)
              : "memory");
      }

      stage += 4;
      if (stage >= STAGES) {
        stage = warp_id % 4;
        phase ^= 1;
      }
    }
    // Wait for pending loads to be consumed before exiting, to avoid race
    for (int i = 0; i < (STAGES / 4) - 1; i++) {
      bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);
      stage += 4;
      if (stage >= STAGES) {
        stage = warp_id % 4;
        phase ^= 1;
      }
    }
  }
  // Compute threads
  else if (warp_id < 4) {
    // Sneak the bias load into the compute warps since they're just waiting for
    // stuff anyway
    if (threadIdx.x < TILE_M) sh_bias[threadIdx.x] = bias[mib + threadIdx.x];

    int stage = warp_id;

    int phase = 0;
    int lane_id_div8 = lane_id / 8;
    int lane_id_mod8 = lane_id % 8;

    int lane_row_offset_wt = (lane_id_div8 % 2) ? 8 : 0;
    int lane_col_offset_wt = (lane_id_div8 / 2) ? 1 : 0;

    int row_wt = lane_id_mod8 + lane_row_offset_wt;
    int row_act = lane_id_mod8;

    int row_offset_wt = (reinterpret_cast<uintptr_t>(sh_weights) / 128) % 8;
    int row_offset_act = row_offset_wt;

    uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
    uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);

    bool weight_ready = bar_try_wait(bar_ptr_wt, phase);
    bool act_ready = bar_try_wait(bar_ptr_act, phase);

  #pragma unroll 2
    for (int ki = 0; ki < K_LOOPS_COMPUTE; ki++) {
      int next_stage = stage + 4;
      int next_phase = phase;
      if (next_stage >= STAGES) {
        next_stage = warp_id;
        next_phase ^= 1;
      }

      while (!weight_ready || !act_ready) {
        weight_ready = bar_try_wait(bar_ptr_wt, phase);
        act_ready = bar_try_wait(bar_ptr_act, phase);
      }

      if (PROFILE && blockIdx.y == 0 && threadIdx.x == 0 && ki == 0)
        profile[blockIdx.x].compute_start = gclock64();

      if (ki + 1 < K_LOOPS_COMPUTE) {
        weight_ready = bar_try_wait(
            __cvta_generic_to_shared(&bar_wt_ready[next_stage]), next_phase);
        act_ready = bar_try_wait(
            __cvta_generic_to_shared(&bar_act_ready[next_stage]), next_phase);
      }

  #pragma unroll
      for (int su = 0; su < STAGE_UNROLL; su++) {
        __nv_bfloat16* ptr_weights =
            &sh_weights[(stage * STAGE_UNROLL + su) * TILE_M * TILE_K];
        __nv_bfloat16* ptr_act =
            &sh_activations[(stage * STAGE_UNROLL + su) * TILE_N * TILE_K];

  #pragma unroll
        for (int kii = 0; kii < TILE_K / 16; kii++) {
          __nv_bfloat16 a[8];
          __nv_bfloat16 b[4];

          int col = 2 * kii + lane_col_offset_wt;
          int col_sw = ((row_wt + row_offset_wt) % 8) ^ col;

          ldmatrix4(a, __cvta_generic_to_shared(
                           &ptr_weights[row_wt * TILE_K + col_sw * 8]));

          col = 2 * kii + lane_id_div8;
          col_sw = ((row_act + row_offset_act) % 8) ^ col;

          ldmatrix2(b, __cvta_generic_to_shared(
                           &ptr_act[row_act * TILE_K + 8 * col_sw]));

          HMMA_16816(accum, a, b, accum);
        }
      }

      uint32_t bar_c = __cvta_generic_to_shared(&bar_data_consumed[stage]);
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(bar_c));

      stage = next_stage;
      phase = next_phase;
    }

    float4 accum4;
    accum4.x = accum[0];
    accum4.y = accum[1];
    accum4.z = accum[2];
    accum4.w = accum[3];
    reduction_buffer[threadIdx.x] = accum4;

    __syncthreads();

    if (warp_id == 0) {
      int mi = mib + warp_id * WARP_TILE_M;
      int tm = mi + lane_id / 4;
      int tn = ni + 2 * (lane_id % 4);

      float4 accum1 = reduction_buffer[32 + threadIdx.x];
      float4 accum2 = reduction_buffer[64 + threadIdx.x];
      float4 accum3 = reduction_buffer[96 + threadIdx.x];

      accum[0] = accum[0] + accum1.x + accum2.x + accum3.x;
      accum[1] = accum[1] + accum1.y + accum2.y + accum3.y;
      accum[2] = accum[2] + accum1.z + accum2.z + accum3.z;
      accum[3] = accum[3] + accum1.w + accum2.w + accum3.w;

      float bias_lo = __bfloat162float(sh_bias[tm - mib]);
      float bias_hi = __bfloat162float(sh_bias[tm + 8 - mib]);

      if (tn < N && tm < M)
        output[tn * M + tm] = __float2bfloat16(accum[0] + bias_lo);
      if (tn + 1 < N && tm < M)
        output[(tn + 1) * M + tm] = __float2bfloat16(accum[1] + bias_lo);
      if (tn < N && tm + 8 < M)
        output[tn * M + tm + 8] = __float2bfloat16(accum[2] + bias_hi);
      if (tn + 1 < N && tm + 8 < M)
        output[(tn + 1) * M + tm + 8] = __float2bfloat16(accum[3] + bias_hi);

      if (PROFILE && blockIdx.y == 0 && threadIdx.x == 0)
        profile[blockIdx.x].complete = gclock64();
    }
  }
#endif  // end if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
