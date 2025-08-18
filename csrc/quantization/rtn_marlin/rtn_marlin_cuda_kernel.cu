/*
 * Copyright © 2025, Oracle and/or its affiliates.
 *
 * The implementation of GEMM routines for RTN is based on Marlin
 * (https://github.com/IST-DASLab/marlin)
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <assert.h>

#include "core/scalar_type.hpp"

#include "quantization/gptq_marlin/marlin.cuh"
#include "quantization/gptq_marlin/marlin_dtypes.cuh"
#include "quantization/gptq_marlin/marlin_template.h"

using namespace marlin;

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor rtn_marlin_dequantize(const torch::Tensor& q_weight,
                                    const torch::Tensor& q_scale,
                                    int8_t q_bits);

template <typename T>
struct MarlinUtils {
  using FragA = typename ScalarType<T>::FragA;
  using FragB = typename ScalarType<T>::FragB;
  using T2 = typename ScalarType<T>::scalar_t2;

  template <int qbits>
  __device__ inline static FragB dequant(int q) {
    FragB frag_b;
    if constexpr (qbits == 4)
      marlin::dequant<T2, vllm::kU4B8.id(), false>(q, &frag_b[0]);
    else
      marlin::dequant<T2, vllm::kU8B128.id(), false>(q, &frag_b[0]);
    return frag_b;
  }

  // Instruction for loading a full 16x16 matrix fragment of operand A from
  // shared memory, directly in tensor core layout.
  __device__ inline static void ldsm4(FragA& frag_a, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  }
};

// Asynchronous global->shared copy with a cache hint indicating that the values
// may be evicted immediately; used for quantized weights B, which are only
// accessed precisely once and should thus not pollute the L2 cache which we
// need for inputs A and outputs C.
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

template <const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks = -1,  // number of consecutive 16x16 blocks
                                        // with a separate quantization scale
          typename WEIGHT_DATA_TYPE = __half,
          int WEIGHT_QBITS = 4>
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    const int4* __restrict__ s,  // fp16 quantization scales of shape
                                 // (k/groupsize)xn
    int prob_m,                  // batch dimension m
    int prob_n,                  // output dimension n
    int prob_k,                  // reduction dimension k
    int* locks  // extra global storage for barrier synchronization
) {
  using FragA = typename ScalarType<WEIGHT_DATA_TYPE>::FragA;
  using FragB = typename ScalarType<WEIGHT_DATA_TYPE>::FragB;
  using FragC = typename ScalarType<WEIGHT_DATA_TYPE>::FragC;
  using FragS = typename ScalarType<WEIGHT_DATA_TYPE>::FragS;
  using T2 = typename ScalarType<WEIGHT_DATA_TYPE>::scalar_t2;
  constexpr int pack_factor = 32 / WEIGHT_QBITS;

  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices"
  // (of width 16 * `thread_n_blocks`). Stripes are defined as shown in the
  // 3x3 matrix 5 SM example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
  // Ensure that the number of tiles in each stripe is a multiple of the
  // groupsize; this avoids an annoying special case where a stripe starts in
  // the middle of group.
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) *
            ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;      // number of threadblock tiles in the current slice
  int slice_count = 0;  // total number of active threadblocks
                        // in the current slice
  int slice_idx;        // index of threadblock in current slice;
                        // numbered bottom to top

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 8;  // stride of the A matrix in global memory
  // We typically use `constexpr` to indicate that this value is a compile-time
  // constant
  constexpr int a_sh_stride =
      16 * thread_k_blocks / 8;  // stride of an A matrix tile in shared memory
  constexpr int a_gl_rd_delta_o =
      16 * thread_k_blocks / 8;  // delta between subsequent A tiles
                                 // in global memory
  int a_gl_rd_delta_i =
      a_gl_stride * (threads / a_gl_rd_delta_o);  // between subsequent
                                                  // accesses within a tile
  constexpr int a_sh_wr_delta =
      a_sh_stride *
      (threads / a_gl_rd_delta_o);  // between shared memory writes
  constexpr int a_sh_rd_delta_o =
      2 * ((threads / 32) / (thread_n_blocks / 4));  // between shared memory
                                                     // tile reads
  constexpr int a_sh_rd_delta_i =
      a_sh_stride * 16;  // within a shared memory tile
  constexpr int a_sh_stage =
      a_sh_stride * (16 * thread_m_blocks);  // overall size of a tile
  constexpr int a_sh_wr_iters =
      ceildiv(a_sh_stage, a_sh_wr_delta);  // number of shared write
                                           // iterations for a tile

  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = WEIGHT_QBITS == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_rd_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_tb_groups =
      group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;

  int s_gl_rd_delta = s_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x * b_thread_vecs;
  int b_sh_rd = threadIdx.x * b_thread_vecs;

  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd;
  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  if (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) % 4;

  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int k_iter_size = tb_k / b_sh_wr_iters;

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependicies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);
  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  int4 frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  // Zero accumulators.
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j);
        }
        B_ptr[i] += b_gl_rd_delta_o;
      }

      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        if constexpr (group_blocks >= thread_k_blocks) {
          // Only fetch scales if this tile starts a new group
          if (pipe % (group_blocks / thread_k_blocks) == 0) {
            if (s_sh_wr_pred) {
#if __CUDA_ARCH__ == 800
              cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
#elif __CUDA_ARCH__ > 800
              cp_async4(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
#else
              assert(false);
#endif
            }
            s_gl_rd += s_gl_rd_delta;
          }
        } else {
          for (int i = 0; i < s_tb_groups; i++) {
            if (s_sh_wr_pred) {
#if __CUDA_ARCH__ == 800
              cp_async4_stream(&sh_s_stage[i * s_sh_stride + s_sh_wr],
                               &s[s_gl_rd]);
#elif __CUDA_ARCH__ > 800
              cp_async4(&sh_s_stage[i * s_sh_stride + s_sh_wr], &s[s_gl_rd]);
#else
              assert(false);
#endif
            }
            s_gl_rd += s_gl_rd_delta;
          }
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    // It may seem inefficient that we reload the groups for every sub-tile;
    // however, this does not seem to be a significant bottleneck, while some
    // theoretically better attempts have lead to bad instruction ordering by
    // the compiler and correspondingly a noticable drop in performance.
    if (group_blocks != -1) {
      if constexpr (group_blocks >= thread_k_blocks) {
        int4* sh_s_stage =
            sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                                 (pipe / (group_blocks / thread_k_blocks)));
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
      } else {
        int warp_id = threadIdx.x / 32;
        int n_warps = thread_n_blocks / 4;

        int warp_row = warp_id / n_warps;

        int cur_k = warp_row * 16;
        cur_k += k_iter_size * (k % b_sh_wr_iters);

        int k_blocks = cur_k / 16;
        int cur_group_id = k_blocks / group_blocks;

        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
      }
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      MarlinUtils<WEIGHT_DATA_TYPE>::ldsm4(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&](int k) {
// We have the m dimension as the inner loop in order to encourage overlapping
// dequantization and matmul operations.
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (WEIGHT_QBITS == 4) {
        b_quant_0 = frag_b_quant[k % 2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(WEIGHT_QBITS == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k % 2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      frag_b0 = MarlinUtils<WEIGHT_DATA_TYPE>::dequant<WEIGHT_QBITS>(b_quant_0);

      // If there are no groups, we can just scale the final output once and can
      // avoid doing so for each weight.
      if (group_blocks != -1)
        marlin::scale<WEIGHT_DATA_TYPE>(frag_b0, frag_s[k % 2][j], 0);

      frag_b1 = MarlinUtils<WEIGHT_DATA_TYPE>::dequant<WEIGHT_QBITS>(b_quant_1);

      if (group_blocks != -1)
        marlin::scale<WEIGHT_DATA_TYPE>(frag_b1, frag_s[k % 2][j], 1);

#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        marlin::mma<WEIGHT_DATA_TYPE>(
            frag_a[k % 2][i], frag_b0,
            *reinterpret_cast<FragC*>(&frag_c[i][j][0]));
        marlin::mma<WEIGHT_DATA_TYPE>(
            frag_a[k % 2][i], frag_b1,
            *reinterpret_cast<FragC*>(&frag_c[i][j][1]));
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

// Parallel logarithmic shared memory reduction. We make sure to avoid any
// unnecessary read or write iterations, e.g., for two warps we write only once
// by warp 1 and read only once by warp 0.
#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd =
                    reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd =
                reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                    4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
// Interestingly, doing direct global accesses here really seems to mess up the
// compiler and lead to slowdowns, hence we also use async-copies even though
// these fetches are not actually asynchronous.
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
              &sh[c_sh_wr + c_sh_wr_delta * i],
              &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                 c_gl_wr_delta_i * (i % 2)],
              i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(
                  &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                  marlin::ScalarType<WEIGHT_DATA_TYPE>::num2float(
                      reinterpret_cast<WEIGHT_DATA_TYPE*>(&c_red)[j]);
            }
          }
          if (!last) {
            int4 c;
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<WEIGHT_DATA_TYPE*>(&c)[j] =
                  marlin::ScalarType<WEIGHT_DATA_TYPE>::float2num(
                      reinterpret_cast<float*>(
                          &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] =
                c;
          }
        }
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr =
        (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s) {
      T2 res = marlin::ScalarType<WEIGHT_DATA_TYPE>::nums2num2(
          marlin::ScalarType<WEIGHT_DATA_TYPE>::float2num(c0),
          marlin::ScalarType<WEIGHT_DATA_TYPE>::float2num(c1));
      if (group_blocks == -1) {
        // for per-column quantization we finally apply the scale here
        res = __hmul2(res, s[0]);
      }
      ((T2*)sh)[idx] = res;
    };
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0,
                *reinterpret_cast<float*>(&frag_c[i][j][0].x),
                *reinterpret_cast<float*>(&frag_c[i][j][0].y),
                frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0,
                *reinterpret_cast<float*>(&frag_c[i][j][0].z),
                *reinterpret_cast<float*>(&frag_c[i][j][0].w),
                frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4,
                *reinterpret_cast<float*>(&frag_c[i][j][1].x),
                *reinterpret_cast<float*>(&frag_c[i][j][1].y),
                frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4,
                *reinterpret_cast<float*>(&frag_c[i][j][1].z),
                *reinterpret_cast<float*>(&frag_c[i][j][1].w),
                frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0;
         i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {
#pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.
#pragma unroll
    for (int pipe = 0; pipe < stages;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compliation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if (group_blocks == -1) {
        if constexpr (WEIGHT_QBITS == 8) {
          if (s_sh_wr_pred) cp_async4(&sh_s[s_sh_wr], &s[s_gl_rd]);
          cp_async_fence();
        } else if (last) {
          if (s_sh_wr_pred) cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);
          cp_async_fence();
        }
      }
      thread_block_reduce();

      if constexpr (group_blocks == -1) {
        if constexpr (WEIGHT_QBITS == 8) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
          }

        } else {
          if (last) {
            cp_async_wait<0>();
            __syncthreads();
            if (threadIdx.x / 32 < thread_n_blocks / 4) {
              reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
              reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (group_blocks == -1 && WEIGHT_QBITS == 8) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              marlin::scale_float<WEIGHT_DATA_TYPE>(
                  reinterpret_cast<float*>(&frag_c[i][j][0].x),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              marlin::scale_float<WEIGHT_DATA_TYPE>(
                  reinterpret_cast<float*>(&frag_c[i][j][0].z),
                  frag_s[j / 2][2 * (j % 2) + 0]);

              marlin::scale_float<WEIGHT_DATA_TYPE>(
                  reinterpret_cast<float*>(&frag_c[i][j][1].x),
                  frag_s[j / 2][2 * (j % 2) + 1]);
              marlin::scale_float<WEIGHT_DATA_TYPE>(
                  reinterpret_cast<float*>(&frag_c[i][j][1].z),
                  frag_s[j / 2][2 * (j % 2) + 1]);
            }
          }
        }
      }
      if (slice_count > 1) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last)  // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int THREADS = 256;
const int STAGES = 4;  // 4 pipeline stages fit into shared memory

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,             \
                GROUP_BLOCKS)                                                  \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                               \
           thread_n_blocks == THREAD_N_BLOCKS &&                               \
           thread_k_blocks == THREAD_K_BLOCKS &&                               \
           group_blocks == GROUP_BLOCKS) {                                     \
    cudaFuncSetAttribute(                                                      \
        Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,     \
               STAGES, GROUP_BLOCKS, WEIGHT_DATA_TYPE, WEIGHT_QBITS>,          \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);          \
    Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, \
           GROUP_BLOCKS, WEIGHT_DATA_TYPE, WEIGHT_QBITS>                       \
        <<<blocks, THREADS, max_shared_mem, stream>>>(                         \
            A_ptr, B_ptr, C_ptr, s_ptr, prob_m, prob_n, prob_k, locks);        \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

template <typename WEIGHT_DATA_TYPE = __half, int WEIGHT_QBITS = 4>
int marlin_cuda(const void* A, const void* B, void* C, void* s, int prob_m,
                int prob_n, int prob_k, void* workspace, int groupsize = -1,
                int dev = 0, cudaStream_t stream = 0, int thread_k = -1,
                int thread_n = -1, int sms = -1, int max_par = 16) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16 || prob_n % 256 != 0) {
      // For small batchizes, better partitioning is slightly more important
      // than better compute utilization
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 ||
      (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) return 0;

  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  const int4* s_ptr = (const int4*)s;

  int* locks = (int*)workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par) par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

    // For compilation speed, we only define the kernel configurations that have
    // seemed useful (in terms of performance) in our testing, however many more
    // are, in principle, possible.
    if (false) {
    }
    CALL_IF(1, 8, 8, -1)
    CALL_IF(1, 8, 8, 8)
    CALL_IF(2, 8, 8, 8)
    CALL_IF(3, 8, 8, 8)
    CALL_IF(4, 8, 8, 8)
    CALL_IF(1, 16, 4, -1)
    CALL_IF(1, 16, 4, 8)
    CALL_IF(2, 16, 4, -1)
    CALL_IF(2, 16, 4, 8)
    CALL_IF(3, 16, 4, -1)
    CALL_IF(3, 16, 4, 8)
    CALL_IF(4, 16, 4, -1)
    CALL_IF(4, 16, 4, 8)
    else ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}

#define INSTANTIATE_MARLIN_CUDA(T, BITS)                                    \
  template int marlin_cuda<T, BITS>(const void*, const void*, void*, void*, \
                                    int, int, int, void*, int, int,         \
                                    cudaStream_t, int, int, int, int);

INSTANTIATE_MARLIN_CUDA(__half, 4)
INSTANTIATE_MARLIN_CUDA(__nv_bfloat16, 4)

INSTANTIATE_MARLIN_CUDA(__half, 8)
INSTANTIATE_MARLIN_CUDA(__nv_bfloat16, 8)

#define CALL_MARLIN_CUDA(WEIGHT_DATA_TYPE, WEIGHT_QBITS)                      \
  marlin_cuda<WEIGHT_DATA_TYPE, WEIGHT_QBITS>(                                \
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), b_scales.data_ptr(), \
      size_m, size_n, size_k, workspace.data_ptr(), groupsize, dev,           \
      at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms, max_par)

torch::Tensor rtn_marlin_gemm(const torch::Tensor& a,
                              const torch::Tensor& b_q_weight,
                              const torch::Tensor& b_scales,
                              torch::Tensor& workspace, int64_t size_m,
                              int64_t size_n, int64_t size_k) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);

  int groupsize = b_scales.size(0) == 1 ? -1 : size_k / b_scales.size(1);

  // Verify groupsize (for now, we only supports group size of 128)
  TORCH_CHECK(groupsize == 128, "Unexpected groupsize = " + str(groupsize));

  int qbits = b_q_weight.size(0) == b_scales.size(0) ? 8 : 4;

  // Switch to a Dequantize-and-GEMM path for long inputs
  // (see https://arxiv.org/pdf/2505.15909 for details)
  const int SLOW_PATH_MATMUL_HEURISTIC_CONDITION = 1024;
  if (size_m >= SLOW_PATH_MATMUL_HEURISTIC_CONDITION) {
    auto weight = rtn_marlin_dequantize(b_q_weight, b_scales, qbits);
    return at::matmul(a, weight.t());
  }

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  int max_par = 16;

  int dev = a.get_device();
  int err;

  auto dtype = a.options().dtype();
  if (dtype == torch::kFloat16) {
    if (qbits == 4)
      err = CALL_MARLIN_CUDA(__half, 4);
    else
      err = CALL_MARLIN_CUDA(__half, 8);
  } else {
    if (qbits == 4)
      err = CALL_MARLIN_CUDA(__nv_bfloat16, 4);
    else
      err = CALL_MARLIN_CUDA(__nv_bfloat16, 8);
  }

  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("RTN Marlin internal error: Problem (m=", size_m, ", n=", size_n,
             ", k=", size_k, ")", " not compatible with thread_k=", thread_k,
             ", thread_n=", thread_n, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
        "RTN Marlin internal error: No kernel implementation for thread_k=",
        thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, ".");
  }

  return c;
}

template <typename T, int BITS>
__global__ void dequantize_marlin_kernel(T* output, const int8_t* input,
                                         const T* qscale, int hidden_dim,
                                         int output_size, int group_size,
                                         int cnt) {
  using T2 = typename std::conditional<std::is_same<T, __half>::value, half2,
                                       __nv_bfloat162>::type;

  unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
  unsigned tid = threadIdx.x;

  const int32_t* input_cast = (const int32_t*)input;
  T2* output_cast = reinterpret_cast<T2*>(output);

  int scaled_hidden_dim = hidden_dim / 2;
  output_cast += bid * scaled_hidden_dim;

  int factor1 = hidden_dim / 128;
  float factor2 = 8.0 / factor1;
  int factor3 = output_size * 16;

  int row = bid;
  int first_row = int((row / 64) * factor2) + (row % 8) / factor1;
  int first_col_base = ((row % 8) % factor1) * 128 + (row % 64) / 8 * 4 +
                       ((row / 64) * 1024) % hidden_dim;

  static constexpr uint32_t OFFSET =
      BITS == 8 ?
                // FP16/BF16 encoding of 128 is 0x5800/0x4300
          (std::is_same<T, __half>::value ? 0x58005800 : 0x43004300)
                :
                // FP16/BF16 encoding of 8 is 0x4800/0x4100
          (std::is_same<T, __half>::value ? 0x48004800 : 0x41004100);

  static constexpr int offsets[] = {0, 0, +2, +2, -2, -2, 0, 0};

  int scale_elem_base = (row / 64) * 64 + (row % 64) / 8 + (row % 8) * 8;

  int first_col = 2 * tid;
  first_col_base +=
      ((first_col % 8) / 2) * 32 + (first_col % 2) * 2 + (first_col % 16) / 8;
  if constexpr (BITS == 4) first_col_base += offsets[first_col_base % 8];

  int shift = (first_col_base % 2) * 4;

  for (int c = 0; c < cnt; c++) {
    if (tid < scaled_hidden_dim) {
      int col = 2 * tid;

      int i = (col / 16) * 16 * output_size / hidden_dim + first_row;
      int j = first_col_base + (factor3 * (col / 16)) % hidden_dim;

      int off = i * hidden_dim + j;
      if constexpr (BITS == 4) off /= 2;

      int scale_elem = scale_elem_base + (col / 128) * output_size;

      T scale = qscale[scale_elem];

      int32_t src_32 = input_cast[off / 4];
      uint8_t* src_8 = (uint8_t*)&src_32;

      T src_val =
          BITS == 8 ? marlin::ScalarType<T>::float2num((float)(src_8[off % 2]))
                    : marlin::ScalarType<T>::float2num(
                          (float)(((src_8[off % 2]) >> shift) & 0x0f));
      T src_val2 =
          BITS == 8
              ? marlin::ScalarType<T>::float2num((float)(src_8[(off % 2) + 2]))
              : marlin::ScalarType<T>::float2num(
                    (float)(((src_8[(off % 2) + 2]) >> shift) & 0x0f));

      T2 scale2 = T2(scale, scale);
      T2 src2 = T2(src_val, src_val2);

      src2 = __hsub2(src2, *reinterpret_cast<const T2*>(&OFFSET));
      T2 q_f = __hmul2(src2, scale2);

      output_cast[tid] = q_f;
      tid += blockDim.x;
    }
  }
}

template <typename T>
void launch_dequantize_marlin(T* output, const int8_t* input, const T* qscale,
                              unsigned output_size, unsigned hidden_dim,
                              unsigned groups, int q_bits,
                              cudaStream_t stream) {
  unsigned threads = 64;
  int group_size = hidden_dim * output_size / groups;

  unsigned thd_cnt = ((hidden_dim - 1) / threads + 1);

  dim3 block_dims(threads);
  dim3 grid_dims(output_size);

  if (q_bits == 4)
    dequantize_marlin_kernel<T, 4><<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, output_size, group_size, thd_cnt);
  else
    dequantize_marlin_kernel<T, 8><<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, output_size, group_size, thd_cnt);
}

#define LAUNCH_DEQUANTIZE_MARLIN(T)                                           \
  launch_dequantize_marlin(                                                   \
      (T*)output.data_ptr(), (const int8_t*)q_weight.data_ptr(),              \
      (const T*)q_scale.data_ptr(), out_size, hidden_dim, num_groups, q_bits, \
      at::cuda::getCurrentCUDAStream())

torch::Tensor rtn_marlin_dequantize(const torch::Tensor& q_weight,
                                    const torch::Tensor& q_scale,
                                    int8_t q_bits) {
  int hidden_dim = q_weight.size(1);
  int out_size = q_weight.size(0);
  if (q_bits == 4) out_size *= 2;

  int num_groups = q_scale.size(0) * q_scale.size(1);

  auto dtype = q_scale.options().dtype();

  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .layout(at::kStrided)
                     .device(at::kCUDA)
                     .requires_grad(false);

  auto output = torch::empty({out_size, hidden_dim}, options);

  if (dtype == torch::kFloat16)
    LAUNCH_DEQUANTIZE_MARLIN(__half);
  else
    LAUNCH_DEQUANTIZE_MARLIN(__nv_bfloat16);

  return output;
}

#define INSTANTIATE_DEQUANTIZE_MARLIN(T)                                       \
  template void launch_dequantize_marlin<T>(T*, const int8_t*, const T*,       \
                                            unsigned, unsigned, unsigned, int, \
                                            cudaStream_t);

INSTANTIATE_DEQUANTIZE_MARLIN(__nv_bfloat16);
INSTANTIATE_DEQUANTIZE_MARLIN(__half);
