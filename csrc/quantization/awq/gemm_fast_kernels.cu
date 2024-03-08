#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "semaphore.h"
#include "dequantize_fast.cuh"
#include <torch/extension.h>
#include <cuda_pipeline_primitives.h>

namespace vllm {
namespace awq {

#define kInterleave 4
#define OP_M 16
#define OP_N 8
#define OP_K 16
#define INTRIN_M 16
#define INTRIN_N 16
#define INTRIN_K 16
#define WARP_SIZE 32
#define SMEM_PAD_A 0
#define SMEM_PAD_B 0
#define PACK_SIZE 8
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

#define KERNEL_LAUNCH_CODE                                                                                                                              \
  int num_mn_tiles = (num_in_feats + CTA_M - 1) / CTA_M * (num_out_channels + CTA_N - 1) / CTA_N;                                                       \
  torch::Tensor _semaphores = torch::empty({num_mn_tiles}, options_int);                                                                                \
  auto semaphores = reinterpret_cast<int *>(_semaphores.data_ptr<int>());                                                                               \
  constexpr int NUM_WARPS = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);                                                                     \
  constexpr int SCALES_SMEM_SIZE = (G >= CTA_K) ? (CTA_N / (G / CTA_K) * STAGES * 2) : (CTA_N * (CTA_K / G) * STAGES * 2);                              \
  constexpr int kSmemByteSize = (CTA_M * (CTA_K + SMEM_PAD_A) + CTA_N * (CTA_K + SMEM_PAD_B) / kInterleave + SCALES_SMEM_SIZE) * STAGES * sizeof(half); \
  if (kSmemByteSize >= 99 * 1024)                                                                                                                       \
  {                                                                                                                                                     \
    printf("This kernel requires %d Bytes of shared memory, which exceeds device limit.\n", kSmemByteSize);                                             \
    return _out_feats;                                                                                                                                  \
  }                                                                                                                                                     \
  int j_factors1 = num_out_channels / CTA_N / 1;                                                                                                        \
  dim3 num_blocks((num_out_feats + CTA_M - 1) / CTA_M * j_factors1 * SPLITK);                                                                           \
  dim3 threads_per_block(WARP_SIZE, NUM_WARPS);                                                                                                         \
  auto kernel_func = vllm::awq::gemm_w4a16_T1<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G, SPLITK>;                                          \
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemByteSize);                                                        \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                                                         \
  kernel_func<<<num_blocks, threads_per_block, kSmemByteSize, stream>>>(                                                                             \
      in_feats, kernel, scales, zeros, out_feats, semaphores, num_in_feats, num_out_channels, num_in_channels);

template <int N>
__inline__ __host__ __device__ int get_log_tile(int n)
{
  if (N >= 8 && n >= 6)
    return 3;
  else if (N >= 4 && n >= 3)
    return 2;
  else if (N >= 2 && n >= 2)
    return 1;
  else
    return 0;
}

__inline__ __device__ uint2 get_block_idx_mapping(int blockIdx_x, int blockIdx_y, int log_tile)
{
  return make_uint2((blockIdx_x >> log_tile), (blockIdx_y << log_tile) + ((blockIdx_x) & ((1 << (log_tile)) - 1)));
}

template <int SLICES, int NUM_WARPS_MN>
__device__ void sync_slice(int slice_id)
{
  if constexpr (SLICES == 1)
  {
    __syncthreads();
  }
  else
  {
    constexpr int SLICE_GROUP = (SLICES + 7) / 8;
    constexpr uint32_t num_threads = NUM_WARPS_MN * WARP_SIZE;
    const uint32_t barrier_id = slice_id / SLICE_GROUP + 1;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "n"(num_threads));
  }
}

__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr)
{
  uint32_t smem_int_ptr;

  asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_int_ptr)
      : "l"(ptr));

  return smem_int_ptr;
}

__inline__ __device__ void ldmatrix_m8n8_x4_b16(half *shared_warp, int ax0_0, uint32_t addr)
{
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[3])
      : "r"(addr));
}

__inline__ __device__ void ldmatrix_m8n8_x4_trans_b16(half *shared_warp, int ax0_0, uint32_t addr)
{
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(shared_warp + (ax0_0 * 8)))[3])
      : "r"(addr));
}

__inline__ __device__ void cp_async_cg_A(uint32_t smem_int_ptr, const uint4 *__restrict__ src, bool mask)
{
  const int cp_size = 16;
  asm volatile("{"
               "  .reg .pred p;"
               "  setp.ne.b32 p, %0, 0;"
               "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;"
                                                                  "}" ::"r"((int)mask),
               "r"(smem_int_ptr),
               "l"(src),
               "n"(cp_size));
}

__device__ __inline__ void mma_m16n8k16(float *C_warp, half *A_shared_warp, half *B_shared_warp)
{
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(((float *)C_warp)[0]), "=f"(((float *)C_warp)[1]), "=f"(((float *)C_warp)[2]), "=f"(((float *)C_warp)[3])
      : "r"(((unsigned *)A_shared_warp)[0]), "r"(((unsigned *)A_shared_warp)[1]), "r"(((unsigned *)A_shared_warp)[2]), "r"(((unsigned *)A_shared_warp)[3]), "r"(((unsigned *)B_shared_warp)[0]), "r"(((unsigned *)B_shared_warp)[1]), "f"(((float *)C_warp)[0]), "f"(((float *)C_warp)[1]), "f"(((float *)C_warp)[2]), "f"(((float *)C_warp)[3]));
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS, int STAGES>
__device__ __inline__ void global_to_share_one_stage_A(half *src, half *dst, int global_nrows, int global_ncols, int cta_offset_m, int cta_offset_n, int cta_offset_k, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int threads_needed = (CTA_M * CTA_K) / PACK_SIZE / SHARED_K_ITERS;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = (CTA_M * CTA_K) / PACK_SIZE / threads_used;
  constexpr int partial_global_iters = (total_global_iters + SHARED_K_ITERS - 1) / SHARED_K_ITERS;
  constexpr int cta_step_m_or_n = (threads_used * PACK_SIZE) / CTA_K;
  constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
  constexpr int threads_per_row = CTA_K / PACK_SIZE;
  constexpr int kSmemCol = CTA_K + SMEM_PAD_A;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
  int ld_col = (threadIdx.x % threads_per_row);
#pragma unroll
  for (int _global_iter = 0; _global_iter < partial_global_iters; ++_global_iter)
  {
    int global_iter = shared_iter_k * partial_global_iters + _global_iter;
    int ld_row = global_iter * cta_step_m_or_n + threadIdx.y * warp_step_m_or_n + (threadIdx.x / threads_per_row);
    int ld_col_swizzled = (ld_col ^ (ld_row) & 7) * PACK_SIZE;
    void *dst_ptr = (void *)(dst + ld_row * kSmemCol + ld_col_swizzled);
    uint4 *src_ptr = (uint4 *)(src + (ld_row + cta_offset_m) * global_ncols + ld_col * PACK_SIZE + global_iter_k * CTA_K + cta_offset_k); // cta_offset_m * global_ncols + global_iter * cta_step_m_or_n * global_ncols + threadIdx.y * warp_step_m_or_n * global_ncols + (threadIdx.x / threads_per_row) * global_ncols + global_iter_k * CTA_K + (threadIdx.x % threads_per_row) * PACK_SIZE);
    if constexpr (STAGES > 1)
    {
      uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
      cp_async_cg_A(addr, src_ptr, local_mask & (ld_row + cta_offset_m < global_nrows));
    }
    else
    {
      if (local_mask & (ld_row + cta_offset_m < global_nrows))
        *(uint4 *)dst_ptr = *src_ptr;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS, int STAGES>
__device__ __inline__ void global_to_share_one_stage_B(half *src, half *dst, int global_ncols, int cta_offset_m, int cta_offset_n, int cta_offset_k, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int threads_needed = (CTA_N / kInterleave * CTA_K) / PACK_SIZE / SHARED_K_ITERS;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = (CTA_N / kInterleave * CTA_K) / PACK_SIZE / threads_used;
  constexpr int partial_global_iters = (total_global_iters + SHARED_K_ITERS - 1) / SHARED_K_ITERS;
  constexpr int cta_step_m_or_n = (threads_used * PACK_SIZE) / CTA_K;
  constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
  constexpr int threads_per_row = CTA_K / PACK_SIZE;
  constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
#pragma unroll
  for (int _global_iter = 0; _global_iter < partial_global_iters; ++_global_iter)
  {
    int global_iter = shared_iter_k * partial_global_iters + _global_iter;

    int ld_row = global_iter * cta_step_m_or_n + threadIdx.y * warp_step_m_or_n + (threadIdx.x / threads_per_row);
    int ld_col = (threadIdx.x % threads_per_row);
    int ld_col_swizzled = ld_col ^ (ld_row % 2) & 7;
    void *dst_ptr = (void *)(dst + (ld_row * kSmemCol + ld_col_swizzled * PACK_SIZE));
    uint4 *src_ptr = (uint4 *)(src + global_iter_k * CTA_K + cta_offset_n / kInterleave * global_ncols + ld_row * global_ncols + ld_col * PACK_SIZE + cta_offset_k);
    if constexpr (STAGES > 1)
    {
      uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
      cp_async_cg_A(addr, src_ptr, local_mask);
    }
    else
    {
      if (local_mask)
        *(uint4 *)dst_ptr = *src_ptr;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int STAGES, int G>
__device__ __inline__ void global_to_share_one_stage_scales(half *src, half *dst, half *src_z, half *dst_z, int global_ncols, int cta_offset_m, int cta_offset_n, int cta_offset_k, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int LD_AMOUNT = (G >= CTA_K) ? CTA_N : CTA_N * CTA_K / G;
  constexpr int threads_needed = LD_AMOUNT / PACK_SIZE / 1;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = LD_AMOUNT / PACK_SIZE / threads_used;
  constexpr int threads_per_row = CTA_N / PACK_SIZE;
  constexpr int kSmemCol = CTA_N;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
  int g_idx = (cta_offset_k + global_iter_k * CTA_K) / G;

  void *dst_ptr = (void *)(dst + (threadIdx.x / threads_per_row) * kSmemCol + (threadIdx.x % threads_per_row) * PACK_SIZE);
  uint4 *src_ptr = (uint4 *)(src + g_idx * global_ncols + cta_offset_n + (threadIdx.x / threads_per_row) * global_ncols + (threadIdx.x % threads_per_row) * PACK_SIZE);
  void *dst_ptr_z = (void *)(dst_z + (threadIdx.x / threads_per_row) * kSmemCol + (threadIdx.x % threads_per_row) * PACK_SIZE);
  uint4 *src_ptr_z = (uint4 *)(src_z + g_idx * global_ncols + cta_offset_n + (threadIdx.x / threads_per_row) * global_ncols + (threadIdx.x % threads_per_row) * PACK_SIZE);
  if (STAGES > 1)
  {
    uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
    cp_async_cg_A(addr, src_ptr, local_mask);
    uint32_t addr_z = cast_smem_ptr_to_uint(dst_ptr_z);
    cp_async_cg_A(addr_z, src_ptr_z, local_mask);
  }
  else
  {
    if (local_mask)
    {
      *(uint4 *)dst_ptr = *src_ptr;
      *(uint4 *)dst_ptr_z = *src_ptr_z;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int STAGES, int shared_iters>
__device__ __inline__ void share_to_reg_one_stage_A(half *src, half *dst, int warp_offset_m, int warp_offset_n, int warp_offset_k, int k_0_1)
{
  constexpr int kSmemCol = CTA_K + SMEM_PAD_A;

  for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
  {

    int ld_row = warp_offset_m + shared_iter * OP_M + (threadIdx.x % 16);
    int ld_col = k_0_1 * 16 + (threadIdx.x / 16) * 8 + warp_offset_k;
    int ld_col_swizzled = ((ld_col / PACK_SIZE) ^ (ld_row) & 7) * PACK_SIZE;
    void *addr_ptr = (void *)(src + ld_row * kSmemCol + ld_col_swizzled);

    uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
    ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int STAGES, bool ldmatrix, int shared_iters, int G>
__device__ __inline__ void share_to_reg_one_stage_B(half *src, half *src_scales, half *src_zeros, half *dst, half *dst_fp16, int warp_offset_m, int warp_offset_n, int warp_offset_k, int k_0_1)
{
  constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
  int r0 = ((threadIdx.x / 8 / 2) * 8 + threadIdx.x % 8);
  int c0 = ((threadIdx.x / 8) % 2) * 8;
  int r = r0 / 4;
  int c = (r0 % 4) * 16 + c0;
  int c_swizzled = ((c / PACK_SIZE) ^ (r % 2) & 7) * PACK_SIZE;

  if constexpr (ldmatrix)
  {
#pragma unroll
    for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
    {
      void *addr_ptr = (void *)(src + warp_offset_n / kInterleave * kSmemCol + shared_iter * 16 / kInterleave * kSmemCol + k_0_1 * 16 + r * kSmemCol + c_swizzled + warp_offset_k);
      uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
      ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
    }
  }

#pragma unroll
  for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
  {
    half scale = src_scales[(warp_offset_k / G) * CTA_N + warp_offset_n + 16 * shared_iter + 8 * (k_0_1 % 2) + threadIdx.x / 4];
    half zero = src_zeros[(warp_offset_k / G) * CTA_N + warp_offset_n + 16 * shared_iter + 8 * (k_0_1 % 2) + threadIdx.x / 4];
    half2 scale2 = make_half2(scale, scale);
    half2 zero2 = make_half2(zero, zero);
    half2 loaded[4];

    dequantize_s4_to_fp16x2_fast(*reinterpret_cast<half2 *>(dst + (k_0_1 % 2) * 4 + (k_0_1 / 2 * 2) + shared_iter * 8), reinterpret_cast<uint4 *>(loaded));
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
      loaded[i] = __hfma2(loaded[i], scale2, zero2);
    }
    *reinterpret_cast<uint4 *>(dst_fp16 + shared_iter * 16 + 8 * (k_0_1 % 2)) = *reinterpret_cast<uint4 *>(loaded);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int G, int SPLITK>
__global__ void gemm_w4a16_T1(half *__restrict__ A, half *__restrict__ B, half *__restrict__ scales, half *__restrict__ zeros, half *__restrict__ C, int *__restrict__ semaphores, int M, int N, int K)
{
  constexpr int NUM_WARPS_MN = CTA_M / WARP_M * CTA_N / WARP_N;
  constexpr int NUM_WARPS = NUM_WARPS_MN * CTA_K / WARP_K;
  constexpr int CTA_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int CTA_SIZE_MN = NUM_WARPS_MN * WARP_SIZE;
  constexpr int SLICES = CTA_K / WARP_K;
  int num_blocks_n = (N + CTA_N - 1) / CTA_N;
  int num_blocks_m = (M + CTA_M - 1) / CTA_M;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % (num_blocks_m * num_blocks_n);
  int blockIdx_z = blockIdx.x / (num_blocks_m * num_blocks_n);
  const int log_tile = get_log_tile<1>((N + CTA_N - 1) / CTA_N);
  int blockIdx_m = blockIdx_y / (num_blocks_n >> log_tile);
  int blockIdx_n = blockIdx_y % (num_blocks_n >> log_tile);
  const uint2 block_idx_mapping = get_block_idx_mapping(blockIdx_m, blockIdx_n, log_tile);
  blockIdx_m = block_idx_mapping.x;
  blockIdx_n = block_idx_mapping.y;

  float C_warp[CTA_M * CTA_N / CTA_SIZE_MN];
  constexpr int kSmemPadKA = CTA_K + SMEM_PAD_A;
  constexpr int kSmemPadKB = CTA_K + SMEM_PAD_B;
  constexpr int kSmemSizeAPerStage = CTA_M * kSmemPadKA;
  constexpr int kSmemSizeBPerStage = CTA_N / kInterleave * kSmemPadKB;
  constexpr int kSmemSizeA = kSmemSizeAPerStage * STAGES;
  constexpr int kSmemSizeB = kSmemSizeBPerStage * STAGES;
  constexpr int scales_load_interval = G >= CTA_K ? G / CTA_K : 1;
  constexpr int scales_per_load = G < CTA_K ? CTA_K / G : 1;
  constexpr int kSmemSizeScales = CTA_N * STAGES / scales_load_interval * scales_per_load;
  constexpr int kSmemSizeZeros = CTA_N * STAGES / scales_load_interval * scales_per_load;
  extern __shared__ half mem_shared[];
  half *A_shared = mem_shared;
  half *B_shared = mem_shared + kSmemSizeA;
  half *scales_shared = mem_shared + kSmemSizeA + kSmemSizeB;
  half *zeros_shared = mem_shared + kSmemSizeA + kSmemSizeB + kSmemSizeScales;
  float *C_shared = reinterpret_cast<float *>(mem_shared);
  half A_shared_warp_[2][WARP_M * INTRIN_K /
                         WARP_SIZE];
  half B_shared_warp_[2][WARP_N * 32 /
                         WARP_SIZE];
  half B_shared_warp_tmp_[2][WARP_N * 16 /
                             WARP_SIZE];
  int cta_offset_m = blockIdx_m * CTA_M;
  int cta_offset_n = blockIdx_n * CTA_N;
  int cta_offset_k = blockIdx_z * (K / SPLITK);
  int warp_mn = threadIdx.y % NUM_WARPS_MN;
  int slice_id = threadIdx.y / NUM_WARPS_MN;
  int warp_offset_n = (warp_mn % (CTA_N / WARP_N)) * WARP_N;
  int warp_offset_m = (warp_mn / (CTA_N / WARP_N)) * WARP_M;
  int warp_offset_k = slice_id * WARP_K;

  for (int i = 0; i < CTA_M * CTA_N / CTA_SIZE_MN; i++)
    C_warp[i] = 0.0;

  int gemm_iters = (K + CTA_K - 1) / CTA_K / SPLITK;
  int k_0_0_ld = 0;
  int k_0_0 = 0;
  constexpr int prologue_stages = STAGES == 1 ? 1 : STAGES - 1;
#pragma unroll
  for (k_0_0_ld = 0; k_0_0_ld < prologue_stages; ++k_0_0_ld)
  {
    global_to_share_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(A, A_shared + k_0_0_ld * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, cta_offset_k, k_0_0_ld, 0, true);
    global_to_share_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(B, B_shared + k_0_0_ld * kSmemSizeBPerStage, K, cta_offset_m, cta_offset_n, cta_offset_k, k_0_0_ld, 0, true);
    global_to_share_one_stage_scales<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
        scales, scales_shared + (k_0_0_ld / scales_load_interval * scales_per_load) * CTA_N,
        zeros, zeros_shared + (k_0_0_ld / scales_load_interval * scales_per_load) * CTA_N,
        N, cta_offset_m, cta_offset_n, cta_offset_k,
        k_0_0_ld, 0, k_0_0_ld < gemm_iters && k_0_0_ld % scales_load_interval == 0);
    if constexpr (STAGES > 1)
      __pipeline_commit();
  }
  if constexpr (STAGES > 1)
    __pipeline_wait_prior(STAGES - 2);
  __syncthreads();

  share_to_reg_one_stage_A<CTA_M, CTA_N, CTA_K, STAGES, WARP_M / INTRIN_M>(A_shared, A_shared_warp_[0], warp_offset_m, warp_offset_n, warp_offset_k, 0);
  share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(B_shared, scales_shared, zeros_shared, B_shared_warp_tmp_[0], B_shared_warp_[0], warp_offset_m, warp_offset_n, warp_offset_k, 0);
  constexpr int SHARED_K_ITERS = WARP_K / INTRIN_K;

  for (; k_0_0 < gemm_iters; ++k_0_0, ++k_0_0_ld)
  {
    int ld_stage = k_0_0_ld % STAGES;
    int compute_stage = k_0_0 % STAGES;
    half *A_shared_this_compute_stage;
    half *B_shared_this_compute_stage;
    half *scales_shared_this_compute_stage;
    half *zeros_shared_this_compute_stage;

#pragma unroll
    for (int iter_k = 0; iter_k < SHARED_K_ITERS; ++iter_k)
    {
      A_shared_this_compute_stage = A_shared + compute_stage * kSmemSizeAPerStage;
      B_shared_this_compute_stage = B_shared + compute_stage * kSmemSizeBPerStage;
      scales_shared_this_compute_stage = scales_shared + (compute_stage / scales_load_interval * scales_per_load) * CTA_N;
      zeros_shared_this_compute_stage = zeros_shared + (compute_stage / scales_load_interval * scales_per_load) * CTA_N;
      share_to_reg_one_stage_A<CTA_M, CTA_N, CTA_K, STAGES, WARP_M / INTRIN_M>(A_shared_this_compute_stage, A_shared_warp_[(iter_k + 1) % 2], warp_offset_m, warp_offset_n, warp_offset_k, (iter_k + 1) % SHARED_K_ITERS);
      if ((iter_k + 1) % kInterleave == 0)
      {
        if (compute_stage % 2 == 1)
        {
          share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[1], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, warp_offset_k, (iter_k + 1) % SHARED_K_ITERS);
        }
        else
        {
          share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[0], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, warp_offset_k, (iter_k + 1) % SHARED_K_ITERS);
        }
      }
      else
      {
        if (compute_stage % 2 == 1)
        {
          share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, STAGES, false, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[1], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, warp_offset_k, (iter_k + 1) % SHARED_K_ITERS);
        }
        else
        {
          share_to_reg_one_stage_B<CTA_M, CTA_N, CTA_K, STAGES, false, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[0], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, warp_offset_k, (iter_k + 1) % SHARED_K_ITERS);
        }
      }
      half *A_shared_warp = A_shared_warp_[iter_k % 2];
      half *B_shared_warp = B_shared_warp_[(iter_k / 2) % 2];

      for (int i_0_3 = 0; i_0_3 < WARP_M / INTRIN_M; ++i_0_3)
      {
        for (int j_0_4 = 0; j_0_4 < WARP_N / INTRIN_N; ++j_0_4)
        {
          mma_m16n8k16(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8, A_shared_warp + i_0_3 * 8, B_shared_warp + j_0_4 * 16 + (iter_k % 2) * 4);
          mma_m16n8k16(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8 + 4, A_shared_warp + i_0_3 * 8, B_shared_warp + j_0_4 * 16 + (iter_k % 2) * 4 + 8);
        }
      }

      if (iter_k < WARP_K / INTRIN_K - 1)
      {
        if constexpr (STAGES == 1)
          __syncthreads();
        global_to_share_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(A, A_shared + ld_stage * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, cta_offset_k, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(B, B_shared + ld_stage * kSmemSizeBPerStage, K, cta_offset_m, cta_offset_n, cta_offset_k, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters);
      }

      if (iter_k == WARP_K / INTRIN_K - 2)
      {
        if constexpr (STAGES == 1 && WARP_K / INTRIN_K > 2)
        {
          __syncthreads();
        }
        global_to_share_one_stage_A<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(A, A_shared + ld_stage * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, cta_offset_k, k_0_0_ld, iter_k + 1, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_B<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(B, B_shared + ld_stage * kSmemSizeBPerStage, K, cta_offset_m, cta_offset_n, cta_offset_k, k_0_0_ld, iter_k + 1, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_scales<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
            scales, scales_shared + (ld_stage / scales_load_interval * scales_per_load) * CTA_N,
            zeros, zeros_shared + (ld_stage / scales_load_interval * scales_per_load) * CTA_N,
            N, cta_offset_m, cta_offset_n, cta_offset_k,
            k_0_0_ld, iter_k, k_0_0_ld < gemm_iters && k_0_0_ld % scales_load_interval == 0);
        if constexpr (STAGES > 1)
        {
          __pipeline_commit();
          __pipeline_wait_prior(STAGES - 2);
        }
        compute_stage = (k_0_0 + 1) % STAGES;
        __syncthreads();
      }
    }
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
  if constexpr (SLICES > 1)
  {
#pragma unroll
    for (int z = 0; z < SLICES; ++z)
    {
      if (slice_id == z)
      {
#pragma unroll
        for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
        {
#pragma unroll
          for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
          {
#pragma unroll
            for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; ++local_id)
            {
              if (z > 0)
              {
                C_warp[ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id] += C_shared[warp_offset_m * CTA_N + ax0_0_1 * OP_M * CTA_N + warp_offset_n + ax1_0_1 * 16 + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4)) * CTA_N + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2];
              }
              C_shared[warp_offset_m * CTA_N + ax0_0_1 * OP_M * CTA_N + warp_offset_n + ax1_0_1 * 16 + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4)) * CTA_N + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2] = C_warp[ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id];
            };
          }
        }
      }
      __syncthreads();
    }
    if (slice_id == 0)
    {
#pragma unroll
      for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
      {
#pragma unroll
        for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
        {
#pragma unroll
          for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; ++local_id)
          {
            C_warp[ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id] = C_shared[warp_offset_m * CTA_N + ax0_0_1 * OP_M * CTA_N + warp_offset_n + ax1_0_1 * 16 + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4)) * CTA_N + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2];
          };
        }
      }
    }
  }

  if (slice_id == 0)
  {
    Semaphore semaphore(semaphores + blockIdx_y, threadIdx.x);

    if constexpr (SPLITK > 1)
    {
      semaphore.fetch();
    }

    if (blockIdx_z != 0)
    {
      semaphore.wait(blockIdx_z);
      for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
      {
        for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
        {
          for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; local_id += 2)
          {
            int write_row = cta_offset_m + warp_offset_m + ax0_0_1 * OP_M + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4));

            if (write_row < M)
            {
              half2 *existing_psum_ptr = reinterpret_cast<half2 *>(
                  C + write_row * N +
                  cta_offset_n + warp_offset_n + ax1_0_1 * 16 +
                  (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2);

              *existing_psum_ptr = __hadd2(*existing_psum_ptr,
                                           __float22half2_rn(*reinterpret_cast<float2 *>(C_warp + ax0_0_1 * WARP_N / INTRIN_N * 8 +
                                                                                         ax1_0_1 * 8 + local_id)));
            }
          };
        }
      }
    }
    else
    {
      for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
      {
        for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
        {
          for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; local_id += 2)
          {
            int write_row = cta_offset_m + warp_offset_m + ax0_0_1 * OP_M + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4));
            if (write_row < M)
            {
              *reinterpret_cast<half2 *>(
                  C + write_row * N +
                  cta_offset_n + warp_offset_n + ax1_0_1 * 16 +
                  (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2) =
                  __float22half2_rn(*reinterpret_cast<float2 *>(C_warp + ax0_0_1 * WARP_N / INTRIN_N * 8 +
                                                                ax1_0_1 * 8 + local_id));
            }
          };
        }
      }
    }

    if constexpr (SPLITK > 1)
    {

      int lock = 0;
      if (SPLITK == blockIdx_z + 1)
      {

        lock = 0;
      }
      else
      {
        lock = blockIdx_z + 1;
      }
      semaphore.release(lock);
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS, int STAGES>
__device__ __inline__ void global_to_share_one_stage_A_T2(half *src, half *dst, int global_nrows, int global_ncols, int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int threads_needed = (CTA_M * CTA_K) / PACK_SIZE / SHARED_K_ITERS;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = (CTA_M * CTA_K) / PACK_SIZE / threads_used;
  constexpr int partial_global_iters = (total_global_iters + SHARED_K_ITERS - 1) / SHARED_K_ITERS;
  constexpr int cta_step_m_or_n = (threads_used * PACK_SIZE) / CTA_K;
  constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
  constexpr int threads_per_row = CTA_K / PACK_SIZE;
  constexpr int kSmemCol = CTA_K + SMEM_PAD_A;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
  int ld_col = (threadIdx.x % threads_per_row);
#pragma unroll
  for (int _global_iter = 0; _global_iter < partial_global_iters; ++_global_iter)
  {
    int global_iter = shared_iter_k * partial_global_iters + _global_iter;
    int ld_row = global_iter * cta_step_m_or_n + threadIdx.y * warp_step_m_or_n + (threadIdx.x / threads_per_row);
    int ld_col_swizzled = (ld_col ^ (ld_row) & 7) * PACK_SIZE;
    void *dst_ptr = (void *)(dst + ld_row * kSmemCol + ld_col_swizzled);
    uint4 *src_ptr = (uint4 *)(src + (ld_row + cta_offset_m) * global_ncols + ld_col * PACK_SIZE + global_iter_k * CTA_K); // cta_offset_m * global_ncols + global_iter * cta_step_m_or_n * global_ncols + threadIdx.y * warp_step_m_or_n * global_ncols + (threadIdx.x / threads_per_row) * global_ncols + global_iter_k * CTA_K + (threadIdx.x % threads_per_row) * PACK_SIZE);
    if constexpr (STAGES > 1)
    {
      uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
      cp_async_cg_A(addr, src_ptr, local_mask & (ld_row + cta_offset_m < global_nrows));
    }
    else
    {
      if (local_mask & (ld_row + cta_offset_m < global_nrows))
        *(uint4 *)dst_ptr = *src_ptr;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS, int STAGES>
__device__ __inline__ void global_to_share_one_stage_B_T2(half *src, half *dst, int global_ncols, int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int threads_needed = (CTA_N / kInterleave * CTA_K) / PACK_SIZE / SHARED_K_ITERS;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = (CTA_N / kInterleave * CTA_K) / PACK_SIZE / threads_used;
  constexpr int partial_global_iters = (total_global_iters + SHARED_K_ITERS - 1) / SHARED_K_ITERS;
  constexpr int cta_step_m_or_n = (threads_used * PACK_SIZE) / CTA_K;
  constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
  constexpr int threads_per_row = CTA_K / PACK_SIZE;
  constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
#pragma unroll
  for (int _global_iter = 0; _global_iter < partial_global_iters; ++_global_iter)
  {
    int global_iter = shared_iter_k * partial_global_iters + _global_iter;

    int ld_row = global_iter * cta_step_m_or_n + threadIdx.y * warp_step_m_or_n + (threadIdx.x / threads_per_row);
    int ld_col = (threadIdx.x % threads_per_row);
    int ld_col_swizzled = ld_col ^ (ld_row % 2) & 7;
    void *dst_ptr = (void *)(dst + (ld_row * kSmemCol + ld_col_swizzled * PACK_SIZE));
    uint4 *src_ptr = (uint4 *)(src + global_iter_k * CTA_K + cta_offset_n / kInterleave * global_ncols + ld_row * global_ncols + ld_col * PACK_SIZE);
    if constexpr (STAGES > 1)
    {
      uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
      cp_async_cg_A(addr, src_ptr, local_mask);
    }
    else
    {
      if (local_mask)
        *(uint4 *)dst_ptr = *src_ptr;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int STAGES, int G>
__device__ __inline__ void global_to_share_one_stage_scales_T2(half *src, half *dst, half *src_z, half *dst_z, int global_ncols, int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
  constexpr int threads_needed = CTA_N / PACK_SIZE / 1;
  constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
  constexpr int total_global_iters = CTA_N / PACK_SIZE / threads_used;
  constexpr int threads_per_row = CTA_N / PACK_SIZE;
  constexpr int kSmemCol = CTA_N;
  bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
  int g_idx = global_iter_k * CTA_K / G;

  void *dst_ptr = (void *)(dst + (threadIdx.x % threads_per_row) * PACK_SIZE);
  uint4 *src_ptr = (uint4 *)(src + g_idx * global_ncols + cta_offset_n + (threadIdx.x % threads_per_row) * PACK_SIZE);
  void *dst_ptr_z = (void *)(dst_z + (threadIdx.x % threads_per_row) * PACK_SIZE);
  uint4 *src_ptr_z = (uint4 *)(src_z + g_idx * global_ncols + cta_offset_n + (threadIdx.x % threads_per_row) * PACK_SIZE);
  if (STAGES > 1)
  {
    uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
    cp_async_cg_A(addr, src_ptr, local_mask);
    uint32_t addr_z = cast_smem_ptr_to_uint(dst_ptr_z);
    cp_async_cg_A(addr_z, src_ptr_z, local_mask);
  }
  else
  {
    if (local_mask)
    {
      *(uint4 *)dst_ptr = *src_ptr;
      *(uint4 *)dst_ptr_z = *src_ptr_z;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int STAGES, int shared_iters>
__device__ __inline__ void share_to_reg_one_stage_A_T2(half *src, half *dst, int warp_offset_m, int warp_offset_n, int k_0_1)
{
  constexpr int kSmemCol = CTA_K + SMEM_PAD_A;

  for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
  {

    int ld_row = warp_offset_m + shared_iter * OP_M + (threadIdx.x % 16);
    int ld_col = k_0_1 * 16 + (threadIdx.x / 16) * 8;
    int ld_col_swizzled = ((ld_col / PACK_SIZE) ^ (ld_row) & 7) * PACK_SIZE;
    void *addr_ptr = (void *)(src + ld_row * kSmemCol + ld_col_swizzled);

    uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
    ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int STAGES, bool ldmatrix, int shared_iters, int G>
__device__ __inline__ void share_to_reg_one_stage_B_T2(half *src, half *src_scales, half *src_zeros, half *dst, half *dst_fp16, int warp_offset_m, int warp_offset_n, int k_0_1)
{
  constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
  int r0 = ((threadIdx.x / 8 / 2) * 8 + threadIdx.x % 8);
  int c0 = ((threadIdx.x / 8) % 2) * 8;
  int r = r0 / 4;
  int c = (r0 % 4) * 16 + c0;
  int c_swizzled = ((c / PACK_SIZE) ^ (r % 2) & 7) * PACK_SIZE;

  if constexpr (ldmatrix)
  {
#pragma unroll
    for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
    {
      void *addr_ptr = (void *)(src + warp_offset_n / kInterleave * kSmemCol + shared_iter * 16 / kInterleave * kSmemCol + k_0_1 * 16 + r * kSmemCol + c_swizzled);
      uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
      ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
    }
  }

#pragma unroll
  for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
  {
    half scale = src_scales[warp_offset_n + 16 * shared_iter + 8 * (k_0_1 % 2) + threadIdx.x / 4];
    half zero = src_zeros[warp_offset_n + 16 * shared_iter + 8 * (k_0_1 % 2) + threadIdx.x / 4];
    half2 scale2 = make_half2(scale, scale);
    half2 zero2 = make_half2(zero, zero);
    half2 loaded[4];
    dequantize_s4_to_fp16x2_fast(*reinterpret_cast<half2 *>(dst + (k_0_1 % 2) * 4 + (k_0_1 / 2 * 2) + shared_iter * 8), reinterpret_cast<uint4 *>(loaded));
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
      loaded[i] = __hfma2(loaded[i], scale2, zero2);
    }
    *reinterpret_cast<uint4 *>(dst_fp16 + shared_iter * 16 + 8 * (k_0_1 % 2)) = *reinterpret_cast<uint4 *>(loaded);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int G>
__global__ void gemm_w4a16_T2(half *__restrict__ A, half *__restrict__ B, half *__restrict__ scales, half *__restrict__ zeros, half *__restrict__ C, int M, int N, int K)
{
  constexpr int NUM_WARPS = CTA_M / WARP_M * CTA_N / WARP_N;
  constexpr int CTA_SIZE = NUM_WARPS * WARP_SIZE;
  int num_blocks_n = (N + CTA_N - 1) / CTA_N;
  int num_blocks_m = (M + CTA_M - 1) / CTA_M;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % (num_blocks_m * num_blocks_n);
  int blockIdx_z = blockIdx.x / (num_blocks_m * num_blocks_n);
  const int log_tile = get_log_tile<1>((N + CTA_N - 1) / CTA_N);
  int blockIdx_m = blockIdx_y / (num_blocks_n >> log_tile);
  int blockIdx_n = blockIdx_y % (num_blocks_n >> log_tile);
  const uint2 block_idx_mapping = get_block_idx_mapping(blockIdx_m, blockIdx_n, log_tile);
  blockIdx_m = block_idx_mapping.x;
  blockIdx_n = block_idx_mapping.y;

  float C_warp[CTA_M * CTA_N / CTA_SIZE];
  constexpr int kSmemPadKA = CTA_K + SMEM_PAD_A;
  constexpr int kSmemPadKB = CTA_K + SMEM_PAD_B;
  constexpr int kSmemSizeAPerStage = CTA_M * kSmemPadKA;
  constexpr int kSmemSizeBPerStage = CTA_N / kInterleave * kSmemPadKB;
  constexpr int kSmemSizeA = kSmemSizeAPerStage * STAGES;
  constexpr int kSmemSizeB = kSmemSizeBPerStage * STAGES;
  constexpr int kSmemSizeScales = CTA_N * STAGES / 2;
  constexpr int kSmemSizeZeros = CTA_N * STAGES / 2;
  constexpr int scales_load_interval = G / CTA_K;
  extern __shared__ half mem_shared[];
  half *A_shared = mem_shared;
  half *B_shared = mem_shared + kSmemSizeA;
  half *scales_shared = mem_shared + kSmemSizeA + kSmemSizeB;
  half *zeros_shared = mem_shared + kSmemSizeA + kSmemSizeB + kSmemSizeScales;
  half A_shared_warp_[2][WARP_M * INTRIN_K /
                         WARP_SIZE];
  half B_shared_warp_[2][WARP_N * 32 /
                         WARP_SIZE];
  half B_shared_warp_tmp_[2][WARP_N * 16 /
                             WARP_SIZE];
  int cta_offset_m = blockIdx_m * CTA_M;
  int cta_offset_n = blockIdx_n * CTA_N;
  int warp_offset_m = (threadIdx.y % (CTA_M / WARP_M)) * WARP_M;
  int warp_offset_n = (threadIdx.y / (CTA_M / WARP_M)) * WARP_N;

  for (int i = 0; i < CTA_M * CTA_N / CTA_SIZE; i++)
    C_warp[i] = 0.0;

  int gemm_iters = (K + CTA_K - 1) / CTA_K;
  int k_0_0_ld = 0;
  int k_0_0 = 0;
  constexpr int prologue_stages = STAGES == 1 ? 1 : STAGES - 1;
#pragma unroll
  for (k_0_0_ld = 0; k_0_0_ld < prologue_stages; ++k_0_0_ld)
  {
    global_to_share_one_stage_A_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(A, A_shared + k_0_0_ld * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, k_0_0_ld, 0, true);
    global_to_share_one_stage_B_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(B, B_shared + k_0_0_ld * kSmemSizeBPerStage, K, cta_offset_m, cta_offset_n, k_0_0_ld, 0, true);
    global_to_share_one_stage_scales_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
        scales, scales_shared + (k_0_0_ld / scales_load_interval) * CTA_N,
        zeros, zeros_shared + (k_0_0_ld / scales_load_interval) * CTA_N,
        N, cta_offset_m, cta_offset_n, k_0_0_ld, 0, k_0_0_ld < gemm_iters && k_0_0_ld % scales_load_interval == 0);
    if constexpr (STAGES > 1)
      __pipeline_commit();
  }
  if constexpr (STAGES > 1)
    __pipeline_wait_prior(STAGES - 2);
  __syncthreads();

  share_to_reg_one_stage_A_T2<CTA_M, CTA_N, CTA_K, STAGES, WARP_M / INTRIN_M>(A_shared, A_shared_warp_[0], warp_offset_m, warp_offset_n, 0);
  share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(B_shared, scales_shared, zeros_shared, B_shared_warp_tmp_[0], B_shared_warp_[0], warp_offset_m, warp_offset_n, 0);
  constexpr int SHARED_K_ITERS = WARP_K / INTRIN_K;

  for (; k_0_0 < gemm_iters; ++k_0_0, ++k_0_0_ld)
  {
    int ld_stage = k_0_0_ld % STAGES;
    int compute_stage = k_0_0 % STAGES;
    half *A_shared_this_compute_stage;
    half *B_shared_this_compute_stage;
    half *scales_shared_this_compute_stage;
    half *zeros_shared_this_compute_stage;

    for (int iter_k = 0; iter_k < SHARED_K_ITERS; ++iter_k)
    {
      A_shared_this_compute_stage = A_shared + compute_stage * kSmemSizeAPerStage;
      B_shared_this_compute_stage = B_shared + compute_stage * kSmemSizeBPerStage;
      scales_shared_this_compute_stage = scales_shared + (compute_stage / scales_load_interval) * CTA_N;
      zeros_shared_this_compute_stage = zeros_shared + (compute_stage / scales_load_interval) * CTA_N;
      share_to_reg_one_stage_A_T2<CTA_M, CTA_N, CTA_K, STAGES, WARP_M / INTRIN_M>(A_shared_this_compute_stage, A_shared_warp_[(iter_k + 1) % 2], warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS);
      if ((iter_k + 1) % kInterleave == 0)
      {
        if (compute_stage % 2 == 1)
        {
          share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[1], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS);
        }
        else
        {
          share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[0], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS);
        }
      }
      else
      {
        if (compute_stage % 2 == 1)
        {
          share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, false, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[1], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS);
        }
        else
        {
          share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, false, WARP_N / INTRIN_N, G>(
              B_shared_this_compute_stage, scales_shared_this_compute_stage, zeros_shared_this_compute_stage,
              B_shared_warp_tmp_[0], B_shared_warp_[((iter_k + 1) / 2) % 2],
              warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS);
        }
      }
      __syncthreads();
      half *A_shared_warp = A_shared_warp_[iter_k % 2];
      half *B_shared_warp = B_shared_warp_[(iter_k / 2) % 2];
      for (int i_0_3 = 0; i_0_3 < WARP_M / INTRIN_M; ++i_0_3)
      {
        for (int j_0_4 = 0; j_0_4 < WARP_N / INTRIN_N; ++j_0_4)
        {
          mma_m16n8k16(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8, A_shared_warp + i_0_3 * 8, B_shared_warp + j_0_4 * 16 + (iter_k % 2) * 4);
          mma_m16n8k16(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8 + 4, A_shared_warp + i_0_3 * 8, B_shared_warp + j_0_4 * 16 + (iter_k % 2) * 4 + 8);
        }
      }

      if (iter_k < WARP_K / INTRIN_K - 1)
      {
        if constexpr (STAGES == 1)
          __syncthreads();
        global_to_share_one_stage_A_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(A, A_shared + ld_stage * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_B_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(B, B_shared + ld_stage * kSmemSizeBPerStage, K, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters);
      }

      if (iter_k == WARP_K / INTRIN_K - 2)
      {
        if constexpr (STAGES == 1 && WARP_K / INTRIN_K > 2)
        {
          __syncthreads();
        }
        global_to_share_one_stage_A_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(A, A_shared + ld_stage * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k + 1, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_B_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(B, B_shared + ld_stage * kSmemSizeBPerStage, K, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k + 1, k_0_0_ld < gemm_iters);
        global_to_share_one_stage_scales_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(
            scales, scales_shared + (ld_stage / scales_load_interval) * CTA_N,
            zeros, zeros_shared + (ld_stage / scales_load_interval) * CTA_N,
            N, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k, k_0_0_ld < gemm_iters && k_0_0_ld % scales_load_interval == 0);
        if constexpr (STAGES > 1)
        {
          __pipeline_commit();
          __pipeline_wait_prior(STAGES - 2);
        }
        compute_stage = (k_0_0 + 1) % STAGES;
        __syncthreads();
      }
    }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
  {
    for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
    {
      for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; local_id += 2)
      {
        int write_row = cta_offset_m + warp_offset_m + ax0_0_1 * OP_M + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4));
        if (write_row < M)
        {
          *reinterpret_cast<half2 *>(
              C + write_row * N +
              cta_offset_n + warp_offset_n + ax1_0_1 * 16 +
              (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2) =
              __float22half2_rn(*reinterpret_cast<float2 *>(C_warp + ax0_0_1 * WARP_N / INTRIN_N * 8 +
                                                            ax1_0_1 * 8 + local_id));
        }
      };
    }
  }
}

} // namespace awq
} // namespace vllm

torch::Tensor awq_gemm_fast(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros)
{
  std::vector<int64_t> output_shape = _in_feats.sizes().vec();
  output_shape.back() = _kernel.size(0) * kInterleave;
  int num_in_feats = _in_feats.numel() / _in_feats.size(-1);
  int num_in_channels = _in_feats.size(-1);
  auto in_feats = reinterpret_cast<half *>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<half *>(_kernel.data_ptr<int16_t>());
  auto scales = reinterpret_cast<half *>(_scales.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<half *>(_zeros.data_ptr<at::Half>());
  auto options =
      torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt32).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty(output_shape, options);
  int num_out_feats = _out_feats.numel() / _out_feats.size(-1);
  int num_out_channels = _out_feats.size(-1);
  auto out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());

  if (num_out_feats <= 32)
  {
    constexpr int G = 128;
    constexpr int CTA_M = 16;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 128;
    constexpr int WARP_M = 16;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int SPLITK = 2;
    constexpr int STAGES = 4;
    KERNEL_LAUNCH_CODE
  }
  else if (num_out_feats <= 64)
  {

    constexpr int G = 128;
    constexpr int CTA_M = 16;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 128;
    constexpr int WARP_M = 16;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int SPLITK = 1;
    constexpr int STAGES = 3;
    KERNEL_LAUNCH_CODE
  }
  else if (num_out_feats <= 128)
  {
    constexpr int G = 128;
    constexpr int CTA_M = 32;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 128;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int SPLITK = 1;
    constexpr int STAGES = 4;
    KERNEL_LAUNCH_CODE
  }
  else if (num_out_feats <= 192)
  {
    constexpr int G = 128;
    constexpr int CTA_M = 64;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int SPLITK = 1;
    constexpr int STAGES = 4;
    KERNEL_LAUNCH_CODE
  }
  else
  {
    constexpr int G = 128;
    constexpr int CTA_M = 64;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 4;

    constexpr int NUM_WARPS = (CTA_M / WARP_M) * (CTA_N / WARP_N);
    constexpr int kSmemByteSize = (CTA_M * (CTA_K + SMEM_PAD_A) + CTA_N * (CTA_K + SMEM_PAD_B) / kInterleave + CTA_N) * STAGES * sizeof(half);
    if (kSmemByteSize >= 99 * 1024)
    {
      printf("This kernel requires %d Bytes of shared memory, which exceeds device limit.\n", kSmemByteSize);
      return _out_feats;
    }
    int j_factors1 = num_out_channels / CTA_N / 1;
    dim3 num_blocks((num_out_feats + CTA_M - 1) / CTA_M * j_factors1);
    dim3 threads_per_block(WARP_SIZE, NUM_WARPS);
    auto kernel_func = vllm::awq::gemm_w4a16_T2<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G>;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemByteSize);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kernel_func<<<num_blocks, threads_per_block, kSmemByteSize, stream>>>(
        in_feats, kernel, scales, zeros, out_feats, num_in_feats, num_out_channels, num_in_channels);
  }

  return _out_feats;
}