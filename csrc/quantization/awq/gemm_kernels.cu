/*
Adapted from https://github.com/mit-han-lab/llm-awq
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "dequantize.cuh" 
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vllm {
namespace awq {

template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32_fp16(int G, int split_k_iters,
                                        half* __restrict__ A, int* __restrict__ B,
                                        half* __restrict__ scaling_factors,
                                        int* __restrict__ zeros, int M, int IC,
                                        int OC, half* __restrict__ C) {
  // Only support matrix n = 64 or 128
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (N + 8)];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  // Warp-level registers for MMA operands
  unsigned A_shared_warp[4]; // 4 * 32-bit = 16 bytes = 8 halfs. For m16n8k8, we need 2 regs. For m16n8k16, 4 regs.
  unsigned B_shared_warp[2]; // For m16n8k16, B is kx8, needs 2 regs.

  for (int i = 0; i < 32; ++i) {
    C_warp[i] = 0.0f;
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;

  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;

  half* A_ptr =
      A +
      (((int)blockIdx_y) / j_factors1 * 16 +
       (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
          IC +
      (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8)) * 1;

  half* A_shared_ptr = A_shared +
                       ((int)threadIdx.y) * row_stride_warp * (32 + 8) +
                       (((int)threadIdx.x) / (32 / 8)) * (32 + 8) +
                       (((int)threadIdx.x) % (32 / 8)) * 8;

  half* B_shared_ptr = B_shared +
                       ((int)threadIdx.y) * (row_stride / 2) * (N + 8) +
                       (((int)threadIdx.x) / (N / 8)) * (N + 8) +
                       (((int)threadIdx.x) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)threadIdx.x) % (N / 8);

  half* scaling_factors_ptr = scaling_factors +
                              (((int)blockIdx_y) % j_factors1) * N +
                              (((int)threadIdx.x) % (N / 8)) * 8;

  half* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC +
      (((int)blockIdx_y) % j_factors1) * N + ((int)threadIdx.y) * (N / 2) +
      (((int)threadIdx.x) % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;

  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();

    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));

    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      unsigned int A_addr;
      void* A_s_ptr = &A_shared[k_0_1 * 16];
      __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(A_addr) : "l"(A_s_ptr));
      A_addr += (threadIdx.x % 16) * 4 + (threadIdx.x / 16) * 16 * (32 + 8) * sizeof(half);

      __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(A_shared_warp[0]), "=r"(A_shared_warp[1]), "=r"(A_shared_warp[2]), "=r"(A_shared_warp[3])
          : "r"(A_addr));

      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
        unsigned int B_addr;
        void* B_s_ptr = &B_shared[k_0_1 * 16 + j_0_4 * 32 * (32 + 8)];
         __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(B_addr) : "l"(B_s_ptr));
        B_addr += (threadIdx.x % 16) * 4 + (threadIdx.x / 16) * 8 * (N + 8) * sizeof(half);

        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
             : "=r"(B_shared_warp[0]), "=r"(B_shared_warp[1])
             : "r"(B_addr));
        
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 // Turing (SM75)
        // TODO: add m16n8k8 logic here if needed
#else // Ampere+ (SM80+)
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(((float*)(C_warp + j_0_4 * 8))[0]), "=f"(((float*)(C_warp + j_0_4 * 8))[1]),
              "=f"(((float*)(C_warp + j_0_4 * 8))[2]), "=f"(((float*)(C_warp + j_0_4 * 8))[3])
            : "r"(A_shared_warp[0]), "r"(A_shared_warp[1]), "r"(A_shared_warp[2]), "r"(A_shared_warp[3]),
              "r"(B_shared_warp[0]), "r"(B_shared_warp[1]),
              "f"(((float*)(C_warp + j_0_4 * 8))[0]), "f"(((float*)(C_warp + j_0_4 * 8))[1]),
              "f"(((float*)(C_warp + j_0_4 * 8))[2]), "f"(((float*)(C_warp + j_0_4 * 8))[3]));
        
        // ... second MMA for the other half of the N dimension ...
#endif
      }
    }
  }

  for (int ax1_0_1 = 0; ax1_0_1 < (N / 32); ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
#endif
}

__global__ void __launch_bounds__(64)
    dequantize_weights_fp16(int* __restrict__ B, half* __restrict__ scaling_factors,
                            int* __restrict__ zeros, half* __restrict__ C, int G) {
  static constexpr uint32_t ZERO = 0x0;
  
  int N = gridDim.x * blockDim.x;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Each thread dequantizes 8 values
  half* C_ptr = C + 8 * (col + row * N);
  int* B_ptr = B + (col + row * N);
  int* zeros_ptr = zeros + (col + (row / G) * N);
  half* scaling_factors_ptr = scaling_factors + 8 * (col + (row / G) * N);

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr);
  uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
  uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr);

  uint32_t B_loaded = *(uint32_t*)B_ptr;
  uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

  *(uint4*)C_ptr = B_loaded_fp16;
}

template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32_bf16(int G, int split_k_iters,
                                         __nv_bfloat16* __restrict__ A, int* __restrict__ B,
                                         __nv_bfloat16* __restrict__ scaling_factors,
                                         int* __restrict__ zeros, int M, int IC,
                                         int OC, __nv_bfloat16* __restrict__ C) {
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false); // BF16 kernels require Ampere (SM80) or newer
#else
  static constexpr uint32_t ZERO_U32 = 0;
  float C_warp[32];
  __shared__ __nv_bfloat16 A_shared[16 * (32 + 8)];
  __shared__ __nv_bfloat16 B_shared[32 * (N + 8)];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  unsigned A_shared_warp[4];
  unsigned B_shared_warp[2];

  for (int i = 0; i < 32; ++i) {
    C_warp[i] = 0.0f;
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;

  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;

  __nv_bfloat16* A_ptr =
      A +
      (((int)blockIdx_y) / j_factors1 * 16 +
       (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
          IC +
      (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8)) * 1;

  __nv_bfloat16* A_shared_ptr = A_shared +
                              ((int)threadIdx.y) * row_stride_warp * (32 + 8) +
                              (((int)threadIdx.x) / (32 / 8)) * (32 + 8) +
                              (((int)threadIdx.x) % (32 / 8)) * 8;

  __nv_bfloat16* B_shared_ptr = B_shared +
                              ((int)threadIdx.y) * (row_stride / 2) * (N + 8) +
                              (((int)threadIdx.x) / (N / 8)) * (N + 8) +
                              (((int)threadIdx.x) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)threadIdx.x) % (N / 8);

  __nv_bfloat16* scaling_factors_ptr = scaling_factors +
                                     (((int)blockIdx_y) % j_factors1) * N +
                                     (((int)threadIdx.x) % (N / 8)) * 8;

  __nv_bfloat16* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC +
      (((int)blockIdx_y) % j_factors1) * N +
      ((int)threadIdx.y) * (N / 2) +
      (((int)threadIdx.x) % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;

  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();

    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_bf16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));

    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_bf16 = dequantize_s4_to_bf16x2(B_loaded);

      asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.x) : "r"(B_loaded_bf16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.x) : "r"(B_loaded_bf16.x), "r"(B_loaded_scale.x), "r"(ZERO_U32));
      asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.y) : "r"(B_loaded_bf16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.y) : "r"(B_loaded_bf16.y), "r"(B_loaded_scale.y), "r"(ZERO_U32));
      asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.z) : "r"(B_loaded_bf16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.z) : "r"(B_loaded_bf16.z), "r"(B_loaded_scale.z), "r"(ZERO_U32));
      asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.w) : "r"(B_loaded_bf16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.w) : "r"(B_loaded_bf16.w), "r"(B_loaded_scale.w), "r"(ZERO_U32));

      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) = B_loaded_bf16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
        // Load A tile from shared into registers
        {
            unsigned int addr;
            void* s_ptr = &A_shared[k_0_1 * 16]; // Base address for the 16x32 tile
            __asm__ __volatile__("{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n" : "=r"(addr) : "l"(s_ptr));
            addr += (threadIdx.x % 16) * sizeof(__nv_bfloat162) + (threadIdx.x / 16) * 16 * (32 + 8) * sizeof(__nv_bfloat16); // Thread-specific offset
            __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(A_shared_warp[0]), "=r"(A_shared_warp[1]), "=r"(A_shared_warp[2]), "=r"(A_shared_warp[3])
                                 : "r"(addr));
        }

      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
        {
            unsigned int addr;
            void* s_ptr = &B_shared[k_0_1 * 16 + j_0_4 * 32 * (32 + 8)];
            __asm__ __volatile__("{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n" : "=r"(addr) : "l"(s_ptr));
            addr += (threadIdx.x % 16) * sizeof(__nv_bfloat162) + (threadIdx.x / 16) * 8 * (N + 8) * sizeof(__nv_bfloat16);
            __asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(B_shared_warp[0]), "=r"(B_shared_warp[1])
                                 : "r"(addr));
        }

        // Perform MMA using correct bf16 instruction
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(((float*)(C_warp + j_0_4 * 8))[0]), "=f"(((float*)(C_warp + j_0_4 * 8))[1]),
              "=f"(((float*)(C_warp + j_0_4 * 8))[2]), "=f"(((float*)(C_warp + j_0_4 * 8))[3])
            : "r"(A_shared_warp[0]), "r"(A_shared_warp[1]), "r"(A_shared_warp[2]), "r"(A_shared_warp[3]),
              "r"(B_shared_warp[0]), "r"(B_shared_warp[1]),
              "f"(((float*)(C_warp + j_0_4 * 8))[0]), "f"(((float*)(C_warp + j_0_4 * 8))[1]),
              "f"(((float*)(C_warp + j_0_4 * 8))[2]), "f"(((float*)(C_warp + j_0_4 * 8))[3]));
      }
    }
  }

  for (int ax1_0_1 = 0; ax1_0_1 < (N / 32); ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) = __float2bfloat16(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
#endif
}

__global__ void __launch_bounds__(64)
    dequantize_weights_bf16(int* __restrict__ B, __nv_bfloat16* __restrict__ scaling_factors,
                            int* __restrict__ zeros, __nv_bfloat16* __restrict__ C, int G) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false); // BF16 requires Ampere or newer
#else
  static constexpr uint32_t ZERO_U32 = 0;

  int N = gridDim.x * blockDim.x;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  __nv_bfloat16* C_ptr = C + 8 * (col + row * N);
  int* B_ptr = B + (col + row * N);
  int* zeros_ptr = zeros + (col + (row / G) * N);
  __nv_bfloat16* scaling_factors_ptr = scaling_factors + 8 * (col + (row / G) * N);

  uint32_t B_loaded = *B_ptr;
  uint32_t zeros_loaded = *zeros_ptr;
  uint4 B_loaded_scale = *(uint4*)scaling_factors_ptr;

  uint4 B_loaded_bf16 = dequantize_s4_to_bf16x2(B_loaded);
  uint4 B_loaded_zero = dequantize_s4_to_bf16x2(zeros_loaded);
  
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.x) : "r"(B_loaded_bf16.x), "r"(B_loaded_zero.x));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.x) : "r"(B_loaded_bf16.x), "r"(B_loaded_scale.x), "r"(ZERO_U32));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.y) : "r"(B_loaded_bf16.y), "r"(B_loaded_zero.y));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.y) : "r"(B_loaded_bf16.y), "r"(B_loaded_scale.y), "r"(ZERO_U32));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.z) : "r"(B_loaded_bf16.z), "r"(B_loaded_zero.z));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.z) : "r"(B_loaded_bf16.z), "r"(B_loaded_scale.z), "r"(ZERO_U32));
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "=r"(B_loaded_bf16.w) : "r"(B_loaded_bf16.w), "r"(B_loaded_zero.w));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_bf16.w) : "r"(B_loaded_bf16.w), "r"(B_loaded_scale.w), "r"(ZERO_U32));

  *(uint4*)C_ptr = B_loaded_bf16;
#endif
}


} // namespace awq
} // namespace vllm


torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, 
                             int64_t split_k_iters,
                             int64_t thx, 
                             int64_t thy,
                             const std::string& dtype = "fp16") {
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);

  int x_thread = thx;
  int y_thread = thy;
  int x_blocks = 1;
  int y_blocks = 1;

  if (thx == 0 || thy == 0) {
      x_thread = 8;
      y_thread = 8;
      x_blocks = qout_c / x_thread;
      y_blocks = in_c / y_thread;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

  c10::ScalarType output_dtype;
  if (dtype == "bf16") {
    output_dtype = c10::kBFloat16;
  } else if (dtype == "fp16") {
    output_dtype = c10::kHalf;
  } else {
    throw std::invalid_argument("Unsupported dtype, must be 'fp16' or 'bf16'");
  }

  auto options = torch::TensorOptions()
                     .dtype(output_dtype)
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

  dim3 num_blocks(x_blocks, y_blocks);
  dim3 threads_per_block(x_thread, y_thread);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (dtype == "bf16") {
    auto scaling_factors = reinterpret_cast<__nv_bfloat16*>(_scaling_factors.data_ptr<at::BFloat16>());
    auto de_kernel = reinterpret_cast<__nv_bfloat16*>(_de_kernel.data_ptr<at::BFloat16>());
    vllm::awq::dequantize_weights_bf16<<<num_blocks, threads_per_block, 0, stream>>>(
        kernel, scaling_factors, zeros, de_kernel, G);
  } else {
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    vllm::awq::dequantize_weights_fp16<<<num_blocks, threads_per_block, 0, stream>>>(
        kernel, scaling_factors, zeros, de_kernel, G);
  }

  return _de_kernel;
}

torch::Tensor awq_gemm(torch::Tensor _in_feats, 
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, 
                       torch::Tensor _zeros,
                       int64_t split_k_iters,
                       const std::string& dtype = "fp16") {
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  int num_out_channels = _kernel.size(1) * 8;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

  c10::ScalarType output_dtype;
  if (dtype == "bf16") {
    output_dtype = c10::kBFloat16;
  } else if (dtype == "fp16") {
    output_dtype = c10::kHalf;
  } else {
    throw std::invalid_argument("Unsupported dtype, must be 'fp16' or 'bf16'");
  }

  auto options = torch::TensorOptions()
                     .dtype(output_dtype)
                     .device(_in_feats.device());
  at::Tensor _out_feats =
      torch::empty({split_k_iters, num_in_feats, num_out_channels}, options);

  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
  int group_size = num_in_channels / _scaling_factors.size(0);

  if (num_out_channels % 64 != 0)
    throw std::invalid_argument("OC must be a multiple of 64");
  if (group_size % 32 != 0)
    throw std::invalid_argument("Group size must be a multiple of 32");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 threads_per_block(32, 2);

  if (dtype == "bf16") {
    auto in_feats = reinterpret_cast<__nv_bfloat16*>(_in_feats.data_ptr<at::BFloat16>());
    auto out_feats = reinterpret_cast<__nv_bfloat16*>(_out_feats.data_ptr<at::BFloat16>());
    auto scaling_factors = reinterpret_cast<__nv_bfloat16*>(_scaling_factors.data_ptr<at::BFloat16>());

    if (num_out_channels % 128 == 0) {
      int j_factors1 = num_out_channels / 128;
      dim3 num_blocks((num_in_feats + 15) / 16 * j_factors1 * split_k_iters);
      vllm::awq::gemm_forward_4bit_cuda_m16nXk32_bf16<128>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
              num_in_feats, num_in_channels, num_out_channels, out_feats);
    } else { // num_out_channels % 64 == 0
      int j_factors1 = num_out_channels / 64;
      dim3 num_blocks((num_in_feats + 15) / 16 * j_factors1 * split_k_iters);
      vllm::awq::gemm_forward_4bit_cuda_m16nXk32_bf16<64>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
              num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
  } else { // fp16
    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());

    if (num_out_channels % 128 == 0) {
      int j_factors1 = num_out_channels / 128;
      dim3 num_blocks((num_in_feats + 15) / 16 * j_factors1 * split_k_iters);
      vllm::awq::gemm_forward_4bit_cuda_m16nXk32_fp16<128>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
              num_in_feats, num_in_channels, num_out_channels, out_feats);
    } else { // num_out_channels % 64 == 0
      int j_factors1 = num_out_channels / 64;
      dim3 num_blocks((num_in_feats + 15) / 16 * j_factors1 * split_k_iters);
      vllm::awq::gemm_forward_4bit_cuda_m16nXk32_fp16<64>
          <<<num_blocks, threads_per_block, 0, stream>>>(
              group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
              num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
  }

  return _out_feats.sum(0);
}