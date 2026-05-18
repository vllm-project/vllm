// Persistent FP8 TMA launcher — compiled separately to avoid register
// allocation interference with the strided kernel in
// silu_mul_fp8_quant_launcher.cu. Uses 3D TMA descriptors with SWIZZLE_128B for
// efficient data loading.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "silu_mul_fp8_quant_tma_ws_persistent_kernel.cuh"
#include "silu_mul_fp8_quant_launcher.h"

namespace vllm {

static constexpr int FP8_PERSISTENT_STAGES = 2;

namespace {
__device__ int32_t g_fp8_persistent_counters[64];
}

template <int N_COMPUTE, int BATCH_SIZE>
void launch_tma_ws_fp8_persistent_dispatch(void* input, void* input_scales,
                                           void* output, void* output_scales,
                                           int32_t n_tokens, int64_t H,
                                           int64_t scale_stride,
                                           bool use_tanh_silu, int64_t N,
                                           cudaStream_t stream) {
  constexpr int GROUP_SIZE = 128;
  int const G = H / GROUP_SIZE;
  int const gridX = (G + N_COMPUTE - 1) / N_COMPUTE;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  constexpr int MBAR_REGION =
      ((2 * FP8_PERSISTENT_STAGES * 8 + sizeof(int32_t)) + 127) & ~127;
  constexpr int TOKEN_BYTES = 2 * N_COMPUTE * 128;
  constexpr int STAGE_DATA = BATCH_SIZE * TOKEN_BYTES;
  int smem = MBAR_REGION + FP8_PERSISTENT_STAGES * STAGE_DATA;
  constexpr int blockThreads = (N_COMPUTE + 1) * 32;

  auto kernel_fn = tma_v5::silu_mul_fp8_quant_tma_ws_persistent_kernel<
      N_COMPUTE, FP8_PERSISTENT_STAGES, BATCH_SIZE, false>;
  if (use_tanh_silu) {
    kernel_fn = tma_v5::silu_mul_fp8_quant_tma_ws_persistent_kernel<
        N_COMPUTE, FP8_PERSISTENT_STAGES, BATCH_SIZE, true>;
  }

  cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);

  int smemPerSM = 0;
  cudaDeviceGetAttribute(&smemPerSM,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
  int maxBySmem = smem > 0 ? smemPerSM / smem : 1;
  int maxByThreads = 2048 / blockThreads;
  int maxCTAsPerSM = maxBySmem < maxByThreads ? maxBySmem : maxByThreads;
  if (maxCTAsPerSM < 1) maxCTAsPerSM = 1;
  int totalDesiredCTAs = maxCTAsPerSM * numSms;
  int gridY = (totalDesiredCTAs + gridX - 1) / gridX;
  if (gridY < numSms) gridY = numSms;

  int32_t* d_counters;
  cudaGetSymbolAddress((void**)&d_counters, g_fp8_persistent_counters);
  cudaMemsetAsync(d_counters, 0, gridX * sizeof(int32_t), stream);

  CUtensorMap tensorMap;
  cuuint64_t globalDim[3] = {128, static_cast<cuuint64_t>(2 * H / 128),
                             static_cast<cuuint64_t>(N)};
  cuuint64_t globalStrides[2] = {128, static_cast<cuuint64_t>(2 * H)};
  cuuint32_t boxDim[3] = {128, static_cast<cuuint32_t>(N_COMPUTE), 1};
  cuuint32_t elementStrides[3] = {1, 1, 1};
  cuTensorMapEncodeTiled(
      &tensorMap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, input, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  dim3 grid(gridX, gridY);
  dim3 block(blockThreads);

  kernel_fn<<<grid, block, smem, stream>>>(
      reinterpret_cast<__nv_fp8_e4m3*>(input),
      reinterpret_cast<float*>(input_scales),
      reinterpret_cast<__nv_fp8_e4m3*>(output),
      reinterpret_cast<float*>(output_scales), n_tokens, H, scale_stride,
      d_counters, tensorMap);
}

template <int N_COMPUTE>
void launch_tma_ws_fp8_persistent_bs_dispatch(
    void* input, void* input_scales, void* output, void* output_scales,
    int32_t n_tokens, int64_t H, int64_t scale_stride, int64_t batch_size,
    bool use_tanh_silu, int64_t N, cudaStream_t stream) {
  switch (batch_size) {
    case 1:
      launch_tma_ws_fp8_persistent_dispatch<N_COMPUTE, 1>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, N, stream);
      break;
    case 2:
      launch_tma_ws_fp8_persistent_dispatch<N_COMPUTE, 2>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, N, stream);
      break;
    case 4:
      launch_tma_ws_fp8_persistent_dispatch<N_COMPUTE, 4>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, N, stream);
      break;
    case 8:
      launch_tma_ws_fp8_persistent_dispatch<N_COMPUTE, 8>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, N, stream);
      break;
    case 16:
      launch_tma_ws_fp8_persistent_dispatch<N_COMPUTE, 16>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, N, stream);
      break;
    default:
      launch_tma_ws_fp8_persistent_dispatch<N_COMPUTE, 2>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, N, stream);
      break;
  }
}

void launch_silu_mul_fp8_quant_tma_ws_persistent(
    void* input, void* input_scales, void* output, void* output_scales,
    int32_t n_tokens, int64_t H, int64_t scale_stride, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu, int64_t N, void* stream_ptr) {
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  switch (n_compute) {
    case 1:
      launch_tma_ws_fp8_persistent_bs_dispatch<1>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    case 2:
      launch_tma_ws_fp8_persistent_bs_dispatch<2>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    case 4:
      launch_tma_ws_fp8_persistent_bs_dispatch<4>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    case 7:
      launch_tma_ws_fp8_persistent_bs_dispatch<7>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    case 8:
      launch_tma_ws_fp8_persistent_bs_dispatch<8>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    case 14:
      launch_tma_ws_fp8_persistent_bs_dispatch<14>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    case 28:
      launch_tma_ws_fp8_persistent_bs_dispatch<28>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, N, stream);
      break;
    default:
      break;
  }
}

}  // namespace vllm
