// Persistent NVFP4 TMA launcher — compiled separately to avoid register
// allocation interference with the strided kernel in
// silu_mul_fp8_quant_launcher.cu.

#include <cuda_runtime.h>
#include <cstdint>

#include "silu_mul_nvfp4_quant_tma_ws_persistent_kernel.cuh"
#include "silu_mul_fp8_quant_launcher.h"

namespace vllm {

static constexpr int NVFP4_PERSISTENT_STAGES = 2;

namespace {
__device__ int32_t g_nvfp4_persistent_counters[64];
}

template <int N_COMPUTE, int BATCH_SIZE>
void launch_tma_ws_nvfp4_bf16_persistent_dispatch(void* input, void* output,
                                                  void* output_sf,
                                                  void* global_scale,
                                                  int32_t n_tokens, int64_t H,
                                                  int64_t N, bool use_tanh_silu,
                                                  cudaStream_t stream) {
  constexpr int WARP_ELTS = 512;
  int const numGroups = H / WARP_ELTS;
  int const gridX = (numGroups + N_COMPUTE - 1) / N_COMPUTE;

  constexpr int MBAR_REGION =
      ((2 * NVFP4_PERSISTENT_STAGES * 8) + 1023) & ~1023;
  constexpr int NC_SLICE_BYTES = N_COMPUTE * WARP_ELTS * 2;
  constexpr int ROW_BYTES = 2 * NC_SLICE_BYTES;
  constexpr int smem =
      MBAR_REGION + NVFP4_PERSISTENT_STAGES * BATCH_SIZE * ROW_BYTES;
  constexpr int blockThreads = (N_COMPUTE + 1) * 32;

  // Static one-time init — these CUDA runtime calls are not safe during
  // CUDA graph stream capture. Caching them here makes the hot path
  // graph-capturable.
  struct StaticState {
    int numSms;
    int smemPerSM;
    int32_t* d_counters;
  };
  static StaticState state = []() {
    StaticState s{};
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&s.numSms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&s.smemPerSM,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaGetSymbolAddress((void**)&s.d_counters, g_nvfp4_persistent_counters);
    cudaFuncSetAttribute(
        tma_v5::silu_mul_nvfp4_quant_tma_ws_persistent_kernel_bf16<
            N_COMPUTE, NVFP4_PERSISTENT_STAGES, BATCH_SIZE, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(
        tma_v5::silu_mul_nvfp4_quant_tma_ws_persistent_kernel_bf16<
            N_COMPUTE, NVFP4_PERSISTENT_STAGES, BATCH_SIZE, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    return s;
  }();

  auto kernel_fn = tma_v5::silu_mul_nvfp4_quant_tma_ws_persistent_kernel_bf16<
      N_COMPUTE, NVFP4_PERSISTENT_STAGES, BATCH_SIZE, false>;
  if (use_tanh_silu) {
    kernel_fn = tma_v5::silu_mul_nvfp4_quant_tma_ws_persistent_kernel_bf16<
        N_COMPUTE, NVFP4_PERSISTENT_STAGES, BATCH_SIZE, true>;
  }

  int maxBySmem = smem > 0 ? state.smemPerSM / smem : 1;
  int maxByThreads = 2048 / blockThreads;
  int maxCTAsPerSM = maxBySmem < maxByThreads ? maxBySmem : maxByThreads;
  if (maxCTAsPerSM < 1) maxCTAsPerSM = 1;
  int totalDesiredCTAs = maxCTAsPerSM * state.numSms;
  int gridY = (totalDesiredCTAs + gridX - 1) / gridX;
  if (gridY < state.numSms) gridY = state.numSms;

  cudaMemsetAsync(state.d_counters, 0, gridX * sizeof(int32_t), stream);

  CUtensorMap tensorMap;
  cuuint64_t globalDim[3] = {64, static_cast<cuuint64_t>(2 * H / 64),
                             static_cast<cuuint64_t>(N)};
  cuuint64_t globalStrides[2] = {128, static_cast<cuuint64_t>(2 * H * 2)};
  cuuint32_t boxDim[3] = {64, 8, 1};
  cuuint32_t elementStrides[3] = {1, 1, 1};
  cuTensorMapEncodeTiled(
      &tensorMap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, input, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  dim3 grid(gridX, gridY);
  dim3 block(blockThreads);

  kernel_fn<<<grid, block, smem, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(input),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_sf),
      reinterpret_cast<float*>(global_scale), n_tokens, H, state.d_counters,
      tensorMap);
}

template <int N_COMPUTE>
void launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch(
    void* input, void* output, void* output_sf, void* global_scale,
    int32_t n_tokens, int64_t H, int64_t N, int64_t batch_size,
    bool use_tanh_silu, cudaStream_t stream) {
  switch (batch_size) {
    case 1:
      launch_tma_ws_nvfp4_bf16_persistent_dispatch<N_COMPUTE, 1>(
          input, output, output_sf, global_scale, n_tokens, H, N, use_tanh_silu,
          stream);
      break;
    case 2:
      launch_tma_ws_nvfp4_bf16_persistent_dispatch<N_COMPUTE, 2>(
          input, output, output_sf, global_scale, n_tokens, H, N, use_tanh_silu,
          stream);
      break;
    case 4:
      launch_tma_ws_nvfp4_bf16_persistent_dispatch<N_COMPUTE, 4>(
          input, output, output_sf, global_scale, n_tokens, H, N, use_tanh_silu,
          stream);
      break;
    case 8:
      launch_tma_ws_nvfp4_bf16_persistent_dispatch<N_COMPUTE, 8>(
          input, output, output_sf, global_scale, n_tokens, H, N, use_tanh_silu,
          stream);
      break;
    default:
      launch_tma_ws_nvfp4_bf16_persistent_dispatch<N_COMPUTE, 2>(
          input, output, output_sf, global_scale, n_tokens, H, N, use_tanh_silu,
          stream);
      break;
  }
}

void launch_silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
    void* input, void* output, void* output_sf, void* global_scale,
    int32_t n_tokens, int64_t H, int64_t N, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu, void* stream_ptr) {
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  switch (n_compute) {
    case 1:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<1>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 2:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<2>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 4:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<4>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 7:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<7>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 8:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<8>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 14:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<14>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 28:
      launch_tma_ws_nvfp4_bf16_persistent_bs_dispatch<28>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    default:
      break;
  }
}

}  // namespace vllm
