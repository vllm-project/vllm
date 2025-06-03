#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp8.h>

template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

#define ELTS_PER_THREAD 8

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
#else
  return 0;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx,
                                                       int numCols,
                                                       SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 16.
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // Compute the global offset.
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride +
                       outerMIdx * outerMStride + innerMIdx * innerMStride +
                       innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal,
                                         uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

  // Local maximum value.
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    // Extract the 8 exponent bits from float32.
    // float 32bits = 1 sign bit + 8 exponent bits + 23 mantissa bits.
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // Convert back to fp32.
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // Convert back to fp32.
    SFValue = float(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
  //                       reciprocal(SFScaleVal))
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(
                         SFValue * reciprocal_approximate_ftz(SFScaleVal))
                   : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows, int32_t numCols, Type const* in, float const* SFScale,
    uint32_t* out, uint32_t* SFout, uint32_t* input_offset_by_experts,
    uint32_t* output_scale_offset_by_experts, int n_experts) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // Find index within the experts.
      int rowIdx_in_expert = 0;
      int expert_idx = 0;
      for (int i = 0; i < n_experts; i++) {
        if (rowIdx >= input_offset_by_experts[i] &&
            rowIdx < input_offset_by_experts[i + 1]) {
          rowIdx_in_expert = rowIdx - input_offset_by_experts[i];
          expert_idx = i;
          break;
        }
      }

      // Get the global scaling factor, which will be applied to the SF.
      // Note SFScale is the same as next GEMM's alpha, which is
      // (448.f / (Alpha_A / 6.f)).
      float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

      int factor = CVT_FP4_SF_VEC_SIZE * 4;
      // The actual output_scales dim is computed from the padded numCols.
      int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
      int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
      uint32_t* SFout_in_expert =
          SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

      out_pos =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

template <typename T>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  // TODO: this multiProcessorCount should be cached.
  int device;
  cudaGetDevice(&device);
  int multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount,
                         device);

  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(k / ELTS_PER_THREAD), 512));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m_topk), multiProcessorCount * numBlocksPerSM));

  cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
      m_topk, k, reinterpret_cast<T*>(input),
      reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<uint32_t*>(input_offset_by_experts),
      reinterpret_cast<uint32_t*>(output_scale_offset_by_experts), n_experts);
}

/*Quantization entry for fp4 experts quantization*/
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

constexpr auto HALF = at::ScalarType::Half;
constexpr auto BF16 = at::ScalarType::BFloat16;
constexpr auto FLOAT = at::ScalarType::Float;
constexpr auto INT = at::ScalarType::Int;
constexpr auto UINT8 = at::ScalarType::Byte;

void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(input_offset_by_experts,
              "input_offset_by_experts must be a CUDA tensor");
  CHECK_INPUT(output_scale_offset_by_experts,
              "output_scale_offset_by_experts must be a CUDA tensor");

  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);
  TORCH_CHECK(input_offset_by_experts.dim() == 1);
  TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output is uint8 (two nvfp4 values are packed into one uint8)
  // output_scale is int32 (four fp8 values are packed into one int32)
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  const int BLOCK_SIZE = 16;
  auto m_topk = input.size(0);
  auto k = input.size(1);
  TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / BLOCK_SIZE;
  // 4 means the swizzle requirement by nvidia nvfp4.
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4 means 4 fp8 values are packed into one int32
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.get_device());
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(output.data_ptr(), output_scale.data_ptr(),
                     input.data_ptr(), input_global_scale.data_ptr(),
                     input_offset_by_experts.data_ptr(),
                     output_scale_offset_by_experts.data_ptr(), m_topk, k,
                     n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(output.data_ptr(), output_scale.data_ptr(),
                              input.data_ptr(), input_global_scale.data_ptr(),
                              input_offset_by_experts.data_ptr(),
                              output_scale_offset_by_experts.data_ptr(), m_topk,
                              k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}