// SPDX-License-Identifier: Apache-2.0
// Fused RoPE + FP8 quantization kernel for MLA decode.
// Computes cos/sin on-the-fly from inv_freq via __sincosf,
// eliminating the precomputed cos_sin_cache.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "mla_rope_quant_kernel.cuh"

// Runtime bool -> compile-time constexpr dispatch
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    constexpr bool INTERLEAVE = true;                    \
    __VA_ARGS__                                          \
  } else {                                               \
    constexpr bool INTERLEAVE = false;                   \
    __VA_ARGS__                                          \
  }

template <typename DType, typename IdType, typename QuantType>
static cudaError_t launch_rope_quantize(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in,
    QuantType* q_rope_out, QuantType* k_rope_out, QuantType* q_nope_out,
    QuantType* k_nope_out, const float* inv_freq, IdType* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim,
    uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h,
    size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride,
    size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv,
    bool interleave, bool enable_pdl, cudaStream_t stream) {
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunk_size = rope_dim;
    uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t no_rope_chunks =
        (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t total_blocks_y =
        num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
        num_kv_heads * no_rope_chunks + num_qo_heads * no_rope_chunks;

    auto kernel =
        vllm::mla_rope::RopeQuantizeTiledKernel<INTERLEAVE, vec_size, 1, DType,
                                                IdType, QuantType>;
    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed =
        enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    return cudaLaunchKernelEx(
        &config, kernel, q_rope_in, k_rope_in, q_nope_in, k_nope_in, q_rope_out,
        k_rope_out, q_nope_out, k_nope_out, inv_freq, pos_ids, nnz,
        num_qo_heads, num_kv_heads, rope_dim, no_rope_dim, q_rope_in_stride_n,
        q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
        q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n,
        q_nope_out_stride_h, k_rope_in_stride, k_rope_in_stride_h,
        k_nope_in_stride, k_nope_in_stride_h, k_rope_out_stride,
        k_rope_out_stride_h, k_nope_out_stride, k_nope_out_stride_h,
        quant_scale_q, quant_scale_kv);
  });

  return cudaSuccess;
}

void mla_rope_quantize_fp8(torch::Tensor& q_rope_in, torch::Tensor& k_rope_in,
                           torch::Tensor& q_nope_in, torch::Tensor& k_nope_in,
                           torch::Tensor& q_rope_out, torch::Tensor& k_rope_out,
                           torch::Tensor& q_nope_out, torch::Tensor& k_nope_out,
                           torch::Tensor& inv_freq, torch::Tensor& pos_ids,
                           double quant_scale_q, double quant_scale_kv,
                           bool interleave, bool enable_pdl) {
  TORCH_CHECK(q_rope_in.dim() == 3, "q_rope_in must be 3D");
  TORCH_CHECK(q_nope_in.dim() == 3, "q_nope_in must be 3D");
  TORCH_CHECK(inv_freq.scalar_type() == at::kFloat, "inv_freq must be float32");
  TORCH_CHECK(q_rope_in.scalar_type() == at::kHalf ||
                  q_rope_in.scalar_type() == at::kBFloat16,
              "Input dtype must be float16 or bfloat16");
  TORCH_CHECK(q_rope_out.scalar_type() == at::kFloat8_e4m3fn,
              "Output dtype must be float8_e4m3fn");

  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);
  uint32_t rope_dim = q_rope_in.size(-1);
  uint32_t no_rope_dim = q_nope_in.size(-1);
  uint32_t num_kv_heads = (k_rope_in.dim() == 2) ? 1 : k_rope_in.size(1);

  const size_t q_rope_in_stride_n = q_rope_in.stride(0);
  const size_t q_rope_in_stride_h = q_rope_in.stride(1);
  const size_t q_nope_in_stride_n = q_nope_in.stride(0);
  const size_t q_nope_in_stride_h = q_nope_in.stride(1);
  const size_t q_rope_out_stride_n = q_rope_out.stride(0);
  const size_t q_rope_out_stride_h = q_rope_out.stride(1);
  const size_t q_nope_out_stride_n = q_nope_out.stride(0);
  const size_t q_nope_out_stride_h = q_nope_out.stride(1);

  size_t k_rope_in_stride, k_nope_in_stride, k_rope_out_stride,
      k_nope_out_stride;
  size_t k_rope_in_stride_h, k_nope_in_stride_h, k_rope_out_stride_h,
      k_nope_out_stride_h;

  if (k_rope_in.dim() == 2) {
    k_rope_in_stride = k_rope_in.stride(0);
    k_nope_in_stride = k_nope_in.stride(0);
    k_rope_out_stride = k_rope_out.stride(0);
    k_nope_out_stride = k_nope_out.stride(0);
    k_rope_in_stride_h = k_rope_in_stride;
    k_nope_in_stride_h = k_nope_in_stride;
    k_rope_out_stride_h = k_rope_out_stride;
    k_nope_out_stride_h = k_nope_out_stride;
  } else {
    k_rope_in_stride = k_rope_in.stride(0);
    k_rope_in_stride_h = k_rope_in.stride(1);
    k_nope_in_stride = k_nope_in.stride(0);
    k_nope_in_stride_h = k_nope_in.stride(1);
    k_rope_out_stride = k_rope_out.stride(0);
    k_rope_out_stride_h = k_rope_out.stride(1);
    k_nope_out_stride = k_nope_out.stride(0);
    k_nope_out_stride_h = k_nope_out.stride(1);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float scale_q = static_cast<float>(quant_scale_q);
  const float scale_kv = static_cast<float>(quant_scale_kv);

  cudaError_t status;
  if (q_rope_in.scalar_type() == at::kBFloat16) {
    status = launch_rope_quantize(
        reinterpret_cast<nv_bfloat16*>(q_rope_in.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(k_rope_in.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(q_nope_in.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(k_nope_in.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q_rope_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(k_rope_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q_nope_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(k_nope_out.data_ptr()),
        reinterpret_cast<const float*>(inv_freq.data_ptr()),
        pos_ids.data_ptr<int64_t>(), nnz, num_qo_heads, num_kv_heads, rope_dim,
        no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n,
        q_nope_in_stride_h, q_rope_out_stride_n, q_rope_out_stride_h,
        q_nope_out_stride_n, q_nope_out_stride_h, k_rope_in_stride,
        k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
        k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride,
        k_nope_out_stride_h, scale_q, scale_kv, interleave, enable_pdl, stream);
  } else {
    status = launch_rope_quantize(
        reinterpret_cast<half*>(q_rope_in.data_ptr()),
        reinterpret_cast<half*>(k_rope_in.data_ptr()),
        reinterpret_cast<half*>(q_nope_in.data_ptr()),
        reinterpret_cast<half*>(k_nope_in.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q_rope_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(k_rope_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q_nope_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(k_nope_out.data_ptr()),
        reinterpret_cast<const float*>(inv_freq.data_ptr()),
        pos_ids.data_ptr<int64_t>(), nnz, num_qo_heads, num_kv_heads, rope_dim,
        no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n,
        q_nope_in_stride_h, q_rope_out_stride_n, q_rope_out_stride_h,
        q_nope_out_stride_n, q_nope_out_stride_h, k_rope_in_stride,
        k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
        k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride,
        k_nope_out_stride_h, scale_q, scale_kv, interleave, enable_pdl, stream);
  }

  TORCH_CHECK(status == cudaSuccess,
              "mla_rope_quantize_fp8 failed: ", cudaGetErrorString(status));
}

// ============================================================================
// Split-fused RoPE+quant+cache launchers
//
// Two independent kernels that can be launched on separate streams:
//   mla_fused_cache_rope  (bdx=4, bdy=32): Q rope + K rope -> cache
//   mla_fused_cache_nope  (bdx=32, bdy=4): Q nope + K nope -> cache
//
// They operate on disjoint input slices and disjoint cache regions,
// so overlapping on separate streams is safe.
// ============================================================================

// Helper: extract k strides handling 2D (single KV head) vs 3D tensors.
static inline void get_k_strides(const torch::Tensor& k, size_t& stride_n,
                                 size_t& stride_h) {
  if (k.dim() == 2) {
    stride_n = k.stride(0);
    stride_h = stride_n;
  } else {
    stride_n = k.stride(0);
    stride_h = k.stride(1);
  }
}

void mla_fused_cache_rope(torch::Tensor& q_rope_in, torch::Tensor& q_rope_out,
                          torch::Tensor& k_rope_in, torch::Tensor& kv_cache,
                          torch::Tensor& slot_mapping, torch::Tensor& inv_freq,
                          torch::Tensor& pos_ids, int64_t num_kv_heads,
                          int64_t no_rope_dim, double quant_scale_q,
                          double quant_scale_kv, bool interleave) {
  TORCH_CHECK(q_rope_in.dim() == 3, "q_rope_in must be 3D");
  TORCH_CHECK(q_rope_in.scalar_type() == at::kBFloat16);

  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_actual_tokens = slot_mapping.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);
  uint32_t rope_dim = q_rope_in.size(-1);
  constexpr uint32_t vec_size = 32 / sizeof(nv_bfloat16);

  size_t k_stride_n, k_stride_h;
  get_k_strides(k_rope_in, k_stride_n, k_stride_h);

  const float sq = static_cast<float>(quant_scale_q);
  const float skv = static_cast<float>(quant_scale_kv);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t bdx = 4, bdy = 32;
    dim3 grid((nnz + bdy - 1) / bdy,
              num_qo_heads + static_cast<uint32_t>(num_kv_heads));
    dim3 block(bdx, bdy);

    auto kernel = vllm::mla_rope::RopeOnlyFusedCacheKernel<
        INTERLEAVE, vec_size, bdx, nv_bfloat16, int64_t, __nv_fp8_e4m3>;
    kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(q_rope_in.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q_rope_out.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(k_rope_in.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(kv_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(),
        reinterpret_cast<const float*>(inv_freq.data_ptr()),
        pos_ids.data_ptr<int64_t>(), nnz, num_actual_tokens, num_qo_heads,
        static_cast<uint32_t>(num_kv_heads), rope_dim,
        static_cast<uint32_t>(no_rope_dim), q_rope_in.stride(0),
        q_rope_in.stride(1), q_rope_out.stride(0), q_rope_out.stride(1),
        k_stride_n, k_stride_h, kv_cache.stride(0), kv_cache.stride(1),
        static_cast<int>(kv_cache.size(1)), sq, skv);
  });
}

void mla_fused_cache_nope(torch::Tensor& q_nope_in, torch::Tensor& q_nope_out,
                          torch::Tensor& k_nope_in, torch::Tensor& kv_cache,
                          torch::Tensor& slot_mapping, int64_t num_kv_heads,
                          double quant_scale_q, double quant_scale_kv) {
  TORCH_CHECK(q_nope_in.dim() == 3, "q_nope_in must be 3D");
  TORCH_CHECK(q_nope_in.scalar_type() == at::kBFloat16);

  uint32_t nnz = q_nope_in.size(0);
  uint32_t num_actual_tokens = slot_mapping.size(0);
  uint32_t num_qo_heads = q_nope_in.size(1);
  uint32_t dim = q_nope_in.size(-1);
  constexpr uint32_t vec_size = 32 / sizeof(nv_bfloat16);

  size_t k_stride_n, k_stride_h;
  get_k_strides(k_nope_in, k_stride_n, k_stride_h);

  const float sq = static_cast<float>(quant_scale_q);
  const float skv = static_cast<float>(quant_scale_kv);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr uint32_t bdx = 32, bdy = 4;
  dim3 grid((nnz + bdy - 1) / bdy,
            num_qo_heads + static_cast<uint32_t>(num_kv_heads));
  dim3 block(bdx, bdy);

  auto kernel =
      vllm::mla_rope::NopeScaleQuantFusedCacheKernel<vec_size, nv_bfloat16,
                                                     __nv_fp8_e4m3>;
  kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const nv_bfloat16*>(q_nope_in.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(q_nope_out.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(k_nope_in.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(kv_cache.data_ptr()),
      slot_mapping.data_ptr<int64_t>(), nnz, num_actual_tokens, num_qo_heads,
      static_cast<uint32_t>(num_kv_heads), dim, q_nope_in.stride(0),
      q_nope_in.stride(1), q_nope_out.stride(0), q_nope_out.stride(1),
      k_stride_n, k_stride_h, kv_cache.stride(0), kv_cache.stride(1),
      static_cast<int>(kv_cache.size(1)), sq, skv);
}

// Legacy combined launcher (calls both on same stream).
void mla_rope_quantize_fp8_fused_cache(
    torch::Tensor& q_rope_in, torch::Tensor& q_nope_in,
    torch::Tensor& q_rope_out, torch::Tensor& q_nope_out,
    torch::Tensor& k_rope_in, torch::Tensor& k_nope_in, torch::Tensor& kv_cache,
    torch::Tensor& slot_mapping, torch::Tensor& inv_freq,
    torch::Tensor& pos_ids, double quant_scale_q, double quant_scale_kv,
    bool interleave, bool enable_pdl) {
  uint32_t num_kv_heads = (k_rope_in.dim() == 2) ? 1 : k_rope_in.size(1);
  uint32_t no_rope_dim = q_nope_in.size(-1);

  mla_fused_cache_rope(q_rope_in, q_rope_out, k_rope_in, kv_cache, slot_mapping,
                       inv_freq, pos_ids, num_kv_heads, no_rope_dim,
                       quant_scale_q, quant_scale_kv, interleave);

  mla_fused_cache_nope(q_nope_in, q_nope_out, k_nope_in, kv_cache, slot_mapping,
                       num_kv_heads, quant_scale_q, quant_scale_kv);
}
