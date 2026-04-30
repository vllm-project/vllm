# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int MX_BLOCK_SIZE = 32;

__device__ inline
void ldg_f32x8(float *data, const void *ptr) {
  asm volatile("ld.global.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=f"(data[0]), "=f"(data[1]), "=f"(data[2]), "=f"(data[3]),
                "=f"(data[4]), "=f"(data[5]), "=f"(data[6]), "=f"(data[7])
              : "l"(ptr));
}

__device__ inline
void ldg_b32x8_fast(int *data, const void *ptr) {
  asm volatile("ld.global.relaxed.cta.L1::no_allocate.v8.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
                "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7])
              : "l"(ptr));
}

__device__ inline
void bf16x2_to_fp32x2(float *out, uint32_t data) {
  asm volatile("shl.b32 %0, %2, 16;\n"        // low 16-bit
               "and.b32 %1, %2, 0xFFFF0000;"  // high 16-bit
              : "=f"(out[0]), "=f"(out[1]) : "r"(data));
}

__device__ inline
int fp32x2_to_bf16x2(float a, float b) {
  int tmp;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(tmp) : "f"(b), "f"(a));
  return tmp;
}

__device__ inline
int bf16x2_abs(int a) {
  int d;
  asm volatile("abs.bf16x2 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}

__device__ inline
int bf16x2_max(int a, int b) {
  int d;
  asm volatile("max.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ inline
int fp32x8_to_fp4x8(const float *x) {
  int out;
  asm volatile(
    "{\n"
    ".reg .b8 x0, x1, x2, x3;\n"
    "cvt.rn.satfinite.e2m1x2.f32 x0, %2, %1;\n"
    "cvt.rn.satfinite.e2m1x2.f32 x1, %4, %3;\n"
    "cvt.rn.satfinite.e2m1x2.f32 x2, %6, %5;\n"
    "cvt.rn.satfinite.e2m1x2.f32 x3, %8, %7;\n"
    "mov.b32 %0, {x0, x1, x2, x3};\n"
    "}"
    : "=r"(out)
    : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]),
      "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7])
  );
  return out;
}

template <int HEAD_DIM, int ROPE_DIM, int TB_SIZE>
__block_size__((TB_SIZE, 1, 1))
__global__
void fused_indexer_q_rope_mxfp4_kernel(
  const int64_t     *positions_ptr,  // [num_tokens]
  const nv_bfloat16 *q_ptr,          // [num_tokens, num_heads, head_dim]
  const float       *cos_sin_ptr,    // [max_pos, rope_dim]
  const nv_bfloat16 *weights,        // [num_tokens, num_heads]
        char        *q_fp4_ptr,      // [num_tokens, num_heads, head_dim/2]
        uint8_t     *q_scale_ptr,    // [num_tokens, num_heads, head_dim/32]
        float       *weights_out,    // [num_tokens, num_heads]
  float scale,
  int num_tokens,
  int num_heads,
  int q_stride0, int q_stride1,
  int cos_sin_stride,
  int weights_stride,
  int q_fp4_stride0, int q_fp4_stride1,
  int q_scale_stride0, int q_scale_stride1,
  int weights_out_stride
) {
  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;

  // we will use 32B load per thread = 16 BF16 elems.
  // hence, we need 8 threads to load single head (128 elems).
  // let's call subwarp = 8 threads -> 1 subwarp handles 1 token
  constexpr int SUBWARP_SIZE = HEAD_DIM / 16;
  static_assert(SUBWARP_SIZE <= WARP_SIZE);

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int global_tid = bid * blockDim.x + tid;
  const int global_subwarp_id = global_tid / SUBWARP_SIZE;
  const int sublane_id = tid % SUBWARP_SIZE;

  const int token_id = global_subwarp_id / num_heads;
  const int head_id = global_subwarp_id % num_heads;

  // load Q
  int q[8];
  float q_f32[16];
  const int q_offset = token_id * q_stride0 + head_id * q_stride1 + sublane_id * 16;
  ldg_b32x8_fast(q, q_ptr + q_offset);
  int64_t pos = positions_ptr[token_id];

  // apply rope
  // NOTE: warp divergence
  if (sublane_id * 16 >= NOPE_DIM) {
    float cos[8], sin[8];
    const int rope_idx = (sublane_id * 16 - NOPE_DIM) / 2;
    ldg_f32x8(cos, cos_sin_ptr + (pos * cos_sin_stride + rope_idx));
    ldg_f32x8(sin, cos_sin_ptr + (pos * cos_sin_stride + ROPE_DIM / 2 + rope_idx));

    // unpack
    for (int i = 0; i < 8; i++)
      bf16x2_to_fp32x2(q_f32 + i * 2, q[i]);

    for (int i = 0; i < 8; i++) {
      float q0 = q_f32[i * 2 + 0] * cos[i] - q_f32[i * 2 + 1] * sin[i];
      float q1 = q_f32[i * 2 + 0] * sin[i] + q_f32[i * 2 + 1] * cos[i];
      q_f32[i * 2 + 0] = q0;
      q_f32[i * 2 + 1] = q1;
    }

    // BF16 round-trip to match reference
    for (int i = 0; i < 8; i++)
      q[i] = fp32x2_to_bf16x2(q_f32[i * 2], q_f32[i * 2 + 1]);
  }

  // absmax in BF16 to save instructions
  int q_amax = bf16x2_abs(q[0]);
  for (int i = 1; i < 8; i++)
    q_amax = bf16x2_max(q_amax, bf16x2_abs(q[i]));

  // each thread holds 16 elems -> 2 threads hold 32 elems
  // warp shuffle among 2 threads
  constexpr int NUM_THREADS_PER_MX = MX_BLOCK_SIZE / 16;
  for (int stride = NUM_THREADS_PER_MX / 2; stride > 0; stride /= 2)
    q_amax = bf16x2_max(q_amax, __shfl_xor_sync(0xFFFF'FFFF, q_amax, stride));

  // final amax in FP32
  float q_amax_f32[2];
  bf16x2_to_fp32x2(q_amax_f32, q_amax);
  float amax = max(q_amax_f32[0], q_amax_f32[1]);
  
  constexpr float amax_eps = 0x6p-126f;   // 6.0f * 2^-126                     
  constexpr float inv_fp4_max = 1.0f / 6.0f;
  float fp4_scale = max(amax, amax_eps) * inv_fp4_max;

  // compute ceil_log2 with bit manipulation
  // add a magic number so that exponent increments by 1
  // when mantissa bits > 0
  uint32_t bits = __float_as_uint(fp4_scale);
  uint32_t ue8m0 = ((bits + 0x7FFFFFU) >> 23U) & 0xFFU;

  // only 1 out of 2 threads need to store SF (rmb, 2 threads = 32 elems)
  if (tid % 2 == 0) {
    const int q_scale_offset = token_id * q_scale_stride0
                             + head_id * q_scale_stride1
                             + sublane_id / 2;
    q_scale_ptr[q_scale_offset] = ue8m0;
  }

  // unpack
  for (int i = 0; i < 8; i++)
    bf16x2_to_fp32x2(q_f32 + i * 2, q[i]);

  // let A = ceil(log2(fp4_scale)) be the actual mathematical value
  // fp4_scale = 2^A, and ue8m0 = A + 127, where 127 is the exponent bias
  // we want 1/fp4_scale = 2^(-A), whose exponent bits = -A + 127 = 254 - ue8m0
  float inv_fp4_scale = __uint_as_float((254U - ue8m0) << 23U);
  for (int i = 0; i < 16; i++)
    q_f32[i] *= inv_fp4_scale;

  int2 packed_fp4;
  packed_fp4.x = fp32x8_to_fp4x8(q_f32);
  packed_fp4.y = fp32x8_to_fp4x8(q_f32 + 8);
  const int q_fp4_offset = token_id * q_fp4_stride0
                         + head_id * q_fp4_stride1
                         + sublane_id * 8;
  __stcs(reinterpret_cast<int2 *>(q_fp4_ptr + q_fp4_offset), packed_fp4);

  // scale weights
  if (global_tid < num_tokens * num_heads) {
    const int token_id = global_tid / num_heads;
    const int head_id = global_tid % num_heads;
    float w = __bfloat162float(weights[token_id * weights_stride + head_id]);
    weights_out[token_id * weights_out_stride + head_id] = w * scale;
  }
}

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

at::Tensor fused_indexer_q_rope_mxfp4(
    const at::Tensor& positions,
    const at::Tensor& q,
    const at::Tensor& cos_sin,
    const at::Tensor& weights,
          at::Tensor& q_fp4,
          at::Tensor& q_scale,
          at::Tensor& weights_out,
    double scale) {
  TORCH_CHECK(positions.is_cuda(), "positions must be CUDA");
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(cos_sin.is_cuda(), "cos_sin must be CUDA");
  TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
  TORCH_CHECK(q_fp4.is_cuda(), "q_fp4 must be CUDA");
  TORCH_CHECK(q_scale.is_cuda(), "q_scale must be CUDA");
  TORCH_CHECK(weights_out.is_cuda(), "weights_out must be CUDA");

  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat, "cos_sin must be float32");
  TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
  TORCH_CHECK(q_fp4.scalar_type() == at::kByte, "q_fp4 must be uint8");
  TORCH_CHECK(q_scale.scalar_type() == at::kByte, "q_scale must be uint8");
  TORCH_CHECK(weights_out.scalar_type() == at::kFloat, "weights_out must be float32");

  TORCH_CHECK(positions.dim() == 1, "positions must be rank 1");
  TORCH_CHECK(q.dim() == 3, "q must have shape [num_tokens, num_heads, 128]");
  TORCH_CHECK(cos_sin.dim() == 2, "cos_sin must have shape [max_pos, 64]");
  TORCH_CHECK(weights.dim() == 2, "weights must have shape [num_tokens, num_heads]");
  TORCH_CHECK(q.size(2) == 128, "q head_dim must be 128");
  TORCH_CHECK(cos_sin.size(1) == 64, "cos_sin rope_dim must be 64");

  const int num_tokens = static_cast<int>(positions.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  TORCH_CHECK(q.size(0) == num_tokens, "q and positions token counts differ");
  TORCH_CHECK(weights.size(0) == num_tokens, "weights token count differs");
  TORCH_CHECK(weights.size(1) == num_heads, "weights head count differs");
  TORCH_CHECK(q_fp4.sizes() == at::IntArrayRef({num_tokens, num_heads, 64}),
              "q_fp4 must have shape [num_tokens, num_heads, 64]");
  TORCH_CHECK(q_scale.sizes() == at::IntArrayRef({num_tokens, num_heads, 4}),
              "q_scale must have shape [num_tokens, num_heads, 4]");
  TORCH_CHECK(weights_out.sizes() == weights.sizes(),
              "weights_out must have the same shape as weights");

  TORCH_CHECK(num_heads % 32 == 0,
              "num_heads must be divisible by 32 for this launch wrapper");

  c10::cuda::CUDAGuard device_guard(q.device());
  constexpr int kHeadDim = 128;
  constexpr int kRopeDim = 64;
  constexpr int kSubwarpSize = kHeadDim / 16;
  constexpr int kBlockSize = 256;
  const int total_threads = num_tokens * num_heads * kSubwarpSize;
  const int grid = (total_threads + kBlockSize - 1) / kBlockSize;

  fused_indexer_q_rope_mxfp4_kernel<kHeadDim, kRopeDim, kBlockSize>
      <<<grid, kBlockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
          positions.data_ptr<int64_t>(),
          reinterpret_cast<const nv_bfloat16*>(q.data_ptr()),
          cos_sin.data_ptr<float>(),
          reinterpret_cast<const nv_bfloat16*>(weights.data_ptr()),
          reinterpret_cast<char*>(q_fp4.data_ptr()),
          q_scale.data_ptr<uint8_t>(),
          weights_out.data_ptr<float>(),
          static_cast<float>(scale),
          num_tokens,
          num_heads,
          static_cast<int>(q.stride(0)),
          static_cast<int>(q.stride(1)),
          static_cast<int>(cos_sin.stride(0)),
          static_cast<int>(weights.stride(0)),
          static_cast<int>(q_fp4.stride(0)),
          static_cast<int>(q_fp4.stride(1)),
          static_cast<int>(q_scale.stride(0)),
          static_cast<int>(q_scale.stride(1)),
          static_cast<int>(weights_out.stride(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return q_fp4;
}

TORCH_LIBRARY(indexer_q_mxfp4, m) {
  m.def("fused_indexer_q_rope_mxfp4("
        "Tensor positions, Tensor q, Tensor cos_sin, Tensor weights, "
        "Tensor(a!) q_fp4, Tensor(b!) q_scale, Tensor(c!) weights_out, "
        "float scale) -> Tensor");
  m.impl("fused_indexer_q_rope_mxfp4",
         torch::dispatch(c10::DispatchKey::CUDA,
                         TORCH_FN(fused_indexer_q_rope_mxfp4)));
}
"""

HEAD_DIM = 128
ROPE_DIM = 64

load_inline(
    "indexer_q_mxfp4",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    verbose=False,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v",
    ],
    extra_ldflags=["-lcuda"],
)
_fused_indexer_q_rope_mxfp4 = torch.ops.indexer_q_mxfp4.fused_indexer_q_rope_mxfp4


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if not use_fp4:
        raise NotImplementedError("indexer_q_mxfp4 only implements use_fp4=True")
    assert index_q.ndim == 3 and index_q.shape[-1] == HEAD_DIM
    assert index_q_cos_sin_cache.ndim == 2
    assert index_q_cos_sin_cache.shape[-1] == ROPE_DIM

    num_tokens, num_heads, _ = index_q.shape
    q_fp4 = torch.empty(
        (num_tokens, num_heads, HEAD_DIM // 2),
        dtype=torch.uint8,
        device=index_q.device,
    )
    q_scale = torch.empty(
        (num_tokens, num_heads, HEAD_DIM // 32),
        dtype=torch.uint8,
        device=index_q.device,
    )
    weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    scale = float(index_weights_softmax_scale * index_weights_head_scale)
    _fused_indexer_q_rope_mxfp4(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        q_fp4,
        q_scale,
        weights_out,
        scale,
    )
    return (q_fp4, q_scale.view(torch.int32).squeeze(-1)), weights_out
