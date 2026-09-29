// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused RHT + INT4 quantize + nibble pack for RDNA3 (gfx1100).
// Single HIP kernel replaces: 2× matmul RHT + Triton INT4 reshape.
// Uses parallel Hadamard butterfly in shared memory (128 threads/block).

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>

// Templated load/store helpers for fp16 and bf16 K/V dtypes.
// Both __half and __hip_bfloat16 are implicitly convertible to/from float.
template <typename T>
__device__ __forceinline__ float to_float(T x) {
  return (float)x;
}
template <>
__device__ __forceinline__ float to_float<half>(half x) {
  return __half2float(x);
}
template <typename T>
__device__ __forceinline__ T from_float(float x) {
  return (T)x;
}
template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

template <int HEAD_SIZE, typename T>
__global__ void reshape_cache_int4_rht_kernel(
    const T* __restrict__ key, const T* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    float* __restrict__ k_scale_cache, float* __restrict__ v_scale_cache,
    const float* __restrict__ rht_signs, const int* __restrict__ slot_mapping,
    int block_size, int64_t sk0, int64_t sk1,  // key strides: token, head
    int64_t sv0, int64_t sv1,                  // value strides
    int64_t scb, int64_t scs,
    int64_t sch,  // key_cache strides: blk, slot, head
    int64_t svb, int64_t svs, int64_t svh,       // value_cache strides
    int64_t ssb, int64_t sss, int64_t ssh,       // k_scale strides
    int64_t svvb, int64_t svvs, int64_t svvh) {  // v_scale strides

  const int tok = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;  // 0..HEAD_SIZE-1

  const int slot = slot_mapping[tok];
  if (slot < 0) return;
  const int blk = slot / block_size;
  const int slot_in_blk = slot % block_size;

  // Shared memory for parallel butterfly (K and V processed sequentially)
  __shared__ float s[HEAD_SIZE];

  // ---- Process K ----
  // Load + D₁ sign flip
  float kval = to_float<T>(key[tok * sk0 + head * sk1 + tid]);
  s[tid] = kval * rht_signs[tid];
  __syncthreads();

  // Parallel Hadamard butterfly (log2(HEAD_SIZE) stages)
  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = s[tid];
    float other = s[partner];
    __syncthreads();
    s[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }

  float k_rotated = s[tid];

  // Cooperative min/max reduction via shared mem
  // Reuse s[] for reduction
  __syncthreads();
  s[tid] = k_rotated;
  __syncthreads();

  // Parallel reduction for min and max
  for (int stride = HEAD_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] = fminf(s[tid], s[tid + stride]);
    }
    __syncthreads();
  }
  float k_min = s[0];
  __syncthreads();
  s[tid] = k_rotated;
  __syncthreads();
  for (int stride = HEAD_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] = fmaxf(s[tid], s[tid + stride]);
    }
    __syncthreads();
  }
  float k_max = s[0];

  // Asymmetric quantization with steganographic zp (original format, 16
  // levels). Full [0, 15] range with per-token-head zero-point for maximum
  // quality.
  float k_scale = fmaxf((k_max - k_min) / 15.0f, 1e-6f);
  float k_inv = 1.0f / k_scale;
  float k_zp_f = fminf(fmaxf((int)(-k_min * k_inv + 0.5f) >= 0
                                 ? (float)(int)(-k_min * k_inv + 0.5f)
                                 : (float)(int)(-k_min * k_inv - 0.5f),
                             0.0f),
                       15.0f);
  int k_zp = (int)k_zp_f;

  // Quantize to [0, 15]
  int k_q = (int)fminf(fmaxf(k_rotated * k_inv + k_zp_f + 0.5f, 0.0f), 15.0f);

  // Pack nibbles via shared memory (safe across wave boundaries).
  {
    __syncthreads();
    int* si = reinterpret_cast<int*>(s);
    si[tid] = k_q;
    __syncthreads();
    if ((tid & 1) == 0) {
      uint8_t packed = (uint8_t)((si[tid] & 0xF) | ((si[tid + 1] & 0xF) << 4));
      key_cache[blk * scb + slot_in_blk * scs + head * sch + tid / 2] = packed;
    }
  }

  // Store steganographed scale: zp in low 4 bits of float bit-pattern.
  if (tid == 0) {
    int sb;
    __builtin_memcpy(&sb, &k_scale, 4);
    int packed = (sb & ~0xF) | (k_zp & 0xF);
    float sp;
    __builtin_memcpy(&sp, &packed, 4);
    k_scale_cache[blk * ssb + slot_in_blk * sss + head * ssh] = sp;
  }

  // ---- Process V (same logic) ----
  __syncthreads();
  float vval = to_float<T>(value[tok * sv0 + head * sv1 + tid]);
  s[tid] = vval * rht_signs[tid];
  __syncthreads();

  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = s[tid];
    float other = s[partner];
    __syncthreads();
    s[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }

  float v_rotated = s[tid];
  __syncthreads();
  s[tid] = v_rotated;
  __syncthreads();
  for (int stride = HEAD_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] = fminf(s[tid], s[tid + stride]);
    __syncthreads();
  }
  float v_min = s[0];
  __syncthreads();
  s[tid] = v_rotated;
  __syncthreads();
  for (int stride = HEAD_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
    __syncthreads();
  }
  float v_max = s[0];

  // Asymmetric quantization for V (same as K)
  float v_scale = fmaxf((v_max - v_min) / 15.0f, 1e-6f);
  float v_inv = 1.0f / v_scale;
  float v_zp_f = fminf(fmaxf((int)(-v_min * v_inv + 0.5f) >= 0
                                 ? (float)(int)(-v_min * v_inv + 0.5f)
                                 : (float)(int)(-v_min * v_inv - 0.5f),
                             0.0f),
                       15.0f);
  int v_zp = (int)v_zp_f;
  int v_q = (int)fminf(fmaxf(v_rotated * v_inv + v_zp_f + 0.5f, 0.0f), 15.0f);

  {
    __syncthreads();
    int* si = reinterpret_cast<int*>(s);
    si[tid] = v_q;
    __syncthreads();
    if ((tid & 1) == 0) {
      uint8_t packed = (uint8_t)((si[tid] & 0xF) | ((si[tid + 1] & 0xF) << 4));
      value_cache[blk * svb + slot_in_blk * svs + head * svh + tid / 2] =
          packed;
    }
  }
  if (tid == 0) {
    int sb;
    __builtin_memcpy(&sb, &v_scale, 4);
    int packed = (sb & ~0xF) | (v_zp & 0xF);
    float sp;
    __builtin_memcpy(&sp, &packed, 4);
    v_scale_cache[blk * svvb + slot_in_blk * svvs + head * svvh] = sp;
  }
}

void reshape_cache_int4_rdna3(torch::Tensor key, torch::Tensor value,
                              torch::Tensor key_cache,
                              torch::Tensor value_cache,
                              torch::Tensor k_scale_cache,
                              torch::Tensor v_scale_cache,
                              torch::Tensor rht_signs,
                              torch::Tensor slot_mapping) {
  const int num_tokens = key.size(0);
  const int num_kv_heads = key.size(1);
  const int head_size = key.size(2);
  const int block_size = key_cache.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(head_size == 128 || head_size == 256,
              "reshape_cache_int4_rdna3: head_size must be 128 or 256, got ",
              head_size);
  TORCH_CHECK(key.dtype() == at::kHalf || key.dtype() == at::kBFloat16,
              "reshape_cache_int4_rdna3: key/value must be fp16 or bf16");
  TORCH_CHECK(value.dtype() == key.dtype(),
              "reshape_cache_int4_rdna3: key and value must share dtype");

  dim3 grid(num_tokens, num_kv_heads);
  #define LAUNCH_RESHAPE(HS, T)                                                \
    reshape_cache_int4_rht_kernel<HS, T><<<grid, dim3(HS), 0, stream>>>(       \
        (const T*)key.data_ptr(), (const T*)value.data_ptr(),                  \
        (uint8_t*)key_cache.data_ptr(), (uint8_t*)value_cache.data_ptr(),      \
        (float*)k_scale_cache.data_ptr(), (float*)v_scale_cache.data_ptr(),    \
        (const float*)rht_signs.data_ptr(),                                    \
        (const int*)slot_mapping.data_ptr(), block_size, key.stride(0),        \
        key.stride(1), value.stride(0), value.stride(1), key_cache.stride(0),  \
        key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),       \
        value_cache.stride(1), value_cache.stride(2), k_scale_cache.stride(0), \
        k_scale_cache.stride(1), k_scale_cache.stride(2),                      \
        v_scale_cache.stride(0), v_scale_cache.stride(1),                      \
        v_scale_cache.stride(2))
  const bool is_bf16 = (key.dtype() == at::kBFloat16);
  if (head_size == 128) {
    if (is_bf16) {
      LAUNCH_RESHAPE(128, __hip_bfloat16);
    } else {
      LAUNCH_RESHAPE(128, half);
    }
  } else {
    if (is_bf16) {
      LAUNCH_RESHAPE(256, __hip_bfloat16);
    } else {
      LAUNCH_RESHAPE(256, half);
    }
  }
  #undef LAUNCH_RESHAPE
}

// ---------------------------------------------------------------------------
// Lightweight inplace RHT butterfly for decode Q rotation / output unrotation.
// Grid: (num_rows), Block: (HEAD_SIZE). Each block = 1 row = 1 head.
// Forward: data = H × D₁ × data (sign flip, then butterfly)
// Inverse: data = D₁ × H × data × scale (butterfly, then sign flip + scale)
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename T>
__global__ void rht_butterfly_inplace_kernel(
    T* __restrict__ data, const float* __restrict__ signs,
    float post_scale,  // 1.0 for forward, 1/HEAD_SIZE for inverse
    bool inverse, int64_t stride_row, int64_t stride_head) {
  const int row = blockIdx.x;   // token index
  const int head = blockIdx.y;  // head index
  const int tid = threadIdx.x;

  __shared__ float s[HEAD_SIZE];

  float val = to_float<T>(data[row * stride_row + head * stride_head + tid]);

  if (!inverse) {
    // Forward RHT: sign flip first, then butterfly
    s[tid] = val * signs[tid];
  } else {
    s[tid] = val;
  }
  __syncthreads();

  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = s[tid];
    float other = s[partner];
    __syncthreads();
    s[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }

  if (inverse) {
    // Inverse RHT: sign flip after butterfly, fuse scale
    data[row * stride_row + head * stride_head + tid] =
        from_float<T>(s[tid] * signs[tid] * post_scale);
  } else {
    data[row * stride_row + head * stride_head + tid] =
        from_float<T>(s[tid] * post_scale);
  }
}

void rht_rotate_inplace_rdna3(torch::Tensor data, torch::Tensor rht_signs,
                              bool inverse, double post_scale) {
  TORCH_CHECK(data.dtype() == at::kHalf || data.dtype() == at::kBFloat16,
              "rht_rotate_inplace_rdna3: data must be fp16 or bf16");
  TORCH_CHECK(data.dim() == 3, "expected [tokens, heads, head_size]");
  const int num_tokens = data.size(0);
  const int num_heads = data.size(1);
  const int head_size = data.size(2);
  TORCH_CHECK(head_size == 128 || head_size == 256,
              "rht_rotate_inplace_rdna3: head_size must be 128 or 256, got ",
              head_size);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid(num_tokens, num_heads);
  #define LAUNCH_RHT(HS, T)                                             \
    rht_butterfly_inplace_kernel<HS, T><<<grid, dim3(HS), 0, stream>>>( \
        (T*)data.data_ptr(), (const float*)rht_signs.data_ptr(),        \
        (float)post_scale, inverse, data.stride(0), data.stride(1))
  const bool is_bf16 = (data.dtype() == at::kBFloat16);
  if (head_size == 128) {
    if (is_bf16) {
      LAUNCH_RHT(128, __hip_bfloat16);
    } else {
      LAUNCH_RHT(128, half);
    }
  } else {
    if (is_bf16) {
      LAUNCH_RHT(256, __hip_bfloat16);
    } else {
      LAUNCH_RHT(256, half);
    }
  }
  #undef LAUNCH_RHT
}

#else
void reshape_cache_int4_rdna3(torch::Tensor key, torch::Tensor value,
                              torch::Tensor key_cache,
                              torch::Tensor value_cache,
                              torch::Tensor k_scale_cache,
                              torch::Tensor v_scale_cache,
                              torch::Tensor rht_signs,
                              torch::Tensor slot_mapping) {
  TORCH_CHECK(false, "requires ROCm");
}
void rht_rotate_inplace_rdna3(torch::Tensor data, torch::Tensor rht_signs,
                              bool inverse, double post_scale) {
  TORCH_CHECK(false, "requires ROCm");
}
#endif
