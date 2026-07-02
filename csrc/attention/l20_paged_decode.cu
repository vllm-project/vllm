#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <cmath>

namespace {

constexpr int kHeadDim = 128;
__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_decode_kernel(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    const int* block_table,
    const int* seq_lens,
    half* output,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int max_pages) {
  const int batch = blockIdx.y;
  const int q_head = blockIdx.x;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  __shared__ float scores[16];
  __shared__ float probabilities[16];
  __shared__ float alpha_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;

  const int q_base = (batch * num_q_heads + q_head) * 128;
  const int pair0 = lane * 2;
  const int pair1 = pair0 + 64;
  const half2 q01 = *reinterpret_cast<const half2*>(query + q_base + pair0);
  const half2 q23 = *reinterpret_cast<const half2*>(query + q_base + pair1);
  const float2 q01f = __half22float2(q01);
  const float2 q23f = __half22float2(q23);
  float2 accumulator = make_float2(0.0f, 0.0f);
  const int seq_len = seq_lens[batch];
  if (thread == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < seq_len; tile_start += 16) {
#pragma unroll
    for (int warp_token = 0; warp_token < 2; ++warp_token) {
      const int token_index = warp + warp_token * 8;
      const int token = tile_start + token_index;
      float dot = 0.0f;
      if (token < seq_len) {
        const int logical_page = token / page_size;
        const int page_offset = token - logical_page * page_size;
        const int physical_page = block_table[batch * max_pages + logical_page];
        const int cache_base =
            ((physical_page * page_size + page_offset) * num_kv_heads + kv_head) *
            128;
        const half2 k01 =
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair0);
        const half2 k23 =
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair1);
        const float2 k01f = __half22float2(k01);
        const float2 k23f = __half22float2(k23);
        dot = q01f.x * k01f.x + q01f.y * k01f.y +
              q23f.x * k23f.x + q23f.y * k23f.y;
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        scores[token_index] = token < seq_len
            ? dot * 0.08838834764831845f
            : -INFINITY;
      }
    }
    __syncthreads();
    if (thread == 0) {
      float tile_max = scores[0];
#pragma unroll
      for (int index = 1; index < 16; ++index) {
        tile_max = fmaxf(tile_max, scores[index]);
      }
      const float next_max = fmaxf(running_max_shared, tile_max);
      alpha_shared = expf(running_max_shared - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        probabilities[index] = expf(scores[index] - next_max);
        tile_sum += probabilities[index];
      }
      running_sum_shared = running_sum_shared * alpha_shared + tile_sum;
      running_max_shared = next_max;
    }
    __syncthreads();
    if (thread < 64) {
      accumulator.x *= alpha_shared;
      accumulator.y *= alpha_shared;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        const int value_token = tile_start + index;
        if (value_token < seq_len) {
          const int logical_page = value_token / page_size;
          const int page_offset = value_token - logical_page * page_size;
          const int physical_page =
              block_table[batch * max_pages + logical_page];
          const int cache_offset =
              ((physical_page * page_size + page_offset) * num_kv_heads +
               kv_head) *
                  128 +
              thread * 2;
          const half2 value =
              *reinterpret_cast<const half2*>(value_cache + cache_offset);
          const float2 value_float = __half22float2(value);
          accumulator.x += probabilities[index] * value_float.x;
          accumulator.y += probabilities[index] * value_float.y;
        }
      }
    }
    __syncthreads();
  }
  if (thread < 64) {
    const half2 result = __floats2half2_rn(
        accumulator.x / running_sum_shared,
        accumulator.y / running_sum_shared);
    *reinterpret_cast<half2*>(
        output + (batch * num_q_heads + q_head) * 128 + thread * 2) = result;
  }
}

__global__ void paged_decode_partial_kernel(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    const int* block_table,
    const int* seq_lens,
    half* partial_output,
    float* partial_max,
    float* partial_sum,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int max_pages,
    int num_splits,
    int split_size,
    const int* page_indptr,
    bool use_indptr) {
  const int q_head = blockIdx.x;
  const int split = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = q_head / (num_q_heads / num_kv_heads);
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int split_start = split * split_size;
  const int split_end = min(split_start + split_size, seq_lens[batch]);
  __shared__ float scores[16];
  __shared__ float probabilities[16];
  __shared__ float alpha_shared;
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;
  __shared__ int physical_page_shared;

  const int q_base = (batch * num_q_heads + q_head) * kHeadDim;
  const int pair0 = lane * 2;
  const int pair1 = pair0 + 64;
  const float2 q01f = __half22float2(
      *reinterpret_cast<const half2*>(query + q_base + pair0));
  const float2 q23f = __half22float2(
      *reinterpret_cast<const half2*>(query + q_base + pair1));
  float2 accumulator = make_float2(0.0f, 0.0f);
  if (thread == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int tile_start = split_start; tile_start < split_end; tile_start += 16) {
    if (thread == 0) {
      const int logical_page = tile_start / page_size;
      const int page_index = use_indptr
          ? page_indptr[batch] + logical_page
          : batch * max_pages + logical_page;
      physical_page_shared = block_table[page_index];
    }
    __syncthreads();
#pragma unroll
    for (int warp_token = 0; warp_token < 2; ++warp_token) {
      const int token_index = warp + warp_token * 8;
      const int token = tile_start + token_index;
      float dot = 0.0f;
      if (token < split_end) {
        const int page_offset = token - tile_start;
        const int cache_base =
            ((physical_page_shared * page_size + page_offset) * num_kv_heads +
             kv_head) *
            kHeadDim;
        const float2 k01f = __half22float2(
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair0));
        const float2 k23f = __half22float2(
            *reinterpret_cast<const half2*>(key_cache + cache_base + pair1));
        dot = q01f.x * k01f.x + q01f.y * k01f.y +
              q23f.x * k23f.x + q23f.y * k23f.y;
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        scores[token_index] = token < split_end
            ? dot * 0.08838834764831845f
            : -INFINITY;
      }
    }
    __syncthreads();
    if (thread == 0) {
      float tile_max = scores[0];
#pragma unroll
      for (int index = 1; index < 16; ++index) {
        tile_max = fmaxf(tile_max, scores[index]);
      }
      const float next_max = fmaxf(running_max_shared, tile_max);
      alpha_shared = expf(running_max_shared - next_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        probabilities[index] = expf(scores[index] - next_max);
        tile_sum += probabilities[index];
      }
      running_sum_shared = running_sum_shared * alpha_shared + tile_sum;
      running_max_shared = next_max;
    }
    __syncthreads();
    if (thread < 64) {
      accumulator.x *= alpha_shared;
      accumulator.y *= alpha_shared;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        const int token = tile_start + index;
        if (token < split_end) {
          const int cache_offset =
              ((physical_page_shared * page_size + index) * num_kv_heads +
               kv_head) *
                  kHeadDim +
              thread * 2;
          const float2 value = __half22float2(
              *reinterpret_cast<const half2*>(value_cache + cache_offset));
          accumulator.x += probabilities[index] * value.x;
          accumulator.y += probabilities[index] * value.y;
        }
      }
    }
    __syncthreads();
  }

  const int partial_index =
      ((batch * num_q_heads + q_head) * num_splits + split);
  if (thread < 64) {
    *reinterpret_cast<half2*>(
        partial_output + partial_index * kHeadDim + thread * 2) =
        __floats2half2_rn(accumulator.x, accumulator.y);
  }
  if (thread == 0) {
    partial_max[partial_index] = running_max_shared;
    partial_sum[partial_index] = running_sum_shared;
  }
}

__global__ void paged_decode_merge_kernel(
    const half* partial_output,
    const float* partial_max,
    const float* partial_sum,
    half* output,
    int num_q_heads,
    int num_splits) {
  const int q_head = blockIdx.x;
  const int batch = blockIdx.y;
  const int pair = threadIdx.x;
  const int base = (batch * num_q_heads + q_head) * num_splits;
  __shared__ float global_max;
  __shared__ float denominator;
  __shared__ float corrections[64];
  if (pair == 0) {
    float max_value = partial_max[base];
    for (int split = 1; split < num_splits; ++split) {
      max_value = fmaxf(max_value, partial_max[base + split]);
    }
    float sum = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      corrections[split] = expf(partial_max[base + split] - max_value);
      sum += partial_sum[base + split] * corrections[split];
    }
    global_max = max_value;
    denominator = sum;
  }
  __syncthreads();
  float2 numerator = make_float2(0.0f, 0.0f);
  for (int split = 0; split < num_splits; ++split) {
    const float2 partial = __half22float2(
        *reinterpret_cast<const half2*>(
            partial_output + (base + split) * kHeadDim + pair * 2));
    numerator.x += partial.x * corrections[split];
    numerator.y += partial.y * corrections[split];
  }
  *reinterpret_cast<half2*>(
      output + (batch * num_q_heads + q_head) * kHeadDim + pair * 2) =
      __floats2half2_rn(numerator.x / denominator, numerator.y / denominator);
}

}  // namespace

torch::Tensor l20_paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16, "FP16 only");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 128, "Q must be [B,H,128]");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(3) == 128, "NHD cache only");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "K/V cache mismatch");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "int32 block table");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "int32 sequence lengths");
  const at::cuda::CUDAGuard guard(query.device());
  auto output = torch::empty_like(query);
  const dim3 grid(query.size(1), query.size(0));
  paged_decode_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      key_cache.size(2),
      key_cache.size(1),
      block_table.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void l20_paged_decode_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size);

torch::Tensor l20_paged_decode_split_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_seq_len,
    int64_t split_size) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16, "FP16 only");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kHeadDim, "invalid query");
  TORCH_CHECK(
      split_size >= 64 && split_size <= 1024 && split_size % 16 == 0,
      "split_size must be a multiple of 16 from 64 through 1024");
  const int num_splits = (max_seq_len + split_size - 1) / split_size;
  auto partial_output = torch::empty(
      {query.size(0), query.size(1), num_splits, kHeadDim},
      query.options());
  auto float_options = query.options().dtype(torch::kFloat32);
  auto partial_max =
      torch::empty({query.size(0), query.size(1), num_splits}, float_options);
  auto partial_sum = torch::empty_like(partial_max);
  auto output = torch::empty_like(query);
  l20_paged_decode_split_out_cuda(
      query,
      key_cache,
      value_cache,
      block_table,
      seq_lens,
      partial_output,
      partial_max,
      partial_sum,
      output,
      max_seq_len,
      split_size);
  return output;
}

void l20_paged_decode_split_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size) {
  const at::cuda::CUDAGuard guard(query.device());
  const int num_splits = (max_seq_len + split_size - 1) / split_size;
  TORCH_CHECK(
      partial_output.size(2) >= num_splits &&
          partial_output.size(3) == kHeadDim,
      "partial output workspace has wrong shape");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(query.size(1), num_splits, query.size(0));
  paged_decode_partial_kernel<<<partial_grid, 256, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      query.size(1),
      key_cache.size(2),
      key_cache.size(1),
      block_table.size(1),
      num_splits,
      split_size,
      nullptr,
      false);
  paged_decode_merge_kernel<<<
      dim3(query.size(1), query.size(0)),
      kHeadDim / 2,
      0,
      stream>>>(
      reinterpret_cast<const half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void l20_paged_decode_split_indices_out_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor page_indptr,
    torch::Tensor page_indices,
    torch::Tensor seq_lens,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t max_seq_len,
    int64_t split_size) {
  const at::cuda::CUDAGuard guard(query.device());
  const int num_splits = (max_seq_len + split_size - 1) / split_size;
  const auto stream = at::cuda::getCurrentCUDAStream();
  paged_decode_partial_kernel<<<
      dim3(query.size(1), num_splits, query.size(0)),
      256,
      0,
      stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      page_indices.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      query.size(1),
      key_cache.size(2),
      key_cache.size(1),
      0,
      num_splits,
      split_size,
      page_indptr.data_ptr<int>(),
      true);
  paged_decode_merge_kernel<<<
      dim3(query.size(1), query.size(0)),
      kHeadDim / 2,
      0,
      stream>>>(
      reinterpret_cast<const half*>(partial_output.data_ptr<at::Half>()),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      query.size(1),
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
