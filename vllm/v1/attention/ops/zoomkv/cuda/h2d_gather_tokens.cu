#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <mutex>
#include <unordered_map>

namespace {
void* mapped_device_pointer(const at::Tensor& host) {
  TORCH_CHECK(!host.is_cuda() && host.is_pinned(), "source must be pinned CPU");
  static std::mutex mutex;
  static std::unordered_map<void*, void*> aliases;
  void* ptr = host.data_ptr();
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = aliases.find(ptr);
    if (it != aliases.end()) return it->second;
  }
  void* alias = nullptr;
  auto error = cudaHostGetDevicePointer(&alias, ptr, 0);
  TORCH_CHECK(error == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(error));
  {
    std::lock_guard<std::mutex> guard(mutex);
    aliases.emplace(ptr, alias);
  }
  return alias;
}

template <typename scalar_t>
__global__ void gather_keys_kernel(const scalar_t* src_k, const int64_t* slots,
                                   const int64_t* offsets, scalar_t* out_k,
                                   int64_t ss, int64_t st, int64_t sh,
                                   int64_t out_token_stride,
                                   int64_t out_head_stride, int num_slots,
                                   int block_size, int num_heads,
                                   int head_dim) {
  int token = blockIdx.x, head = blockIdx.y, dim = threadIdx.x;
  if (dim >= head_dim) return;
  int64_t slot = slots[token], offset = offsets[token];
  bool valid =
      slot >= 0 && slot < num_slots && offset >= 0 && offset < block_size;
  int64_t dst = token * out_token_stride + head * out_head_stride + dim;
  if (!valid) {
    out_k[dst] = scalar_t(0);
    return;
  }
  int64_t src = slot * ss + offset * st + head * sh + dim;
  out_k[dst] = src_k[src];
}

template <typename scalar_t>
__global__ void gather_keys_hybrid_kernel(
    const scalar_t* src_k, const int64_t* logical_ids,
    const int32_t* block_table, const int64_t* cpu_slots,
    const uint8_t* offloaded_mask, scalar_t* out_k, int64_t ss, int64_t st,
    int64_t sh, int64_t lh, int64_t out_head_stride, int64_t out_token_stride,
    int start_block, int num_cpu_blocks, int num_slots, int block_size,
    int head_dim, int num_gpu_blocks) {
  int head = blockIdx.x, token = blockIdx.y, dim = threadIdx.x;
  if (dim >= head_dim) return;
  int64_t logical = logical_ids[head * lh + token];
  if (logical < 0) return;
  int64_t lb = logical / block_size;
  int phys = block_table[lb];
  if (phys < 0 || phys >= num_gpu_blocks) return;
  if (!offloaded_mask[phys]) return;
  int rel = lb - start_block;
  if (rel < 0 || rel >= num_cpu_blocks) return;
  int64_t slot = cpu_slots[rel];
  if (slot < 0 || slot >= num_slots) return;
  int offset = logical - lb * block_size;
  int64_t src = slot * ss + offset * st + head * sh + dim;
  int64_t dst = head * out_head_stride + token * out_token_stride + dim;
  out_k[dst] = src_k[src];
}
}  // namespace

void h2d_gather_keys(const at::Tensor& src_k, const at::Tensor& slots,
                     const at::Tensor& offsets, at::Tensor& out_k) {
  TORCH_CHECK(slots.is_cuda() && offsets.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(out_k.is_cuda(), "outputs must be CUDA");
  TORCH_CHECK(src_k.dim() == 4 && out_k.dim() == 3, "invalid rank");
  TORCH_CHECK(src_k.scalar_type() == out_k.scalar_type(), "dtype mismatch");
  c10::cuda::CUDAGuard guard(out_k.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int n = slots.numel(), heads = src_k.size(2), dim = src_k.size(3);
  dim3 grid(n, heads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_k.scalar_type(),
      "zoomkv_h2d_gather_keys", [&] {
        gather_keys_kernel<scalar_t><<<grid, dim, 0, stream>>>(
            static_cast<const scalar_t*>(mapped_device_pointer(src_k)),
            slots.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            out_k.data_ptr<scalar_t>(), src_k.stride(0), src_k.stride(1),
            src_k.stride(2), out_k.stride(0), out_k.stride(1), src_k.size(0),
            src_k.size(1), heads, dim);
      });
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "ZoomKV H2D key gather failed: ", cudaGetErrorString(error));
}

void h2d_gather_keys_hybrid(const at::Tensor& src_k,
                            const at::Tensor& logical_ids,
                            const at::Tensor& block_table,
                            const at::Tensor& cpu_slots,
                            const at::Tensor& offloaded_mask,
                            int64_t start_block, at::Tensor& out_k) {
  TORCH_CHECK(logical_ids.is_cuda() && block_table.is_cuda() &&
                  cpu_slots.is_cuda() && offloaded_mask.is_cuda(),
              "indices must be CUDA");
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int,
              "block_table must be int32");
  TORCH_CHECK(offloaded_mask.scalar_type() == at::ScalarType::Bool ||
                  offloaded_mask.scalar_type() == at::ScalarType::Byte,
              "offloaded_mask must be bool/uint8");
  c10::cuda::CUDAGuard guard(out_k.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int heads = logical_ids.size(0), tokens = logical_ids.size(1);
  int dim = src_k.size(3);
  dim3 grid(heads, tokens);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_k.scalar_type(),
      "zoomkv_h2d_gather_keys_hybrid", [&] {
        gather_keys_hybrid_kernel<scalar_t><<<grid, dim, 0, stream>>>(
            static_cast<const scalar_t*>(mapped_device_pointer(src_k)),
            logical_ids.data_ptr<int64_t>(), block_table.data_ptr<int32_t>(),
            cpu_slots.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t*>(offloaded_mask.data_ptr()),
            out_k.data_ptr<scalar_t>(), src_k.stride(0), src_k.stride(1),
            src_k.stride(2), logical_ids.stride(0), out_k.stride(0),
            out_k.stride(1), start_block, cpu_slots.numel(), src_k.size(0),
            src_k.size(1), dim, offloaded_mask.size(0));
      });
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "hybrid key gather failed: ", cudaGetErrorString(error));
}

#ifndef ZOOMKV_UNIFIED_EXTENSION
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h2d_gather_keys", &h2d_gather_keys);
  m.def("h2d_gather_keys_hybrid", &h2d_gather_keys_hybrid);
}
#endif
