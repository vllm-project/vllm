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

template <typename scalar_t>
__global__ void gather_kv_hybrid_kernel(
    const scalar_t* src_k, const scalar_t* src_v, const scalar_t* gpu_k,
    const scalar_t* gpu_v,
    const int64_t* logical_ids, const int32_t* block_table,
    const int64_t* physical_to_slot, const uint8_t* offloaded_mask,
    scalar_t* out_k, scalar_t* out_v, int64_t kss, int64_t kst,
    int64_t ksh, int64_t vss, int64_t vst, int64_t vsh, int64_t gkss,
    int64_t gkst, int64_t gksh, int64_t gvss, int64_t gvst, int64_t gvsh,
    int64_t lh,
    int64_t out_k_head_stride, int64_t out_k_token_stride,
    int64_t out_v_head_stride, int64_t out_v_token_stride, int num_slots,
    int block_size, int head_dim, int num_gpu_blocks) {
  int head = blockIdx.x, token = blockIdx.y, dim = threadIdx.x;
  if (dim >= head_dim) return;
  int64_t logical = logical_ids[head * lh + token];
  if (logical < 0) return;
  int64_t lb = logical / block_size;
  int phys = block_table[lb];
  if (phys < 0 || phys >= num_gpu_blocks) return;
  int offset = logical - lb * block_size;
  int64_t dst_k =
      head * out_k_head_stride + token * out_k_token_stride + dim;
  int64_t dst_v =
      head * out_v_head_stride + token * out_v_token_stride + dim;
  int kv = blockIdx.z;
  if (offloaded_mask[phys]) {
    int64_t slot = physical_to_slot[phys];
    if (slot < 0 || slot >= num_slots) return;
    if (kv == 0) {
      int64_t src_k_idx = slot * kss + offset * kst + head * ksh + dim;
      out_k[dst_k] = src_k[src_k_idx];
    } else {
      int64_t src_v_idx = slot * vss + offset * vst + head * vsh + dim;
      out_v[dst_v] = src_v[src_v_idx];
    }
  } else {
    if (kv == 0) {
      int64_t src_k_idx = phys * gkss + offset * gkst + head * gksh + dim;
      out_k[dst_k] = gpu_k[src_k_idx];
    } else {
      int64_t src_v_idx = phys * gvss + offset * gvst + head * gvsh + dim;
      out_v[dst_v] = gpu_v[src_v_idx];
    }
  }
}

template <int HEAD_DIM>
__global__ void gather_kv_hybrid_uva_vec_kernel(
    const void* src_k_ptr, const void* src_v_ptr, const void* gpu_k_ptr,
    const void* gpu_v_ptr, const int64_t* logical_ids,
    const int32_t* block_table, const int64_t* physical_to_slot,
    const uint8_t* offloaded_mask, void* out_k_ptr, void* out_v_ptr,
    int64_t kss, int64_t kst, int64_t ksh, int64_t vss, int64_t vst,
    int64_t vsh, int64_t gkss, int64_t gkst, int64_t gksh, int64_t gvss,
    int64_t gvst, int64_t gvsh, int64_t lh, int64_t out_k_head_stride,
    int64_t out_k_token_stride, int64_t out_v_head_stride,
    int64_t out_v_token_stride, int num_slots, int block_size,
    int num_gpu_blocks) {
  constexpr int kVecElems = 8;  // uint4 = 16 bytes = 8 bf16/fp16 values.
  constexpr int kVecsPerHead = HEAD_DIM / kVecElems;
  int head = blockIdx.x;
  int token = blockIdx.y;
  int lane = threadIdx.x;
  bool is_value = lane >= kVecsPerHead;
  int vec = lane - (is_value ? kVecsPerHead : 0);
  if (vec >= kVecsPerHead) return;

  int64_t logical = logical_ids[head * lh + token];
  if (logical < 0) return;
  int64_t logical_block = logical / block_size;
  int phys = block_table[logical_block];
  if (phys < 0 || phys >= num_gpu_blocks) return;
  int offset = logical - logical_block * block_size;

  int64_t src_elem;
  const uint4* src;
  if (offloaded_mask[phys]) {
    int64_t slot = physical_to_slot[phys];
    if (slot < 0 || slot >= num_slots) return;
    if (is_value) {
      src = static_cast<const uint4*>(src_v_ptr);
      src_elem = slot * vss + offset * vst + head * vsh;
    } else {
      src = static_cast<const uint4*>(src_k_ptr);
      src_elem = slot * kss + offset * kst + head * ksh;
    }
  } else if (is_value) {
    src = static_cast<const uint4*>(gpu_v_ptr);
    src_elem = phys * gvss + offset * gvst + head * gvsh;
  } else {
    src = static_cast<const uint4*>(gpu_k_ptr);
    src_elem = phys * gkss + offset * gkst + head * gksh;
  }

  int64_t dst_elem;
  uint4* dst;
  if (is_value) {
    dst = static_cast<uint4*>(out_v_ptr);
    dst_elem = head * out_v_head_stride + token * out_v_token_stride;
  } else {
    dst = static_cast<uint4*>(out_k_ptr);
    dst_elem = head * out_k_head_stride + token * out_k_token_stride;
  }
  dst[dst_elem / kVecElems + vec] = src[src_elem / kVecElems + vec];
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

void h2d_gather_kv_hybrid(const at::Tensor& src_k, const at::Tensor& src_v,
                          const at::Tensor& gpu_k, const at::Tensor& gpu_v,
                          const at::Tensor& logical_ids,
                          const at::Tensor& block_table,
                          const at::Tensor& physical_to_slot,
                          const at::Tensor& offloaded_mask, at::Tensor& out_k,
                          at::Tensor& out_v) {
  TORCH_CHECK(!src_k.is_cuda() && src_k.is_pinned() && !src_v.is_cuda() &&
                  src_v.is_pinned(),
              "K/V sources must be pinned CPU tensors");
  TORCH_CHECK(logical_ids.is_cuda() && block_table.is_cuda() &&
                  physical_to_slot.is_cuda() && offloaded_mask.is_cuda(),
              "indices and maps must be CUDA tensors");
  TORCH_CHECK(gpu_k.is_cuda() && gpu_v.is_cuda() && out_k.is_cuda() &&
                  out_v.is_cuda(),
              "GPU K/V and outputs must be CUDA");
  TORCH_CHECK(src_k.scalar_type() == src_v.scalar_type() &&
                  src_k.scalar_type() == gpu_k.scalar_type() &&
                  src_k.scalar_type() == gpu_v.scalar_type() &&
                  src_k.scalar_type() == out_k.scalar_type() &&
                  src_k.scalar_type() == out_v.scalar_type(),
              "K/V dtype mismatch");
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int,
              "block_table must be int32");
  TORCH_CHECK(physical_to_slot.scalar_type() == at::ScalarType::Long,
              "physical_to_slot must be int64");
  c10::cuda::CUDAGuard guard(out_k.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int heads = logical_ids.size(0), tokens = logical_ids.size(1);
  int dim = src_k.size(3);
  bool vectorized =
      src_k.element_size() == 2 && (dim == 128 || dim == 256) &&
      src_k.stride(3) == 1 && src_v.stride(3) == 1 && gpu_k.stride(3) == 1 &&
      gpu_v.stride(3) == 1 && out_k.stride(2) == 1 && out_v.stride(2) == 1 &&
      src_k.stride(0) % 8 == 0 && src_k.stride(1) % 8 == 0 &&
      src_k.stride(2) % 8 == 0 && src_v.stride(0) % 8 == 0 &&
      src_v.stride(1) % 8 == 0 && src_v.stride(2) % 8 == 0 &&
      gpu_k.stride(0) % 8 == 0 && gpu_k.stride(1) % 8 == 0 &&
      gpu_k.stride(2) % 8 == 0 && gpu_v.stride(0) % 8 == 0 &&
      gpu_v.stride(1) % 8 == 0 && gpu_v.stride(2) % 8 == 0 &&
      out_k.stride(0) % 8 == 0 && out_k.stride(1) % 8 == 0 &&
      out_v.stride(0) % 8 == 0 && out_v.stride(1) % 8 == 0;
  if (vectorized) {
    void* host_k = mapped_device_pointer(src_k);
    void* host_v = mapped_device_pointer(src_v);
    dim3 grid(heads, tokens);
#define LAUNCH_VEC(DIM)                                                        \
  gather_kv_hybrid_uva_vec_kernel<DIM><<<grid, 2 * (DIM / 8), 0, stream>>>(  \
      host_k, host_v, gpu_k.data_ptr(), gpu_v.data_ptr(),                     \
      logical_ids.data_ptr<int64_t>(), block_table.data_ptr<int32_t>(),       \
      physical_to_slot.data_ptr<int64_t>(),                                   \
      reinterpret_cast<const uint8_t*>(offloaded_mask.data_ptr()),            \
      out_k.data_ptr(), out_v.data_ptr(), src_k.stride(0), src_k.stride(1),   \
      src_k.stride(2), src_v.stride(0), src_v.stride(1), src_v.stride(2),     \
      gpu_k.stride(0), gpu_k.stride(1), gpu_k.stride(2), gpu_v.stride(0),     \
      gpu_v.stride(1), gpu_v.stride(2), logical_ids.stride(0),                \
      out_k.stride(0), out_k.stride(1), out_v.stride(0), out_v.stride(1),    \
      src_k.size(0), src_k.size(1), offloaded_mask.size(0))
    if (dim == 128) {
      LAUNCH_VEC(128);
    } else {
      LAUNCH_VEC(256);
    }
#undef LAUNCH_VEC
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "vectorized hybrid K/V gather failed: ",
                cudaGetErrorString(error));
    return;
  }
  dim3 grid(heads, tokens, 2);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_k.scalar_type(),
      "zoomkv_h2d_gather_kv_hybrid", [&] {
        gather_kv_hybrid_kernel<scalar_t><<<grid, dim, 0, stream>>>(
            static_cast<const scalar_t*>(mapped_device_pointer(src_k)),
            static_cast<const scalar_t*>(mapped_device_pointer(src_v)),
            gpu_k.data_ptr<scalar_t>(), gpu_v.data_ptr<scalar_t>(),
            logical_ids.data_ptr<int64_t>(), block_table.data_ptr<int32_t>(),
            physical_to_slot.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t*>(offloaded_mask.data_ptr()),
            out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
            src_k.stride(0), src_k.stride(1), src_k.stride(2),
            src_v.stride(0), src_v.stride(1), src_v.stride(2),
            gpu_k.stride(0), gpu_k.stride(1), gpu_k.stride(2),
            gpu_v.stride(0), gpu_v.stride(1), gpu_v.stride(2),
            logical_ids.stride(0), out_k.stride(0), out_k.stride(1),
            out_v.stride(0), out_v.stride(1), src_k.size(0), src_k.size(1),
            dim, offloaded_mask.size(0));
      });
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "hybrid K/V gather failed: ", cudaGetErrorString(error));
}

#ifndef ZOOMKV_UNIFIED_EXTENSION
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h2d_gather_keys", &h2d_gather_keys);
  m.def("h2d_gather_keys_hybrid", &h2d_gather_keys_hybrid);
  m.def("h2d_gather_kv_hybrid", &h2d_gather_kv_hybrid);
}
#endif
