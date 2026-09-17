#include "torch_utils.h"

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "custom_all_reduce.cuh"
#include "custom_all_gather_reduce_scatter.cuh"

namespace vllm {

void CustomAllreduce::allgather(cudaStream_t stream, void* input, void* output,
                                int size_bytes, int threads, int block_limit) {
  if (size_bytes % sizeof(CopyPack) != 0)
    throw std::runtime_error(
        "custom allgather requires input byte size to be a multiple of " +
        std::to_string(sizeof(CopyPack)));

  auto ptrs = buffers_.at(input);
  int size_per_rank = size_bytes / sizeof(CopyPack);
  int total_size = size_per_rank * world_size_;
  int blocks = std::min(block_limit, (total_size + threads - 1) / threads);

#define AG_CASE(ngpus)                                                   \
  case ngpus:                                                            \
    cross_device_all_gather<ngpus><<<blocks, threads, 0, stream>>>(      \
        ptrs, sg_, self_sg_, reinterpret_cast<CopyPack*>(output), rank_, \
        size_per_rank);                                                  \
    break;

  switch (world_size_) {
    AG_CASE(2)
    AG_CASE(4)
    AG_CASE(6)
    AG_CASE(8)
    default:
      throw std::runtime_error(
          "custom allgather only supports num gpus in (2,4,6,8)");
  }
#undef AG_CASE
}

template <typename T>
void CustomAllreduce::mnnvl_lamport_allgather(cudaStream_t stream, T* input,
                                              T* output, void* local_buffer,
                                              void* multicast_buffer,
                                              uint32_t* epochs, int size_bytes,
                                              int stage_size_bytes) {
  if (size_bytes % sizeof(typename packed_t<T>::P) != 0 ||
      stage_size_bytes % sizeof(typename packed_t<T>::P) != 0)
    throw std::runtime_error(
        "MNNVL Lamport allgather requires 16-byte aligned sizes");

  auto ptrs = buffers_.at(local_buffer);
  int size_per_rank = size_bytes / sizeof(typename packed_t<T>::P);
  int stage_size = stage_size_bytes / sizeof(typename packed_t<T>::P);
  int blocks =
      (size_per_rank + kMnnvlLamportAgThreads - 1) / kMnnvlLamportAgThreads;

#if !defined(USE_ROCM) && CUDA_VERSION >= 12000
  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{.gridDim = dim3(blocks),
                            .blockDim = dim3(kMnnvlLamportAgThreads),
                            .dynamicSmemBytes = 0,
                            .stream = stream,
                            .attrs = attributes,
                            .numAttrs = 1};
  #define MNNVL_LAMPORT_AG_LAUNCH(ngpus)                                       \
    CUDACHECK(cudaLaunchKernelEx(&config, &mnnvl_lamport_all_gather<T, ngpus>, \
                                 ptrs, input, output,                          \
                                 reinterpret_cast<T*>(multicast_buffer),       \
                                 epochs, rank_, size_per_rank, stage_size))
#else
  #define MNNVL_LAMPORT_AG_LAUNCH(ngpus)                                 \
    mnnvl_lamport_all_gather<T, ngpus>                                   \
        <<<blocks, kMnnvlLamportAgThreads, 0, stream>>>(                 \
            ptrs, input, output, reinterpret_cast<T*>(multicast_buffer), \
            epochs, rank_, size_per_rank, stage_size)
#endif

#define MNNVL_LAMPORT_AG_CASE(ngpus) \
  case ngpus:                        \
    MNNVL_LAMPORT_AG_LAUNCH(ngpus);  \
    break;

  switch (world_size_) {
    MNNVL_LAMPORT_AG_CASE(2)
    MNNVL_LAMPORT_AG_CASE(4)
    MNNVL_LAMPORT_AG_CASE(6)
    MNNVL_LAMPORT_AG_CASE(8)
    MNNVL_LAMPORT_AG_CASE(16)
    default:
      throw std::runtime_error(
          "MNNVL Lamport allgather only supports num gpus in (2,4,6,8,16)");
  }
#undef MNNVL_LAMPORT_AG_CASE
#undef MNNVL_LAMPORT_AG_LAUNCH
}

template <typename T>
void CustomAllreduce::reduce_scatter(cudaStream_t stream, T* input, T* output,
                                     int size, int threads, int block_limit) {
  auto packed_size = packed_t<T>::P::size;
  if (size % (packed_size * world_size_) != 0)
    throw std::runtime_error(
        "custom reduce-scatter requires each output shard byte size to be "
        "a multiple of 16");

  auto ptrs = buffers_.at(input);
  int size_per_rank = size / packed_size / world_size_;
  int blocks = std::min(block_limit, (size_per_rank + threads - 1) / threads);

#define RS_CASE(ngpus)                                                     \
  case ngpus:                                                              \
    cross_device_reduce_scatter<T, ngpus><<<blocks, threads, 0, stream>>>( \
        ptrs, sg_, self_sg_, output, rank_, size_per_rank);                \
    break;

  switch (world_size_) {
    RS_CASE(2)
    RS_CASE(4)
    RS_CASE(6)
    RS_CASE(8)
    default:
      throw std::runtime_error(
          "custom reduce-scatter only supports num gpus in (2,4,6,8)");
  }
#undef RS_CASE
}

template <typename T>
void CustomAllreduce::mnnvl_lamport_reduce_scatter(cudaStream_t stream,
                                                   T* input, T* output,
                                                   void* local_buffer,
                                                   uint32_t* epochs, int size,
                                                   int stage_size_bytes) {
  auto packed_size = packed_t<T>::P::size;
  if (size % (packed_size * world_size_) != 0 ||
      stage_size_bytes % sizeof(typename packed_t<T>::P) != 0)
    throw std::runtime_error(
        "MNNVL Lamport reduce-scatter requires 16-byte aligned sizes");

  auto ptrs = buffers_.at(local_buffer);
  int size_per_rank = size / packed_size / world_size_;
  int stage_size = stage_size_bytes / sizeof(typename packed_t<T>::P);
  int blocks_per_rank =
      (size_per_rank + kMnnvlLamportRsThreads - 1) / kMnnvlLamportRsThreads;
  int blocks = blocks_per_rank * world_size_;

#if !defined(USE_ROCM) && CUDA_VERSION >= 12000
  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{.gridDim = dim3(blocks),
                            .blockDim = dim3(kMnnvlLamportRsThreads),
                            .dynamicSmemBytes = 0,
                            .stream = stream,
                            .attrs = attributes,
                            .numAttrs = 1};
  #define MNNVL_LAMPORT_RS_LAUNCH(ngpus)                                      \
    CUDACHECK(cudaLaunchKernelEx(                                             \
        &config, &mnnvl_lamport_reduce_scatter_kernel<T, ngpus>, ptrs, input, \
        output, epochs, rank_, size_per_rank, stage_size))
#else
  #define MNNVL_LAMPORT_RS_LAUNCH(ngpus)                 \
    mnnvl_lamport_reduce_scatter_kernel<T, ngpus>        \
        <<<blocks, kMnnvlLamportRsThreads, 0, stream>>>( \
            ptrs, input, output, epochs, rank_, size_per_rank, stage_size)
#endif

#define MNNVL_LAMPORT_RS_CASE(ngpus) \
  case ngpus:                        \
    MNNVL_LAMPORT_RS_LAUNCH(ngpus);  \
    break;

  switch (world_size_) {
    MNNVL_LAMPORT_RS_CASE(2)
    MNNVL_LAMPORT_RS_CASE(4)
    MNNVL_LAMPORT_RS_CASE(6)
    MNNVL_LAMPORT_RS_CASE(8)
    MNNVL_LAMPORT_RS_CASE(16)
    default:
      throw std::runtime_error(
          "MNNVL Lamport reduce-scatter only supports num gpus in "
          "(2,4,6,8,16)");
  }
#undef MNNVL_LAMPORT_RS_CASE
#undef MNNVL_LAMPORT_RS_LAUNCH
}

}  // namespace vllm

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

bool _is_weak_contiguous(torch::stable::Tensor& t);

void custom_all_gather(fptr_t _fa, torch::stable::Tensor& inp,
                       torch::stable::Tensor& out, fptr_t _reg_buffer,
                       int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.numel() * fa->world_size_) == (out.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  STD_TORCH_CHECK(reg_buffer != nullptr);
  STD_TORCH_CHECK((input_size) <= (reg_buffer_sz_bytes));
  STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                 cudaMemcpyDeviceToDevice, stream));
  fa->allgather(stream, reg_buffer, out.mutable_data_ptr(), input_size);
}

void mnnvl_lamport_all_gather(fptr_t _fa, torch::stable::Tensor& inp,
                              torch::stable::Tensor& out, fptr_t _local_buffer,
                              fptr_t _multicast_buffer, fptr_t _epoch_buffer,
                              int64_t stage_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.numel() * fa->world_size_) == (out.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  STD_TORCH_CHECK((input_size * fa->world_size_) <= stage_sz_bytes);
  auto local_buffer = reinterpret_cast<void*>(_local_buffer);
  auto multicast_buffer = reinterpret_cast<void*>(_multicast_buffer);
  auto epochs = reinterpret_cast<uint32_t*>(_epoch_buffer);
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->mnnvl_lamport_allgather<float>(
          stream, reinterpret_cast<float*>(inp.mutable_data_ptr()),
          reinterpret_cast<float*>(out.mutable_data_ptr()), local_buffer,
          multicast_buffer, epochs, input_size, stage_sz_bytes);
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->mnnvl_lamport_allgather<half>(
          stream, reinterpret_cast<half*>(inp.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()), local_buffer,
          multicast_buffer, epochs, input_size, stage_sz_bytes);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->mnnvl_lamport_allgather<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.mutable_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), local_buffer,
          multicast_buffer, epochs, input_size, stage_sz_bytes);
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "MNNVL Lamport allgather only supports float32, float16 and "
          "bfloat16");
  }
}

void custom_reduce_scatter(fptr_t _fa, torch::stable::Tensor& inp,
                           torch::stable::Tensor& out, fptr_t _reg_buffer,
                           int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((out.numel() * fa->world_size_) == (inp.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  STD_TORCH_CHECK(reg_buffer != nullptr);
  STD_TORCH_CHECK((input_size) <= (reg_buffer_sz_bytes));
  STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                 cudaMemcpyDeviceToDevice, stream));
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->reduce_scatter<float>(
          stream, reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(out.mutable_data_ptr()), inp.numel());
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->reduce_scatter<half>(stream, reinterpret_cast<half*>(reg_buffer),
                               reinterpret_cast<half*>(out.mutable_data_ptr()),
                               inp.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->reduce_scatter<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), inp.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom reduce-scatter only supports float32, float16 and bfloat16");
  }
}

void mnnvl_lamport_reduce_scatter(fptr_t _fa, torch::stable::Tensor& inp,
                                  torch::stable::Tensor& out,
                                  fptr_t _local_buffer, fptr_t _epoch_buffer,
                                  int64_t stage_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((out.numel() * fa->world_size_) == (inp.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  STD_TORCH_CHECK(input_size <= stage_sz_bytes);
  auto local_buffer = reinterpret_cast<void*>(_local_buffer);
  auto epochs = reinterpret_cast<uint32_t*>(_epoch_buffer);
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->mnnvl_lamport_reduce_scatter<float>(
          stream, reinterpret_cast<float*>(inp.mutable_data_ptr()),
          reinterpret_cast<float*>(out.mutable_data_ptr()), local_buffer,
          epochs, inp.numel(), stage_sz_bytes);
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->mnnvl_lamport_reduce_scatter<half>(
          stream, reinterpret_cast<half*>(inp.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()), local_buffer, epochs,
          inp.numel(), stage_sz_bytes);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->mnnvl_lamport_reduce_scatter<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.mutable_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), local_buffer,
          epochs, inp.numel(), stage_sz_bytes);
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "MNNVL Lamport reduce-scatter only supports float32, float16 and "
          "bfloat16");
  }
}
