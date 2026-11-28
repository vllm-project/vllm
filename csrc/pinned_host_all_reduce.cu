#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "pinned_host_all_reduce.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

int64_t pinned_host_ar_buffer_size(int64_t ll_max, int64_t flag_max,
                                   int64_t max_size) {
  return vllm::PinnedHostAllreduce::Layout(ll_max, flag_max, max_size).total;
}

/**
 * Registers `host_bytes` of shared host memory at `host_ptr` (page aligned,
 * zero filled, mapped by both ranks) as mapped pinned memory for the current
 * device. Inputs must be a multiple of 16 bytes and at most `max_size`; LL
 * mode serves up to `ll_max` bytes, flag mode up to `flag_max`, the
 * copy-engine mode the rest. Calls are numbered from `start_seq` + 1.
 */
fptr_t init_pinned_host_ar(int64_t host_ptr, int64_t host_bytes, int64_t rank,
                           int64_t ll_max, int64_t flag_max, int64_t max_size,
                           int64_t start_seq) {
  using State = vllm::PinnedHostAllreduce::State;
  STD_TORCH_CHECK(rank == 0 || rank == 1, "pinned-host allreduce is 2-rank");
  STD_TORCH_CHECK(start_seq >= 0);
  STD_TORCH_CHECK(ll_max % 16 == 0 && flag_max % 16 == 0 && max_size % 16 == 0);
  STD_TORCH_CHECK(ll_max <= flag_max && flag_max <= max_size);
  const vllm::PinnedHostAllreduce::Layout layout(ll_max, flag_max, max_size);
  STD_TORCH_CHECK(static_cast<int64_t>(layout.total) <= host_bytes,
                  "host buffer is smaller than the layout");
  auto* host = reinterpret_cast<char*>(host_ptr);
  STD_CUDA_CHECK(cudaHostRegister(
      host, host_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
  void* device = nullptr;
  State* state = nullptr;
  const State init{static_cast<vllm::PinnedHostAllreduce::Seq>(start_seq), 0};
  cudaError_t e = cudaHostGetDevicePointer(&device, host, 0);
  if (e == cudaSuccess) e = cudaMalloc(&state, sizeof(State));
  if (e == cudaSuccess)
    e = cudaMemcpy(state, &init, sizeof(State), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    cudaFree(state);
    cudaHostUnregister(host);
    STD_CUDA_CHECK(e);
  }
  return reinterpret_cast<fptr_t>(new vllm::PinnedHostAllreduce{
      host, static_cast<char*>(device), state, static_cast<int>(rank),
      static_cast<size_t>(ll_max), static_cast<size_t>(flag_max),
      static_cast<size_t>(max_size), layout});
}

void pinned_host_all_reduce(fptr_t _fa, torch::stable::Tensor& inp,
                            torch::stable::Tensor& out) {
  auto fa = reinterpret_cast<const vllm::PinnedHostAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK(inp.scalar_type() == out.scalar_type());
  STD_TORCH_CHECK(inp.numel() == out.numel());
  STD_TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  const size_t bytes = inp.numel() * inp.element_size();
  STD_TORCH_CHECK(bytes > 0 && bytes % 16 == 0 && bytes <= fa->max_size,
                  "pinned-host allreduce needs a multiple of 16 bytes up to ",
                  fa->max_size);
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(inp.const_data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(out.const_data_ptr()) % 16 == 0,
      "pinned-host allreduce needs 16-byte aligned tensors");
  cudaError_t e;
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float:
      e = fa->allreduce<float>(
          stream, reinterpret_cast<const float*>(inp.const_data_ptr()),
          reinterpret_cast<float*>(out.mutable_data_ptr()), bytes);
      break;
    case torch::headeronly::ScalarType::Half:
      e = fa->allreduce<half>(
          stream, reinterpret_cast<const half*>(inp.const_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()), bytes);
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16:
      e = fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<const nv_bfloat16*>(inp.const_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), bytes);
      break;
#endif
    default:
      throw std::runtime_error(
          "pinned-host allreduce only supports float32, float16 and bfloat16");
  }
  STD_CUDA_CHECK(e);
}

void dispose_pinned_host_ar(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::PinnedHostAllreduce*>(_fa);
  const cudaError_t errors[] = {cudaFree(fa->state),
                                cudaHostUnregister(fa->host)};
  delete fa;
  // At interpreter exit the CUDA runtime may already be unloading.
  for (const cudaError_t e : errors)
    if (e != cudaErrorCudartUnloading) STD_CUDA_CHECK(e);
}
