#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "fast_allreduce.cuh"

// fake pointer type
using fptr_t = uint64_t;
static_assert(sizeof(void *) == sizeof(fptr_t));

fptr_t prepare_buffer(fptr_t ptr, const std::vector<std::string> &handles,
                      const std::vector<int64_t> &offsets, int rank,
                      bool full_nvlink) {
  int world_size = offsets.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (world_size != handles.size())
    throw std::invalid_argument(
        "handles length should equal to offsets length");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  cudaIpcMemHandle_t ipc_handles[8];
  for (int i = 0; i < world_size; i++) {
    std::memcpy(&ipc_handles[i], handles[i].data(), sizeof(cudaIpcMemHandle_t));
  }
  return (fptr_t) new vllm::FastAllreduce(
      reinterpret_cast<vllm::Metadata *>(ptr), ipc_handles, offsets, rank,
      full_nvlink);
}

void allreduce(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out) {
  auto fa = reinterpret_cast<vllm::FastAllreduce *>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  switch (inp.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float *>(inp.data_ptr()),
                           reinterpret_cast<float *>(out.data_ptr()),
                           inp.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half *>(inp.data_ptr()),
                          reinterpret_cast<half *>(out.data_ptr()),
                          inp.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16 *>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16 *>(out.data_ptr()), inp.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "Fast allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::FastAllreduce *>(_fa);
  delete fa;
}

int meta_size() { return sizeof(vllm::Metadata); }

void register_buffer(fptr_t _fa, torch::Tensor &t,
                     const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets) {
  auto fa = reinterpret_cast<vllm::FastAllreduce *>(_fa);
  fa->register_buffer(handles, offsets, t.data_ptr());
}

std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::FastAllreduce *>(_fa);
  auto sz = fa->graph_unreg_buffers_.size();
  auto handle_sz = sizeof(cudaIpcMemHandle_t);
  std::vector<uint8_t> handles(handle_sz * sz, 0);
  std::vector<int64_t> offsets(sz);
  for (int i = 0; i < sz; i++) {
    auto ptr = fa->graph_unreg_buffers_[i];
    void *base_ptr;
    // note: must share the base address of each allocation, or we get wrong address
    auto _err = cuPointerGetAttribute(
        &base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr);
    if (_err != CUDA_SUCCESS)
      throw std::runtime_error("failed to get pointer attr");
    CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&handles[i * handle_sz],
                                  base_ptr));
    offsets[i] = ((char *)ptr) - ((char *)base_ptr);
  }
  return std::make_pair(handles, offsets);
}

void register_graph_buffers(fptr_t _fa, const std::vector<std::string> &handles,
                            const std::vector<std::vector<int64_t>> &offsets) {
  auto fa = reinterpret_cast<vllm::FastAllreduce *>(_fa);
  fa->register_graph_buffers(handles, offsets);
}
