// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// C++ bridge functions for push-based allreduce.
// Exposes PushAllReduceManager to Python via torch custom ops.

#include "push_all_reduce.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

using fptr_t = int64_t;
using namespace vllm::push_ar;

// Initialize the manager; returns opaque pointer as int64_t
fptr_t init_push_ar(int64_t rank, int64_t world_size, int64_t push_buffer_bytes,
                    int64_t max_num_cta) {
  auto* mgr = new PushAllReduceManager(
      static_cast<int>(rank), static_cast<int>(world_size), push_buffer_bytes,
      static_cast<int>(max_num_cta));
  return reinterpret_cast<fptr_t>(mgr);
}

// Get IPC handle as a byte tensor
torch::Tensor get_push_ar_ipc_handle(fptr_t _mgr) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  cudaIpcMemHandle_t handle = mgr->get_ipc_handle();
  auto t = torch::from_blob(&handle, {static_cast<int64_t>(sizeof(handle))},
                            torch::kUInt8)
               .clone();
  return t;
}

// Post-init with peer IPC handles
void post_init_push_ar(fptr_t _mgr, torch::Tensor all_handles) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  int world_size = all_handles.size(0);
  std::vector<cudaIpcMemHandle_t> handles(world_size);
  for (int i = 0; i < world_size; i++) {
    memcpy(&handles[i], all_handles[i].data_ptr(), sizeof(cudaIpcMemHandle_t));
  }
  mgr->post_init(handles);
}

// Check weak contiguity (same logic as vLLM custom_all_reduce.cu)
static bool _is_weak_contiguous(const torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          static_cast<size_t>(t.numel()) * t.element_size());
}

// Perform allreduce
void push_ar_all_reduce(fptr_t _mgr, torch::Tensor& inp, torch::Tensor& out) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp), "Input must be contiguous");
  TORCH_CHECK(_is_weak_contiguous(out), "Output must be contiguous");

  switch (out.scalar_type()) {
    case at::ScalarType::BFloat16:
      mgr->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    case at::ScalarType::Half:
      mgr->allreduce<half>(stream, reinterpret_cast<half*>(inp.data_ptr()),
                           reinterpret_cast<half*>(out.data_ptr()),
                           out.numel());
      break;
    case at::ScalarType::Float:
      mgr->allreduce<float>(stream, reinterpret_cast<float*>(inp.data_ptr()),
                            reinterpret_cast<float*>(out.data_ptr()),
                            out.numel());
      break;
    default:
      TORCH_CHECK(false,
                  "push allreduce: unsupported dtype (need bf16/fp16/fp32)");
  }
}

// Dispose the manager
void dispose_push_ar(fptr_t _mgr) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  delete mgr;
}
