#pragma once
#include "graph_capture.cuh"

namespace DOSA_NAMESPACE::native_fp32_while {
static __global__ void initialize(cudaGraphConditionalHandle handle,
                                  int* cursor, const int* count) {
  *cursor = 0;
  cudaGraphSetConditional(handle, *count > 0);
}
static __global__ void advance(cudaGraphConditionalHandle handle, int* cursor,
                               const int* count, int batch) {
  const int limit = *count;
  const int64_t next = int64_t(*cursor) + batch;
  *cursor = next < limit ? int(next) : limit;
  cudaGraphSetConditional(handle, *cursor < limit);
}

// Same-device leaf graph reads cursor/count and owns no conditional controller.
// Its tensors and external scratch outlive the parent graph. Count is stable
// during one replay; cursor is written only by these init/advance kernels.
inline void attach(cudaStream_t stream, int* cursor, const int* count,
                   int batch, cudaGraph_t leaf) {
  const auto graph = current_capture_graph(stream);
  auto handle = create_conditional(graph);
  initialize<<<1, 1, 0, stream>>>(handle, cursor, count);
  DOSA_CUDA_CHECK(cudaGetLastError());
  cudaGraphNodeParams step{};
  step.type = cudaGraphNodeTypeKernel;
  step.kernel.func = reinterpret_cast<void*>(advance);
  step.kernel.gridDim = dim3(1);
  step.kernel.blockDim = dim3(1);
  void* arguments[] = {&handle, &cursor, &count, &batch};
  step.kernel.kernelParams = arguments;
  const int64_t leaf_address = reinterpret_cast<int64_t>(leaf);
  append_conditional_bodies(stream, graph, handle, cudaGraphCondTypeWhile,
                            &leaf_address, 1, &step);
}
}  // namespace DOSA_NAMESPACE::native_fp32_while
