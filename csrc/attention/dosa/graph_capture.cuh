#pragma once
#include "common.cuh"

namespace DOSA_NAMESPACE {
// Ordered conditional siblings can reuse external scratch. Each body embeds
// a leaf graph, followed by an optional loop-controller kernel.
inline void append_conditional_bodies(cudaStream_t stream, cudaGraph_t graph,
                                      cudaGraphConditionalHandle handle,
                                      cudaGraphConditionalNodeType type,
                                      const int64_t* bodies, unsigned size,
                                      cudaGraphNodeParams* after_child = nullptr) {
  cudaStreamCaptureStatus status;
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edges;
  size_t count;
  DOSA_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                           &dependencies, &edges, &count));
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = type;
  params.conditional.size = size;
  cudaGraphNode_t conditional, child;
  DOSA_CUDA_CHECK(cudaGraphAddNode(&conditional, graph, dependencies, edges,
                                   count, &params));
  for (unsigned body = 0; body < size; ++body) {
    const auto graph_body = params.conditional.phGraph_out[body];
    DOSA_CUDA_CHECK(cudaGraphAddChildGraphNode(
        &child, graph_body, nullptr, 0, reinterpret_cast<cudaGraph_t>(bodies[body])));
    if (after_child != nullptr) {
      cudaGraphNode_t next;
      DOSA_CUDA_CHECK(cudaGraphAddNode(&next, graph_body, &child, nullptr, 1, after_child));
    }
  }
  DOSA_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream, &conditional, nullptr, 1, cudaStreamSetCaptureDependencies));
}

inline void append_if_body(cudaStream_t stream, cudaGraph_t graph,
                           cudaGraphConditionalHandle handle,
                           int64_t body_address) {
  append_conditional_bodies(stream, graph, handle, cudaGraphCondTypeIf,
                            &body_address, 1);
}

inline cudaGraph_t current_capture_graph(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  DOSA_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                           nullptr, nullptr, nullptr));
  STD_TORCH_CHECK(status == cudaStreamCaptureStatusActive,
                  "conditional attachment requires active stream capture");
  return graph;
}

inline cudaGraphConditionalHandle create_conditional(cudaGraph_t graph) {
  cudaGraphConditionalHandle handle;
  DOSA_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, graph, 0, cudaGraphCondAssignDefault));
  return handle;
}

}  // namespace DOSA_NAMESPACE
