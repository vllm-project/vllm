#include "common.cuh"

namespace DOSA_NAMESPACE {
namespace native_ties {
constexpr int Threads = 128, Top = 2048;
template <int Items>
using Sort = cub::BlockRadixSort<int, Threads, Items, int>;
struct Args {
  const int *boundary, *counts, *diag, *starts, *ends, *permutation;
  int *error, *output;
  int nk;
  int64_t stride;
};

template <int Items>
__device__ void finish(Args a, int row, int b, int e, void* storage) {
  int keys[Items], values[Items];
  bool bad = false;
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    int at = i * Threads + threadIdx.x;
    int id = at < e ? a.boundary[int64_t(row) * a.stride + at] : -1;
    bool valid = id >= a.starts[row] && id < a.ends[row];
    bad |= at < e && !valid;
    values[i] = id;
    keys[i] = at < e && valid ? a.permutation[id] : INT_MAX;
  }
  if (__syncthreads_or(bad)) {
    if (threadIdx.x == 0) atomicOr(a.error + row, 1);
    return;
  }
  Sort<Items>(*static_cast<typename Sort<Items>::TempStorage*>(storage))
      .SortBlockedToStriped(keys, values);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    int at = i * Threads + threadIdx.x;
    if (at < Top - b) a.output[int64_t(row) * Top + b + at] = values[i];
  }
}

__global__ void complete(Args a) {
  const int row = blockIdx.x;
  if (a.error[row]) return;
  const int b = a.diag[row * 2], e = a.counts[row];
  if (b < 0 || b >= Top || e <= 0 || e > 4096 || b + e < Top ||
      e != a.diag[row * 2 + 1] || a.starts[row] < 0 ||
      a.ends[row] < a.starts[row] || a.ends[row] > a.nk) {
    if (threadIdx.x == 0) atomicOr(a.error + row, 1);
    return;
  }
  extern __shared__ __align__(16) uint8_t storage[];
  // The pivot cell is an exact native BF16 score. Ties need only logical IDs.
  if (e <= 128)
    finish<1>(a, row, b, e, storage);
  else if (e <= 512)
    finish<4>(a, row, b, e, storage);
  else
    finish<32>(a, row, b, e, storage);
}
}  // namespace native_ties

void complete_ties(Tensor boundary, Tensor counts, Tensor diag, Tensor starts,
                   Tensor ends, Tensor error, Tensor output,
                   Tensor permutation) {
  DeviceGuard guard(output.get_device_index());
  native_ties::Args args{
      boundary.const_data_ptr<int>(), counts.const_data_ptr<int>(),
      diag.const_data_ptr<int>(),     starts.const_data_ptr<int>(),
      ends.const_data_ptr<int>(),     permutation.const_data_ptr<int>(),
      error.mutable_data_ptr<int>(),  output.mutable_data_ptr<int>(),
      int(permutation.numel()),       boundary.stride(0)};
  native_ties::complete<<<output.size(0), 128,
                          sizeof(native_ties::Sort<32>::TempStorage),
                          get_current_cuda_stream()>>>(args);
  DOSA_CUDA_CHECK(cudaGetLastError());
}

__global__ void set_if_any(cudaGraphConditionalHandle handle,
                           const int32_t* values, int count, bool positive) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    active |= (positive ? values[i] > 0 : values[i] != 0);
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

__global__ void set_if_any_not_in(cudaGraphConditionalHandle handle,
                                  const int32_t* values, int count,
                                  int inactive0, int inactive1) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int value = values[i];
    active |= value != inactive0 && value != inactive1;
  }
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

// The supplied child graph is immutable and its tensor allocations must remain
// alive. cudaGraphAddChildGraphNode copies its topology into the owning graph.
void attach_conditional(Tensor values, bool positive, int64_t body_address,
                        bool exclude_inactive, int inactive0, int inactive1) {
  // values is the owner's contiguous I32 error or Q4 budget vector. The
  // captured parent leases the immutable child graph and all of its tensors.
  STD_TORCH_CHECK(body_address != 0,
                  "conditional body must name a live child graph");
  DeviceGuard guard(values.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  DOSA_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                           nullptr, nullptr, nullptr));
  STD_TORCH_CHECK(status == cudaStreamCaptureStatusActive,
                  "conditional attachment requires active stream capture");
  cudaGraphConditionalHandle handle;
  DOSA_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                                   cudaGraphCondAssignDefault));
  if (exclude_inactive)
    set_if_any_not_in<<<1, 256, 0, stream>>>(
        handle, values.mutable_data_ptr<int32_t>(), int(values.numel()),
        inactive0, inactive1);
  else
    set_if_any<<<1, 256, 0, stream>>>(handle,
                                      values.mutable_data_ptr<int32_t>(),
                                      int(values.numel()), positive);
  DOSA_CUDA_CHECK(cudaGetLastError());
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edges;
  size_t count;
  DOSA_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                           &dependencies, &edges, &count));
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t conditional, child;
  DOSA_CUDA_CHECK(cudaGraphAddNode(&conditional, graph, dependencies, edges,
                                   count, &params));
  DOSA_CUDA_CHECK(cudaGraphAddChildGraphNode(
      &child, params.conditional.phGraph_out[0], nullptr, 0,
      reinterpret_cast<cudaGraph_t>(body_address)));
  DOSA_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream, &conditional, nullptr, 1, cudaStreamSetCaptureDependencies));
}

void attach_if_any(Tensor values, bool positive, int64_t body_address) {
  attach_conditional(values, positive, body_address, false, 0, 0);
}

void attach_if_any_not_in(Tensor values, int64_t inactive0, int64_t inactive1,
                          int64_t body_address) {
  // Every value outside the explicitly inactive set executes the original body,
  // including malformed/unknown budgets. Original error handling stays inside
  // it.
  attach_conditional(values, false, body_address, true, inactive0, inactive1);
}

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("complete_ties") "(Tensor boundary, Tensor counts, Tensor diag, Tensor starts, Tensor ends, Tensor! error, Tensor! output, Tensor permutation) -> ()");
  ops.def(DOSA_OP(
      "attach_if_any") "(Tensor values, bool positive, int body_address) -> "
                       "()");
  ops.def(DOSA_OP("attach_if_any_not_in") "(Tensor values, int inactive0, int inactive1, int body_address) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("complete_ties"), TORCH_BOX(&DOSA_NAMESPACE::complete_ties));
  ops.impl(DOSA_OP("attach_if_any"), TORCH_BOX(&DOSA_NAMESPACE::attach_if_any));
  ops.impl(DOSA_OP("attach_if_any_not_in"),
           TORCH_BOX(&DOSA_NAMESPACE::attach_if_any_not_in));
}
