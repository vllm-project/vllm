#include "compact_candidates.cuh"
#include "graph_capture.cuh"

namespace DOSA_NAMESPACE {
void compact_spans_live_nk(Tensor pool, Tensor head, Tensor solo_tasks,
                           Tensor common_tasks, Tensor solo_state,
                           Tensor common_state, Tensor error, Tensor words,
                           Tensor budget, int64_t seed, int64_t nk_capacity,
                           Tensor starts, Tensor ends, Tensor live_nk,
                           bool tight, bool defer_capacity_error,
                           int64_t retry_body_address) {
  DeviceGuard guard(words.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  cudaGraphConditionalHandle retry_handle = 0;
  cudaGraph_t graph = nullptr;
  if (retry_body_address != 0) {
    STD_TORCH_CHECK(!tight && defer_capacity_error,
                    "retry body requires ordinary deferred compaction");
    graph = current_capture_graph(stream);
    retry_handle = create_conditional(graph);
  }
  candidate_packets::Args a{
      reinterpret_cast<int*>(pool.mutable_data_ptr<uint8_t>()),
      solo_state.mutable_data_ptr<int>(),
      common_state.mutable_data_ptr<int>(),
      error.mutable_data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(head.mutable_data_ptr<int64_t>()),
      reinterpret_cast<CandidateChunk*>(solo_tasks.mutable_data_ptr<int>()),
      reinterpret_cast<CandidateChunk*>(common_tasks.mutable_data_ptr<int>()),
      reinterpret_cast<const uint4*>(words.const_data_ptr<int>()),
      int(words.size(1)),
      int(words.size(0) * 4),
      int(nk_capacity),
      int(pool.size(0)),
      int(solo_tasks.size(0)),
      int(common_tasks.size(0))};
  auto compact = [&]<bool Tight>() {
    launch<candidate_packets::compact<Tight>, 256>(
        dim3(1, a.nq / 4), a, budget.const_data_ptr<int>(), int(seed),
        starts.const_data_ptr<int>(), ends.const_data_ptr<int>(),
        live_nk.const_data_ptr<int>(), defer_capacity_error, retry_handle);
  };
  if (tight)
    compact.template operator()<true>();
  else
    compact.template operator()<false>();
  if (retry_body_address != 0)
    append_if_body(stream, graph, retry_handle, retry_body_address);
}

}  // namespace DOSA_NAMESPACE

DOSA_REGISTER(
    "spans",
    "(Tensor! pool, Tensor! head, Tensor! solo_tasks, Tensor! common_tasks, "
    "Tensor! solo_state, Tensor! common_state, Tensor! error, Tensor words, "
    "Tensor budget, int seed, int nk_capacity, Tensor starts, Tensor ends, "
    "Tensor live_nk, bool tight, bool defer_capacity_error=False, "
    "int retry_body_address=0) -> ()",
    CUDA, compact_spans_live_nk)
