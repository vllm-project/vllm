// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
// See persistent_topk.cuh for kernel implementation.

#include <cuda_runtime.h>
#include <algorithm>

#include "ops.h"
#include "torch_utils.h"

#ifndef USE_ROCM
  #include "persistent_topk.cuh"
#endif

namespace {

#ifndef USE_ROCM
template <int TopK>
void launch_persistent_topk(const torch::stable::Tensor& logits,
                            const torch::stable::Tensor& lengths,
                            torch::stable::Tensor& output,
                            torch::stable::Tensor& workspace,
                            int64_t max_seq_len) {
  namespace P = vllm::persistent;

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.stride(0);
  const cudaStream_t stream = get_current_cuda_stream();

  // Per call, for the device selected by the caller's guard: get_device_prop()
  // caches the properties per device, and the dynamic-smem cap below must
  // reflect the device this launch runs on.
  const cudaDeviceProp* device_prop = get_device_prop();
  const int num_sms = device_prop->multiProcessorCount;
  const int max_smem_per_block = device_prop->sharedMemPerBlockOptin;

  if (num_rows > 32 && max_smem_per_block >= 128 * 1024) {
    cudaError_t status =
        vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
            logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
            lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(TopK), static_cast<uint32_t>(stride),
            static_cast<uint32_t>(max_seq_len), max_smem_per_block, stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "FilteredTopK failed: ", cudaGetErrorString(status));
  } else {
    STD_TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
    STD_TORCH_CHECK(
        workspace.scalar_type() == torch::headeronly::ScalarType::Byte,
        "workspace must be uint8");

    int effective_max_smem;
    if (num_rows <= 4) {
      effective_max_smem =
          std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
    } else if (num_rows <= 8) {
      constexpr int kSmemCapMedium = 48 * 1024;
      effective_max_smem = std::min(max_smem_per_block, kSmemCapMedium);
    } else {
      effective_max_smem = max_smem_per_block;
    }

    uint32_t vec_size = 1;
    if (stride % 4 == 0)
      vec_size = 4;
    else if (stride % 2 == 0)
      vec_size = 2;

    // The dynamic shared-memory budget is the opt-in minus the kernel's own
    // static __shared__, not the opt-in itself. Sizing the chunk from the
    // opt-in overshoots by exactly that much, and when the row needs only one
    // CTA the resulting request exceeds the cap and the launch is rejected --
    // on this part that is every row of 24576 or 49152 elements at 32 or 64
    // rows. Subtract the static size, queried for the instantiation that will
    // actually launch.
    cudaFuncAttributes chunk_fa{};
    cudaError_t chunk_fa_err =
        (vec_size == 4) ? cudaFuncGetAttributes(
                              &chunk_fa, P::persistent_topk_kernel<TopK, 4>)
        : (vec_size == 2) ? cudaFuncGetAttributes(
                                &chunk_fa, P::persistent_topk_kernel<TopK, 2>)
                          : cudaFuncGetAttributes(
                                &chunk_fa, P::persistent_topk_kernel<TopK, 1>);
    STD_TORCH_CHECK(chunk_fa_err == cudaSuccess,
                    "persistent_topk: cudaFuncGetAttributes failed: ",
                    cudaGetErrorString(chunk_fa_err));
    const size_t static_smem = chunk_fa.sharedSizeBytes;
    size_t available_for_ordered = static_cast<size_t>(effective_max_smem) -
                                   P::kFixedSmemLarge - static_smem;
    uint32_t max_chunk_elements =
        static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

    max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
    uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
    if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;

    // Schedule from the active width, not the padded pitch. The kernel
    // guarantees seq_len <= min(stride, max_seq_len), so sizing groups by the
    // pitch launches CTAs that immediately return -- a 163,840-wide padded
    // tensor whose rows are 3k-12k long would otherwise build its geometry as
    // if every row were 163,840 elements. Below RADIX_THRESHOLD no row can take
    // the cooperative path at all, so one CTA per row is the whole geometry.
    uint32_t force_single_cta = 0u;
    const uint32_t active_width =
        std::min(static_cast<uint32_t>(stride),
                 static_cast<uint32_t>(std::max<int64_t>(max_seq_len, 0)));
    uint32_t ctas_per_group =
        (active_width <= P::RADIX_THRESHOLD)
            ? 1u
            : (active_width + max_chunk_elements - 1) / max_chunk_elements;
    if (ctas_per_group == 0) ctas_per_group = 1;
    uint32_t chunk_size = (active_width + ctas_per_group - 1) / ctas_per_group;
    if (chunk_size == 0) chunk_size = max_chunk_elements;
    chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
    if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;

    size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
    // The large path publishes per-CTA counts and sorts the row
    // in CTA 0's chunk buffer.
    STD_TORCH_CHECK(ctas_per_group <= P::kDetMaxCtasPerGroup,
                    "persistent_topk: ctas_per_group ", ctas_per_group,
                    " exceeds ", P::kDetMaxCtasPerGroup);
    // Only the cooperative large path (max_seq_len > RADIX_THRESHOLD) ranks
    // the final candidates in CTA 0's chunk buffer. Rows at or below the
    // threshold take the single-CTA select (or the trivial seq_len <= TopK
    // case) and never touch that buffer, so a row shorter than TopK is legal
    // there -- the block-level QSA indexer calls this with TopK 512 over a
    // few hundred blocks at warm-up.
    STD_TORCH_CHECK(static_cast<uint32_t>(max_seq_len) <= P::RADIX_THRESHOLD ||
                        chunk_size >= static_cast<uint32_t>(TopK),
                    "persistent_topk: chunk_size ", chunk_size,
                    " smaller than TopK ", TopK, " on the cooperative path");
    if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;
    // Let det_select_row keep the row's keys in shared memory
    // (single-CTA rows are <= RADIX_THRESHOLD); capped by the device optin.
    {
      const uint32_t det_rows = std::min<uint32_t>(
          static_cast<uint32_t>(max_seq_len), P::RADIX_THRESHOLD);
      const size_t det_want =
          P::det_select_row_bytes<TopK, P::kThreadsPerBlock>(
              det_rows);  // same expression as the kernel
      // The kernel also owns static __shared__ storage (the large path's
      // BlockScan scratch); the dynamic request must leave room for it.
      cudaFuncAttributes fa{};
      cudaError_t fa_err =
          (vec_size == 4)
              ? cudaFuncGetAttributes(&fa, P::persistent_topk_kernel<TopK, 4>)
          : (vec_size == 2)
              ? cudaFuncGetAttributes(&fa, P::persistent_topk_kernel<TopK, 2>)
              : cudaFuncGetAttributes(&fa, P::persistent_topk_kernel<TopK, 1>);
      STD_TORCH_CHECK(fa_err == cudaSuccess,
                      "persistent_topk: cudaFuncGetAttributes failed: ",
                      cudaGetErrorString(fa_err));
      const size_t dyn_cap =
          static_cast<size_t>(max_smem_per_block) - fa.sharedSizeBytes;
      if (det_want > smem_size) smem_size = std::min(det_want, dyn_cap);
      STD_TORCH_CHECK(smem_size <= dyn_cap, "persistent_topk: dynamic smem ",
                      smem_size, " exceeds ", dyn_cap, " (optin ",
                      max_smem_per_block, " - static ", fa.sharedSizeBytes,
                      ")");
    }

    // Query occupancy for the instantiation that will actually launch;
    // overestimating it deadlocks the cooperative barrier.
    int occupancy = 1;
    cudaError_t occ_err = cudaSuccess;
    if (vec_size == 4) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 4>, P::kThreadsPerBlock,
          smem_size);
    } else if (vec_size == 2) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 2>, P::kThreadsPerBlock,
          smem_size);
    } else {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock,
          smem_size);
    }
    STD_TORCH_CHECK(occ_err == cudaSuccess,
                    "persistent_topk occupancy query failed: ",
                    cudaGetErrorString(occ_err));
    if (occupancy < 1) occupancy = 1;

    // The cooperative spin-wait barrier only runs when at least one row hits
    // the radix path (seq_len > RADIX_THRESHOLD). Below that, non-CTA-0 CTAs
    // early-exit, so oversubscription can't deadlock and headroom is wasted.
    const bool needs_cooperative =
        static_cast<uint32_t>(max_seq_len) > P::RADIX_THRESHOLD;

    const uint32_t hw_resident_cap =
        static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
    uint32_t max_resident_ctas = hw_resident_cap;
    if (needs_cooperative) {
      // Reserve one CTA per SM when occupancy allows; fall back to a single
      // CTA when occupancy == 1 (the most deadlock-prone case — any straggler
      // kernel that takes the only slot on one SM hangs the barrier). Never
      // drop below one full group's worth.
      uint32_t headroom = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
      if (max_resident_ctas >= headroom + ctas_per_group) {
        max_resident_ctas -= headroom;
      }
    }
    uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group,
                                   static_cast<uint32_t>(num_rows));
    if (num_groups == 0) num_groups = 1;
    uint32_t total_ctas = num_groups * ctas_per_group;

    // If the cooperative launch wouldn't fit, use the generic decode kernel on
    // low-smem devices or FilteredTopK where its 128 KiB requirement is met.
    if (needs_cooperative && total_ctas > hw_resident_cap) {
      if (max_smem_per_block < 128 * 1024) {
        // Fall back to one CTA per row running the same deterministic select
        // rather than to top_k_per_row_decode, which hands out slots by
        // arrival order and would break this op's contract on exactly the
        // hardware that cannot host the cooperative launch. Rescanning a long
        // row from global memory is slower, but this is already an
        // exceptional-occupancy case.
        force_single_cta = 1u;
        ctas_per_group = 1u;
        chunk_size = max_chunk_elements;
        num_groups =
            std::min(max_resident_ctas, static_cast<uint32_t>(num_rows));
        if (num_groups == 0) num_groups = 1;
        total_ctas = num_groups;
      } else {
        cudaError_t status =
            vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
                logits.const_data_ptr<float>(),
                output.mutable_data_ptr<int32_t>(),
                lengths.const_data_ptr<int32_t>(),
                static_cast<uint32_t>(num_rows), static_cast<uint32_t>(TopK),
                static_cast<uint32_t>(stride),
                static_cast<uint32_t>(max_seq_len), max_smem_per_block, stream);
        STD_TORCH_CHECK(status == cudaSuccess, "FilteredTopK fallback failed: ",
                        cudaGetErrorString(status));
        return;
      }
    }

    size_t state_bytes = num_groups * sizeof(P::RadixRowState);
    STD_TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(state_bytes),
                    "workspace too small, need ", state_bytes, " bytes, have ",
                    workspace.numel());

    // Zero the per-group RadixRowState region before launch.
    //
    // Issued UNCONDITIONALLY so the memset is captured as its own node in
    // the cudagraph (a separate cudaMemsetAsync node, sequenced before the
    // persistent_topk_kernel launch on the same stream). The previous
    // host-side guard `if (needs_cooperative)` was evaluated at capture time;
    // when capture-time max_seq_len <= RADIX_THRESHOLD (always true under
    // FULL_DECODE_ONLY with max_model_len < 32 K) the memset would NOT be
    // captured, leaving the workspace state to accumulate across replays.
    // That's a latent correctness bug if the runtime data ever takes the
    // radix path, and removes one variable while debugging hangs in the
    // decode/medium paths.
    //
    // Cost is sub-microsecond: state_bytes = num_groups * sizeof(RadixRowState)
    // is ~3 KB per group, ~100 KB for the largest grids on this hardware.
    //
    // Why the memset is required (regardless of which path the kernel takes):
    //   1. arrival_counter accumulates within a launch and is never reset,
    //      so a prior call leaves it at a large positive value. Without this
    //      reset, the very first wait_ge in the next call sees counter >>
    //      target and returns instantly, breaking the barrier.
    //   2. The previous in-kernel init only ran in CTA-0 with intra-CTA
    //      __syncthreads(), so it had no happens-before edge to CTA-1+'s
    //      first red_release. cudaMemsetAsync is stream-ordered: the zero
    //      is globally visible before any CTA runs.
    {
      cudaError_t mz_err = cudaMemsetAsync(
          workspace.mutable_data_ptr<uint8_t>(), 0, state_bytes, stream);
      STD_TORCH_CHECK(mz_err == cudaSuccess,
                      "row_states memset failed: ", cudaGetErrorString(mz_err));
    }

    P::PersistentTopKParams params;
    params.input = logits.const_data_ptr<float>();
    params.output = output.mutable_data_ptr<int32_t>();
    params.lengths = lengths.const_data_ptr<int32_t>();
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.stride = static_cast<uint32_t>(stride);
    params.top_k = static_cast<uint32_t>(TopK);
    params.chunk_size = chunk_size;
    params.row_states = reinterpret_cast<P::RadixRowState*>(
        workspace.mutable_data_ptr<uint8_t>());
    params.ctas_per_group = ctas_per_group;
    params.max_seq_len = static_cast<uint32_t>(max_seq_len);
    params.det_smem_bytes = static_cast<uint32_t>(smem_size);
    params.force_single_cta = force_single_cta;

  #define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                     \
    do {                                                                      \
      auto kernel = &P::persistent_topk_kernel<TOPK_VAL, VS>;                 \
      cudaError_t err = cudaFuncSetAttribute(                                 \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);    \
      STD_TORCH_CHECK(err == cudaSuccess,                                     \
                      "Failed to set smem: ", cudaGetErrorString(err));       \
      kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params); \
    } while (0)

    if (vec_size == 4) {
      LAUNCH_PERSISTENT(TopK, 4);
    } else if (vec_size == 2) {
      LAUNCH_PERSISTENT(TopK, 2);
    } else {
      LAUNCH_PERSISTENT(TopK, 1);
    }
  #undef LAUNCH_PERSISTENT
  }

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "persistent_topk failed: ", cudaGetErrorString(err));
}
#endif

}  // anonymous namespace

void persistent_topk(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& lengths,
                     torch::stable::Tensor& output,
                     torch::stable::Tensor& workspace, int64_t k,
                     int64_t max_seq_len) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  STD_TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "Only float32 supported");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int,
                  "lengths must be int32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "output must be int32");
  STD_TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  STD_TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
                  "lengths must be 1D or 2D");
  STD_TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  STD_TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);

  // Assumptions the kernel makes and used to take on trust.
  STD_TORCH_CHECK(logits.stride(1) == 1,
                  "logits must be row-contiguous (stride(1) == 1), got ",
                  logits.stride(1));
  STD_TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  STD_TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  STD_TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative, got ",
                  max_seq_len);
  STD_TORCH_CHECK(max_seq_len <= logits.stride(0), "max_seq_len ", max_seq_len,
                  " exceeds the row pitch ", logits.stride(0));
  // A DeviceGuard is taken on the logits device; every other tensor must live
  // there too or the kernel reads another device's memory.
  STD_TORCH_CHECK(lengths.get_device_index() == logits.get_device_index() &&
                      output.get_device_index() == logits.get_device_index() &&
                      workspace.get_device_index() == logits.get_device_index(),
                  "all tensors must be on the same CUDA device");

  STD_TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  STD_TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
                  "output size mismatch");
  STD_TORCH_CHECK(
      k == 512 || k == 1024 || k == 2048,
      "persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

  // Nothing to do, and the geometry below divides by quantities derived from
  // the row count. Checked last so an empty batch still validates the contract
  // rather than accepting calls a non-empty one would reject.
  if (num_rows == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());

  if (k == 512) {
    launch_persistent_topk<512>(logits, lengths, output, workspace,
                                max_seq_len);
  } else if (k == 1024) {
    launch_persistent_topk<1024>(logits, lengths, output, workspace,
                                 max_seq_len);
  } else {
    launch_persistent_topk<2048>(logits, lengths, output, workspace,
                                 max_seq_len);
  }
#else
  STD_TORCH_CHECK(false, "persistent_topk is not supported on ROCm");
#endif
}
