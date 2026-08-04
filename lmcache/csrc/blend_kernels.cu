// SPDX-License-Identifier: Apache-2.0

#include "blend_kernels.cuh"
#include "mem_kernels.cuh"
#include "pos_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

void execute_cb_retrieve_plan(const torch::Device& device,
                              size_t host_buffer_alignment,
                              const std::vector<CBGroupSpec>& group_specs,
                              const std::vector<CBRetrieveStep>& steps) {
  // Set the device guard once for the whole plan. Staging runs on a pool
  // stream, overlapping the previous step's kernels on the caller's stream;
  // the planner alternates slot halves so step w's staging only conflicts
  // with step w-2 (ordered by the per-parity events below). Kernels stay on
  // the caller's stream, so its completion event covers the whole plan.
  const at::cuda::OptionalCUDAGuard device_guard(device);
  at::cuda::CUDAStream compute_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStream copy_stream =
      at::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());

  at::cuda::CUDAEvent copy_done[2];     // staging(w) finished, parity w%2
  at::cuda::CUDAEvent compute_done[2];  // kernels(w) finished, parity w%2
  bool compute_recorded[2] = {false, false};

  // Launch-coalescing buffers, reused across steps and groups (every flush
  // clears them; a step never leaves entries behind).
  std::vector<uintptr_t> rope_keys, sc_bufs, sc_maps;
  std::vector<int64_t> rope_old, rope_new;
  std::vector<int> sc_toks;
  rope_keys.reserve(MAX_FUSED_TRANSFER_CHUNKS);
  rope_old.reserve(MAX_FUSED_TRANSFER_CHUNKS);
  rope_new.reserve(MAX_FUSED_TRANSFER_CHUNKS);
  sc_bufs.reserve(MAX_FUSED_TRANSFER_CHUNKS);
  sc_maps.reserve(MAX_FUSED_TRANSFER_CHUNKS);
  sc_toks.reserve(MAX_FUSED_TRANSFER_CHUNKS);

  size_t step_idx = 0;
  for (const auto& step : steps) {
    const size_t parity = step_idx % 2;
    ++step_idx;

    {
      // Stage on the copy stream after the kernels of the step that last
      // used this slot half (two back).
      const at::cuda::CUDAStreamGuard stream_guard(copy_stream);
      if (compute_recorded[parity]) {
        compute_done[parity].block(copy_stream);
      }
      for (const auto& copy : step.staging) {
        lmcache_memcpy_async(copy.dest, copy.src, copy.nbytes,
                             TransferDirection::H2D, copy.host_offset,
                             host_buffer_alignment);
      }
      copy_done[parity].record(copy_stream);
    }
    // Kernels read the staged slots: order them after the staging.
    copy_done[parity].block(compute_stream);

    // Coalesce this step's ropes/scatters per group into fused launches
    // (~1000 tiny launches -> ~4 per wave); ropes still precede scatters.
    for (int group_idx = 0; group_idx < static_cast<int>(group_specs.size());
         ++group_idx) {
      const CBGroupSpec& group = group_specs[group_idx];

      if (group.cos_sin_cache != 0) {
        const auto flush_ropes = [&]() {
          if (rope_keys.empty()) {
            return;
          }
          rotary_embedding_k_fused_ramp_multi_ptr(
              rope_keys, static_cast<at::ScalarType>(group.key_scalar_type),
              static_cast<int64_t>(group.num_layers) * group.slot_tokens,
              rope_old, rope_new, group.slot_tokens,
              static_cast<int64_t>(group.head_size), group.rope_head_stride,
              group.rope_num_kv_heads, group.cos_sin_cache, group.rot_dim,
              group.is_neox);
          rope_keys.clear();
          rope_old.clear();
          rope_new.clear();
        };
        for (const auto& rope : step.ropes) {
          if (rope.group_idx != group_idx) {
            continue;
          }
          TORCH_CHECK(rope.slot_idx >= 0 &&
                          rope.slot_idx <
                              static_cast<int>(group.temp_buffer_ptrs.size()),
                      "CBRopeVar.slot_idx out of range: ", rope.slot_idx);
          // The K plane is the slot buffer's first plane, so its base pointer
          // is the slot base for both split K/V (kv_size 2) and fused-packed
          // / key-only (kv_size 1) layouts. rope_base_offset shifts the base
          // to the first rope-carrying element for layouts whose rope dims
          // trail the row (MLA latents); 0 everywhere else.
          rope_keys.push_back(
              static_cast<uintptr_t>(group.temp_buffer_ptrs[rope.slot_idx]) +
              static_cast<uintptr_t>(group.rope_base_offset));
          rope_old.push_back(rope.old_st);
          rope_new.push_back(rope.cur_st);
          if (static_cast<int>(rope_keys.size()) == MAX_FUSED_TRANSFER_CHUNKS) {
            flush_ropes();
          }
        }
        flush_ropes();
      }

      const auto flush_scatters = [&]() {
        if (sc_bufs.empty()) {
          return;
        }
        multi_layer_kv_transfer_fused_ptr(
            sc_bufs, sc_maps, sc_toks, group.paged_kv_ptrs, group.num_layers,
            group.slot_tokens, group.hidden_elems, group.element_size, device,
            group.page_buffer_size, TransferDirection::H2D,
            group.engine_kv_format, group.block_size, group.head_size);
        sc_bufs.clear();
        sc_maps.clear();
        sc_toks.clear();
      };
      for (const auto& scatter : step.scatters) {
        if (scatter.group_idx != group_idx) {
          continue;
        }
        TORCH_CHECK(scatter.slot_idx >= 0 &&
                        scatter.slot_idx <
                            static_cast<int>(group.temp_buffer_ptrs.size()),
                    "CBScatterVar.slot_idx out of range: ", scatter.slot_idx);
        TORCH_CHECK(scatter.n_tok >= 0 && scatter.n_tok <= group.slot_tokens,
                    "CBScatterVar.n_tok (", scatter.n_tok,
                    ") exceeds slot capacity ", group.slot_tokens);
        // Bounds-check the slot_mapping slice before the kernel dereferences
        // it on device: an out-of-range offset/length would otherwise be a
        // silent out-of-bounds device read (CUDA fault or garbage), not a
        // clean error.
        TORCH_CHECK(scatter.slot_mapping_offset >= 0 &&
                        scatter.slot_mapping_offset + scatter.n_tok <=
                            group.slot_mapping_capacity,
                    "CBScatterVar slot_mapping slice [",
                    scatter.slot_mapping_offset, ", ",
                    scatter.slot_mapping_offset + scatter.n_tok,
                    ") exceeds capacity ", group.slot_mapping_capacity);
        sc_bufs.push_back(
            static_cast<uintptr_t>(group.temp_buffer_ptrs[scatter.slot_idx]));
        sc_maps.push_back(group.slot_mapping_base +
                          static_cast<uintptr_t>(scatter.slot_mapping_offset) *
                              sizeof(int64_t));
        sc_toks.push_back(scatter.n_tok);
        if (static_cast<int>(sc_bufs.size()) == MAX_FUSED_TRANSFER_CHUNKS) {
          flush_scatters();
        }
      }
      flush_scatters();
    }

    // The step after next reuses these slots; its staging waits here.
    compute_done[parity].record(compute_stream);
    compute_recorded[parity] = true;
  }
}
