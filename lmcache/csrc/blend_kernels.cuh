// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

#include "kv_transfer_types.h"
#include "mp_mem_kernels.cuh"  // StagingCopy

// ---------------------------------------------------------------------------
// CacheBlend retrieve plan.
//
// A whole CB retrieve (all staging copies + re-RoPE + scatter launches) is
// described as a plan on the Python side, then executed in a single native
// call (execute_cb_retrieve_plan) that releases the GIL once for the entire
// request instead of once per copy/launch. Same plan-then-execute shape as
// the object-group transfer in mp_mem_kernels.cuh, plus K-only re-RoPE and a
// per-token scatter (CB matches are not block-aligned). Plans are built in
// blend_v3.py (cb_retrieve_pre_computed).
// ---------------------------------------------------------------------------

// Per-kernel-group invariants, resolved once on the Python side; only the
// slot-mapping fields are re-stamped per request.
struct CBGroupSpec {
  uintptr_t paged_kv_ptrs;  // device ptr-array base (per-layer paged ptrs)
  std::vector<int64_t> temp_buffer_ptrs;  // temp GPU buffer base per tmp slot
  // Scatter geometry: the tmp slot buffer is [kv_size, num_layers,
  // slot_tokens, hidden_elems] of element_size-byte scalars.
  int num_layers;
  int slot_tokens;   // token capacity of one tmp slot (slots per chunk)
  int hidden_elems;  // scalars per token per layer per K/V plane
  int element_size;
  EngineKVFormat engine_kv_format;
  int page_buffer_size;
  int block_size;
  int head_size;                  // scatter kernel head_size (element units)
  uintptr_t slot_mapping_base;    // device int64*, whole-request slot mapping
  int64_t slot_mapping_capacity;  // int64 elements behind slot_mapping_base
  // Re-RoPE (cos_sin_cache == 0 disables rope for this group). Rotation
  // width is `rot_dim` (from the cos/sin cache); `head_size` above is the
  // scatter geometry.
  uintptr_t cos_sin_cache;  // device ptr, [max_position, rot_dim] scalars
  int rot_dim;
  int rope_num_kv_heads;
  int64_t rope_head_stride;  // == head_size, or 2*head_size for fused packed
  int key_scalar_type;       // at::ScalarType of the KV data
  bool is_neox;
  // Byte offset from the slot's K-plane base to the first rope-carrying
  // element: 0 unless MLA, where rope dims trail the latent row.
  int64_t rope_base_offset;
};

// One K-only re-RoPE launch: rotate tmp slot `slot_idx` of group `group_idx`
// in place from stored position `old_st` to new position `cur_st`.
struct CBRopeVar {
  int group_idx;
  int slot_idx;
  int64_t old_st;
  int64_t cur_st;
};

// One per-token scatter launch: write `n_tok` tokens of tmp slot `slot_idx`
// to the paged KV slots at `slot_mapping_base + slot_mapping_offset`.
struct CBScatterVar {
  int group_idx;
  int slot_idx;
  int64_t slot_mapping_offset;
  int n_tok;
};

// One batch of tmp slots: H2D staging, then re-RoPE, then scatter. The order
// is load-bearing (slots are reused by the next step).
struct CBRetrieveStep {
  std::vector<StagingCopy> staging;
  std::vector<CBRopeVar> ropes;
  std::vector<CBScatterVar> scatters;
};

/**
 * Execute one CB retrieve plan on the caller's current CUDA stream.
 *
 * Enqueues every staging copy, re-RoPE, and scatter launch described by
 * `steps` within a single GIL release (configured at the pybind layer),
 * eliminating the per-copy/per-launch GIL handoffs of the equivalent Python
 * loop. Staging runs on a pool stream, overlapping the previous step's
 * kernels; per-parity CUDA events order tmp-slot reuse across steps.
 *
 * @param device                CUDA device of the transfer
 * @param host_buffer_alignment Host buffer alignment for staging copies
 *                              (power of two)
 * @param group_specs           Per-kernel-group invariants
 * @param steps                 Ordered per-step staging + rope + scatter work
 */
void execute_cb_retrieve_plan(const torch::Device& device,
                              size_t host_buffer_alignment,
                              const std::vector<CBGroupSpec>& group_specs,
                              const std::vector<CBRetrieveStep>& steps);
