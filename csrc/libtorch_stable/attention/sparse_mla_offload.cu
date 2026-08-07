#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include <torch/headeronly/core/ScalarType.h>

#include "../torch_utils.h"

namespace {

using torch::headeronly::ScalarType;

__device__ bool slot_reserved(int slot, int topk, const int32_t* resident_hits,
                              const int32_t* chosen_slots) {
  for (int index = 0; index < topk; ++index) {
    if (resident_hits[index] == slot) return true;
  }
  for (int index = 0; index < topk; ++index) {
    if (chosen_slots[index] == slot) return true;
  }
  return false;
}

__device__ int choose_victim(const int64_t* resident_ids,
                             const int64_t* resident_access, int resident_rows,
                             int topk, const int32_t* resident_hits,
                             const int32_t* chosen_slots) {
  int best_slot = -1;
  int best_valid = 2;
  int64_t best_access = std::numeric_limits<int64_t>::max();
  for (int slot = 0; slot < resident_rows; ++slot) {
    if (slot_reserved(slot, topk, resident_hits, chosen_slots)) continue;
    const bool valid = resident_ids[slot] >= 0;
    const int valid_key = valid ? 1 : 0;
    const int64_t access_key = valid ? resident_access[slot] : 0;
    if (valid_key < best_valid ||
        (valid_key == best_valid && access_key < best_access) ||
        (valid_key == best_valid && access_key == best_access &&
         (best_slot < 0 || slot < best_slot))) {
      best_slot = slot;
      best_valid = valid_key;
      best_access = access_key;
    }
  }
  return best_slot;
}

__global__ void validate_rows_kernel(
    const int32_t* request_block_ids, const int32_t* request_num_blocks,
    const int32_t* request_num_tokens, const bool* request_active,
    const int32_t* req_id_per_token, const int32_t* live_topk_logical_ids,
    const int64_t* positions, const int32_t* saved_topk_logical_ids,
    const int32_t* tp_fence_token, int32_t* row_valid, int token_rows,
    int request_slots, int request_block_width, int window, int topk,
    int num_host_blocks) {
  const int row = blockIdx.x;
  if (row >= token_rows || threadIdx.x != 0) return;
  const int request = req_id_per_token[row];
  bool valid = request >= 0 && request < request_slots;
  if (valid) {
    valid = request_active[request] && request_num_tokens[request] > 0 &&
            request_num_blocks[request] > 0 &&
            request_num_blocks[request] <= request_block_width;
  }
  const int slot = valid ? static_cast<int>(positions[row] % window) : 0;
  valid = valid && slot >= 0;
  const int num_tokens = valid ? request_num_tokens[request] : 0;
  const int num_blocks = valid ? request_num_blocks[request] : 0;
  for (int block = 0; valid && block < num_blocks; ++block) {
    const int physical =
        request_block_ids[request * request_block_width + block];
    valid = physical >= 0 && physical < num_host_blocks;
  }
  const int saved_request = valid ? request : 0;
  const bool use_saved_topk = window > 1 && tp_fence_token[0] > 0;
  const int32_t* topk_logical_ids =
      use_saved_topk ? saved_topk_logical_ids + saved_request * window * topk
                     : live_topk_logical_ids + row * topk;
  for (int item = 0; valid && item < topk; ++item) {
    const int logical = topk_logical_ids[item];
    if (logical == -1) continue;
    valid = logical >= 0 && logical < num_tokens;
  }
  if (valid) row_valid[row] = 1;
}

__global__ void plan_rows_kernel(
    const uint16_t* current_main_kv, const int32_t* request_num_tokens,
    const int32_t* req_id_per_token, const int32_t* live_topk_logical_ids,
    const int64_t* positions, int32_t* saved_topk_logical_ids,
    const int32_t* tp_fence_token, uint16_t* resident_main_kv,
    int64_t* resident_logical_ids, int64_t* resident_last_access,
    uint16_t* newest_main_kv, int64_t* newest_logical_ids,
    int32_t* topk_physical_ids, bool* topk_hit_mask, int32_t* miss_logical_ids,
    int32_t* miss_victim_slots, int32_t* miss_counts, int32_t* hit_counts,
    int token_rows, int scratch_rows, int window_slot, int window, int topk,
    int resident_rows, int head_dim) {
  const int request = blockIdx.x;
  if (request >= scratch_rows / window || threadIdx.x != 0) return;
  const int state_row = request * window + window_slot;
  int token_row = -1;
  for (int candidate = 0; candidate < token_rows; ++candidate) {
    if (req_id_per_token[candidate] == request &&
        positions[candidate] % window == window_slot) {
      token_row = candidate;
      break;
    }
  }
  if (token_row < 0) return;
  int32_t* hit_indices = topk_physical_ids + token_row * topk;
  bool* hit_mask = topk_hit_mask + token_row * topk;
  int32_t* miss_ids = miss_logical_ids + token_row * topk;
  int32_t* victim_slots = miss_victim_slots + token_row * topk;
  const bool valid = miss_counts[token_row] == 1;
  miss_counts[token_row] = 0;
  hit_counts[token_row] = 0;
  for (int item = 0; item < topk; ++item) {
    hit_indices[item] = -1;
    hit_mask[item] = false;
    miss_ids[item] = -1;
    victim_slots[item] = -1;
  }
  if (!valid) return;

  const int64_t access = request_num_tokens[request];
  const int64_t current_id = positions[token_row];
  const int resident_base = request * resident_rows;
  const bool use_saved_topk = window > 1 && tp_fence_token[0] > 0;
  const int32_t* topk_logical_ids =
      use_saved_topk ? saved_topk_logical_ids + request * window * topk
                     : live_topk_logical_ids + token_row * topk;

  for (int item = 0; item < topk; ++item) {
    const int logical = topk_logical_ids[item];
    if (logical < 0) continue;
    for (int slot = 0; slot < resident_rows; ++slot) {
      const int index = resident_base + slot;
      if (resident_logical_ids[index] == logical) {
        hit_indices[item] = slot;
        break;
      }
    }
  }

  for (int item = 0; item < topk; ++item) {
    const int logical = topk_logical_ids[item];
    if (logical < 0) continue;
    int duplicate = -1;
    for (int prior = 0; prior < item; ++prior) {
      if (topk_logical_ids[prior] == logical) {
        duplicate = prior;
        break;
      }
    }
    if (duplicate >= 0) continue;

    int slot = hit_indices[item];
    if (slot >= 0) {
      resident_last_access[resident_base + slot] = access;
      hit_mask[item] = true;
      victim_slots[item] = slot;
      continue;
    }
    int newest_slot = -1;
    for (int slot = 0; slot < window; ++slot) {
      if (newest_logical_ids[request * window + slot] == logical) {
        newest_slot = slot;
        break;
      }
    }
    const bool newest_hit = newest_slot >= 0;
    const bool current_hit = logical == current_id;
    if (!newest_hit && !current_hit) continue;
    slot = choose_victim(resident_logical_ids + resident_base,
                         resident_last_access + resident_base, resident_rows,
                         topk, hit_indices, victim_slots);
    if (slot < 0) continue;
    victim_slots[item] = slot;

    const uint16_t* source =
        current_hit
            ? current_main_kv + token_row * head_dim
            : newest_main_kv + (request * window + newest_slot) * head_dim;
    uint16_t* target = resident_main_kv + (resident_base + slot) * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) target[dim] = source[dim];
    resident_logical_ids[resident_base + slot] = logical;
    resident_last_access[resident_base + slot] = access;
    hit_mask[item] = true;
  }

  for (int item = 0; item < topk; ++item) {
    const int logical = topk_logical_ids[item];
    if (logical < 0 || hit_mask[item]) continue;
    bool duplicate = false;
    for (int prior = 0; prior < item; ++prior) {
      duplicate |= topk_logical_ids[prior] == logical;
    }
    if (duplicate) continue;
    victim_slots[item] =
        choose_victim(resident_logical_ids + resident_base,
                      resident_last_access + resident_base, resident_rows, topk,
                      hit_indices, victim_slots);
  }

  int hit_count = 0;
  int miss_count = 0;
  for (int item = 0; item < topk; ++item) {
    const int logical = topk_logical_ids[item];
    if (logical < 0) continue;
    bool duplicate = false;
    for (int prior = 0; prior < item; ++prior) {
      duplicate |= topk_logical_ids[prior] == logical;
    }
    if (duplicate) continue;
    const int global_slot = resident_base + victim_slots[item];
    if (hit_mask[item]) {
      hit_indices[hit_count++] = global_slot;
    } else {
      miss_ids[miss_count] = logical;
      victim_slots[miss_count++] = global_slot;
    }
  }
  for (int item = hit_count; item < topk; ++item) hit_indices[item] = 0;
  for (int item = miss_count; item < topk; ++item) {
    miss_ids[item] = -1;
    victim_slots[item] = 0;
  }

  const uint16_t* current = current_main_kv + token_row * head_dim;
  uint16_t* newest = newest_main_kv + state_row * head_dim;
  for (int dim = 0; dim < head_dim; ++dim) newest[dim] = current[dim];
  newest_logical_ids[state_row] = current_id;
  bool is_last_request_token = true;
  for (int candidate = token_row + 1; candidate < token_rows; ++candidate) {
    if (req_id_per_token[candidate] == request) {
      is_last_request_token = false;
      break;
    }
  }
  if (window > 1 && tp_fence_token[0] == 0 && is_last_request_token) {
    int32_t* saved = saved_topk_logical_ids + request * window * topk;
    for (int item = 0; item < topk; ++item)
      saved[item] = topk_logical_ids[item];
  }
  hit_counts[token_row] = hit_count;
  miss_counts[token_row] = miss_count;
}

__global__ void transfer_misses_kernel(
    const uint16_t* main_host_kv, const int32_t* request_block_ids,
    const int32_t* request_num_blocks, const int32_t* request_num_tokens,
    const bool* request_active, const int32_t* req_id_per_token,
    const int64_t* positions, const int32_t* miss_logical_ids,
    const int32_t* miss_victim_slots, const int32_t* miss_counts,
    uint16_t* resident_main_kv, int64_t* resident_logical_ids,
    int64_t* resident_last_access, int token_rows, int window, int topk,
    int request_slots, int request_block_width, int resident_rows,
    int num_host_blocks, int block_size, int head_dim) {
  const int work = blockIdx.x;
  const int row = work / topk;
  const int miss = work % topk;
  if (row >= token_rows) return;
  const int request = req_id_per_token[row];
  if (request < 0 || request >= request_slots || !request_active[request])
    return;
  if (miss >= miss_counts[row]) return;
  const int num_blocks = request_num_blocks[request];
  if (num_blocks <= 0 || num_blocks > request_block_width) return;
  const int logical = miss_logical_ids[row * topk + miss];
  const int logical_block = logical / block_size;
  if (logical < 0 || logical >= request_num_tokens[request] ||
      logical_block >= num_blocks)
    return;
  const int physical_block =
      request_block_ids[request * request_block_width + logical_block];
  const int global_slot = miss_victim_slots[row * topk + miss];
  if (physical_block < 0 || physical_block >= num_host_blocks ||
      global_slot < 0 || global_slot >= request_slots * resident_rows)
    return;
  const uint16_t* source =
      main_host_kv +
      (physical_block * block_size + logical % block_size) * head_dim;
  uint16_t* target = resident_main_kv + global_slot * head_dim;
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    target[dim] = source[dim];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    resident_logical_ids[global_slot] = logical;
    resident_last_access[global_slot] = request_num_tokens[request];
  }
}

__global__ void record_provisional_slots_kernel(
    const int32_t* req_id_per_token, const int64_t* positions,
    const int32_t* miss_victim_slots, int32_t* provisional_slots,
    int token_rows, int window, int topk, int request_slots) {
  const int row = blockIdx.x;
  if (row >= token_rows) return;
  const int request = req_id_per_token[row];
  if (request < 0 || request >= request_slots) return;
  const int window_slot = positions[row] % window;
  if (window_slot < 0) return;
  const int state_row = request * window + window_slot;
  for (int item = threadIdx.x; item < topk; item += blockDim.x) {
    provisional_slots[state_row * topk + item] =
        miss_victim_slots[row * topk + item];
  }
}

__global__ void writeback_current_kernel(
    uint16_t* main_host_kv, const int32_t* request_block_ids,
    const int32_t* request_num_blocks, const int32_t* request_num_tokens,
    const bool* request_active, const int32_t* req_id_per_token,
    const int64_t* positions, const uint16_t* newest_main_kv,
    const int64_t* newest_logical_ids, const int32_t* tp_fence_token,
    int token_rows, int window, int request_slots, int request_block_width,
    int num_host_blocks, int block_size, int head_dim) {
  const int row = blockIdx.x;
  if (row >= token_rows || window != 1 || tp_fence_token[0] != 0) return;
  const int request = req_id_per_token[row];
  if (request < 0 || request >= request_slots || !request_active[request])
    return;
  const int window_slot = positions[row] % window;
  if (window_slot < 0) return;
  const int scratch_row = request * window + window_slot;
  const int num_blocks = request_num_blocks[request];
  if (num_blocks <= 0 || num_blocks > request_block_width) return;
  const int logical = newest_logical_ids[scratch_row];
  const int logical_block = logical / block_size;
  if (logical < 0 || logical_block >= num_blocks ||
      logical != request_num_tokens[request] - 1)
    return;
  const int physical_block =
      request_block_ids[request * request_block_width + logical_block];
  if (physical_block < 0 || physical_block >= num_host_blocks) return;
  const uint16_t* source = newest_main_kv + scratch_row * head_dim;
  uint16_t* target =
      main_host_kv +
      (physical_block * block_size + logical % block_size) * head_dim;
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    target[dim] = source[dim];
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
}

__global__ void finalize_mtp_kernel(
    uint16_t* main_host_kv, const int32_t* request_block_ids,
    const int32_t* request_num_blocks, const int32_t* request_num_tokens,
    const bool* request_active, const uint16_t* newest_main_kv,
    int64_t* newest_logical_ids, int64_t* resident_logical_ids,
    int64_t* resident_last_access, int32_t* provisional_slots,
    const int32_t* finalize_status, bool is_host_writer, int request_slots,
    int request_block_width, int resident_rows, int window, int topk,
    int num_host_blocks, int block_size, int head_dim) {
  const int request = blockIdx.x;
  if (finalize_status[0] != 0) return;
  if (request >= request_slots || !request_active[request]) return;
  const int boundary = request_num_tokens[request];
  const int num_blocks = request_num_blocks[request];

  for (int slot = 0; slot < window; ++slot) {
    const int index = request * window + slot;
    const int64_t logical = newest_logical_ids[index];
    if (logical >= 0 && logical < boundary && is_host_writer) {
      const int logical_block = logical / block_size;
      const int physical =
          request_block_ids[request * request_block_width + logical_block];
      const uint16_t* source = newest_main_kv + index * head_dim;
      uint16_t* target =
          main_host_kv +
          (physical * block_size + logical % block_size) * head_dim;
      for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x)
        target[dim] = source[dim];
    }
    if (threadIdx.x == 0 && logical >= boundary) newest_logical_ids[index] = -1;
    for (int item = threadIdx.x; item < topk; item += blockDim.x)
      provisional_slots[(index * topk) + item] = -1;
  }
  for (int slot = threadIdx.x; slot < resident_rows; slot += blockDim.x) {
    const int index = request * resident_rows + slot;
    if (resident_logical_ids[index] >= boundary) {
      resident_logical_ids[index] = -1;
      resident_last_access[index] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && is_host_writer) __threadfence_system();
}

__global__ void validate_finalize_mtp_kernel(
    const int32_t* request_block_ids, const int32_t* request_num_blocks,
    const int32_t* request_num_tokens, const bool* request_active,
    const uint16_t* newest_main_kv, const int64_t* newest_logical_ids,
    int32_t* finalize_status, int request_slots, int request_block_width,
    int window, int num_host_blocks, int block_size, int head_dim) {
  const int request = blockIdx.x;
  if (request >= request_slots || !request_active[request] || threadIdx.x != 0)
    return;
  const int boundary = request_num_tokens[request];
  const int num_blocks = request_num_blocks[request];
  bool valid =
      boundary >= 0 && num_blocks > 0 && num_blocks <= request_block_width;
  for (int slot = 0; valid && slot < window; ++slot) {
    const int64_t logical = newest_logical_ids[request * window + slot];
    valid = logical < 0 || logical % window == slot;
    const int logical_block = logical / block_size;
    if (logical >= 0 && logical < boundary) {
      valid = logical_block < num_blocks;
      const int physical =
          valid
              ? request_block_ids[request * request_block_width + logical_block]
              : -1;
      valid = valid && physical >= 0 && physical < num_host_blocks;
    }
    for (int prior = 0; valid && prior < slot; ++prior) {
      if (newest_logical_ids[request * window + prior] != logical ||
          logical < 0)
        continue;
      const uint16_t* left =
          newest_main_kv + (request * window + prior) * head_dim;
      const uint16_t* right =
          newest_main_kv + (request * window + slot) * head_dim;
      for (int dim = 0; valid && dim < head_dim; ++dim)
        valid = left[dim] == right[dim];
    }
  }
  if (!valid) atomicExch(finalize_status, 1);
}

void check_cuda_contiguous(const torch::stable::Tensor& tensor,
                           const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_contiguous_on_device(const torch::stable::Tensor& tensor,
                                     const char* name, int device_index) {
  check_cuda_contiguous(tensor, name);
  STD_TORCH_CHECK(tensor.get_device_index() == device_index, name,
                  " must be on the same CUDA device");
}

}  // namespace

void sparse_mla_cache_plan(const torch::stable::Tensor& current_main_kv,
                           const torch::stable::Tensor& request_block_ids,
                           const torch::stable::Tensor& request_num_blocks,
                           const torch::stable::Tensor& request_num_tokens,
                           const torch::stable::Tensor& request_active,
                           const torch::stable::Tensor& req_id_per_token,
                           const torch::stable::Tensor& live_topk_logical_ids,
                           const torch::stable::Tensor& positions,
                           torch::stable::Tensor& saved_topk_logical_ids,
                           const torch::stable::Tensor& tp_fence_token,
                           torch::stable::Tensor& resident_main_kv,
                           torch::stable::Tensor& resident_logical_ids,
                           torch::stable::Tensor& resident_last_access,
                           torch::stable::Tensor& newest_main_kv,
                           torch::stable::Tensor& newest_logical_ids,
                           torch::stable::Tensor& topk_physical_ids,
                           torch::stable::Tensor& topk_hit_mask,
                           torch::stable::Tensor& miss_logical_ids,
                           torch::stable::Tensor& miss_victim_slots,
                           torch::stable::Tensor& miss_counts,
                           torch::stable::Tensor& hit_counts,
                           int64_t num_host_blocks) {
  check_cuda_contiguous(current_main_kv, "current_main_kv");
  const int device_index = current_main_kv.get_device_index();
  check_cuda_contiguous_on_device(request_block_ids, "request_block_ids",
                                  device_index);
  check_cuda_contiguous_on_device(request_num_blocks, "request_num_blocks",
                                  device_index);
  check_cuda_contiguous_on_device(request_num_tokens, "request_num_tokens",
                                  device_index);
  check_cuda_contiguous_on_device(request_active, "request_active",
                                  device_index);
  check_cuda_contiguous_on_device(req_id_per_token, "req_id_per_token",
                                  device_index);
  check_cuda_contiguous_on_device(live_topk_logical_ids,
                                  "live_topk_logical_ids", device_index);
  check_cuda_contiguous_on_device(positions, "positions", device_index);
  check_cuda_contiguous_on_device(saved_topk_logical_ids,
                                  "saved_topk_logical_ids", device_index);
  check_cuda_contiguous_on_device(tp_fence_token, "tp_fence_token",
                                  device_index);
  check_cuda_contiguous_on_device(resident_main_kv, "resident_main_kv",
                                  device_index);
  check_cuda_contiguous_on_device(resident_logical_ids, "resident_logical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(resident_last_access, "resident_last_access",
                                  device_index);
  check_cuda_contiguous_on_device(newest_main_kv, "newest_main_kv",
                                  device_index);
  check_cuda_contiguous_on_device(newest_logical_ids, "newest_logical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(topk_physical_ids, "topk_physical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(topk_hit_mask, "topk_hit_mask", device_index);
  check_cuda_contiguous_on_device(miss_logical_ids, "miss_logical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(miss_victim_slots, "miss_victim_slots",
                                  device_index);
  check_cuda_contiguous_on_device(miss_counts, "miss_counts", device_index);
  check_cuda_contiguous_on_device(hit_counts, "hit_counts", device_index);
  STD_TORCH_CHECK(current_main_kv.scalar_type() == ScalarType::BFloat16 &&
                      resident_main_kv.scalar_type() == ScalarType::BFloat16 &&
                      newest_main_kv.scalar_type() == ScalarType::BFloat16,
                  "Main KV tensors must be bfloat16");
  STD_TORCH_CHECK(request_block_ids.scalar_type() == ScalarType::Int &&
                      request_num_blocks.scalar_type() == ScalarType::Int &&
                      request_num_tokens.scalar_type() == ScalarType::Int &&
                      req_id_per_token.scalar_type() == ScalarType::Int &&
                      live_topk_logical_ids.scalar_type() == ScalarType::Int &&
                      saved_topk_logical_ids.scalar_type() == ScalarType::Int &&
                      tp_fence_token.scalar_type() == ScalarType::Int &&
                      topk_physical_ids.scalar_type() == ScalarType::Int &&
                      miss_logical_ids.scalar_type() == ScalarType::Int &&
                      miss_victim_slots.scalar_type() == ScalarType::Int &&
                      miss_counts.scalar_type() == ScalarType::Int &&
                      hit_counts.scalar_type() == ScalarType::Int,
                  "sparse MLA indices and counts must be int32");
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Long &&
                      resident_logical_ids.scalar_type() == ScalarType::Long &&
                      resident_last_access.scalar_type() == ScalarType::Long &&
                      newest_logical_ids.scalar_type() == ScalarType::Long,
                  "sparse MLA persistent metadata must be int64");
  STD_TORCH_CHECK(request_active.scalar_type() == ScalarType::Bool &&
                      topk_hit_mask.scalar_type() == ScalarType::Bool,
                  "sparse MLA masks must be bool");
  STD_TORCH_CHECK(
      current_main_kv.dim() == 2 && request_block_ids.dim() == 2 &&
          request_num_blocks.dim() == 1 && request_num_tokens.dim() == 1 &&
          request_active.dim() == 1 && req_id_per_token.dim() == 1 &&
          live_topk_logical_ids.dim() == 3 && positions.dim() == 1 &&
          saved_topk_logical_ids.dim() == 3 && tp_fence_token.dim() == 1 &&
          resident_main_kv.dim() == 3 && resident_logical_ids.dim() == 2 &&
          resident_last_access.dim() == 2 && newest_main_kv.dim() == 3 &&
          newest_logical_ids.dim() == 2 && topk_physical_ids.dim() == 3 &&
          topk_hit_mask.dim() == 3 && miss_logical_ids.dim() == 3 &&
          miss_victim_slots.dim() == 3 && miss_counts.dim() == 2 &&
          hit_counts.dim() == 2,
      "invalid sparse MLA tensor rank");
  const int token_rows = current_main_kv.size(0);
  const int request_slots = resident_main_kv.size(0);
  const int resident_rows = resident_main_kv.size(1);
  const int head_dim = resident_main_kv.size(2);
  const int topk = live_topk_logical_ids.size(2);
  const int window = saved_topk_logical_ids.size(1);
  const int request_block_width = request_block_ids.size(1);
  const int scratch_rows = topk_physical_ids.size(0) * window;
  STD_TORCH_CHECK(
      token_rows > 0 && token_rows <= scratch_rows && request_slots > 0 &&
          resident_rows >= topk && topk > 0 && head_dim > 0 &&
          request_block_width > 0 &&
          topk_physical_ids.size(0) == request_slots &&
          current_main_kv.size(1) == head_dim &&
          request_block_ids.size(0) == request_slots &&
          request_num_blocks.size(0) == request_slots &&
          request_num_tokens.size(0) == request_slots &&
          request_active.size(0) == request_slots &&
          req_id_per_token.size(0) == token_rows &&
          live_topk_logical_ids.size(0) >= token_rows &&
          live_topk_logical_ids.size(1) == 1 &&
          positions.size(0) == token_rows &&
          saved_topk_logical_ids.size(0) == request_slots && window > 0 &&
          saved_topk_logical_ids.size(2) == topk &&
          tp_fence_token.size(0) == 1 &&
          newest_main_kv.size(0) == request_slots &&
          newest_main_kv.size(1) == window &&
          newest_main_kv.size(2) == head_dim &&
          resident_logical_ids.size(0) == request_slots &&
          resident_logical_ids.size(1) == resident_rows &&
          resident_last_access.sizes().equals(resident_logical_ids.sizes()) &&
          newest_logical_ids.size(0) == request_slots &&
          newest_logical_ids.size(1) == window &&
          topk_physical_ids.size(1) == window &&
          topk_physical_ids.size(2) == topk &&
          topk_hit_mask.sizes().equals(topk_physical_ids.sizes()) &&
          miss_logical_ids.sizes().equals(topk_physical_ids.sizes()) &&
          miss_victim_slots.sizes().equals(topk_physical_ids.sizes()) &&
          miss_counts.size(0) == request_slots &&
          miss_counts.size(1) == window &&
          hit_counts.sizes().equals(miss_counts.sizes()) &&
          num_host_blocks > 0 &&
          num_host_blocks <= std::numeric_limits<int>::max(),
      "invalid sparse MLA static shape");
  const torch::stable::accelerator::DeviceGuard guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  cudaMemsetAsync(miss_counts.mutable_data_ptr<int32_t>(), 0,
                  scratch_rows * sizeof(int32_t), stream);
  cudaMemsetAsync(hit_counts.mutable_data_ptr<int32_t>(), 0,
                  scratch_rows * sizeof(int32_t), stream);
  validate_rows_kernel<<<token_rows, 1, 0, stream>>>(
      request_block_ids.const_data_ptr<int32_t>(),
      request_num_blocks.const_data_ptr<int32_t>(),
      request_num_tokens.const_data_ptr<int32_t>(),
      request_active.const_data_ptr<bool>(),
      req_id_per_token.const_data_ptr<int32_t>(),
      live_topk_logical_ids.const_data_ptr<int32_t>(),
      positions.const_data_ptr<int64_t>(),
      saved_topk_logical_ids.const_data_ptr<int32_t>(),
      tp_fence_token.const_data_ptr<int32_t>(),
      miss_counts.mutable_data_ptr<int32_t>(), token_rows, request_slots,
      request_block_width, window, topk, num_host_blocks);
  for (int window_slot = 0; window_slot < window; ++window_slot) {
    plan_rows_kernel<<<request_slots, 1, 0, stream>>>(
        static_cast<const uint16_t*>(current_main_kv.const_data_ptr()),
        request_num_tokens.const_data_ptr<int32_t>(),
        req_id_per_token.const_data_ptr<int32_t>(),
        live_topk_logical_ids.const_data_ptr<int32_t>(),
        positions.const_data_ptr<int64_t>(),
        saved_topk_logical_ids.mutable_data_ptr<int32_t>(),
        tp_fence_token.const_data_ptr<int32_t>(),
        static_cast<uint16_t*>(resident_main_kv.mutable_data_ptr()),
        resident_logical_ids.mutable_data_ptr<int64_t>(),
        resident_last_access.mutable_data_ptr<int64_t>(),
        static_cast<uint16_t*>(newest_main_kv.mutable_data_ptr()),
        newest_logical_ids.mutable_data_ptr<int64_t>(),
        topk_physical_ids.mutable_data_ptr<int32_t>(),
        topk_hit_mask.mutable_data_ptr<bool>(),
        miss_logical_ids.mutable_data_ptr<int32_t>(),
        miss_victim_slots.mutable_data_ptr<int32_t>(),
        miss_counts.mutable_data_ptr<int32_t>(),
        hit_counts.mutable_data_ptr<int32_t>(), token_rows, scratch_rows,
        window_slot, window, topk, resident_rows, head_dim);
  }
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "sparse_mla_cache_plan failed: ", cudaGetErrorString(error));
}

void sparse_mla_offload_transfer(
    torch::stable::Tensor& main_host_kv_uva,
    const torch::stable::Tensor& request_block_ids,
    const torch::stable::Tensor& request_num_blocks,
    const torch::stable::Tensor& request_num_tokens,
    const torch::stable::Tensor& request_active,
    const torch::stable::Tensor& req_id_per_token,
    const torch::stable::Tensor& newest_main_kv,
    torch::stable::Tensor& newest_logical_ids,
    const torch::stable::Tensor& miss_logical_ids,
    const torch::stable::Tensor& miss_victim_slots,
    const torch::stable::Tensor& miss_counts,
    const torch::stable::Tensor& hit_counts,
    torch::stable::Tensor& resident_main_kv,
    torch::stable::Tensor& resident_logical_ids,
    torch::stable::Tensor& resident_last_access,
    torch::stable::Tensor& provisional_slots,
    const torch::stable::Tensor& positions,
    torch::stable::Tensor& tp_fence_token, bool is_host_writer,
    int64_t block_size, bool finalize, int64_t finalize_phase) {
  check_cuda_contiguous(resident_main_kv, "resident_main_kv");
  const int device_index = resident_main_kv.get_device_index();
  check_cuda_contiguous_on_device(main_host_kv_uva, "main_host_kv_uva",
                                  device_index);
  check_cuda_contiguous_on_device(request_block_ids, "request_block_ids",
                                  device_index);
  check_cuda_contiguous_on_device(request_num_blocks, "request_num_blocks",
                                  device_index);
  check_cuda_contiguous_on_device(request_num_tokens, "request_num_tokens",
                                  device_index);
  check_cuda_contiguous_on_device(request_active, "request_active",
                                  device_index);
  check_cuda_contiguous_on_device(req_id_per_token, "req_id_per_token",
                                  device_index);
  if (!finalize)
    check_cuda_contiguous_on_device(positions, "positions", device_index);
  check_cuda_contiguous_on_device(newest_main_kv, "newest_main_kv",
                                  device_index);
  check_cuda_contiguous_on_device(newest_logical_ids, "newest_logical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(miss_logical_ids, "miss_logical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(miss_victim_slots, "miss_victim_slots",
                                  device_index);
  check_cuda_contiguous_on_device(miss_counts, "miss_counts", device_index);
  check_cuda_contiguous_on_device(hit_counts, "hit_counts", device_index);
  check_cuda_contiguous_on_device(resident_logical_ids, "resident_logical_ids",
                                  device_index);
  check_cuda_contiguous_on_device(resident_last_access, "resident_last_access",
                                  device_index);
  check_cuda_contiguous_on_device(provisional_slots, "provisional_slots",
                                  device_index);
  check_cuda_contiguous_on_device(tp_fence_token, "tp_fence_token",
                                  device_index);
  STD_TORCH_CHECK(main_host_kv_uva.scalar_type() == ScalarType::BFloat16 &&
                      newest_main_kv.scalar_type() == ScalarType::BFloat16 &&
                      resident_main_kv.scalar_type() == ScalarType::BFloat16,
                  "Main KV tensors must be bfloat16");
  STD_TORCH_CHECK(request_block_ids.scalar_type() == ScalarType::Int &&
                      request_num_blocks.scalar_type() == ScalarType::Int &&
                      request_num_tokens.scalar_type() == ScalarType::Int &&
                      req_id_per_token.scalar_type() == ScalarType::Int &&
                      miss_logical_ids.scalar_type() == ScalarType::Int &&
                      miss_victim_slots.scalar_type() == ScalarType::Int &&
                      miss_counts.scalar_type() == ScalarType::Int &&
                      hit_counts.scalar_type() == ScalarType::Int &&
                      provisional_slots.scalar_type() == ScalarType::Int &&
                      tp_fence_token.scalar_type() == ScalarType::Int,
                  "sparse MLA indices and counts must be int32");
  STD_TORCH_CHECK((finalize || positions.scalar_type() == ScalarType::Long) &&
                      newest_logical_ids.scalar_type() == ScalarType::Long &&
                      resident_logical_ids.scalar_type() == ScalarType::Long &&
                      resident_last_access.scalar_type() == ScalarType::Long,
                  "sparse MLA persistent metadata must be int64");
  STD_TORCH_CHECK(request_active.scalar_type() == ScalarType::Bool,
                  "request_active must be bool");
  STD_TORCH_CHECK(!finalize || finalize_phase == 0 || finalize_phase == 1,
                  "invalid sparse MLA finalize phase");
  STD_TORCH_CHECK(
      main_host_kv_uva.dim() == 3 && request_block_ids.dim() == 2 &&
          request_num_blocks.dim() == 1 && request_num_tokens.dim() == 1 &&
          request_active.dim() == 1 && req_id_per_token.dim() == 1 &&
          (finalize || positions.dim() == 1) && newest_main_kv.dim() == 3 &&
          newest_logical_ids.dim() == 2 && miss_logical_ids.dim() == 3 &&
          miss_victim_slots.dim() == 3 && miss_counts.dim() == 2 &&
          hit_counts.dim() == 2 && resident_main_kv.dim() == 3 &&
          resident_logical_ids.dim() == 2 && resident_last_access.dim() == 2 &&
          provisional_slots.dim() == 3 && tp_fence_token.dim() == 1,
      "invalid sparse MLA transfer tensor rank");
  const int token_rows = req_id_per_token.size(0);
  const int topk = miss_logical_ids.size(2);
  const int request_slots = resident_main_kv.size(0);
  const int resident_rows = resident_main_kv.size(1);
  const int request_block_width = request_block_ids.size(1);
  const int num_host_blocks = main_host_kv_uva.size(0);
  const int head_dim = resident_main_kv.size(2);
  const int window = newest_logical_ids.size(1);
  STD_TORCH_CHECK(
      token_rows > 0 && token_rows <= request_slots * window &&
          request_slots > 0 && resident_rows >= topk && topk > 0 &&
          head_dim > 0 && request_block_width > 0 && num_host_blocks > 0 &&
          block_size > 0 && block_size <= std::numeric_limits<int>::max() &&
          block_size == main_host_kv_uva.size(1) &&
          main_host_kv_uva.size(2) == head_dim &&
          request_block_ids.size(0) == request_slots &&
          request_num_blocks.size(0) == request_slots &&
          request_num_tokens.size(0) == request_slots &&
          request_active.size(0) == request_slots &&
          (finalize || positions.size(0) == token_rows) &&
          newest_main_kv.size(0) == request_slots &&
          newest_main_kv.size(1) == window &&
          newest_main_kv.size(2) == head_dim &&
          newest_logical_ids.size(0) == request_slots &&
          newest_logical_ids.size(1) == window && window > 0 &&
          miss_logical_ids.size(0) == request_slots &&
          miss_logical_ids.size(1) == window &&
          miss_victim_slots.sizes().equals(miss_logical_ids.sizes()) &&
          miss_counts.size(0) == request_slots &&
          miss_counts.size(1) == window &&
          hit_counts.sizes().equals(miss_counts.sizes()) &&
          resident_logical_ids.size(0) == request_slots &&
          resident_logical_ids.size(1) == resident_rows &&
          resident_last_access.sizes().equals(resident_logical_ids.sizes()) &&
          provisional_slots.sizes().equals(miss_victim_slots.sizes()) &&
          tp_fence_token.size(0) == 1,
      "invalid sparse MLA transfer static shape");
  const torch::stable::accelerator::DeviceGuard guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  if (finalize) {
    const int provisional_topk = provisional_slots.size(2);
    if (finalize_phase == 0) {
      validate_finalize_mtp_kernel<<<request_slots, 1, 0, stream>>>(
          request_block_ids.const_data_ptr<int32_t>(),
          request_num_blocks.const_data_ptr<int32_t>(),
          request_num_tokens.const_data_ptr<int32_t>(),
          request_active.const_data_ptr<bool>(),
          static_cast<const uint16_t*>(newest_main_kv.const_data_ptr()),
          newest_logical_ids.const_data_ptr<int64_t>(),
          tp_fence_token.mutable_data_ptr<int32_t>(), request_slots,
          request_block_width, window, num_host_blocks, block_size, head_dim);
    } else {
      finalize_mtp_kernel<<<request_slots, 256, 0, stream>>>(
          static_cast<uint16_t*>(main_host_kv_uva.mutable_data_ptr()),
          request_block_ids.const_data_ptr<int32_t>(),
          request_num_blocks.const_data_ptr<int32_t>(),
          request_num_tokens.const_data_ptr<int32_t>(),
          request_active.const_data_ptr<bool>(),
          static_cast<const uint16_t*>(newest_main_kv.const_data_ptr()),
          newest_logical_ids.mutable_data_ptr<int64_t>(),
          resident_logical_ids.mutable_data_ptr<int64_t>(),
          resident_last_access.mutable_data_ptr<int64_t>(),
          provisional_slots.mutable_data_ptr<int32_t>(),
          tp_fence_token.const_data_ptr<int32_t>(), is_host_writer,
          request_slots, request_block_width, resident_rows, window,
          provisional_topk, num_host_blocks, block_size, head_dim);
    }
    const cudaError_t error = cudaGetLastError();
    STD_TORCH_CHECK(error == cudaSuccess,
                    "sparse_mla_offload_transfer finalize failed: ",
                    cudaGetErrorString(error));
    return;
  }
  transfer_misses_kernel<<<token_rows * topk, 256, 0, stream>>>(
      static_cast<const uint16_t*>(main_host_kv_uva.const_data_ptr()),
      request_block_ids.const_data_ptr<int32_t>(),
      request_num_blocks.const_data_ptr<int32_t>(),
      request_num_tokens.const_data_ptr<int32_t>(),
      request_active.const_data_ptr<bool>(),
      req_id_per_token.const_data_ptr<int32_t>(),
      positions.const_data_ptr<int64_t>(),
      miss_logical_ids.const_data_ptr<int32_t>(),
      miss_victim_slots.const_data_ptr<int32_t>(),
      miss_counts.const_data_ptr<int32_t>(),
      static_cast<uint16_t*>(resident_main_kv.mutable_data_ptr()),
      resident_logical_ids.mutable_data_ptr<int64_t>(),
      resident_last_access.mutable_data_ptr<int64_t>(), token_rows, window,
      topk, request_slots, request_block_width, resident_rows, num_host_blocks,
      block_size, head_dim);
  if (is_host_writer) {
    writeback_current_kernel<<<token_rows, 256, 0, stream>>>(
        static_cast<uint16_t*>(main_host_kv_uva.mutable_data_ptr()),
        request_block_ids.const_data_ptr<int32_t>(),
        request_num_blocks.const_data_ptr<int32_t>(),
        request_num_tokens.const_data_ptr<int32_t>(),
        request_active.const_data_ptr<bool>(),
        req_id_per_token.const_data_ptr<int32_t>(),
        positions.const_data_ptr<int64_t>(),
        static_cast<const uint16_t*>(newest_main_kv.const_data_ptr()),
        newest_logical_ids.const_data_ptr<int64_t>(),
        tp_fence_token.const_data_ptr<int32_t>(), token_rows, window,
        request_slots, request_block_width, num_host_blocks, block_size,
        head_dim);
    record_provisional_slots_kernel<<<token_rows, 256, 0, stream>>>(
        req_id_per_token.const_data_ptr<int32_t>(),
        positions.const_data_ptr<int64_t>(),
        miss_victim_slots.const_data_ptr<int32_t>(),
        provisional_slots.mutable_data_ptr<int32_t>(), token_rows, window, topk,
        request_slots);
  }
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, "sparse_mla_offload_transfer failed: ",
                  cudaGetErrorString(error));
}
