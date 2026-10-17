#ifndef VLLM_NUMA_DISABLED
  #include <numa.h>
  #include <numaif.h>
  #include <unistd.h>
  #include <cerrno>
  #include <cstring>
  #include <string>
  #include <sched.h>
#endif
#include <algorithm>

#include "cpu/moe_numa_shard.hpp"
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
  #include <unistd.h>
  #include <sys/syscall.h>
  #define gettid() syscall(SYS_gettid)
#endif

#include "cpu/utils.hpp"

#ifdef VLLM_NUMA_DISABLED
void init_cpu_memory_env(std::vector<int64_t> node_ids) {}
#else
void init_cpu_memory_env(std::vector<int64_t> node_ids) {
  // Memory node binding
  if (numa_available() != -1) {
    // Concatenate all node_ids into a single comma-separated string
    if (!node_ids.empty()) {
      std::string node_ids_str;
      for (const int node_id : node_ids) {
        if (!node_ids_str.empty()) {
          node_ids_str += ",";
        }
        node_ids_str += std::to_string(node_id);
      }

      bitmask* mask = numa_parse_nodestring(node_ids_str.c_str());
      bitmask* src_mask = numa_get_mems_allowed();

      int pid = getpid();

      if (mask && src_mask) {
        // move all existing pages to the specified numa node.
        *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
        int page_num = numa_migrate_pages(pid, src_mask, mask);
        if (page_num == -1) {
          TORCH_WARN("numa_migrate_pages failed. errno: " +
                     std::to_string(errno));
        }

        // Restrict memory allocation to the selected NUMA node(s).
        // Enhances memory locality for the threads bound to those NUMA CPUs.
        if (node_ids.size() > 1) {
          errno = 0;
          numa_set_interleave_mask(mask);
          if (errno != 0) {
            TORCH_WARN("numa_set_interleave_mask failed. errno: " +
                       std::to_string(errno));
          } else {
            TORCH_WARN(
                "NUMA binding: Using INTERLEAVE policy for memory "
                "allocation across multiple NUMA nodes (nodes: " +
                node_ids_str +
                "). Memory allocations will be "
                "interleaved across the specified NUMA nodes.");
          }
        } else {
          errno = 0;
          numa_set_membind(mask);
          if (errno != 0) {
            TORCH_WARN("numa_set_membind failed. errno: " +
                       std::to_string(errno));
          } else {
            TORCH_WARN(
                "NUMA binding: Using MEMBIND policy for memory "
                "allocation on the NUMA nodes (" +
                node_ids_str +
                "). Memory allocations will be "
                "strictly bound to these NUMA nodes.");
          }
        }

        numa_set_strict(1);

        numa_free_nodemask(mask);
        numa_free_nodemask(src_mask);
      } else {
        TORCH_WARN(
            "numa_parse_nodestring or numa_get_run_node_mask failed. errno: " +
            std::to_string(errno));
      }
    }
  }
}
#endif  // VLLM_NUMA_DISABLED

namespace cpu_utils {
ScratchPadManager::ScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

void ScratchPadManager::realloc(size_t new_size) {
  new_size = round(new_size);
  if (new_size > size_) {
    void* new_ptr = std::aligned_alloc(64, new_size);
    TORCH_CHECK(new_ptr != nullptr,
                "ScratchPadManager: aligned_alloc failed for size ", new_size);
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = new_ptr;
    size_ = new_size;
  }
}

ScratchPadManager* ScratchPadManager::get_scratchpad_manager() {
  static ScratchPadManager manager;
  return &manager;
}
}  // namespace cpu_utils

void compute_slot_mapping_kernel_impl(const torch::Tensor query_start_loc,
                                      const torch::Tensor positions,
                                      const torch::Tensor block_table,
                                      torch::Tensor slot_mapping,
                                      const int64_t block_size) {
  const int32_t req_num = query_start_loc.size(0) - 1;
  const int64_t block_table_stride = block_table.stride(0);

  const int32_t* __restrict__ query_start_loc_ptr =
      query_start_loc.data_ptr<int32_t>();
  const int64_t* __restrict__ positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* __restrict__ blocktable_ptr = block_table.data_ptr<int32_t>();
  int64_t* __restrict__ slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();

#pragma omp parallel for
  for (int32_t req_idx = 0; req_idx < req_num; ++req_idx) {
    int32_t token_start_idx = query_start_loc_ptr[req_idx];
    int32_t token_end_idx = query_start_loc_ptr[req_idx + 1];
    int32_t token_num = token_end_idx - token_start_idx;
    const int64_t* __restrict__ curr_position_ptr =
        positions_ptr + token_start_idx;
    int64_t* __restrict__ curr_slot_mapping_ptr =
        slot_mapping_ptr + token_start_idx;
    const int32_t* __restrict__ curr_block_table_ptr =
        blocktable_ptr + req_idx * block_table_stride;

    for (int32_t token_idx = 0; token_idx < token_num; ++token_idx) {
      int64_t token_position = curr_position_ptr[token_idx];
      int64_t block_id = curr_block_table_ptr[token_position / block_size];
      curr_slot_mapping_ptr[token_idx] =
          block_id * block_size + token_position % block_size;
    }
  }
}

// ---------------------------------------------------------------------------
// NUMA sharding for the CPU MoE experts. See csrc/cpu/moe_numa_shard.hpp for
// why the split lives where it does and what the kernel expects of it.
// ---------------------------------------------------------------------------

namespace vllm_cpu_moe_numa {
int& shard_count() {
  static int shards = 1;
  return shards;
}
}  // namespace vllm_cpu_moe_numa

// The NUMA node each shard's threads run on, in shard order. Shard `s` is not
// assumed to live on node `s`: under a cpuset the usable nodes can be any
// subset, e.g. {2, 3}, and binding those shards to nodes 0 and 1 would place
// every page on a node whose CPUs are not running the work.
static std::vector<int64_t>& moe_numa_shard_nodes() {
  static std::vector<int64_t> nodes;
  return nodes;
}

int64_t cpu_moe_numa_shards() { return vllm_cpu_moe_numa::shard_count(); }

#ifdef VLLM_NUMA_DISABLED
void set_cpu_moe_numa_nodes(std::vector<int64_t> node_ids) {}

void place_moe_expert_weight(at::Tensor& weight, int64_t block_size) {}
#else
void set_cpu_moe_numa_nodes(std::vector<int64_t> node_ids) {
  const int shards = static_cast<int>(node_ids.size());
  // Only what has to hold for the *kernel*: distinct, non-negative ids. Whether
  // the machine actually has those nodes is checked where it matters, in
  // place_moe_expert_weight, and warned about rather than raised. Splitting the
  // loop is meaningful without them -- it is how the split is tested on a
  // single-node machine, where no placement is possible but the arithmetic
  // still has to come out bit for bit the same.
  for (size_t i = 0; i < node_ids.size(); ++i) {
    TORCH_CHECK(node_ids[i] >= 0, "node id must be >= 0, got ", node_ids[i]);
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(node_ids[i] != node_ids[j], "node id ", node_ids[i],
                  " appears twice in the MoE shard node list.");
    }
  }
  moe_numa_shard_nodes() = std::move(node_ids);
  vllm_cpu_moe_numa::shard_count() = std::max(shards, 1);
}

// Move the pages of one expert weight so that each shard's rows sit on the node
// that will compute them.
//
// `weight` is [E, rows, ...] with `rows` the parallel axis of the GEMM that
// reads it -- 2N for w1, K for w2 -- already in the kernel's packed layout. The
// split has to be the *same* one the kernel makes at run time, so both go
// through moe_numa_block_split on the block count rather than on the rows.
//
// MPOL_MF_MOVE is what makes this work after the weights are filled: without it
// mbind only affects future faults and an already-populated tensor would not
// move. Failures are reported and not fatal -- a weight that did not move is
// slow, not wrong.
void place_moe_expert_weight(at::Tensor& weight, int64_t block_size) {
  const int shards = vllm_cpu_moe_numa::shard_count();
  if (shards <= 1 || numa_available() == -1) {
    return;
  }
  // Best-effort all the way down: a weight this does not understand is left
  // where it is, which costs speed, rather than aborting a model load that
  // would otherwise have worked.
  if (!weight.is_contiguous() || weight.dim() < 2) {
    TORCH_WARN_ONCE(
        "place_moe_expert_weight: skipping an expert weight that is not a "
        "contiguous tensor of rank >= 2; it stays where it was allocated and "
        "will be read across NUMA nodes.");
    return;
  }
  const std::vector<int64_t>& nodes = moe_numa_shard_nodes();
  if (static_cast<int>(nodes.size()) != shards) {
    return;
  }
  const int64_t max_node = numa_max_node();
  for (const int64_t node : nodes) {
    if (node > max_node) {
      TORCH_WARN_ONCE(
          "place_moe_expert_weight: node ", node,
          " does not exist on this machine (nodes are 0..", max_node,
          "); leaving the expert weights where they were allocated.");
      return;
    }
  }

  const int64_t experts = weight.size(0);
  const int64_t rows = weight.size(1);
  const int64_t row_bytes = weight.nbytes() / (experts * rows);
  const int64_t blocks = (rows + block_size - 1) / block_size;
  if (blocks < shards) {
    return;
  }

  const long page = sysconf(_SC_PAGESIZE);
  char* base = static_cast<char*>(weight.data_ptr());
  int failures = 0;

  for (int64_t e = 0; e < experts; ++e) {
    char* expert = base + e * rows * row_bytes;
    for (int shard = 0; shard < shards; ++shard) {
      int b0 = 0, b1 = 0;
      vllm_cpu_moe_numa::moe_numa_block_split(static_cast<int>(blocks), shards,
                                              shard, &b0, &b1);
      const int64_t r0 = std::min<int64_t>(int64_t(b0) * block_size, rows);
      const int64_t r1 = std::min<int64_t>(int64_t(b1) * block_size, rows);
      if (r1 <= r0) {
        continue;
      }
      // mbind needs a page-aligned start; round the slice inward so a slice
      // never claims a page that belongs to its neighbour. The few rows in the
      // partial pages at each edge stay wherever they were, which costs at most
      // one page per boundary per expert.
      char* lo = expert + r0 * row_bytes;
      char* hi = expert + r1 * row_bytes;
      char* aligned_lo = reinterpret_cast<char*>(
          (reinterpret_cast<uintptr_t>(lo) + page - 1) & ~uintptr_t(page - 1));
      char* aligned_hi = reinterpret_cast<char*>(
          reinterpret_cast<uintptr_t>(hi) & ~uintptr_t(page - 1));
      if (aligned_hi <= aligned_lo) {
        continue;
      }
      bitmask* mask = numa_allocate_nodemask();
      if (mask == nullptr) {
        ++failures;
        continue;
      }
      numa_bitmask_clearall(mask);
      numa_bitmask_setbit(mask, static_cast<unsigned int>(nodes[shard]));
      // `mask->size + 1` is libnuma's own maxnode convention, not an overrun:
      // the kernel decrements maxnode before sizing the copy (`--maxnode` at
      // the top of get_nodes() in mm/mempolicy.c), so this reads exactly the
      // longs numa_allocate_nodemask() allocated.
      if (mbind(aligned_lo, aligned_hi - aligned_lo, MPOL_BIND, mask->maskp,
                mask->size + 1, MPOL_MF_MOVE) != 0) {
        ++failures;
      }
      numa_free_nodemask(mask);
    }
  }
  if (failures > 0) {
    TORCH_WARN("place_moe_expert_weight: ", failures,
               " mbind calls failed; those pages stay where they were and the "
               "shard that owns them will read across nodes. errno: ",
               std::strerror(errno));
  }
}
#endif
