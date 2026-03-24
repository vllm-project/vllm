#ifndef VLLM_NUMA_DISABLED
  #include <numa.h>
  #include <unistd.h>
  #include <string>
  #include <sched.h>
#endif
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
  #include <unistd.h>
  #include <sys/syscall.h>
  #define gettid() syscall(SYS_gettid)
#endif

#include "cpu/utils.hpp"

#ifdef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  return std::string(
      "Warning: NUMA is not enabled in this build. `init_cpu_threads_env` has "
      "no effect to setup thread affinity.");
}

#endif

#ifndef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  bitmask* omp_cpu_mask = numa_parse_cpustring_all(cpu_ids.c_str());
  TORCH_CHECK(omp_cpu_mask != nullptr,
              "Failed to parse CPU string: " + cpu_ids);
  TORCH_CHECK(omp_cpu_mask->size > 0);
  std::vector<int> omp_cpu_ids;
  omp_cpu_ids.reserve(omp_cpu_mask->size);

  constexpr int group_size = 8 * sizeof(*omp_cpu_mask->maskp);

  for (int offset = 0; offset < omp_cpu_mask->size; offset += group_size) {
    unsigned long group_mask = omp_cpu_mask->maskp[offset / group_size];
    int i = 0;
    while (group_mask) {
      if (group_mask & 1) {
        omp_cpu_ids.emplace_back(offset + i);
      }
      ++i;
      group_mask >>= 1;
    }
  }

  // Memory node binding
  if (numa_available() != -1) {
    std::set<int> node_ids;
    for (const auto& cpu_id : omp_cpu_ids) {
      int node_id = numa_node_of_cpu(cpu_id);
      if (node_id != -1) {
        node_ids.insert(node_id);
      }
    }
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

  // OMP threads binding
  omp_set_num_threads((int)omp_cpu_ids.size());
  torch::set_num_threads((int)omp_cpu_ids.size());
  TORCH_CHECK_EQ(omp_cpu_ids.size(), torch::get_num_threads());
  TORCH_CHECK_EQ(omp_cpu_ids.size(), omp_get_max_threads());

  std::vector<std::pair<int, int>> thread_core_mapping;
  thread_core_mapping.reserve(omp_cpu_ids.size());
  omp_lock_t writelock;
  omp_init_lock(&writelock);

  #pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < omp_cpu_ids.size(); ++i) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(omp_cpu_ids[i], &mask);
    int ret = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
    if (ret == -1) {
      TORCH_CHECK(false,
                  "sched_setaffinity failed. errno: " + std::to_string(errno));
    }

    omp_set_lock(&writelock);
    thread_core_mapping.emplace_back(gettid(), omp_cpu_ids[i]);
    omp_unset_lock(&writelock);
  }

  omp_destroy_lock(&writelock);

  numa_free_nodemask(omp_cpu_mask);

  std::stringstream ss;
  ss << "OMP threads binding of Process " << getpid() << ":\n";
  std::sort(thread_core_mapping.begin(), thread_core_mapping.end(),
            [](auto&& a, auto&& b) { return a.second < b.second; });
  for (auto&& item : thread_core_mapping) {
    ss << "\t"
       << "OMP tid: " << item.first << ", core " << item.second << "\n";
  }

  return ss.str();
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
