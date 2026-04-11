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
