#ifndef VLLM_NUMA_DISABLED
  #include <numa.h>
  #include <unistd.h>
  #include <string>
  #include <sched.h>
#endif

#include "cpu_types.hpp"

#ifdef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  return std::string(
      "Warning: NUMA is not enabled in this build. `init_cpu_threads_env` has "
      "no effect to setup thread affinity.");
}

#endif

#ifndef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  bitmask* omp_cpu_mask = numa_parse_cpustring(cpu_ids.c_str());
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
    int mem_node_id = numa_node_of_cpu(omp_cpu_ids.front());
    bitmask* mask = numa_parse_nodestring(std::to_string(mem_node_id).c_str());
    bitmask* src_mask = numa_get_membind();

    int pid = getpid();

    // move all existing pages to the specified numa node.
    *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
    int page_num = numa_migrate_pages(pid, src_mask, mask);
    if (page_num == -1) {
      TORCH_CHECK(false,
                  "numa_migrate_pages failed. errno: " + std::to_string(errno));
    }

    // restrict memory allocation node.
    numa_set_membind(mask);
    numa_set_strict(1);
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
#endif