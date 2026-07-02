#include "cpu/cpu_types.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#ifdef VLLM_CPU_RDMA_HAR
#include <infiniband/verbs.h>
#endif
#include <numa.h>
#include <sched.h>
#include <sstream>

#if defined(__aarch64__) || defined(__powerpc64__)
  #include <atomic>
  #include <arm_neon.h>
#endif

namespace {
#define MAX_SHM_RANK_NUM 8

#define PER_THREAD_SHM_BUFFER_BYTES (4 * 1024 * 1024)
static_assert(PER_THREAD_SHM_BUFFER_BYTES % 2 == 0);
#define PER_THREAD_SHM_BUFFER_OFFSET (PER_THREAD_SHM_BUFFER_BYTES >> 1)
#define MIN_THREAD_PROCESS_SIZE (256)
#define MAX_P2P_SEND_TENSOR_NUM 8

template <typename scalar_t>
struct KernelVecType {
  using scalar_vec_t = void;
};

template <>
struct KernelVecType<float> {
  using scalar_vec_t = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::BFloat16> {
  using scalar_vec_t = vec_op::BF16Vec16;
};

template <>
struct KernelVecType<c10::Half> {
  using scalar_vec_t = vec_op::FP16Vec16;
};

struct ThreadSHMContext {
#if defined(__aarch64__) || defined(__powerpc64__)
  // memory model is weaker on AArch64, so we use atomic variables for
  // consumer (load-acquire) and producer (store-release) to make sure
  // that a stamp cannot be ready before the corresponding data is ready.
  std::atomic<char> _curr_thread_stamp[2];
  std::atomic<char> _ready_thread_stamp[2];
  static_assert(std::atomic<char>::is_always_lock_free);
#else
  volatile char _curr_thread_stamp[2];
  volatile char _ready_thread_stamp[2];
#endif  // __aarch64__
  int local_stamp_buffer_idx;
  int remote_stamp_buffer_idx;
  int thread_id;
  int thread_num;
  int rank;
  int group_size;
  size_t _spinning_count;
  int swizzled_ranks[MAX_SHM_RANK_NUM];
  void* thread_shm_ptrs[MAX_SHM_RANK_NUM];
  ThreadSHMContext* shm_contexts[MAX_SHM_RANK_NUM];
  size_t _thread_buffer_mask[2];
  char _padding2[40];

  ThreadSHMContext(const int thread_id, const int thread_num, const int rank,
                   const int group_size, void* thread_shm_ptr)
      : local_stamp_buffer_idx(0),
        remote_stamp_buffer_idx(0),
        thread_id(thread_id),
        thread_num(thread_num),
        rank(rank),
        group_size(group_size),
        _spinning_count(0) {
    static_assert(sizeof(ThreadSHMContext) % 64 == 0);
    TORCH_CHECK(group_size <= MAX_SHM_RANK_NUM);
    TORCH_CHECK((size_t)this % 64 == 0);
    TORCH_CHECK((size_t)thread_shm_ptr % 64 == 0);
#if defined(__aarch64__) || defined(__powerpc64__)
    _curr_thread_stamp[0].store(1, std::memory_order_relaxed);
    _curr_thread_stamp[1].store(1, std::memory_order_relaxed);
    _ready_thread_stamp[0].store(0, std::memory_order_relaxed);
    _ready_thread_stamp[1].store(0, std::memory_order_relaxed);
#else
    _curr_thread_stamp[0] = 1;
    _curr_thread_stamp[1] = 1;
    _ready_thread_stamp[0] = 0;
    _ready_thread_stamp[1] = 0;
#endif  // __aarch64__
    _thread_buffer_mask[0] = 0;
    _thread_buffer_mask[1] = 0;
    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      shm_contexts[i] = nullptr;
      thread_shm_ptrs[i] = nullptr;
      swizzled_ranks[i] = (i + rank) % group_size;
    }
    set_context(rank, this, thread_shm_ptr);
  }

  void set_stamp_buffer_idx(int local, int remote) {
    local_stamp_buffer_idx = local;
    remote_stamp_buffer_idx = remote;
  }

  void set_context(int rank, ThreadSHMContext* ptr, void* thread_shm_ptr) {
    TORCH_CHECK(rank < MAX_SHM_RANK_NUM);
    TORCH_CHECK(ptr);
    TORCH_CHECK(thread_shm_ptr);
    TORCH_CHECK_EQ(ptr->thread_num, thread_num);
    TORCH_CHECK_EQ(ptr->thread_id, thread_id);
    shm_contexts[rank] = ptr;
    thread_shm_ptrs[rank] = thread_shm_ptr;
  }

  template <typename T>
  T* get_thread_shm_ptr(int rank) {
    return reinterpret_cast<T*>(
        reinterpret_cast<int8_t*>(thread_shm_ptrs[rank]) +
        (PER_THREAD_SHM_BUFFER_OFFSET &
         _thread_buffer_mask[local_stamp_buffer_idx]));
  }

  void next_buffer() {
    _thread_buffer_mask[local_stamp_buffer_idx] ^= 0xFFFFFFFFFFFFFFFF;
  }

  char get_curr_stamp(int idx) const {
#if defined(__aarch64__) || defined(__powerpc64__)
    return _curr_thread_stamp[idx].load(std::memory_order_acquire);
#else
    return _curr_thread_stamp[idx];
#endif  // __aarch64__
  }

  char get_ready_stamp(int idx) const {
#if defined(__aarch64__) || defined(__powerpc64__)
    return _ready_thread_stamp[idx].load(std::memory_order_acquire);
#else
    return _ready_thread_stamp[idx];
#endif  // __aarch64__
  }

  void next_stamp() {
#if defined(__aarch64__) || defined(__powerpc64__)
    _curr_thread_stamp[local_stamp_buffer_idx].fetch_add(
        1, std::memory_order_release);
#else
    _mm_mfence();
    _curr_thread_stamp[local_stamp_buffer_idx] += 1;
#endif  // __aarch64__
  }

  void commit_ready_stamp() {
#if defined(__aarch64__) || defined(__powerpc64__)
    _ready_thread_stamp[local_stamp_buffer_idx].store(
        _curr_thread_stamp[local_stamp_buffer_idx].load(
            std::memory_order_relaxed),
        std::memory_order_release);
#else
    _mm_mfence();
    _ready_thread_stamp[local_stamp_buffer_idx] =
        _curr_thread_stamp[local_stamp_buffer_idx];
#endif  // __aarch64__
  }

  int get_swizzled_rank(int idx) { return swizzled_ranks[idx]; }

  template <typename Cond>
  void wait_for_all(Cond&& cond) {
    for (int idx = 1; idx < group_size; ++idx) {
      int rank = get_swizzled_rank(idx);
      wait_for_one(rank, std::forward<Cond>(cond));
    }
  }

  template <typename Cond>
  void wait_for_one(int rank, Cond&& cond) {
    ThreadSHMContext* rank_ctx = shm_contexts[rank];
    for (;;) {
      char local_curr_stamp = get_curr_stamp(local_stamp_buffer_idx);
      char local_ready_stamp = get_ready_stamp(local_stamp_buffer_idx);
      char rank_curr_stamp = rank_ctx->get_curr_stamp(remote_stamp_buffer_idx);
      char rank_ready_stamp =
          rank_ctx->get_ready_stamp(remote_stamp_buffer_idx);
      if (cond(local_curr_stamp, local_ready_stamp, rank_curr_stamp,
               rank_ready_stamp)) {
        break;
      }
      ++_spinning_count;
#if defined(__aarch64__)
      __asm__ __volatile__("yield");
#elif defined(__powerpc64__)
      __asm__ __volatile__("or 1,1,1");
#else
      _mm_pause();
#endif  // __aarch64__
    }
  }

  static bool check_no_buffer_conflict(char local_curr_stamp,
                                       char local_ready_stamp,
                                       char rank_curr_stamp,
                                       char rank_ready_stamp) {
    char temp = rank_curr_stamp + 2;
    return local_curr_stamp != temp;
  }

  static bool check_stamp_ready(char local_curr_stamp, char local_ready_stamp,
                                char rank_curr_stamp, char rank_ready_stamp) {
    char temp = local_curr_stamp + 1;
    return (local_curr_stamp == rank_ready_stamp) || (temp == rank_ready_stamp);
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "SHMContext:";
    ss << "\nrank: " << rank;
    ss << "\ngroup_size: " << group_size;
    ss << "\nthread_num: " << thread_num;
    ss << "\nthread_id: " << thread_id;

    ss << "\nshm_ctx_stat_loop_seq: [";
    for (int i = 0; i < group_size; ++i) {
      ss << swizzled_ranks[i] << ", ";
    }
    ss << "]";

    ss << "\nshm_contexts: [";
    for (int i = 0; i < group_size; ++i) {
      if (shm_contexts[i]) {
        ss << shm_contexts[i]->rank << ", ";
      }
    }
    ss << "]";

    return ss.str();
  }
};

class SHMManager {
 public:
  explicit SHMManager(const std::string& name, const int rank,
                      const int group_size, const int thread_num)
      : _rank(rank),
        _group_size(group_size),
        _thread_num(thread_num),
        _shm_names({""}),
        _shared_mem_ptrs({nullptr}),
        _shm_ctx(nullptr) {
    _shm_names[rank] = get_shm_name(name, rank);
    _shared_mem_ptrs[rank] = init_shm(rank);
    _shm_ctx = reinterpret_cast<ThreadSHMContext*>(_shared_mem_ptrs[rank]);

    for (int i = 0; i < _thread_num; ++i) {
      ThreadSHMContext* ctx = new (_shm_ctx + i)
          ThreadSHMContext(i, _thread_num, _rank, _group_size,
                           compute_thread_shm_ptr(_shm_ctx, i));
    }
  }

  void join(const std::string& name) {
    for (int rank_idx = 0; rank_idx < _group_size; ++rank_idx) {
      if (rank_idx != _rank) {
        TORCH_CHECK(_shm_names[rank_idx].empty());
        TORCH_CHECK(_shared_mem_ptrs[rank_idx] == nullptr);
        _shm_names[rank_idx] = get_shm_name(name, rank_idx);
        _shared_mem_ptrs[rank_idx] = init_shm(rank_idx);
        ThreadSHMContext* target_ctx =
            reinterpret_cast<ThreadSHMContext*>(_shared_mem_ptrs[rank_idx]);
        for (int thread_idx = 0; thread_idx < _thread_num; ++thread_idx) {
          _shm_ctx[thread_idx].set_context(
              rank_idx, target_ctx + thread_idx,
              compute_thread_shm_ptr(target_ctx, thread_idx));
        }
      }
    }
  }

  ~SHMManager() { destroy_shm(); }

  ThreadSHMContext* get_shm_ctx() const { return _shm_ctx; }

  static std::string get_shm_name(const std::string& name, int rank) {
    return name + "_" + std::to_string(rank);
  }

  static int64_t create_singleton_instance(const std::string& name,
                                           const int group_size, const int rank,
                                           const int thread_num) {
    std::lock_guard<std::mutex> guard(SingletonInstancesLock);
    SingletonInstances.emplace_back(
        std::make_unique<SHMManager>(name, rank, group_size, thread_num));
    return static_cast<int64_t>(SingletonInstances.size() - 1);
  }

  static SHMManager* get_singleton_instance(int64_t handle) {
    return SingletonInstances[handle].get();
  }

 protected:
  static std::vector<std::unique_ptr<SHMManager>> SingletonInstances;
  static std::mutex SingletonInstancesLock;

 private:
  static size_t round_to_alignment(size_t num) {
    return ((num + 63) / 64) * 64;
  }

  int8_t* compute_thread_shm_ptr(ThreadSHMContext* ctx, int thread_id) {
    int8_t* thread_shm_ptr =
        reinterpret_cast<int8_t*>(ctx) +
        round_to_alignment(_thread_num * sizeof(ThreadSHMContext));
    return thread_shm_ptr +
           thread_id * round_to_alignment(PER_THREAD_SHM_BUFFER_BYTES);
  }

  size_t compute_shm_size() {
    const size_t rounded_rank_buffer_size =
        round_to_alignment(PER_THREAD_SHM_BUFFER_BYTES) * _thread_num;
    const size_t rounded_thread_shm_ctx_size =
        round_to_alignment(_thread_num * sizeof(ThreadSHMContext));
    const size_t shm_size =
        rounded_thread_shm_ctx_size + rounded_rank_buffer_size;
    return shm_size;
  }

  void* init_shm(int target_rank) {
    const std::string& shm_name = _shm_names[target_rank];
    const int local_rank = _rank;
    const size_t shm_size = compute_shm_size();

    int fd = -1;
    if (local_rank == target_rank) {
      fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
                    S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "create shm in SHMManager failed. errno: " +
                               std::to_string(errno));

      if (ftruncate(fd, shm_size) == -1)
        TORCH_CHECK(false, "ftruncate in SHMManager failed. errno: " +
                               std::to_string(errno));
    } else {
      fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "open shm in SHMManager failed. errno: " +
                               std::to_string(errno));
    }

    void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, 0);

    if (shm_ptr == MAP_FAILED) {
      TORCH_CHECK(false,
                  "mmap in SHMManager failed. errno: " + std::to_string(errno));
    }

    if (close(fd) != 0) {
      TORCH_CHECK(
          false, "close in SHMManager failed. errno: " + std::to_string(errno));
    }

    TORCH_CHECK((size_t)shm_ptr % 64 == 0);

    return shm_ptr;
  }

  void destroy_shm() {
    std::stringstream ss;
    ss << "local rank " << _rank << ": [";
    for (int thread_id = 0; thread_id < _thread_num; ++thread_id) {
      ss << _shm_ctx[thread_id]._spinning_count << ", ";
    }
    ss << "]\n";

    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      if (_shared_mem_ptrs[i] != nullptr) {
        munmap(_shared_mem_ptrs[i], compute_shm_size());
      }

      if (!_shm_names[i].empty()) {
        shm_unlink(_shm_names[i].c_str());
      }
    }
  }

  int _rank;
  int _group_size;
  int _thread_num;
  std::array<std::string, MAX_SHM_RANK_NUM> _shm_names;
  std::array<void*, MAX_SHM_RANK_NUM> _shared_mem_ptrs;
  ThreadSHMContext* _shm_ctx;
};

namespace shm_cc_ops {
template <typename scalar_t, typename F>
void shm_cc_loop(ThreadSHMContext* ctx, int64_t elem_num, F&& inner_func) {
  int thread_num = ctx->thread_num;
  int64_t total_bytes = elem_num * sizeof(scalar_t);
  int64_t total_units_num =
      (total_bytes + MIN_THREAD_PROCESS_SIZE - 1) / MIN_THREAD_PROCESS_SIZE;
  int64_t per_thread_units_num =
      (total_units_num + thread_num - 1) / thread_num;
  int64_t per_unit_elem_num = MIN_THREAD_PROCESS_SIZE / sizeof(scalar_t);
  int64_t max_per_thread_iteration_elem_num =
      (PER_THREAD_SHM_BUFFER_BYTES >> 1) /
      sizeof(scalar_t);  // Note: double buffer
  int64_t per_thread_elem_num = per_unit_elem_num * per_thread_units_num;

#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    int64_t offset = i * per_thread_elem_num;
    int64_t end = std::min(elem_num, offset + per_thread_elem_num);
    int64_t curr_elem_num =
        std::min(max_per_thread_iteration_elem_num, end - offset);
    ThreadSHMContext* thread_ctx = ctx + i;
    bool fast_mode = ((end - offset) <= max_per_thread_iteration_elem_num);

    while (curr_elem_num > 0) {
      inner_func(thread_ctx, offset, curr_elem_num, fast_mode);

      thread_ctx->next_stamp();
      thread_ctx->next_buffer();
      offset += max_per_thread_iteration_elem_num;
      curr_elem_num = std::min(max_per_thread_iteration_elem_num, end - offset);
    }
  }
}

void reset_threads_stamp_buffer_idx(ThreadSHMContext* ctx, int local,
                                    int remote) {
  int thread_num = ctx->thread_num;
  for (int i = 0; i < thread_num; ++i) {
    ThreadSHMContext* thread_ctx = ctx + i;
    thread_ctx->set_stamp_buffer_idx(local, remote);
  }
}
};  // namespace shm_cc_ops

namespace shm_cc_ops {

void memcpy_from_shm(void* dst, void* src, const int64_t bytes) {
  const int64_t aligned_bytes = ((bytes >> 6) << 6);  // 64 bytes aligned
  int64_t i = 0;
#pragma GCC unroll 4
  for (; i < aligned_bytes; i += 64) {
    vec_op::INT8Vec64 data(
        true, (int8_t*)src + i);  // stream loading shm to avoid caching
    data.save((int8_t*)dst + i);
  }
  if (aligned_bytes < bytes) {
    vec_op::INT8Vec64 data(true, (int8_t*)src + aligned_bytes);
    data.save((int8_t*)dst + aligned_bytes, bytes - aligned_bytes);
  }
}

void memcpy_to_shm(void* dst, void* src, const int64_t bytes) {
#pragma GCC unroll 4
  for (int64_t i = 0; i < bytes; i += 64) {
    vec_op::INT8Vec64 data((int8_t*)src + i);
    data.nt_save((int8_t*)dst + i);
  }
}

void memcpy(void* dst, void* src, const int64_t bytes) {
  const int64_t aligned_bytes = ((bytes >> 6) << 6);  // 64 bytes aligned
  int64_t i = 0;
#pragma GCC unroll 4
  for (; i < aligned_bytes; i += 64) {
    vec_op::INT8Vec64 data((int8_t*)src + i);
    data.save((int8_t*)dst + i);
  }
  if (aligned_bytes < bytes) {
    vec_op::INT8Vec64 data((int8_t*)src + aligned_bytes);
    data.save((int8_t*)dst + aligned_bytes, bytes - aligned_bytes);
  }
}

template <typename scalar_t, int RANKS>
void all_reduce_sum_impl(ThreadSHMContext* ctx, scalar_t* data,
                         size_t elem_num) {
  CPU_KERNEL_GUARD_IN(all_reduce_sum_impl)
  using vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int64_t vec_elem_num = vec_t::get_elem_num();
  const int worldsize = ctx->group_size;

  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);
        scalar_t* thread_data_ptr = data + data_offset;
        int64_t thread_data_elem_num = data_elem_num * sizeof(scalar_t);

        scalar_t* remote_data_ptrs[RANKS - 1];
        vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
          remote_data_ptrs[idx] = thread_ctx->get_thread_shm_ptr<scalar_t>(
              thread_ctx->get_swizzled_rank(idx + 1));
        });

        if (!fast_mode) {
          thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
        }

        shm_cc_ops::memcpy_to_shm(thread_shm_ptr, thread_data_ptr,
                                  thread_data_elem_num);
        thread_ctx->commit_ready_stamp();
        int64_t aligned_data_elem_num =
            (data_elem_num / vec_elem_num) * vec_elem_num;
        int64_t i = 0;
        thread_ctx->wait_for_all(ThreadSHMContext::check_stamp_ready);
#pragma GCC unroll 4
        for (; i < aligned_data_elem_num; i += vec_elem_num) {
          vec_t local_data(thread_data_ptr + i);  // load from cache
          vec_op::FP32Vec16 local_data_fp32(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(
                true, remote_data_ptrs[idx] + i);  // stream load from shm
            vec_op::FP32Vec16 remote_data_fp32(remote_data);
            local_data_fp32 = local_data_fp32 + remote_data_fp32;  // sum reduce
          });
          vec_t reduced_data(local_data_fp32);
          reduced_data.save(thread_data_ptr + i);
        }

        if (i < data_elem_num) {
          vec_t local_data(thread_data_ptr + i);  // load from cache
          vec_op::FP32Vec16 local_data_fp32(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(
                true, remote_data_ptrs[idx] + i);  // stream load from shm
            vec_op::FP32Vec16 remote_data_fp32(remote_data);
            local_data_fp32 = local_data_fp32 + remote_data_fp32;  // sum reduce
          });
          vec_t reduced_data(local_data_fp32);
          reduced_data.save(thread_data_ptr + i,
                            data_elem_num - aligned_data_elem_num);
        }
      });

  return;
}
};  // namespace shm_cc_ops

std::vector<std::unique_ptr<SHMManager>> SHMManager::SingletonInstances = {};
std::mutex SHMManager::SingletonInstancesLock = {};

template <typename scalar_t>
void shm_allreduce_sum(ThreadSHMContext* ctx, scalar_t* data, size_t elem_num) {
  switch (ctx->group_size) {
    case 2:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 2>(ctx, data, elem_num);
      break;
    case 3:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 3>(ctx, data, elem_num);
      break;
    case 4:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 4>(ctx, data, elem_num);
      break;
    case 8:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 8>(ctx, data, elem_num);
      break;
    default:
      TORCH_CHECK(false,
                  "Invalid world size: " + std::to_string(ctx->group_size));
  }
}

template <typename scalar_t>
void shm_gather_impl(ThreadSHMContext* ctx, scalar_t* data, size_t elem_num,
                     scalar_t** outputs, const int dst) {
  CPU_KERNEL_GUARD_IN(shm_gather_impl)
  const int worldsize = ctx->group_size;
  TORCH_CHECK_LT(dst, worldsize);
  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);

        if (!fast_mode) {
          thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
        }

        shm_cc_ops::memcpy(thread_shm_ptr, data + data_offset,
                           data_elem_num * sizeof(scalar_t));
        thread_ctx->commit_ready_stamp();
        if (rank == dst) {
          shm_cc_ops::memcpy(outputs[rank] + data_offset, data + data_offset,
                             data_elem_num * sizeof(scalar_t));
          for (int i = 1; i < worldsize; ++i) {
            int src_rank = thread_ctx->get_swizzled_rank(i);
            scalar_t* src_ptr =
                thread_ctx->get_thread_shm_ptr<scalar_t>(src_rank);  // shm
            scalar_t* dst_ptr = outputs[src_rank] + data_offset;
            thread_ctx->wait_for_one(src_rank,
                                     ThreadSHMContext::check_stamp_ready);
            shm_cc_ops::memcpy(dst_ptr, src_ptr,
                               data_elem_num * sizeof(scalar_t));
          }
        }
      });

  return;
}

struct MemPiece {
  void* ptr;
  int64_t size;

  template <typename T>
  T* data_ptr() {
    return reinterpret_cast<T*>(ptr);
  }
};

struct TensorListMeta {
  int64_t tensor_bytes[MAX_P2P_SEND_TENSOR_NUM];
  torch::ScalarType tensor_types[MAX_P2P_SEND_TENSOR_NUM];
  int64_t tensor_num;
  int64_t total_bytes;

  TensorListMeta() : tensor_num(0), total_bytes(0) {
    static_assert(sizeof(TensorListMeta) % 64 == 0);
    static_assert(sizeof(TensorListMeta) <
                  MIN_THREAD_PROCESS_SIZE);  // To ensure the metadata always
                                             // hold by the thread 0
    for (int i = 0; i < MAX_P2P_SEND_TENSOR_NUM; ++i) {
      tensor_bytes[i] = 0;
      tensor_ptrs[i] = nullptr;
      tensor_types[i] = torch::ScalarType::Undefined;
    }
  }

  // For send and recv
  void bind_tensor_list(std::vector<torch::Tensor>& tensor_list) {
    TORCH_CHECK(tensor_types[0] == torch::ScalarType::Undefined,
                "Re-bind TensorListMeta is not allowed.")
    TORCH_CHECK_LE(tensor_list.size(), MAX_P2P_SEND_TENSOR_NUM);
    tensor_num = tensor_list.size();
    int64_t bytes_sum = 0;
    for (int i = 0; i < tensor_list.size(); ++i) {
      torch::Tensor& t = tensor_list[i];
      TORCH_CHECK(t.is_contiguous());
      tensor_bytes[i] = t.nbytes();
      tensor_types[i] = t.scalar_type();
      tensor_ptrs[i] = t.data_ptr();
      bytes_sum += t.nbytes();
    }
    total_bytes = bytes_sum;
  }

  // For recv
  std::vector<torch::Tensor> generate_tensor_list() {
    std::vector<torch::Tensor> tensor_list;
    tensor_list.reserve(tensor_num);

    for (int i = 0; i < tensor_num; ++i) {
      int64_t bytes = tensor_bytes[i];
      auto type = tensor_types[i];
      int64_t elem_bytes = torch::elementSize(type);

      TORCH_CHECK_EQ(bytes % elem_bytes, 0);
      int64_t elem_num = bytes / elem_bytes;
      auto options = torch::TensorOptions().dtype(type).device(torch::kCPU);
      tensor_list.emplace_back(torch::empty({elem_num}, options));
    }
    return tensor_list;
  }

  MemPiece get_data(int64_t offset) {
    for (int i = 0; i < tensor_num; ++i) {
      if (offset < tensor_bytes[i]) {
        return {reinterpret_cast<int8_t*>(tensor_ptrs[i]) + offset,
                tensor_bytes[i] - offset};
      }
      offset -= tensor_bytes[i];
    }
    return {nullptr, 0};
  }

 private:
  void* tensor_ptrs[MAX_P2P_SEND_TENSOR_NUM];
  int8_t _padding[40];
};

void shm_send_tensor_list_impl(ThreadSHMContext* ctx, int64_t dst,
                               const std::vector<torch::Tensor>& tensor_list) {
  CPU_KERNEL_GUARD_IN(shm_send_tensor_list_impl)
  std::vector<torch::Tensor> tensor_list_with_metadata;
  tensor_list_with_metadata.reserve(1 + tensor_list.size());

  auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
  tensor_list_with_metadata.emplace_back(
      torch::empty({sizeof(TensorListMeta)}, options));
  tensor_list_with_metadata.insert(tensor_list_with_metadata.end(),
                                   tensor_list.begin(), tensor_list.end());

  torch::Tensor& metadata_tensor = tensor_list_with_metadata[0];
  TORCH_CHECK_EQ(metadata_tensor.nbytes(), sizeof(TensorListMeta));

  TensorListMeta* metadata = new (metadata_tensor.data_ptr()) TensorListMeta();
  metadata->bind_tensor_list(tensor_list_with_metadata);

  shm_cc_ops::reset_threads_stamp_buffer_idx(ctx, 0, 1);
  shm_cc_ops::shm_cc_loop<int8_t>(
      ctx, metadata->total_bytes,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        int64_t curr_shm_offset = 0;
        thread_ctx->wait_for_one(dst,
                                 ThreadSHMContext::check_no_buffer_conflict);
        while (curr_shm_offset < data_elem_num) {
          MemPiece frag = metadata->get_data(data_offset + curr_shm_offset);
          frag.size = std::min(frag.size, data_elem_num - curr_shm_offset);
          shm_cc_ops::memcpy(
              thread_ctx->get_thread_shm_ptr<int8_t>(rank) + curr_shm_offset,
              frag.ptr, frag.size);
          curr_shm_offset += frag.size;
        }
        thread_ctx->commit_ready_stamp();
      });
}

std::vector<torch::Tensor> shm_recv_tensor_list_impl(ThreadSHMContext* ctx,
                                                     int64_t src) {
  CPU_KERNEL_GUARD_IN(shm_recv_tensor_list_impl)
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
  torch::Tensor metadata_tensor =
      torch::empty({sizeof(TensorListMeta)}, options);

  shm_cc_ops::reset_threads_stamp_buffer_idx(ctx, 1, 0);
  ctx->wait_for_one(src, ThreadSHMContext::check_stamp_ready);
  shm_cc_ops::memcpy(metadata_tensor.data_ptr(),
                     ctx->get_thread_shm_ptr<void>(src),
                     sizeof(TensorListMeta));
  TensorListMeta* src_metadata =
      reinterpret_cast<TensorListMeta*>(metadata_tensor.data_ptr());
  std::vector<torch::Tensor> tensor_list_with_metadata =
      src_metadata->generate_tensor_list();

  TensorListMeta metadata;
  metadata.bind_tensor_list(tensor_list_with_metadata);
  TORCH_CHECK_EQ(metadata.tensor_num, src_metadata->tensor_num);
  TORCH_CHECK_EQ(metadata.total_bytes, src_metadata->total_bytes);

  shm_cc_ops::shm_cc_loop<int8_t>(
      ctx, metadata.total_bytes,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        thread_ctx->wait_for_one(src, ThreadSHMContext::check_stamp_ready);
        int64_t curr_shm_offset = 0;
        while (curr_shm_offset < data_elem_num) {
          MemPiece frag = metadata.get_data(data_offset + curr_shm_offset);
          frag.size = std::min(frag.size, data_elem_num - curr_shm_offset);
          shm_cc_ops::memcpy(
              frag.ptr,
              thread_ctx->get_thread_shm_ptr<int8_t>(src) + curr_shm_offset,
              frag.size);
          curr_shm_offset += frag.size;
        }
      });

  std::vector<torch::Tensor> tensor_list;
  tensor_list.reserve(metadata.tensor_num - 1);
  tensor_list.insert(tensor_list.begin(), tensor_list_with_metadata.begin() + 1,
                     tensor_list_with_metadata.end());

  return tensor_list;
}
}  // namespace

void shm_gather(int64_t handle, torch::Tensor& data,
                const std::optional<std::vector<torch::Tensor>>& outputs,
                int64_t dst) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_gather_impl", [&] {
    CPU_KERNEL_GUARD_IN(shm_gather_impl)

    if (outputs.has_value()) {
      TORCH_CHECK_LE(outputs->size(), MAX_SHM_RANK_NUM);
      scalar_t* output_ptrs[MAX_SHM_RANK_NUM] = {nullptr};
      for (int i = 0; i < outputs->size(); ++i) {
        output_ptrs[i] = outputs->at(i).data_ptr<scalar_t>();
      }
      shm_gather_impl(SHMManager::get_singleton_instance(handle)->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel(), output_ptrs,
                      dst);
    } else {
      shm_gather_impl(SHMManager::get_singleton_instance(handle)->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel(), (scalar_t**)(0),
                      dst);
    }

    CPU_KERNEL_GUARD_OUT(shm_gather_impl)
  });
}

void shm_all_gather(int64_t handle, const torch::Tensor& data,
                    torch::Tensor& output) {
  TORCH_CHECK(data.is_contiguous())
  TORCH_CHECK(output.is_contiguous())

  const int64_t input_elem_num = data.numel();
  const int64_t output_elem_num = output.numel();
  TORCH_CHECK_EQ(output_elem_num % input_elem_num, 0);
  const int world_size = output_elem_num / input_elem_num;

  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_all_gather_impl", [&] {
    CPU_KERNEL_GUARD_IN(shm_all_gather_impl)
    auto ctx = SHMManager::get_singleton_instance(handle)->get_shm_ctx();
    TORCH_CHECK_EQ(ctx->group_size, world_size);

    scalar_t* output_ptrs[MAX_SHM_RANK_NUM] = {nullptr};
    for (int i = 0; i < world_size; ++i) {
      output_ptrs[i] = output.data_ptr<scalar_t>() + i * input_elem_num;
    }
    shm_gather_impl(ctx, data.data_ptr<scalar_t>(), data.numel(), output_ptrs,
                    ctx->rank);
    CPU_KERNEL_GUARD_OUT(shm_all_gather_impl)
  });
}

void shm_allreduce(int64_t handle, torch::Tensor& data) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_allreduce_sum", [&] {
    CPU_KERNEL_GUARD_IN(shm_allreduce_sum)
    shm_allreduce_sum(SHMManager::get_singleton_instance(handle)->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel());
    CPU_KERNEL_GUARD_OUT(shm_allreduce_sum)
  });
}

void shm_send_tensor_list(int64_t handle,
                          const std::vector<torch::Tensor>& tensor_list,
                          int64_t dst) {
  CPU_KERNEL_GUARD_IN(shm_send_tensor_list)
  shm_send_tensor_list_impl(
      SHMManager::get_singleton_instance(handle)->get_shm_ctx(), dst,
      tensor_list);
  CPU_KERNEL_GUARD_OUT(shm_send_tensor_list)
}

std::vector<torch::Tensor> shm_recv_tensor_list(int64_t handle, int64_t src) {
  CPU_KERNEL_GUARD_IN(shm_recv_tensor_list)
  auto tensor_list = shm_recv_tensor_list_impl(
      SHMManager::get_singleton_instance(handle)->get_shm_ctx(), src);
  CPU_KERNEL_GUARD_OUT(shm_recv_tensor_list)
  return tensor_list;
}

int64_t init_shm_manager(const std::string& name, const int64_t group_size,
                         const int64_t rank, const int64_t thread_num) {
  return SHMManager::create_singleton_instance(name, group_size, rank,
                                               thread_num);
}

std::string join_shm_manager(int64_t handle, const std::string& name) {
  auto shm_manager = SHMManager::get_singleton_instance(handle);
  TORCH_CHECK(shm_manager);
  shm_manager->join(name);
  return shm_manager->get_shm_ctx()->to_string();
}

// Hierarchical all-reduce: SHM (intra-node) + IBV/Gloo (cross-node)

// ============================================================
// Raw IBVerbs 2-way all-reduce for cross-node communication.
// Bypasses Gloo to avoid OMP thread contention.
// Uses RDMA Write with Immediate for low-latency signaling.
// ============================================================

#ifdef VLLM_CPU_RDMA_HAR

namespace {

static constexpr size_t IBV_BUF_SIZE = 256 * 1024 * 1024;  // 256 MB

struct IBVState {
  struct ibv_context* ctx = nullptr;
  struct ibv_pd* pd = nullptr;
  struct ibv_cq* cq = nullptr;
  struct ibv_qp* qp = nullptr;

  // Pre-registered data buffers
  void* send_buf = nullptr;
  void* recv_buf = nullptr;
  struct ibv_mr* send_mr = nullptr;
  struct ibv_mr* recv_mr = nullptr;
  size_t buf_size = IBV_BUF_SIZE;
  bool numa_allocated = false;  // true if buffers were allocated with numa_alloc_onnode

  // Remote memory info (for RDMA Write target)
  uint64_t remote_addr = 0;
  uint32_t remote_rkey = 0;

  ~IBVState() {
    if (send_mr) ibv_dereg_mr(send_mr);
    if (recv_mr) ibv_dereg_mr(recv_mr);
    if (qp) ibv_destroy_qp(qp);
    if (cq) ibv_destroy_cq(cq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    if (send_buf) { if (numa_allocated) numa_free(send_buf, buf_size); else free(send_buf); }
    if (recv_buf) { if (numa_allocated) numa_free(recv_buf, buf_size); else free(recv_buf); }
  }
};

std::unordered_map<int64_t, std::unique_ptr<IBVState>> g_ibv_states;
int64_t g_ibv_next_id = 0;

}  // namespace

// Create IBVerbs context, QP, and register buffers.
// Returns handle and local connection info string.
// buf_size_bytes: size of RDMA send/recv buffers; 0 means use default (256 MB).
int64_t ibv_ar_create(const std::string& dev_name, int64_t port,
                      int64_t gid_index, int64_t buf_size_bytes) {
  auto state = std::make_unique<IBVState>();
  if (buf_size_bytes > 0) {
    state->buf_size = static_cast<size_t>(buf_size_bytes);
  }

  // Find and open device
  int num_devs = 0;
  struct ibv_device** dev_list = ibv_get_device_list(&num_devs);
  TORCH_CHECK(dev_list && num_devs > 0, "No IBVerbs devices found");
  struct ibv_device* dev = nullptr;
  for (int i = 0; i < num_devs; i++) {
    if (dev_name == ibv_get_device_name(dev_list[i])) {
      dev = dev_list[i];
      break;
    }
  }
  TORCH_CHECK(dev, "IBVerbs device not found: ", dev_name);
  state->ctx = ibv_open_device(dev);
  ibv_free_device_list(dev_list);
  TORCH_CHECK(state->ctx, "Failed to open IBVerbs device");

  // Determine NIC's NUMA node for optimal buffer placement
  int nic_numa = -1;
  {
    std::string numa_path = "/sys/class/infiniband/" + dev_name + "/device/numa_node";
    FILE* f = fopen(numa_path.c_str(), "r");
    if (f) {
      if (fscanf(f, "%d", &nic_numa) != 1) nic_numa = -1;
      fclose(f);
    }
    fprintf(stderr, "IBV: device %s NUMA node = %d\n", dev_name.c_str(), nic_numa);
  }

  // Temporarily bind to NIC's NUMA node for PD/CQ/QP allocation
  // This ensures verbs driver internal structures (CQ buffers etc.) are NIC-local
  cpu_set_t old_mask;
  bool did_rebind = false;
  if (nic_numa >= 0 && numa_available() >= 0) {
    sched_getaffinity(0, sizeof(old_mask), &old_mask);
    cpu_set_t new_mask;
    CPU_ZERO(&new_mask);
    struct bitmask* cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(nic_numa, cpus) == 0) {
      for (int i = 0; i < numa_num_possible_cpus(); i++) {
        if (numa_bitmask_isbitset(cpus, i)) CPU_SET(i, &new_mask);
      }
      sched_setaffinity(0, sizeof(new_mask), &new_mask);
      did_rebind = true;
    }
    numa_free_cpumask(cpus);
  }

  // Allocate PD
  state->pd = ibv_alloc_pd(state->ctx);
  TORCH_CHECK(state->pd, "Failed to alloc PD");

  // Create CQ (shared for send+recv)
  state->cq = ibv_create_cq(state->ctx, 64, nullptr, nullptr, 0);
  TORCH_CHECK(state->cq, "Failed to create CQ");

  // Create RC QP
  struct ibv_qp_init_attr qp_init = {};
  qp_init.send_cq = state->cq;
  qp_init.recv_cq = state->cq;
  qp_init.qp_type = IBV_QPT_RC;
  qp_init.cap.max_send_wr = 16;
  qp_init.cap.max_recv_wr = 16;
  qp_init.cap.max_send_sge = 1;
  qp_init.cap.max_recv_sge = 1;
  state->qp = ibv_create_qp(state->pd, &qp_init);
  TORCH_CHECK(state->qp, "Failed to create QP");

  // Move QP to INIT state
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = (uint8_t)port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
  int ret = ibv_modify_qp(state->qp, &attr,
                           IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                           IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  TORCH_CHECK(ret == 0, "Failed to move QP to INIT, ret=", ret);

  // Allocate buffers on NIC's NUMA node for optimal DMA performance
  if (nic_numa >= 0 && numa_available() >= 0) {
    state->send_buf = numa_alloc_onnode(state->buf_size, nic_numa);
    state->recv_buf = numa_alloc_onnode(state->buf_size, nic_numa);
    state->numa_allocated = true;
    TORCH_CHECK(state->send_buf, "Failed to numa_alloc send_buf on node ", nic_numa);
    TORCH_CHECK(state->recv_buf, "Failed to numa_alloc recv_buf on node ", nic_numa);
    fprintf(stderr, "IBV: allocated send_buf/recv_buf (%zu MB each) on NUMA %d\n",
            state->buf_size / (1024*1024), nic_numa);
  } else {
    ret = posix_memalign(&state->send_buf, 4096, state->buf_size);
    TORCH_CHECK(ret == 0, "Failed to alloc send_buf");
    ret = posix_memalign(&state->recv_buf, 4096, state->buf_size);
    TORCH_CHECK(ret == 0, "Failed to alloc recv_buf");
    state->numa_allocated = false;
  }
  memset(state->send_buf, 0, state->buf_size);
  memset(state->recv_buf, 0, state->buf_size);

  // Restore CPU affinity
  if (did_rebind) {
    sched_setaffinity(0, sizeof(old_mask), &old_mask);
  }

  state->send_mr = ibv_reg_mr(state->pd, state->send_buf, state->buf_size,
                               IBV_ACCESS_LOCAL_WRITE);
  TORCH_CHECK(state->send_mr, "Failed to register send_mr");
  state->recv_mr = ibv_reg_mr(state->pd, state->recv_buf, state->buf_size,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  TORCH_CHECK(state->recv_mr, "Failed to register recv_mr");

  int64_t handle = g_ibv_next_id++;
  g_ibv_states[handle] = std::move(state);
  return handle;
}

// Get local connection info as a serialized string.
std::string ibv_ar_get_local_info(int64_t handle, int64_t port,
                                  int64_t gid_index) {
  auto& s = *g_ibv_states[handle];
  union ibv_gid gid;
  int ret = ibv_query_gid(s.ctx, (uint8_t)port, (int)gid_index, &gid);
  TORCH_CHECK(ret == 0, "Failed to query GID");

  std::ostringstream ss;
  ss << s.qp->qp_num << ",";
  // Serialize GID as 32 hex chars
  for (int i = 0; i < 16; i++) {
    char buf[3];
    snprintf(buf, sizeof(buf), "%02x", gid.raw[i]);
    ss << buf;
  }
  ss << "," << (uint64_t)s.recv_buf
     << "," << s.recv_mr->rkey;
  return ss.str();
}

// Connect QP to remote peer (INIT → RTR → RTS).
void ibv_ar_connect(int64_t handle, const std::string& remote_info,
                    int64_t port, int64_t gid_index) {
  auto& s = *g_ibv_states[handle];

  // Parse remote info: "qpn,gid_hex,addr,rkey"
  uint32_t remote_qpn;
  union ibv_gid remote_gid;
  uint64_t remote_addr;
  uint32_t remote_rkey;

  std::istringstream iss(remote_info);
  std::string tok;
  std::getline(iss, tok, ',');
  remote_qpn = (uint32_t)std::stoul(tok);
  std::getline(iss, tok, ',');
  // Parse GID hex (32 chars)
  for (int i = 0; i < 16; i++) {
    std::string byte_str = tok.substr(i * 2, 2);
    remote_gid.raw[i] = (uint8_t)std::stoul(byte_str, nullptr, 16);
  }
  std::getline(iss, tok, ',');
  remote_addr = std::stoull(tok);
  std::getline(iss, tok, ',');
  remote_rkey = (uint32_t)std::stoul(tok);

  s.remote_addr = remote_addr;
  s.remote_rkey = remote_rkey;

  // Query actual port MTU (CRITICAL: must match link, not hardcode)
  struct ibv_port_attr port_attr;
  int qret = ibv_query_port(s.ctx, (uint8_t)port, &port_attr);
  TORCH_CHECK(qret == 0, "Failed to query port attributes");

  // QP → RTR
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr.active_mtu;  // use actual link MTU
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.dgid = remote_gid;
  attr.ah_attr.grh.sgid_index = (uint8_t)gid_index;
  attr.ah_attr.grh.hop_limit = 64;
  attr.ah_attr.grh.traffic_class = 0;
  attr.ah_attr.dlid = 0;  // RoCE uses GID, not LID
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = (uint8_t)port;

  int ret = ibv_modify_qp(s.qp, &attr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                           IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                           IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  TORCH_CHECK(ret == 0, "Failed to move QP to RTR, ret=", ret,
              " errno=", strerror(errno));

  // QP → RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 1;

  ret = ibv_modify_qp(s.qp, &attr,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                       IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                       IBV_QP_MAX_QP_RD_ATOMIC);
  TORCH_CHECK(ret == 0, "Failed to move QP to RTS, ret=", ret);

  // Pre-post a receive WR so first incoming RDMA Write with Immediate
  // won't get an RNR NAK if it arrives before ibv_ar_allreduce is called.
  struct ibv_recv_wr pre_recv = {};
  struct ibv_recv_wr* bad_pre_recv = nullptr;
  pre_recv.wr_id = 1;
  pre_recv.num_sge = 0;
  pre_recv.sg_list = nullptr;
  ret = ibv_post_recv(s.qp, &pre_recv, &bad_pre_recv);
  TORCH_CHECK(ret == 0, "Failed to pre-post recv, ret=", ret);
}

// The hot-path 2-way all-reduce over raw RDMA.
// Both ranks call this simultaneously.
// A receive WR is always pre-posted (from connect or previous call).
void ibv_ar_allreduce(int64_t handle, torch::Tensor& data) {
  auto& s = *g_ibv_states[handle];
  TORCH_CHECK(data.is_contiguous(),
              "IBV allreduce requires a contiguous tensor");
  size_t nbytes = data.numel() * data.element_size();
  TORCH_CHECK(nbytes <= s.buf_size,
              "IBV allreduce: tensor size ", nbytes, " exceeds buffer ", s.buf_size);

  // 1. Copy tensor data to registered send buffer
  memcpy(s.send_buf, data.data_ptr(), nbytes);

  // 2. Post RDMA Write with Immediate (sends data + signals remote)
  struct ibv_sge send_sge = {};
  send_sge.addr = (uint64_t)s.send_buf;
  send_sge.length = (uint32_t)nbytes;
  send_sge.lkey = s.send_mr->lkey;

  struct ibv_send_wr send_wr = {};
  struct ibv_send_wr* bad_send = nullptr;
  send_wr.wr_id = 0;  // send marker
  send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  send_wr.send_flags = IBV_SEND_SIGNALED;
  send_wr.imm_data = htonl((uint32_t)nbytes);
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.wr.rdma.remote_addr = s.remote_addr;
  send_wr.wr.rdma.rkey = s.remote_rkey;
  int ret = ibv_post_send(s.qp, &send_wr, &bad_send);
  TORCH_CHECK(ret == 0, "ibv_post_send failed, ret=", ret);

  // 3. Poll CQ for both completions (our send + their recv notification)
  int got_send = 0, got_recv = 0;
  struct ibv_wc wc;
  while (!got_send || !got_recv) {
    int n = ibv_poll_cq(s.cq, 1, &wc);
    if (n > 0) {
      TORCH_CHECK(wc.status == IBV_WC_SUCCESS,
                  "IBV WC error: status=", wc.status,
                  " (", ibv_wc_status_str(wc.status), ")");
      if (wc.wr_id == 0) got_send = 1;
      if (wc.wr_id == 1) got_recv = 1;
    }
  }

  // 4. In-place add: data += recv_buf
  auto recv_tensor = torch::from_blob(
      s.recv_buf, data.sizes(), data.strides(), data.options());
  data.add_(recv_tensor);

  // 5. Post receive for the NEXT call
  struct ibv_recv_wr recv_wr = {};
  struct ibv_recv_wr* bad_recv = nullptr;
  recv_wr.wr_id = 1;
  recv_wr.num_sge = 0;
  recv_wr.sg_list = nullptr;
  ret = ibv_post_recv(s.qp, &recv_wr, &bad_recv);
  TORCH_CHECK(ret == 0, "ibv_post_recv failed, ret=", ret);
}

#endif  // VLLM_CPU_RDMA_HAR (IBV functions)

#ifdef VLLM_CPU_RDMA_HAR
// ============================================================
// Hierarchical all-reduce state (SHM + IBV)
// ============================================================

// Lightweight SHM region for 2-rank reduce + broadcast.
// Uses atomic sequence counters instead of OMP barriers.
// Layout: [seq_reduce(64B)] [seq_read(64B)] [slot0(MAX_BUF)] [slot1(MAX_BUF)] [bcast(MAX_BUF)]
static constexpr size_t HIER_SHM_MAX_BUF = 4 * 1024 * 1024;  // 4 MB per slot
static constexpr size_t HIER_SHM_HEADER = 128;  // 2 cache lines for counters

struct alignas(64) HierShmHeader {
  std::atomic<uint64_t> reduce_seq;  // incremented when rank writes its slot
  char _pad1[56];
  std::atomic<uint64_t> read_seq;    // incremented when rank finishes reading both slots
  char _pad2[56];
};
static_assert(sizeof(HierShmHeader) == 128);

namespace {

struct HierARState {
  int64_t shm_handle;            // existing SHM for fallback (large tensors)
  std::string cross_group_name;  // Gloo fallback (empty if using IBV)
  bool is_leader;
  int64_t ibv_handle;            // -1 if not using IBV
  bool has_local_peer;           // false if 1 worker/node (IBV-only)

  // Lightweight SHM for hier allreduce
  int local_rank;                // 0 or 1 within the node
  void* hier_shm_ptr;            // mmap'd shared memory
  std::string hier_shm_name;
  HierShmHeader* header;
  void* slot0;                   // rank 0's write slot
  void* slot1;                   // rank 1's write slot
  void* bcast_buf;               // leader writes result here
  uint64_t reduce_epoch;         // tracks reduce sequence
  uint64_t read_epoch;           // tracks read-barrier sequence
};
std::unordered_map<int64_t, HierARState> g_hier_ar_states;
int64_t g_hier_ar_next_id = 0;

}  // namespace

int64_t init_hier_ar(int64_t shm_handle,
                     const std::string& cross_group_name,
                     bool is_leader,
                     int64_t ibv_handle) {
  int64_t id = g_hier_ar_next_id++;

  HierARState state;
  state.shm_handle = shm_handle;
  state.cross_group_name = cross_group_name;
  state.is_leader = is_leader;
  state.ibv_handle = ibv_handle;
  // If shm_handle < 0, there's no local peer (1 worker per node)
  state.has_local_peer = (shm_handle >= 0);

  // Determine local rank (0 = leader, 1 = follower)
  state.local_rank = is_leader ? 0 : 1;

  // Skip lightweight SHM when there's no local peer (IBV-only mode)
  state.hier_shm_ptr = nullptr;
  state.header = nullptr;
  state.slot0 = nullptr;
  state.slot1 = nullptr;
  state.bcast_buf = nullptr;
  state.reduce_epoch = 0;
  state.read_epoch = 0;

  if (state.has_local_peer) {
    // Create/open lightweight SHM for hier allreduce
    // Use a fixed name derived from shm_handle
    state.hier_shm_name = "/vllm_hier_ar_" + std::to_string(shm_handle);
    size_t shm_size = HIER_SHM_HEADER + 3 * HIER_SHM_MAX_BUF;

    int fd;
    if (is_leader) {
      // Leader creates
      shm_unlink(state.hier_shm_name.c_str());  // clean up any stale
      fd = shm_open(state.hier_shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
                    S_IRUSR | S_IWUSR);
      TORCH_CHECK(fd >= 0, "hier SHM create failed: ", strerror(errno));
      TORCH_CHECK(ftruncate(fd, shm_size) == 0, "hier SHM truncate failed");
    } else {
      // Follower opens (may need to wait for leader)
      for (int tries = 0; tries < 100; tries++) {
        fd = shm_open(state.hier_shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
        if (fd >= 0) break;
        usleep(10000);  // 10ms
      }
      TORCH_CHECK(fd >= 0, "hier SHM open failed: ", strerror(errno));
    }

    state.hier_shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    TORCH_CHECK(state.hier_shm_ptr != MAP_FAILED, "hier SHM mmap failed");

    // Initialize layout
    auto* base = reinterpret_cast<char*>(state.hier_shm_ptr);
    state.header = reinterpret_cast<HierShmHeader*>(base);
    state.slot0 = base + HIER_SHM_HEADER;
    state.slot1 = base + HIER_SHM_HEADER + HIER_SHM_MAX_BUF;
    state.bcast_buf = base + HIER_SHM_HEADER + 2 * HIER_SHM_MAX_BUF;

    if (is_leader) {
      state.header->reduce_seq.store(0, std::memory_order_relaxed);
      state.header->read_seq.store(0, std::memory_order_relaxed);
    }
  }

  g_hier_ar_states[id] = std::move(state);
  return id;
}

// 4-way direct IBV allreduce with lightweight SHM:
// Fused 4-way IBV + lightweight SHM allreduce.
// Writes SHM slots EARLY (before IBV), does IBV exchange, adds IBV result
// to SHM slot, then single atomic sync + NEON reduce.
// Only ONE synchronization point per call (vs TWO in sequential IBV → SHM).

// Helper: bf16 NEON add  dst[i] = a[i] + b[i]
static inline void bf16_neon_add(at::BFloat16* dst,
                                 const at::BFloat16* a,
                                 const at::BFloat16* b,
                                 int64_t n) {
#ifdef __aarch64__
  int64_t i = 0;
  int64_t n8 = (n / 8) * 8;
  for (; i < n8; i += 8) {
    uint16x8_t va = vld1q_u16(reinterpret_cast<const uint16_t*>(a + i));
    uint16x8_t vb = vld1q_u16(reinterpret_cast<const uint16_t*>(b + i));
    float32x4_t a_lo = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(va)), 16));
    float32x4_t a_hi = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_high_u16(va)), 16));
    float32x4_t b_lo = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(vb)), 16));
    float32x4_t b_hi = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_high_u16(vb)), 16));
    float32x4_t sum_lo = vaddq_f32(a_lo, b_lo);
    float32x4_t sum_hi = vaddq_f32(a_hi, b_hi);
    uint16x4_t r_lo = vmovn_u32(
        vshrq_n_u32(vreinterpretq_u32_f32(sum_lo), 16));
    uint16x4_t r_hi = vmovn_u32(
        vshrq_n_u32(vreinterpretq_u32_f32(sum_hi), 16));
    vst1q_u16(reinterpret_cast<uint16_t*>(dst + i),
              vcombine_u16(r_lo, r_hi));
  }
  for (; i < n; i++) {
    dst[i] = at::BFloat16(float(a[i]) + float(b[i]));
  }
#else
  for (int64_t i = 0; i < n; i++) {
    dst[i] = at::BFloat16(float(a[i]) + float(b[i]));
  }
#endif
}

// Generic tensor add helper: dst = a + b
static inline void tensor_add(void* dst, const void* a, const void* b,
                               torch::Tensor& ref_tensor) {
  if (ref_tensor.scalar_type() == at::kBFloat16) {
    bf16_neon_add(reinterpret_cast<at::BFloat16*>(dst),
                  reinterpret_cast<const at::BFloat16*>(a),
                  reinterpret_cast<const at::BFloat16*>(b),
                  ref_tensor.numel());
  } else {
    auto at = torch::from_blob(const_cast<void*>(a), ref_tensor.sizes(),
                                ref_tensor.strides(), ref_tensor.options());
    auto bt = torch::from_blob(const_cast<void*>(b), ref_tensor.sizes(),
                                ref_tensor.strides(), ref_tensor.options());
    auto dt = torch::from_blob(dst, ref_tensor.sizes(),
                                ref_tensor.strides(), ref_tensor.options());
    torch::add_out(dt, at, bt);
  }
}

// Leader-Mediated 4-way allreduce:
// LEADER: wait for FOLLOW data, sum locally, IBV with remote LEADER, broadcast result
// FOLLOW: deposit data to SHM, wait for LEADER's broadcast, read result
// This synchronizes IBV timing between nodes and eliminates cascading wait.
void hier_allreduce(int64_t handle, torch::Tensor& data) {
  auto& state = g_hier_ar_states[handle];
  TORCH_CHECK(data.is_contiguous(),
              "Hierarchical allreduce requires a contiguous tensor");
  size_t nbytes = data.numel() * data.element_size();

  // IBV-only path: no local peer (1 worker per node, e.g. TP=2 cross-node)
  if (!state.has_local_peer) {
    if (state.ibv_handle >= 0) {
      ibv_ar_allreduce(state.ibv_handle, data);
    }
    return;
  }

  // Large tensor or no IBV: fallback
  if (nbytes > HIER_SHM_MAX_BUF || state.ibv_handle < 0) {
    if (state.ibv_handle >= 0) {
      ibv_ar_allreduce(state.ibv_handle, data);
    }
    shm_allreduce(state.shm_handle, data);
    return;
  }

  state.reduce_epoch++;
  uint64_t my_epoch = state.reduce_epoch;
  void* my_slot = (state.local_rank == 0) ? state.slot0 : state.slot1;
  auto& ibv = *g_ibv_states[state.ibv_handle];

  TORCH_CHECK(nbytes <= ibv.buf_size,
              "Hierarchical allreduce: tensor size ", nbytes,
              " exceeds IBV buffer ", ibv.buf_size);

  // Step 1: Copy data to SHM slot AND IBV send_buf
  memcpy(my_slot, data.data_ptr(), nbytes);
  memcpy(ibv.send_buf, data.data_ptr(), nbytes);

  // Step 2: Post IBV RDMA write
  struct ibv_sge sge = {};
  sge.addr = (uint64_t)ibv.send_buf;
  sge.length = (uint32_t)nbytes;
  sge.lkey = ibv.send_mr->lkey;

  struct ibv_send_wr wr = {};
  struct ibv_send_wr* bad = nullptr;
  wr.wr_id = 0;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl((uint32_t)nbytes);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = ibv.remote_addr;
  wr.wr.rdma.rkey = ibv.remote_rkey;
  int ret = ibv_post_send(ibv.qp, &wr, &bad);
  TORCH_CHECK(ret == 0, "ibv_post_send failed");

  // Step 3: Poll IBV CQ - batch poll for both completions
  int remaining = 2;
  struct ibv_wc wcs[2];
  while (remaining > 0) {
    int nc = ibv_poll_cq(ibv.cq, remaining, wcs);
    if (nc > 0) {
      for (int ci = 0; ci < nc; ci++) {
        TORCH_CHECK(wcs[ci].status == IBV_WC_SUCCESS,
                    "IBV WC error: status=", wcs[ci].status,
                    " (", ibv_wc_status_str(wcs[ci].status), ")");
      }
      remaining -= nc;
    }
  }

  // Step 4: Add IBV recv data to my SHM slot (in-place)
  // my_slot now = my_data + partner_data
  if (data.scalar_type() == at::kBFloat16) {
    auto* slot = reinterpret_cast<at::BFloat16*>(my_slot);
    auto* recv = reinterpret_cast<const at::BFloat16*>(ibv.recv_buf);
    bf16_neon_add(slot, slot, recv, data.numel());
  } else {
    auto slot_t = torch::from_blob(my_slot, data.sizes(), data.strides(),
                                   data.options());
    auto recv_t = torch::from_blob(ibv.recv_buf, data.sizes(), data.strides(),
                                   data.options());
    slot_t.add_(recv_t);
  }

  // Post recv for next IBV call
  struct ibv_recv_wr rwr = {};
  struct ibv_recv_wr* bad_r = nullptr;
  rwr.wr_id = 1;
  rwr.num_sge = 0;
  rwr.sg_list = nullptr;
  ret = ibv_post_recv(ibv.qp, &rwr, &bad_r);
  TORCH_CHECK(ret == 0, "ibv_post_recv failed, ret=", ret);

  // Step 5: Signal SHM slot is complete
  std::atomic_thread_fence(std::memory_order_release);
  state.header->reduce_seq.fetch_add(1, std::memory_order_release);

  // Wait for peer's signal
  uint64_t target = my_epoch * 2;
  while (state.header->reduce_seq.load(std::memory_order_acquire) < target) {
#ifdef __aarch64__
    __asm__ __volatile__("yield");
#else
    _mm_pause();
#endif
  }

  // Step 6: Reduce both slots into data
  // data = slot0 + slot1 = (d0+d2) + (d1+d3) = all 4 ranks
  if (data.scalar_type() == at::kBFloat16) {
    bf16_neon_add(reinterpret_cast<at::BFloat16*>(data.data_ptr()),
                  reinterpret_cast<const at::BFloat16*>(state.slot0),
                  reinterpret_cast<const at::BFloat16*>(state.slot1),
                  data.numel());
  } else {
    auto t0 = torch::from_blob(state.slot0, data.sizes(), data.strides(),
                                data.options());
    auto t1 = torch::from_blob(state.slot1, data.sizes(), data.strides(),
                                data.options());
    data.copy_(t0.add(t1));
  }

  // Step 7: Post-read barrier. Both ranks have now read slot0 and slot1, but
  // neither may reuse its slot (step 1 of the next epoch) until the peer has
  // also finished reading. Without this, a faster rank could overwrite its slot
  // while the peer is still in step 6, corrupting shared memory.
  state.read_epoch++;
  std::atomic_thread_fence(std::memory_order_release);
  state.header->read_seq.fetch_add(1, std::memory_order_release);
  uint64_t read_target = state.read_epoch * 2;
  while (state.header->read_seq.load(std::memory_order_acquire) < read_target) {
#ifdef __aarch64__
    __asm__ __volatile__("yield");
#else
    _mm_pause();
#endif
  }
}

#endif  // VLLM_CPU_RDMA_HAR

