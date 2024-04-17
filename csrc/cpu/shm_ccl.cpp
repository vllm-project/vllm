#include "cpu/cpu_types.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace {
#define MAX_SHM_RANK_NUM 8

template <typename scalar_t> struct KernelVecType {
  using scalar_vec_t = void;
};

template <> struct KernelVecType<float> {
  using scalar_vec_t = vec_op::FP32Vec16;
};

template <> struct KernelVecType<c10::BFloat16> {
  using scalar_vec_t = vec_op::BF16Vec16;
};

enum class RankStat : char { READY = 0, EXECUTE, DONE };

struct SHMContext {
  volatile RankStat rank_stat;
  char _padding1[60];
  int rank;
  int group_size;
  size_t rank_buffer_size;
  SHMContext *shm_contexts[MAX_SHM_RANK_NUM];
  char _padding2[48];

  SHMContext(const int rank, const int group_size,
             const size_t rank_buffer_size)
      : rank(rank), group_size(group_size), rank_buffer_size(rank_buffer_size) {
    static_assert(sizeof(SHMContext) % 64 == 0);
    TORCH_CHECK(group_size <= MAX_SHM_RANK_NUM);
    TORCH_CHECK(rank < MAX_SHM_RANK_NUM);
    TORCH_CHECK((size_t)this % 64 == 0);
    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      shm_contexts[i] = nullptr;
    }
    set_context(rank, this);
    rank_stat = RankStat::DONE;
  }

  void set_context(int rank, SHMContext *ptr) {
    TORCH_CHECK(rank < MAX_SHM_RANK_NUM);
    TORCH_CHECK(ptr);
    shm_contexts[rank] = ptr;
  }

  template <typename T> T *rank_ptr(int rank) {
    return reinterpret_cast<T *>(shm_contexts[rank] + 1);
  }

  RankStat get_rank_stat(int rank) const {
    return shm_contexts[rank]->rank_stat;
  }

  bool is_all_done() {
    for (int i = 0; i < group_size; ++i) {
      if (shm_contexts[i]->rank_stat != RankStat::DONE) {
        return false;
      }
    }
    return true;
  }

  bool is_last() const { return rank == (group_size - 1); }

  void set_rank_stat(int rank, RankStat stat) {
    shm_contexts[rank]->rank_stat = stat;
  }

  void barrier(const RankStat next_stat) {
    if (next_stat == RankStat::READY) {
      if (is_last()) {
        for (int i = 0; i < group_size; ++i) {
          set_rank_stat(i, RankStat::READY);
        }
      } else {
        while (get_rank_stat(rank) != RankStat::READY)
          _mm_pause();
      }
      set_rank_stat(rank, RankStat::EXECUTE);
    } else if (next_stat == RankStat::DONE) {
      set_rank_stat(rank, RankStat::DONE);
      if (is_last()) {
        while (!is_all_done())
          _mm_pause();
      }
    } else {
      TORCH_CHECK(false, "Invalid next_stat to barrier.");
    }
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "SHMContext: \nrank_stat: ";
    switch (rank_stat) {
    case RankStat::READY:
      ss << "READY, ";
      break;
    case RankStat::EXECUTE:
      ss << "EXECUTE, ";
      break;
    case RankStat::DONE:
      ss << "DONE, ";
      break;
    default:
      TORCH_CHECK(false, "Invalid RankStat type.");
    }
    ss << "\nrank: " << rank;
    ss << "\ngroup_size: " << group_size;
    ss << "\nrank_buffer_size: " << rank_buffer_size;
    ss << "\nshm_contexts: [";

    for (int i = 0; i < group_size; ++i) {
      ss << shm_contexts[i]->rank << ", ";
    }
    ss << "]";

    return ss.str();
  }
};

namespace shm_cc_ops {

void memcpy_64bytes(void *dst, void *src, size_t len) {
  constexpr size_t align_len = 64;
  TORCH_CHECK(len % align_len == 0);
  TORCH_CHECK((size_t)dst % align_len == 0);
  TORCH_CHECK((size_t)src % align_len == 0);
#pragma GCC unroll 4
  for (size_t i = 0; i < len; i += align_len) {
    vec_op::BF16Vec32 data((char *)src + i);
    vec_op::non_temporal_save(data, (char *)dst + i);
  }
}

void parallel_memcpy(void *dst, void *src, size_t len) {
  int thread_num = omp_get_max_threads();
  const size_t partition_num =
      (len + 512 * thread_num - 1) / (512 * thread_num);

#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    size_t offset = i * partition_num * 512;
    if (offset < len) {
      size_t partition_len = std::min(512 * partition_num, len - offset);
      memcpy_64bytes((char *)dst + offset, (char *)src + offset, partition_len);
    }
  }
}

void gather(SHMContext *ctx, int rank, void *data, size_t len) {
  CPU_KERNEL_GUARD_IN(gather)
  TORCH_CHECK(len <= ctx->rank_buffer_size);
  ctx->barrier(RankStat::READY);
  parallel_memcpy(ctx->rank_ptr<void>(rank), data, len);
  ctx->barrier(RankStat::DONE);
}

void broadcast(SHMContext *ctx, int rank, void *data, size_t len) {
  CPU_KERNEL_GUARD_IN(broatcast)
  ctx->barrier(RankStat::READY);
  parallel_memcpy(data, ctx->rank_ptr<void>(0), len);
  ctx->barrier(RankStat::DONE);
}

void scatter(SHMContext *ctx, int rank, void *data, size_t len) {
  CPU_KERNEL_GUARD_IN(scatter)
  ctx->barrier(RankStat::READY);
  parallel_memcpy(data, ctx->rank_ptr<void>(rank), len);
  ctx->barrier(RankStat::DONE);
}

template <typename scalar_t, int RANKS>
void all_reduce_sum_v1(SHMContext *ctx, int rank, scalar_t *data,
                       size_t elem_num) {
  CPU_KERNEL_GUARD_IN(all_reduce_sum_v1)
  const size_t bytes = elem_num * sizeof(scalar_t);
  TORCH_CHECK(bytes <= ctx->rank_buffer_size);
  shm_cc_ops::gather(ctx, rank, data, bytes);
  using scalar_vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  constexpr int CACHELINE_SIZE = 64;
  constexpr int PACKED_FACTOR =
      CACHELINE_SIZE / (sizeof(scalar_t) * VEC_ELEM_NUM);
  TORCH_CHECK(elem_num % VEC_ELEM_NUM == 0);

  ctx->barrier(RankStat::READY);

  int thread_num = omp_get_max_threads();
  size_t partition_num =
      (elem_num + thread_num * VEC_ELEM_NUM * PACKED_FACTOR - 1) /
      (thread_num * VEC_ELEM_NUM * PACKED_FACTOR);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    size_t offset = i * partition_num * VEC_ELEM_NUM * PACKED_FACTOR;
    if (offset < elem_num) {
      const size_t partition_len = std::min(
          VEC_ELEM_NUM * PACKED_FACTOR * partition_num, elem_num - offset);
      scalar_t *rank_ptrs[RANKS];
      vec_op::unroll_loop<int, RANKS>([&](int idx) {
        rank_ptrs[idx] = ctx->rank_ptr<scalar_t>(idx) + offset;
        TORCH_CHECK((size_t)rank_ptrs[idx] % 64 == 0);
      });

#pragma GCC unroll 4
      for (int i = 0; i < partition_len; i += VEC_ELEM_NUM) {
        size_t curr_offset = i;
        scalar_vec_t data_0(rank_ptrs[0] + curr_offset);
        vec_op::FP32Vec16 fp32_data_0(data_0);
        vec_op::unroll_loop<int, RANKS - 1>([&](int k) {
          scalar_vec_t data_x(rank_ptrs[k + 1] + curr_offset);
          vec_op::FP32Vec16 fp32_data_x(data_x);
          fp32_data_0 = fp32_data_0 + fp32_data_x;
        });
        data_0 = scalar_vec_t(fp32_data_0);
        data_0.save(data + offset + curr_offset);
      }
    }
  }
  ctx->barrier(RankStat::DONE);
}

template <typename scalar_t, int RANKS>
void all_reduce_sum_v2(SHMContext *ctx, int rank, scalar_t *data,
                       size_t elem_num) {
  CPU_KERNEL_GUARD_IN(all_reduce_sum_v2)
  const size_t bytes = elem_num * sizeof(scalar_t);
  TORCH_CHECK(bytes <= ctx->rank_buffer_size);
  shm_cc_ops::gather(ctx, rank, data, bytes);
  using scalar_vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  constexpr int CACHELINE_SIZE = 64;
  constexpr int PACKED_FACTOR =
      CACHELINE_SIZE / (sizeof(scalar_t) * VEC_ELEM_NUM);
  TORCH_CHECK(elem_num % VEC_ELEM_NUM == 0);

  ctx->barrier(RankStat::READY);

  const int world_size = ctx->group_size;
  const size_t rank_partition_num =
      (elem_num + world_size * VEC_ELEM_NUM * PACKED_FACTOR - 1) /
      (world_size * VEC_ELEM_NUM * PACKED_FACTOR);
  const size_t rank_offset =
      rank * rank_partition_num * VEC_ELEM_NUM * PACKED_FACTOR;

  if (rank_offset >= elem_num) {
    ctx->barrier(RankStat::DONE);
    return;
  }

  const size_t rank_elem_num =
      std::min(VEC_ELEM_NUM * PACKED_FACTOR * rank_partition_num,
               elem_num - rank_offset);

  int thread_num = omp_get_max_threads();
  size_t partition_num =
      (rank_elem_num + thread_num * VEC_ELEM_NUM * PACKED_FACTOR - 1) /
      (thread_num * VEC_ELEM_NUM * PACKED_FACTOR);

#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    size_t offset = i * partition_num * VEC_ELEM_NUM * PACKED_FACTOR;
    if (offset < rank_elem_num) {
      const size_t partition_len = std::min(
          VEC_ELEM_NUM * PACKED_FACTOR * partition_num, rank_elem_num - offset);
      scalar_t *rank_ptrs[RANKS];
      vec_op::unroll_loop<int, RANKS>([&](int idx) {
        rank_ptrs[idx] = ctx->rank_ptr<scalar_t>(idx) + rank_offset + offset;
        TORCH_CHECK((size_t)rank_ptrs[idx] % 64 == 0);
      });

#pragma GCC unroll 4
      for (int i = 0; i < partition_len; i += VEC_ELEM_NUM) {
        size_t curr_offset = i;
        scalar_vec_t data_0(rank_ptrs[0] + curr_offset);
        vec_op::FP32Vec16 fp32_data_0(data_0);
        vec_op::unroll_loop<int, RANKS - 1>([&](int k) {
          scalar_vec_t data_x(rank_ptrs[k + 1] + curr_offset);
          vec_op::FP32Vec16 fp32_data_x(data_x);
          fp32_data_0 = fp32_data_0 + fp32_data_x;
        });
        data_0 = scalar_vec_t(fp32_data_0);
        vec_op::unroll_loop<int, RANKS>([&](int k) {
          vec_op::non_temporal_save(data_0, rank_ptrs[k] + curr_offset);
        });
      }
    }
  }
  ctx->barrier(RankStat::DONE);

  shm_cc_ops::scatter(ctx, rank, data, bytes);
}
}; // namespace shm_cc_ops

class SHMManager {
public:
  explicit SHMManager(const std::string &ip_port, const int group_size,
                      const int rank, const size_t rank_buffer_size)
      : _rank(rank), _shm_names({""}), _shared_mem_ptrs({nullptr}),
        _shm_ctx(nullptr) {
    _shm_names[rank] = get_shm_name(ip_port, rank);
    _shared_mem_ptrs[rank] = init_shm(rank, rank_buffer_size);

    _shm_ctx = new (_shared_mem_ptrs[rank])
        SHMContext(rank, group_size, round_size(rank_buffer_size));
  }

  void join(const std::string &ip_port, const int group_size, const int rank,
            const size_t rank_buffer_size) {
    TORCH_CHECK(rank == _rank);
    SHMContext *ctx = get_shm_ctx();
    for (int i = 0; i < group_size; ++i) {
      if (i != rank) {
        TORCH_CHECK(_shm_names[i].empty());
        TORCH_CHECK(_shared_mem_ptrs[i] == nullptr);

        _shm_names[i] = get_shm_name(ip_port, i);
        _shared_mem_ptrs[i] = init_shm(i, rank_buffer_size);
        ctx->set_context(i, (SHMContext *)_shared_mem_ptrs[i]);
      }
    }
  }

  ~SHMManager() { destroy_shm(); }

  SHMContext *get_shm_ctx() const {
    return reinterpret_cast<SHMContext *>(_shared_mem_ptrs[_rank]);
  }

  static std::string get_shm_name(const std::string &ip_port, int rank) {
    return "/vllm_" + ip_port + "_" + std::to_string(rank);
  }

private:
  static size_t round_size(const size_t size) {
    return ((size + 63) >> 6) << 6;
  }

  void *init_shm(int target_rank, const size_t rank_buffer_size) {
    const std::string &shm_name = _shm_names[target_rank];
    const int local_rank = _rank;
    const size_t rounded_rank_buffer_size = round_size(rank_buffer_size);
    const size_t shm_size = sizeof(SHMContext) + rounded_rank_buffer_size;

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

    void *shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, 0);

    if (shm_ptr == MAP_FAILED) {
      TORCH_CHECK(false,
                  "mmap in SHMManager failed. errno: " + std::to_string(errno));
    }

    TORCH_CHECK((size_t)shm_ptr % 64 == 0)

    return shm_ptr;
  }

  void destroy_shm() {
    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      if (!_shm_names[i].empty() && _shared_mem_ptrs[i] != nullptr) {
        shm_unlink(_shm_names[i].c_str());
      }
    }
  }

  int _rank;
  std::array<std::string, MAX_SHM_RANK_NUM> _shm_names;
  std::array<void *, MAX_SHM_RANK_NUM> _shared_mem_ptrs;
  SHMContext *_shm_ctx;
};

static std::unique_ptr<SHMManager> shm_manager_singleton = nullptr;

template <typename scalar_t>
void shm_allreduce_sum(SHMContext *ctx, const int rank, scalar_t *data,
                       size_t elem_num) {
  switch (ctx->group_size) {
  case 2:
    shm_cc_ops::all_reduce_sum_v1<scalar_t, 2>(ctx, rank, data, elem_num);
    break;
  case 4:
    shm_cc_ops::all_reduce_sum_v2<scalar_t, 4>(ctx, rank, data, elem_num);
    break;
  case 8:
    shm_cc_ops::all_reduce_sum_v2<scalar_t, 8>(ctx, rank, data, elem_num);
    break;
  default:
    TORCH_CHECK(false,
                "Invalid world size: " + std::to_string(ctx->group_size));
  }
}

} // namespace

void shm_allreduce(torch::Tensor &data, int rank) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_allreduce_sum", [&] {
    CPU_KERNEL_GUARD_IN(shm_allreduce_sum)
    shm_allreduce_sum(shm_manager_singleton->get_shm_ctx(), rank,
                      data.data_ptr<scalar_t>(), data.numel());
    CPU_KERNEL_GUARD_OUT(shm_allreduce_sum)
  });
}

void init_shm_manager(const std::string &ip_port, const int group_size,
                      const int rank, const size_t rank_buffer_size) {
  if (shm_manager_singleton == nullptr) {
    shm_manager_singleton = std::make_unique<SHMManager>(
        ip_port, group_size, rank, rank_buffer_size);
  } else {
    TORCH_CHECK(
        false,
        "Duplicate initialization of shm_manager_singleton is not allowed.")
  }
}

std::string join_shm_manager(const std::string &ip_port, const int group_size,
                             const int rank, const size_t rank_buffer_size) {
  TORCH_CHECK(shm_manager_singleton);
  shm_manager_singleton->join(ip_port, group_size, rank, rank_buffer_size);
  return shm_manager_singleton->get_shm_ctx()->to_string();
}