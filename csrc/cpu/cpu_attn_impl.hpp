#ifndef CPU_ATTN_HPP
#define CPU_ATTN_HPP

#include <type_traits>
#include <cstddef>
#include <numeric>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#include "cpu/cpu_arch_macros.h"
#include "cpu/utils.hpp"

namespace cpu_attention {
enum class ISA { AMX, VEC, VEC16, NEON, VXE, RVV, VSX, AMX_FP8 };

// Mirrors csrc/attention/dtype_fp8.cuh Fp8KVCacheDataType exactly.
enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

struct AttentionInput;

template <ISA isa, typename scalar_t, int64_t head_dim,
          typename kv_cache_scalar_t = scalar_t>
class AttentionImpl {
 public:
  void init_from_input(const AttentionInput*) {}
  float get_output_v_scale() const noexcept { return 1.0f; }
};

struct AttentionWorkItemGroup {
  int32_t req_id;
  int32_t kv_head_idx;
  int32_t q_head_start;
  int32_t q_head_num;
  int32_t q_token_id_start;
  int32_t q_token_num;
  int32_t kv_split_pos_start;
  int32_t kv_split_pos_end;

  int32_t reduction_id;
  int32_t split_id;
  int32_t local_split_id;
  int32_t _padding;  // Keep work-item records aligned to 16-byte boundaries.

  AttentionWorkItemGroup(const int32_t req_id, const int32_t q_token_id_start,
                         const int32_t kv_split_pos_start,
                         const int32_t kv_split_pos_end,
                         const int32_t q_head_num)
      : req_id(req_id),
        kv_head_idx(-1),
        q_head_start(0),
        q_head_num(q_head_num),
        q_token_id_start(q_token_id_start),
        q_token_num(0),
        kv_split_pos_start(kv_split_pos_start),
        kv_split_pos_end(kv_split_pos_end),
        reduction_id(-1),
        split_id(-1),
        local_split_id(0),
        _padding(0) {}

  std::string to_string() const {
    std::stringstream ss;
    ss << '[' << "req_id: " << req_id << ",\n";
    ss << "kv_head_idx: " << kv_head_idx << ",\n";
    ss << "q_head_start: " << q_head_start << ",\n";
    ss << "q_head_num: " << q_head_num << ",\n";
    ss << "q_token_id_start: " << q_token_id_start << ",\n";
    ss << "q_token_num: " << q_token_num << ",\n";
    ss << "kv_split_pos_start: " << kv_split_pos_start << ",\n";
    ss << "kv_split_pos_end: " << kv_split_pos_end << ",\n";
    ss << "reduction_id: " << reduction_id << ",\n";
    ss << "split_id: " << split_id << ",\n";
    ss << "local_split_id: " << local_split_id << ",\n";
    ss << ']';

    return ss.str();
  }
};

static_assert(sizeof(AttentionWorkItemGroup) == 48);
static_assert(offsetof(AttentionWorkItemGroup, req_id) == 0);
static_assert(offsetof(AttentionWorkItemGroup, q_head_num) == 12);

struct ReductionWorkItemGroup {
  int32_t req_id;
  int32_t q_head_start;
  int32_t q_head_num;
  int32_t q_token_id_start;
  int32_t q_token_id_num;
  int32_t split_num;
  int64_t scratch_offset;

  ReductionWorkItemGroup(const int32_t req_id, const int32_t q_token_id_start,
                         const int32_t q_token_id_num, const int32_t q_head_num)
      : req_id(req_id),
        q_head_start(0),
        q_head_num(q_head_num),
        q_token_id_start(q_token_id_start),
        q_token_id_num(q_token_id_num),
        split_num(0),
        scratch_offset(0) {}

  std::string to_string() const {
    std::stringstream ss;
    ss << '[' << "req_id: " << req_id << ",\n";
    ss << "q_head_start: " << q_head_start << ",\n";
    ss << "q_head_num: " << q_head_num << ",\n";
    ss << "q_token_id_start: " << q_token_id_start << ",\n";
    ss << "q_token_id_num: " << q_token_id_num << ",\n";
    ss << "split_num: " << split_num << ",\n";
    ss << "scratch_offset: " << scratch_offset << ",\n";
    ss << ']';

    return ss.str();
  }
};

struct AttentionTaskSpan {
  int32_t start;
  int32_t count;
};

struct AttentionMetadata {
  std::atomic_int64_t counter;
  char _padding1[56];
  ISA isa;
  int32_t workitem_group_num;
  int32_t reduction_item_num;
  int32_t reduction_split_num;
  int32_t thread_num;
  int64_t attention_scratchpad_size_per_thread;
  int64_t reduction_scratchpad_size;
  AttentionWorkItemGroup* workitem_groups_ptr;
  ReductionWorkItemGroup* reduction_items_ptr;
  AttentionTaskSpan* task_spans_ptr;
  int32_t task_span_num;
  char _padding_tail[60];

  AttentionMetadata(ISA isa, int32_t workitem_group_num,
                    int32_t reduction_item_num, int32_t reduction_split_num,
                    int32_t task_span_num)
      : counter(0),
        _padding1{},
        isa(isa),
        workitem_group_num(workitem_group_num),
        reduction_item_num(reduction_item_num),
        reduction_split_num(reduction_split_num),
        thread_num(cpu_utils::get_max_threads()),
        attention_scratchpad_size_per_thread(0),
        reduction_scratchpad_size(0),
        workitem_groups_ptr(
            (AttentionWorkItemGroup*)((char*)this + sizeof(AttentionMetadata))),
        reduction_items_ptr(
            (ReductionWorkItemGroup*)((char*)this + sizeof(AttentionMetadata) +
                                      workitem_group_num *
                                          sizeof(AttentionWorkItemGroup))),
        task_spans_ptr(
            (AttentionTaskSpan*)((char*)this + sizeof(AttentionMetadata) +
                                 workitem_group_num *
                                     sizeof(AttentionWorkItemGroup) +
                                 reduction_item_num *
                                     sizeof(ReductionWorkItemGroup))),
        task_span_num(task_span_num),
        _padding_tail{} {
    static_assert(sizeof(AttentionMetadata) % 64 == 0);
    TORCH_CHECK(reinterpret_cast<size_t>(this) % 64 == 0);
  }

  void reset_counter() { counter.store(0); }

  int64_t acquire_counter() { return counter++; }

  void print() const {
    std::stringstream ss;
    ss << "ISA: ";
    switch (isa) {
      case ISA::AMX:
        ss << "AMX, ";
        break;
      case ISA::AMX_FP8:
        ss << "AMX_FP8, ";
        break;
      case ISA::VEC:
        ss << "VEC, ";
        break;
      case ISA::VEC16:
        ss << "VEC16, ";
        break;
      case ISA::NEON:
        ss << "NEON, ";
        break;
      case ISA::VXE:
        ss << "VXE, ";
        break;
      case ISA::RVV:
        ss << "RVV, ";
        break;
      case ISA::VSX:
        ss << "VSX, ";
        break;
    }
    ss << "workitem_group_num: " << workitem_group_num
       << ", reduction_item_num: " << reduction_item_num
       << ", reduction_split_num: " << reduction_split_num
       << ", thread_num: " << thread_num
       << ", attention_scratchpad_size_per_thread: "
       << attention_scratchpad_size_per_thread
       << ", reduction_scratchpad_size: " << reduction_scratchpad_size
       << ", workitem groups:\n";
    for (int32_t i = 0; i < workitem_group_num; ++i) {
      ss << (workitem_groups_ptr + i)->to_string() << ",\n";
    }
    ss << "reduction items: \n";
    for (int32_t i = 0; i < reduction_item_num; ++i) {
      ss << (reduction_items_ptr + i)->to_string() << ",\n";
    }
    std::printf("%s", ss.str().c_str());
  }
};

static_assert(sizeof(AttentionMetadata) == 192);
static_assert(offsetof(AttentionMetadata, workitem_group_num) == 68);

// Thread attention scratchpad contains:
//  - Q: q_tile_size * head_dim * q_buffer_elem_size, gather Q heads, especially
//  for GQA
//  - Q@K^T: max_num_q_per_iter * k_tile_size * logits_buffer_elem_size, logits
//  - Intermediate outputs: q_tile_size * head_dim * output_buffer_elem_size + 2
//  * q_tile_size * 4, partial output, max + sum (float)
// Reduction scratchpad contains:
//  - flags: bool array to indicate whether the split is finished
//  - outputs: split_num * q_tile_size * head_dim * output_buffer_elem_size
//  - max, sum: 2 * split_num * q_tile_size * 4
class AttentionScratchPad {
 public:
  AttentionScratchPad(int64_t thread_id,
                      const AttentionMetadata& attention_metadata,
                      void* scratchpad_ptr)
      : thread_scratchpad_ptr(
            static_cast<int8_t*>(scratchpad_ptr) +
            thread_id *
                attention_metadata.attention_scratchpad_size_per_thread),
        reduction_scratchpad_ptr(
            static_cast<int8_t*>(scratchpad_ptr) +
            attention_metadata.thread_num *
                attention_metadata.attention_scratchpad_size_per_thread) {}

  // for attention
  void update(const int64_t head_dim, const int64_t q_buffer_elem_size,
              const int64_t logits_buffer_elem_size,
              const int64_t output_buffer_elem_size,
              const int64_t max_num_q_per_iter, const int64_t q_head_tile_size,
              const int64_t kv_tile_size) {
    int64_t buffer_offset = 0;
    q_buffer_offset_ = buffer_offset;
    buffer_offset +=
        calcu_q_buffer_size(q_head_tile_size, head_dim, q_buffer_elem_size);
    logits_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_logits_buffer_size(max_num_q_per_iter, kv_tile_size,
                                              logits_buffer_elem_size);
    output_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_partial_output_buffer_size(
        q_head_tile_size, head_dim, output_buffer_elem_size);
    max_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_partial_output_max_sum_buffer_size(q_head_tile_size);
    sum_buffer_offset_ = buffer_offset;
  }

  // Keep different splits' max/sum statistics on separate cache lines.
  static int64_t reduction_stats_stride(const int64_t head_num) {
    return ((head_num + 15) / 16) * 16;
  }

  // for reduction
  void update(const int64_t scratch_offset, const int32_t total_split_num,
              const int64_t head_dim, const int64_t q_head_tile_size,
              const int64_t output_buffer_elem_size) {
    int64_t buffer_offset = scratch_offset;
    reduce_flag_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_reduce_flag_buffer_size(total_split_num);
    reduce_output_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_reduce_output_buffer_size(
        total_split_num, q_head_tile_size, head_dim, output_buffer_elem_size);
    reduce_max_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_reduce_max_sum_buffer_size(
        total_split_num, reduction_stats_stride(q_head_tile_size));
    reduce_sum_buffer_offset_ = buffer_offset;
  }

  template <typename T>
  T* get_q_buffer() {
    return reinterpret_cast<T*>(thread_scratchpad_ptr + q_buffer_offset_);
  }

  float* get_logits_buffer() {
    return reinterpret_cast<float*>(thread_scratchpad_ptr +
                                    logits_buffer_offset_);
  }

  float* get_output_buffer() {
    return reinterpret_cast<float*>(thread_scratchpad_ptr +
                                    output_buffer_offset_);
  }

  float* get_max_buffer() {
    return reinterpret_cast<float*>(thread_scratchpad_ptr + max_buffer_offset_);
  }

  float* get_sum_buffer() {
    return reinterpret_cast<float*>(thread_scratchpad_ptr + sum_buffer_offset_);
  }

  volatile bool* get_reduce_flag_buffer() {
    return reinterpret_cast<volatile bool*>(reduction_scratchpad_ptr +
                                            reduce_flag_buffer_offset_);
  }

  float* get_reduce_output_buffer() {
    return reinterpret_cast<float*>(reduction_scratchpad_ptr +
                                    reduce_output_buffer_offset_);
  }

  float* get_reduce_max_buffer() {
    return reinterpret_cast<float*>(reduction_scratchpad_ptr +
                                    reduce_max_buffer_offset_);
  }

  float* get_reduce_sum_buffer() {
    return reinterpret_cast<float*>(reduction_scratchpad_ptr +
                                    reduce_sum_buffer_offset_);
  }

  int64_t get_thread_scratchpad_size() const {
    return 2 * sum_buffer_offset_ - max_buffer_offset_;
  }

  int64_t get_reduction_scratchpad_size() const {
    return 2 * reduce_sum_buffer_offset_ - reduce_max_buffer_offset_;
  }

 private:
  static int64_t round_to_64(const int64_t num) {
    return ((num + 63) >> 6) << 6;
  }

  static int64_t calcu_q_buffer_size(const int64_t q_tile_size,
                                     const int64_t head_dim,
                                     const int64_t elem_size) {
    return round_to_64(q_tile_size * head_dim * elem_size);
  }

  static int64_t calcu_logits_buffer_size(const int64_t max_num_q_per_iter,
                                          const int64_t k_tile_size,
                                          const int64_t elem_size) {
    return round_to_64(elem_size * max_num_q_per_iter * k_tile_size);
  }

  static int64_t calcu_partial_output_buffer_size(const int64_t q_tile_size,
                                                  const int64_t head_dim,
                                                  const int64_t elem_size) {
    return round_to_64(q_tile_size * head_dim * elem_size);
  }

  static int64_t calcu_partial_output_max_sum_buffer_size(
      const int64_t q_tile_size) {
    return round_to_64(q_tile_size * sizeof(float));
  }

  static int64_t calcu_reduce_flag_buffer_size(const int64_t total_split_num) {
    return round_to_64(total_split_num * sizeof(bool));
  }

  static int64_t calcu_reduce_max_sum_buffer_size(
      const int64_t total_split_num, const int32_t q_head_tile_size) {
    return round_to_64(total_split_num * q_head_tile_size * sizeof(float));
  }

  static int64_t calcu_reduce_output_buffer_size(
      const int64_t total_split_num, const int64_t q_head_tile_size,
      const int64_t head_dim, const int64_t output_buffer_elem_size) {
    return round_to_64(total_split_num * q_head_tile_size * head_dim *
                       output_buffer_elem_size);
  }

 private:
  int8_t* thread_scratchpad_ptr;
  int8_t* reduction_scratchpad_ptr;
  // attention buffers
  int64_t q_buffer_offset_;
  int64_t logits_buffer_offset_;
  int64_t output_buffer_offset_;
  int64_t max_buffer_offset_;
  int64_t sum_buffer_offset_;
  // reduction buffers
  int64_t reduce_flag_buffer_offset_;
  int64_t reduce_output_buffer_offset_;
  int64_t reduce_max_buffer_offset_;
  int64_t reduce_sum_buffer_offset_;
};

class AttentionScheduler {
 public:
  struct ScheduleInput {
    int32_t num_reqs;
    int32_t elem_size;
    int32_t q_buffer_elem_size;
    int32_t logits_buffer_elem_size;
    int32_t output_buffer_elem_size;
    int32_t num_heads_q;
    int32_t num_heads_kv;
    int32_t head_dim;
    int32_t* query_start_loc;
    int32_t* seq_lens;
    int32_t sliding_window_size;
    bool causal;
    cpu_attention::ISA isa;
    int32_t max_num_q_per_iter;  // max Q head num can be hold in registers
    int32_t kv_block_alignment;  // context length alignment requirement
    bool enable_kv_split;
    int32_t* dynamic_causal;
  };

  static constexpr int32_t MaxQTileIterNum = 128;

  torch::Tensor schedule(const ScheduleInput& input) const {
    const bool causal = input.causal;
    const bool is_dynamic_causal = input.dynamic_causal != nullptr;
    const int32_t thread_num = cpu_utils::get_max_threads();
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    const int32_t max_num_q_per_iter = input.max_num_q_per_iter;
    const int32_t kv_len_alignment = input.kv_block_alignment;
    const int32_t sliding_window_size = input.sliding_window_size;
    const auto request_q_token_num = [&](const int32_t req_id) {
      return input.query_start_loc[req_id + 1] - input.query_start_loc[req_id];
    };
    const int32_t original_q_head_per_kv =
        input.num_heads_q / input.num_heads_kv;
    const bool supports_gqa = original_q_head_per_kv <= max_num_q_per_iter;
    const bool is_amx = input.isa == ISA::AMX || input.isa == ISA::AMX_FP8;
    const int32_t min_split_kv_len =
        ((max_num_q_per_iter * 8 + kv_len_alignment - 1) / kv_len_alignment) *
        kv_len_alignment;
    const int64_t default_tile_size = calcu_default_tile_size(
        cache_size, input.head_dim, input.elem_size, input.q_buffer_elem_size,
        input.logits_buffer_elem_size, input.output_buffer_elem_size,
        max_num_q_per_iter, max_num_q_per_iter);
    const auto request_causal = [&](const int32_t req_id) {
      return is_dynamic_causal ? input.dynamic_causal[req_id] != 0 : causal;
    };
    const auto group_for_request = [&](const int32_t req_id,
                                       const int32_t group) {
      const int32_t q_tokens = request_q_token_num(req_id);
      if (!supports_gqa) {
        return 1;
      }
      if (q_tokens == 1) {
        return original_q_head_per_kv;
      }
      // Keep long prefill on MHA. Divisible groups fill short-query iterations;
      // other groups pack only one iteration to offset unused query rows.
      if (is_amx && request_causal(req_id) && q_tokens <= max_num_q_per_iter &&
          (max_num_q_per_iter % original_q_head_per_kv == 0 ||
           q_tokens <= max_num_q_per_iter / original_q_head_per_kv)) {
        return group;
      }
      return 1;
    };

    struct PartitionItem {
      int32_t req_id;
      int32_t q_token_id_start;
      int32_t q_token_num = 0;
      int32_t kv_split_pos_start;
      int32_t kv_split_pos_end;
      int64_t total_kv_len = 0;
      int32_t split_id = -1;
      int32_t local_split_id = 0;
      int32_t group;

      PartitionItem(int32_t req, int32_t token, int32_t start, int32_t end,
                    int32_t q_group)
          : req_id(req),
            q_token_id_start(token),
            kv_split_pos_start(start),
            kv_split_pos_end(end),
            group(q_group) {}
    };
    struct PartitionReduction {
      int32_t req_id;
      int32_t token_start;
      int32_t token_num;
      int32_t split_start;
      int32_t group;
      int32_t split_num = 0;
    };
    struct Partitions {
      std::vector<PartitionItem> items;
      std::vector<PartitionReduction> reductions;
      std::vector<int32_t> offsets;
      int32_t total_splits = 0;
      int32_t head_stride = 1;
    };
    const auto partition = [&](const int32_t group) {
      Partitions result;
      result.head_stride = original_q_head_per_kv;

      // get total kv len
      int64_t total_kv_len = 0;
      size_t query_tile_num = 0;
      for (int32_t req_id = 0; req_id < input.num_reqs; ++req_id) {
        result.head_stride =
            std::gcd(result.head_stride, group_for_request(req_id, group));
        const int32_t max_num_q_token_per_iter =
            max_num_q_per_iter / group_for_request(req_id, group);
        const int32_t seq_len = input.seq_lens[req_id];
        const int32_t q_token_num = request_q_token_num(req_id);
        const bool req_causal = request_causal(req_id);
        const int32_t q_start_pos = seq_len - q_token_num;
        const int32_t kv_start_pos = 0;
        const int32_t kv_end_pos = seq_len;

        for (int32_t token_id = 0; token_id < q_token_num;
             token_id += max_num_q_token_per_iter) {
          ++query_tile_num;
          const int32_t q_tile_token_num =
              std::min(max_num_q_token_per_iter, q_token_num - token_id);
          const int32_t q_tile_pos_left = q_start_pos + token_id;
          const int32_t q_tile_pos_right = q_tile_pos_left + q_tile_token_num;
          const auto [kv_tile_pos_left, kv_tile_pos_right] = calcu_kv_tile_pos(
              kv_start_pos, kv_end_pos, q_tile_pos_left, q_tile_pos_right,
              sliding_window_size, req_causal);
          const auto [aligned_kv_tile_pos_left, aligned_kv_tile_pos_right] =
              align_kv_tile_pos(kv_tile_pos_left, kv_tile_pos_right,
                                kv_len_alignment);

          int32_t curr_kv_len =
              aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
          total_kv_len += curr_kv_len;
        }
      }
      const int64_t kv_len_per_thread =
          (((total_kv_len / thread_num) + kv_len_alignment - 1) /
           kv_len_alignment) *
          kv_len_alignment;
      auto& workitems = result.items;
      auto& reduce_workitems = result.reductions;
      workitems.reserve(query_tile_num + thread_num);
      reduce_workitems.reserve(std::min<size_t>(query_tile_num, thread_num));
      std::vector<int32_t> workitem_num_per_thread(thread_num, 0);

      // split tasks
      int32_t curr_thread_id = 0;
      int64_t remaining_kv_len = kv_len_per_thread;
      int32_t cum_split_num = 0;
      for (int32_t req_id = 0; req_id < input.num_reqs; ++req_id) {
        const int32_t request_group = group_for_request(req_id, group);
        const int32_t max_num_q_token_per_iter =
            max_num_q_per_iter / request_group;
        const int32_t default_tile_token_num =
            default_tile_size / request_group;
        const int32_t seq_len = input.seq_lens[req_id];
        const int32_t q_token_num = request_q_token_num(req_id);
        const bool req_causal = request_causal(req_id);
        const int32_t q_start_pos = seq_len - q_token_num;
        const int32_t kv_start_pos = 0;
        const int32_t kv_end_pos = seq_len;
        // Short verification queries can span several packed iterations.
        // Split each iteration separately so their reduction fits one tile.
        const bool packed_query = request_group > 1 && q_token_num > 1;

        PartitionItem curr_workitem(req_id, 0, 0, seq_len, request_group);
        for (int32_t token_id = 0; token_id < q_token_num;
             token_id += max_num_q_token_per_iter) {
          const int32_t q_tile_token_num =
              std::min(max_num_q_token_per_iter, q_token_num - token_id);
          const int32_t q_tile_pos_left = q_start_pos + token_id;
          const int32_t q_tile_pos_right = q_tile_pos_left + q_tile_token_num;
          const auto [kv_tile_pos_left, kv_tile_pos_right] = calcu_kv_tile_pos(
              kv_start_pos, kv_end_pos, q_tile_pos_left, q_tile_pos_right,
              sliding_window_size, req_causal);
          const auto [aligned_kv_tile_pos_left, aligned_kv_tile_pos_right] =
              align_kv_tile_pos(kv_tile_pos_left, kv_tile_pos_right,
                                kv_len_alignment);
          // A packed query already supplies several rows of work per KV load.
          // Discount its partition credit to avoid creating tiny KV splits.
          const int32_t scale = packed_query ? q_tile_token_num : 1;
          const auto charge = [&](const int64_t len) {
            return ((len + scale * kv_len_alignment - 1) /
                    (scale * kv_len_alignment)) *
                   kv_len_alignment;
          };
          const int64_t min_credit = charge(min_split_kv_len);
          int32_t curr_kv_len =
              aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
          int32_t kv_token_pos_start = aligned_kv_tile_pos_left;
          int32_t local_split_id = 0;

          while (curr_kv_len > 0) {
            if (charge(curr_kv_len) <= (remaining_kv_len + min_credit) ||
                curr_thread_id == (thread_num - 1)) {
              curr_workitem.q_token_num += q_tile_token_num;
              curr_workitem.total_kv_len += curr_kv_len;
              remaining_kv_len -= charge(curr_kv_len);
              curr_kv_len = 0;

              if (remaining_kv_len < 0) {
                // stop to accept more workitems
                remaining_kv_len -= min_credit;
              }

              if (curr_workitem.kv_split_pos_start != 0) {
                // got a partial kv spilt, need to create a single workitem
                curr_workitem.split_id = cum_split_num;
                curr_workitem.local_split_id = local_split_id;
                workitems.emplace_back(curr_workitem);
                ++workitem_num_per_thread[curr_thread_id];
                ++reduce_workitems.back().split_num;
                ++cum_split_num;

                curr_workitem =
                    PartitionItem(req_id, token_id + max_num_q_token_per_iter,
                                  0, seq_len, request_group);
              }

              break;
            }

            if (remaining_kv_len < min_credit &&
                (curr_workitem.total_kv_len > 0 ||
                 workitem_num_per_thread[curr_thread_id] > 0)) {
              // remaining_kv_len is too short, and have allocated workitems,
              // just leave to next thread
              if (curr_workitem.total_kv_len > 0) {
                workitems.emplace_back(curr_workitem);
                ++workitem_num_per_thread[curr_thread_id];
                curr_workitem =
                    PartitionItem(req_id, token_id, 0, seq_len, request_group);
              }

              // switch to next thread
              ++curr_thread_id;
              remaining_kv_len = kv_len_per_thread;

              // retry this iteration
              continue;
            }

            // Split a whole packed query or a one-token tail; coalesce the
            // other prefill tiles to amortize scheduling and query copies.
            if (!input.enable_kv_split ||
                (!packed_query &&
                 (token_id + max_num_q_token_per_iter < q_token_num ||
                  q_tile_token_num > 1))) {
              // if requires a new q tile iteration and already has workitems,
              // leave this workitem to next thread
              if (curr_workitem.q_token_num % default_tile_token_num == 0 &&
                  (curr_workitem.total_kv_len > 0 ||
                   workitem_num_per_thread[curr_thread_id] > 0)) {
                if (curr_workitem.total_kv_len > 0) {
                  workitems.emplace_back(curr_workitem);
                  ++workitem_num_per_thread[curr_thread_id];
                }
                curr_workitem =
                    PartitionItem(req_id, token_id, 0, seq_len, request_group);

                // switch to next thread
                ++curr_thread_id;
                remaining_kv_len = kv_len_per_thread;
              }

              curr_workitem.q_token_num += q_tile_token_num;
              curr_workitem.total_kv_len += curr_kv_len;
              remaining_kv_len -= charge(curr_kv_len);
              curr_kv_len = 0;
              break;
            }

            // split kv
            if (curr_workitem.total_kv_len > 0) {
              // write back curr workitem
              workitems.emplace_back(curr_workitem);
              ++workitem_num_per_thread[curr_thread_id];
            }

            if (kv_token_pos_start == aligned_kv_tile_pos_left) {
              // first split, init the workitem
              reduce_workitems.push_back({req_id, token_id, q_tile_token_num,
                                          cum_split_num, request_group});
            }

            int32_t spilt_size = std::min<int64_t>(
                std::max<int64_t>(remaining_kv_len * scale, min_split_kv_len),
                curr_kv_len);
            curr_workitem =
                PartitionItem(req_id, token_id, kv_token_pos_start,
                              kv_token_pos_start + spilt_size, request_group);
            curr_workitem.q_token_num += q_tile_token_num;
            curr_workitem.total_kv_len += spilt_size;
            curr_workitem.split_id = cum_split_num;
            curr_workitem.local_split_id = local_split_id;
            workitems.emplace_back(curr_workitem);
            ++workitem_num_per_thread[curr_thread_id];
            ++reduce_workitems.back().split_num;
            ++cum_split_num;
            ++local_split_id;

            kv_token_pos_start += spilt_size;
            curr_kv_len -= spilt_size;
            curr_workitem = PartitionItem(req_id, token_id, kv_token_pos_start,
                                          seq_len, request_group);

            // switch to next thread
            ++curr_thread_id;
            remaining_kv_len = kv_len_per_thread;
          }

          if (packed_query && curr_workitem.total_kv_len > 0) {
            // Keep packed queries independently claimable by worker threads.
            workitems.emplace_back(curr_workitem);
            ++workitem_num_per_thread[curr_thread_id];
            curr_workitem =
                PartitionItem(req_id, token_id + max_num_q_token_per_iter, 0,
                              seq_len, request_group);
          }
        }

        if (curr_workitem.total_kv_len > 0) {
          // write back curr workitem
          workitems.emplace_back(curr_workitem);
          ++workitem_num_per_thread[curr_thread_id];
        }
      }
      result.total_splits = cum_split_num;
      result.offsets.resize(thread_num + 1, 0);
      std::partial_sum(workitem_num_per_thread.begin(),
                       workitem_num_per_thread.end(),
                       result.offsets.begin() + 1);
      return result;
    };
    const auto for_each_span = [&](const Partitions& partitions,
                                   const int32_t head, auto&& emit) {
      for (int32_t thread = 0; thread < thread_num; ++thread) {
        int32_t start = -1;
        const int32_t end = partitions.offsets[thread + 1];
        for (int32_t idx = partitions.offsets[thread]; idx < end; ++idx) {
          const auto& item = partitions.items[idx];
          if (head % item.group != 0) {
            continue;
          }
          if (item.group > 1 && request_q_token_num(item.req_id) > 1) {
            if (start >= 0) {
              emit(start, idx);
              start = -1;
            }
            emit(idx, idx + 1);
          } else if (start < 0) {
            start = idx;
          }
        }
        if (start >= 0) {
          emit(start, end);
        }
      }
    };
    const auto span_count = [&](const Partitions& partitions) {
      int64_t count = 0;
      for (int32_t head = 0; head < input.num_heads_q;
           head += partitions.head_stride) {
        for_each_span(partitions, head,
                      [&](const int32_t, const int32_t) { ++count; });
      }
      return count;
    };
    // Keep the largest group that fills the threads; otherwise prefer more
    // spans. MHA must improve the span count; ties retain GQA.
    Partitions partitions = partition(original_q_head_per_kv);
    int64_t best_spans = span_count(partitions);
    if (input.enable_kv_split && best_spans < thread_num) {
      for (int32_t req_id = 0; req_id < input.num_reqs; ++req_id) {
        if (request_q_token_num(req_id) <= 1 ||
            group_for_request(req_id, original_q_head_per_kv) == 1) {
          continue;
        }
        for (int32_t group = original_q_head_per_kv - 1; group >= 2; --group) {
          if (original_q_head_per_kv % group != 0) {
            continue;
          }
          auto candidate = partition(group);
          const int64_t spans = span_count(candidate);
          if (spans > best_spans) {
            partitions = std::move(candidate);
            best_spans = spans;
          }
          if (best_spans >= thread_num) {
            break;
          }
        }
        if (best_spans < thread_num) {
          auto mha = partition(1);
          if (span_count(mha) > best_spans) {
            partitions = std::move(mha);
          }
        }
        break;
      }
    }
    std::vector<AttentionWorkItemGroup> workitems;
    std::vector<ReductionWorkItemGroup> reduce_workitems;
    std::vector<AttentionTaskSpan> task_spans;
    size_t workitem_num = 0;
    size_t reduction_item_num = 0;
    for (const auto& item : partitions.items) {
      workitem_num += input.num_heads_q / item.group;
    }
    for (const auto& item : partitions.reductions) {
      reduction_item_num += input.num_heads_q / item.group;
    }
    workitems.reserve(workitem_num);
    reduce_workitems.reserve(reduction_item_num);
    task_spans.reserve(std::min(
        workitem_num, partitions.items.size() + size_t(thread_num) *
                                                    input.num_heads_q /
                                                    partitions.head_stride));
    std::vector<int32_t> reduction_for_split(partitions.total_splits);
    for (size_t idx = 0; idx < partitions.reductions.size(); ++idx) {
      const auto& reduction = partitions.reductions[idx];
      std::fill_n(reduction_for_split.begin() + reduction.split_start,
                  reduction.split_num, idx);
    }
    std::vector<int32_t> reduction_ids(partitions.reductions.size());
    for (int32_t head = 0; head < input.num_heads_q;
         head += partitions.head_stride) {
      for (size_t idx = 0; idx < partitions.reductions.size(); ++idx) {
        const auto& source = partitions.reductions[idx];
        reduction_ids[idx] = -1;
        if (head % source.group != 0) {
          continue;
        }
        reduction_ids[idx] = reduce_workitems.size();
        ReductionWorkItemGroup reduction(source.req_id, source.token_start,
                                         source.token_num, source.group);
        reduction.q_head_start = head;
        reduction.split_num = source.split_num;
        reduce_workitems.emplace_back(reduction);
      }
      for_each_span(partitions, head,
                    [&](const int32_t first, const int32_t last) {
                      const int32_t start = workitems.size();
                      for (int32_t idx = first; idx < last; ++idx) {
                        const auto& source = partitions.items[idx];
                        if (head % source.group != 0) {
                          continue;
                        }
                        AttentionWorkItemGroup item(
                            source.req_id, source.q_token_id_start,
                            source.kv_split_pos_start, source.kv_split_pos_end,
                            source.group);
                        item.kv_head_idx = head / original_q_head_per_kv;
                        item.q_head_start = head;
                        item.q_token_num = source.q_token_num;
                        item.local_split_id = source.local_split_id;
                        if (source.split_id >= 0) {
                          const int32_t reduction_idx =
                              reduction_for_split[source.split_id];
                          item.reduction_id = reduction_ids[reduction_idx];
                          item.split_id =
                              source.split_id -
                              partitions.reductions[reduction_idx].split_start;
                        }
                        workitems.emplace_back(item);
                      }
                      const int32_t count = workitems.size() - start;
                      if (count > 0) {
                        task_spans.push_back({start, count});
                      }
                    });
    }
    TORCH_CHECK_LE(task_spans.size(), std::numeric_limits<int32_t>::max());

    int64_t metadata_tensor_size =
        sizeof(AttentionMetadata) +
        workitems.size() * sizeof(AttentionWorkItemGroup) +
        reduce_workitems.size() * sizeof(ReductionWorkItemGroup) +
        task_spans.size() * sizeof(AttentionTaskSpan);
    auto options =
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    torch::Tensor metadata_tensor =
        torch::empty({metadata_tensor_size}, options);
    const int32_t total_reduction_split_num = std::accumulate(
        reduce_workitems.begin(), reduce_workitems.end(), 0,
        [](const int32_t total, const ReductionWorkItemGroup& item) {
          return total + item.split_num;
        });
    AttentionMetadata* metadata_ptr = new (metadata_tensor.data_ptr())
        AttentionMetadata(input.isa, workitems.size(), reduce_workitems.size(),
                          total_reduction_split_num, task_spans.size());
    AttentionWorkItemGroup* workitem_groups_ptr =
        metadata_ptr->workitem_groups_ptr;
    ReductionWorkItemGroup* reduction_items_ptr =
        metadata_ptr->reduction_items_ptr;
    std::memcpy(workitem_groups_ptr, workitems.data(),
                workitems.size() * sizeof(AttentionWorkItemGroup));
    std::memcpy(reduction_items_ptr, reduce_workitems.data(),
                reduce_workitems.size() * sizeof(ReductionWorkItemGroup));
    if (!task_spans.empty()) {
      std::memcpy(metadata_ptr->task_spans_ptr, task_spans.data(),
                  task_spans.size() * sizeof(AttentionTaskSpan));
    }

    {
      AttentionScratchPad sc(0, *metadata_ptr, 0x0);
      int64_t max_attention_scratchpad_size = 0;

      for (const AttentionWorkItemGroup& item : workitems) {
        const int32_t curr_q_heads_per_kv = item.q_head_num;
        const int32_t curr_default_q_tile_token_num =
            default_tile_size / curr_q_heads_per_kv;

        for (int32_t q_token_offset = 0; q_token_offset < item.q_token_num;
             q_token_offset += curr_default_q_tile_token_num) {
          const int32_t actual_q_token_num = std::min(
              curr_default_q_tile_token_num, item.q_token_num - q_token_offset);
          const int32_t q_head_tile_size =
              actual_q_token_num * curr_q_heads_per_kv;
          const int32_t rounded_q_head_tile_size =
              ((q_head_tile_size + max_num_q_per_iter - 1) /
               max_num_q_per_iter) *
              max_num_q_per_iter;

          const int64_t n = AttentionScheduler::calcu_tile_size_with_constant_q(
              cache_size, input.head_dim, input.elem_size,
              input.q_buffer_elem_size, input.logits_buffer_elem_size,
              input.output_buffer_elem_size, max_num_q_per_iter,
              kv_len_alignment, rounded_q_head_tile_size,
              rounded_q_head_tile_size <= max_num_q_per_iter);

          sc.update(input.head_dim, input.q_buffer_elem_size,
                    input.logits_buffer_elem_size,
                    input.output_buffer_elem_size, max_num_q_per_iter,
                    rounded_q_head_tile_size, n);

          max_attention_scratchpad_size = std::max(
              max_attention_scratchpad_size, sc.get_thread_scratchpad_size());
        }
      }

      metadata_ptr->attention_scratchpad_size_per_thread =
          ((max_attention_scratchpad_size + 63) / 64) * 64;

      int64_t reduction_scratchpad_size = 0;
      for (int32_t item_idx = 0; item_idx < metadata_ptr->reduction_item_num;
           ++item_idx) {
        ReductionWorkItemGroup& item =
            metadata_ptr->reduction_items_ptr[item_idx];
        item.scratch_offset = reduction_scratchpad_size;
        sc.update(item.scratch_offset, item.split_num, input.head_dim,
                  item.q_token_id_num * item.q_head_num,
                  input.output_buffer_elem_size);
        reduction_scratchpad_size =
            ((sc.get_reduction_scratchpad_size() + 63) / 64) * 64;
      }
      metadata_ptr->reduction_scratchpad_size = reduction_scratchpad_size;
    }
    int64_t scratchpad_size =
        metadata_ptr->attention_scratchpad_size_per_thread *
            metadata_ptr->thread_num +
        metadata_ptr->reduction_scratchpad_size;
    cpu_utils::ScratchPadManager::get_scratchpad_manager()->realloc(
        scratchpad_size);

    // metadata_ptr->print();

    // test out of boundary access
    // {
    //     float* cache_ptr =
    //     cpu_utils::ScratchPadManager::getl_scratchpad_manager()->get_data<float>();
    //     for (int64_t i = 0; i < scratchpad_size / sizeof(float); ++i) {
    //         cache_ptr[i] = std::numeric_limits<float>::quiet_NaN();
    //     }
    // }

    return metadata_tensor;
  }

  FORCE_INLINE static std::pair<int32_t, int32_t> calcu_sliding_window_size(
      int32_t window_size, bool causal) {
    int32_t left_sliding_window_size, right_sliding_window_size;
    if (window_size != -1) {
      left_sliding_window_size = window_size - 1;
      if (causal) {
        right_sliding_window_size = 0;
      } else {
        right_sliding_window_size = window_size - 1;
      }
    } else {
      left_sliding_window_size = -1;
      if (causal) {
        right_sliding_window_size = 0;
      } else {
        right_sliding_window_size = -1;
      }
    }

    return {left_sliding_window_size, right_sliding_window_size};
  }

  FORCE_INLINE static std::pair<int32_t, int32_t> calcu_kv_tile_pos(
      int32_t kv_left_pos, int32_t kv_right_pos, int32_t q_left_pos,
      int32_t q_right_pos, int32_t window_size, bool causal) {
    auto [left_sliding_window_size, right_sliding_window_size] =
        calcu_sliding_window_size(window_size, causal);

    if (left_sliding_window_size != -1) {
      kv_left_pos =
          std::max(kv_left_pos, q_left_pos - left_sliding_window_size);
    }
    if (right_sliding_window_size != -1) {
      kv_right_pos =
          std::min(kv_right_pos, q_right_pos + right_sliding_window_size);
    }
    return {kv_left_pos, kv_right_pos};
  }

  FORCE_INLINE static std::pair<int32_t, int32_t> align_kv_tile_pos(
      int32_t kv_left_pos, int32_t kv_right_pos, int32_t align_factor) {
    kv_left_pos = (kv_left_pos / align_factor) * align_factor;
    kv_right_pos =
        ((kv_right_pos + align_factor - 1) / align_factor) * align_factor;
    return {kv_left_pos, kv_right_pos};
  }

  static int64_t calcu_default_tile_size(int64_t cache_size, int64_t head_dim,
                                         int64_t elem_size,
                                         int64_t q_buffer_elem_size,
                                         int64_t logits_buffer_elem_size,
                                         int64_t output_buffer_elem_size,
                                         int64_t max_num_q_per_iter,
                                         int64_t round_size) {
    // For CPU, different from CUDA, Q@K^T results should also be hold in cache,
    // using float32. Intermediate outputs should be float32 to be compatible
    // with AMX Then the cache includes:
    //  - Q: q_tile_size * head_dim * q_buffer_elem_size
    //  - K, V: 2 * k_tile_size * head_dim * elem_size
    //  - Q@K^T: max_num_q_per_iter * k_tile_size * logits_buffer_elem_size
    //  - Intermediate outputs: q_tile_size * head_dim * output_buffer_elem_size
    // By default, let tile_size = q_tile_size = k_tile_size. To record
    // is_first_iter states in a static array, require the default tile <= 128 *
    // max_num_q_per_iter

    int64_t tile_size =
        cache_size / (head_dim * (q_buffer_elem_size + 2 * elem_size +
                                  output_buffer_elem_size) +
                      max_num_q_per_iter * logits_buffer_elem_size);
    tile_size = std::min(tile_size, MaxQTileIterNum * max_num_q_per_iter);
    int64_t rounded_tile_size = (tile_size / round_size) * round_size;
    return std::max(rounded_tile_size, round_size);
  }

  static int64_t calcu_tile_size_with_constant_q(
      int64_t cache_size, int64_t head_dim, int64_t elem_size,
      int64_t q_buffer_elem_size, int64_t logits_buffer_elem_size,
      int64_t output_buffer_elem_size, int64_t max_num_q_per_iter,
      int64_t round_size, int64_t q_tile_size, bool one_round) {
    // calculate tile_size with known q_tile_size
    // If one_round is True, the outer Q tile loop time is 1, then the K,V will
    // not be included in the cache
    int64_t tile_size;
    if (one_round) {
      tile_size =
          (cache_size - q_tile_size * head_dim *
                            (q_buffer_elem_size + output_buffer_elem_size)) /
          (logits_buffer_elem_size * max_num_q_per_iter);
    } else {
      tile_size =
          (cache_size - q_tile_size * head_dim *
                            (q_buffer_elem_size + output_buffer_elem_size)) /
          (logits_buffer_elem_size * max_num_q_per_iter +
           2 * head_dim * elem_size);
    }
    int64_t rounded_tile_size = (tile_size / round_size) * round_size;
    return std::max(rounded_tile_size, round_size);
  }
};

struct AttentionInput {
  AttentionMetadata* metadata;
  int32_t num_tokens;
  int32_t num_heads;
  int32_t num_kv_heads;
  int32_t block_size;
  void* query;
  int64_t query_num_tokens_stride;
  int64_t query_num_heads_stride;
  int64_t cache_num_blocks_stride;
  int64_t cache_num_kv_heads_stride;
  int64_t blt_num_tokens_stride;
  void* key_cache;
  void* value_cache;
  void* output;
  int32_t* query_start_loc;
  int32_t* seq_lens;
  int32_t* block_table;
  float* alibi_slopes;
  // Attention sinks pointer. May reference bf16 or fp32 data depending on
  // `s_aux_is_bf16`. bf16 sinks keep the native bf16 path; anything else is
  // provided as fp32 and executed in full float precision.
  const void* s_aux;
  bool s_aux_is_bf16;
  int32_t* dynamic_causal;
  float scale;
  bool causal;
  int32_t sliding_window_size;
  float softcap;
  // FP8 KV cache scales (used by FP8 attention implementations)
  float k_scale_fp8 = 1.0f;
  float v_scale_fp8 = 1.0f;
  // FP8 query scale (used by AMX_FP8 which quantizes query inside the kernel)
  float q_scale_fp8 = 1.0f;
};

#define DEFINE_CPU_ATTENTION_PARAMS                                         \
  q_buffer_t *__restrict__ q_heads_buffer,                                  \
      kv_cache_t *__restrict__ k_head_cache_ptr,                            \
      kv_cache_t *__restrict__ v_head_cache_ptr,                            \
      logits_buffer_t *__restrict__ logits_buffer,                          \
      float *__restrict__ partial_q_buffer, float *__restrict__ max_buffer, \
      float *__restrict__ sum_buffer, int32_t *__restrict__ block_table,    \
      const int32_t kv_end_pos, const int32_t kv_tile_start_pos,            \
      const int32_t kv_tile_end_pos, const int32_t kv_tile_token_num,       \
      const int64_t kv_cache_num_blocks_stride, const int32_t q_head_num,   \
      const int32_t q_token_num, const int32_t q_tile_start_pos,            \
      const int32_t q_heads_per_kv, const int32_t block_size,               \
      const int32_t left_window_size, const int32_t right_window_size,      \
      float scale, const float softcap_scale,                               \
      const float *__restrict__ alibi_slopes, const bool is_first_iter,     \
      const bool use_sink, const bool debug_info

#define CPU_ATTENTION_PARAMS                                                  \
  q_heads_buffer, k_head_cache_ptr, v_head_cache_ptr, logits_buffer,          \
      partial_q_buffer, max_buffer, sum_buffer, block_table, kv_end_pos,      \
      kv_tile_start_pos, kv_tile_end_pos, kv_tile_token_num,                  \
      kv_cache_num_blocks_stride, q_head_num, q_token_num, q_tile_start_pos,  \
      q_heads_per_kv, block_size, left_window_size, right_window_size, scale, \
      softcap_scale, alibi_slopes, is_first_iter, use_sink, debug_info

enum class AttentionGemmPhase { QK, PV };

template <typename T>
struct VecTypeTrait {
  using vec_t = void;
};

template <>
struct VecTypeTrait<float> {
  using vec_t = vec_op::FP32Vec16;
};

template <>
struct VecTypeTrait<c10::BFloat16> {
  using vec_t = vec_op::BF16Vec16;
};

template <>
struct VecTypeTrait<c10::Half> {
  using vec_t = vec_op::FP16Vec16;
};

template <typename T>
void print_logits(const char* name, T* ptr, int32_t row, int32_t col,
                  int32_t stride) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(5) << name << ": [\n";
  auto* curr_logits_buffer = ptr;
  for (int32_t m = 0; m < row; ++m) {
    for (int32_t n = 0; n < col; ++n) {
      ss << curr_logits_buffer[n] << ", ";
    }
    ss << "\n";
    curr_logits_buffer += stride;
  }
  ss << "]\n";
  std::printf("%s", ss.str().c_str());
}

// This allows us to store probabilities in a packed format
// for GEMM implementations that need packed A.
template <typename attention_impl_t, typename = void>
struct ProbabilityTokenStore {
  using prob_buffer_t = typename attention_impl_t::prob_buffer_t;
  static constexpr int32_t TokenStride = 1;

  FORCE_INLINE static void store_probabilities(
      prob_buffer_t* __restrict__ probability, const vec_op::FP32Vec16& values,
      const int32_t, const int64_t) {
    using prob_buffer_vec_t = typename VecTypeTrait<prob_buffer_t>::vec_t;
    prob_buffer_vec_t output_vec(values);
    output_vec.save(probability);
  }
};

template <typename attention_impl_t>
struct ProbabilityTokenStore<
    attention_impl_t,
    std::void_t<typename attention_impl_t::ProbabilityTokenStore>>
    : attention_impl_t::ProbabilityTokenStore {};

template <typename attention_impl_t>
class AttentionMainLoop {
 public:
  using query_t = typename attention_impl_t::query_t;
  using q_buffer_t = typename attention_impl_t::q_buffer_t;
  using kv_cache_t = typename attention_impl_t::kv_cache_t;
  using logits_buffer_t = typename attention_impl_t::logits_buffer_t;
  using partial_output_buffer_t =
      typename attention_impl_t::partial_output_buffer_t;
  using prob_buffer_t = typename attention_impl_t::prob_buffer_t;
  using probability_token_store_t = ProbabilityTokenStore<attention_impl_t>;

  static constexpr int64_t max_q_head_num_per_iter =
      attention_impl_t::MaxQHeadNumPerIteration;
  static constexpr int64_t blocksize_alignment =
      attention_impl_t::BlockSizeAlignment;
  static constexpr int64_t headdim_alignment =
      attention_impl_t::HeadDimAlignment;
  static constexpr int64_t head_dim = attention_impl_t::HeadDim;
  static constexpr int32_t probability_token_stride =
      probability_token_store_t::TokenStride;
  static constexpr ISA ISAType = attention_impl_t::ISAType;
  static constexpr bool scale_on_logits =
      attention_impl_t::scale_on_logits;  // apply scale on logits, otherwise
                                          // apply scale on q_buffer

  template <typename tile_gemm_t>
  class Attention {
   public:
    // Args:
    //  - q_heads_buffer: [MaxQHeadNumPerIteration, head_dim]
    //  - k_head_cache_ptr: [num_blocks, block_size * head_dim]
    //  - v_head_cache_ptr: [num_blocks, block_size * head_dim]
    //  - logits_buffer: [MaxQHeadNumPerIteration, kv_tile_token_num], store Q@K
    //  - logits partial_q_buffer: [MaxQHeadNumPerIteration, head_dim], store
    //  partial output
    //  - max_buffer: [MaxQHeadNumPerIteration, 1], store max logits
    //  - sum_buffer: [MaxQHeadNumPerIteration, 1], store sum of exp
    //  - block_table
    //  - kv_end_pos: un-aligned end position of KV cache
    //  - kv_tile_start_pos: start position of KV cache, aligned to
    //  BlockSizeAlignment
    //  - kv_tile_end_pos: end position of KV cache, aligned to
    //  BlockSizeAlignment
    //  - kv_tile_token_num: KV token num, aligned to BlockSizeAlignment
    //  - kv_cache_num_blocks_stride
    //  - q_head_num: head num of q_tile
    //  - q_token_num: token num of q_tile, should be q_head_num /
    //  q_heads_per_kv
    //  - q_tile_start_pos: start pos of the first token in q_heads_buffer
    //  - q_heads_per_kv
    //  - block_size
    //  - left_window_size
    //  - right_window_size
    //  - scale
    //  - softcap_scale
    //  - alibi_slopes
    //  - is_first_iter
    //  - use_sink
    //  - debug_info
    void operator()(DEFINE_CPU_ATTENTION_PARAMS) {
      // k_cache_token_group_stride: stride of K cache when move to next
      // BlockSizeAlignment tokens in a block
      const int64_t k_cache_token_group_stride =
          attention_impl_t::k_cache_token_group_stride(block_size);
      // v_cache_token_group_stride: stride of V cache when move to next
      // BlockSizeAlignment tokens in a block
      const int64_t v_cache_token_group_stride =
          attention_impl_t::v_cache_token_group_stride(block_size);
      // v_cache_head_group_stride: stride of V cache when move to next
      // HeadDimAlignment head dims in a block
      const int64_t v_cache_head_group_stride =
          attention_impl_t::v_cache_head_group_stride(block_size);
      const int32_t token_group_num = kv_tile_token_num / blocksize_alignment;
      const int32_t token_group_num_per_block =
          block_size / blocksize_alignment;
      const int32_t start_block_idx = kv_tile_start_pos / block_size;
      const int32_t start_block_offset = kv_tile_start_pos % block_size;
      const int32_t start_block_group_offset =
          start_block_offset / blocksize_alignment;
      const int32_t end_block_idx =
          (kv_tile_start_pos + kv_tile_token_num - 1) / block_size + 1;

      // compute Q@K logits
      {
        int32_t curr_group_offset =
            start_block_group_offset * k_cache_token_group_stride;
        int32_t curr_group_num_in_block =
            token_group_num_per_block - start_block_group_offset;
        int32_t remaining_group_num = token_group_num;
        logits_buffer_t* curr_logits_buffer = logits_buffer;
        for (int32_t block_idx = start_block_idx; block_idx < end_block_idx;
             ++block_idx) {
          int32_t physical_block_idx = block_table[block_idx];
          kv_cache_t* k_cache_block_ptr =
              k_head_cache_ptr +
              physical_block_idx * kv_cache_num_blocks_stride +
              curr_group_offset;
          curr_group_num_in_block =
              std::min(remaining_group_num, curr_group_num_in_block);

          for (int32_t block_group_idx = 0;
               block_group_idx < curr_group_num_in_block; ++block_group_idx) {
            // logits_tile = q_tile @ k_tile, [MaxQHeadNumPerIteration,
            // BlockSizeAlignment] = [MaxQHeadNumPerIteration, head_dim] @
            // [head_dim, BlockSizeAlignment]

            // By default, logits_buffer, q_buffer and k_cache are row-major,
            // but may be packed by ISA implementation.
            tile_gemm_t::template gemm<AttentionGemmPhase::QK, head_dim>(
                q_head_num, q_heads_buffer, k_cache_block_ptr,
                curr_logits_buffer, head_dim, block_size, kv_tile_token_num,
                block_size, head_dim, false);

            if constexpr (scale_on_logits) {
              float* __restrict__ scale_curr_logits_buffer = curr_logits_buffer;
              vec_op::FP32Vec16 scale_vec(scale);
              for (int32_t i = 0; i < q_head_num; ++i) {
                static_assert(blocksize_alignment % 16 == 0);
                constexpr int32_t vec_num = blocksize_alignment / 16;
                vec_op::unroll_loop<int32_t, vec_num>([&](int32_t vec_idx) {
                  vec_op::FP32Vec16 vec(scale_curr_logits_buffer +
                                        vec_idx * 16);
                  vec = vec * scale_vec;
                  vec.save(scale_curr_logits_buffer + vec_idx * 16);
                });
                scale_curr_logits_buffer += kv_tile_token_num;
              }
            }

            // Move buffer ptrs
            k_cache_block_ptr += k_cache_token_group_stride;
            curr_logits_buffer += blocksize_alignment;
          }

          // Update
          remaining_group_num -= curr_group_num_in_block;
          curr_group_offset = 0;
          curr_group_num_in_block = token_group_num_per_block;
        }
      }

      // process logits
      {
        if (softcap_scale != 0.0f) {
          apply_softcap(logits_buffer, kv_tile_token_num, q_head_num,
                        kv_tile_token_num, softcap_scale);
          // print_logits("softcap raw logits", logits_buffer, q_head_num,
          // kv_tile_token_num, kv_tile_token_num);
        }

        if (alibi_slopes != nullptr) {
          apply_alibi_slopes(logits_buffer, alibi_slopes, kv_tile_token_num,
                             q_tile_start_pos, kv_tile_start_pos, q_token_num,
                             kv_tile_token_num, q_heads_per_kv);

          // print_logits("alibi raw logits", logits_buffer, q_head_num,
          // kv_tile_token_num, kv_tile_token_num);
        }

        apply_mask(logits_buffer, kv_tile_token_num, q_tile_start_pos,
                   kv_end_pos, kv_tile_start_pos, kv_tile_end_pos, q_token_num,
                   q_heads_per_kv, left_window_size, right_window_size);

        // if (debug_info){
        // print_logits("masked logits", logits_buffer, q_head_num,
        // kv_tile_token_num, kv_tile_token_num);
        // print_logits("old_max", max_buffer, 1, q_head_num, q_head_num);
        // print_logits("old_sum", sum_buffer, 1, q_head_num, q_head_num);
        // }

        apply_softmax(logits_buffer, partial_q_buffer, max_buffer, sum_buffer,
                      kv_tile_token_num, q_head_num, kv_tile_token_num,
                      is_first_iter, use_sink);
      }

      // compute P@V
      {
        constexpr bool prequantize_probabilities = []() {
          if constexpr (requires { tile_gemm_t::prequantize_probabilities; }) {
            return tile_gemm_t::prequantize_probabilities;
          }
          return false;
        }();
        int32_t curr_group_offset =
            start_block_group_offset * v_cache_token_group_stride;
        int32_t curr_group_num_in_block =
            token_group_num_per_block - start_block_group_offset;
        int32_t remaining_group_num = token_group_num;
        int32_t head_dim_group_num = head_dim / headdim_alignment;
        using pv_prob_buffer_t = std::conditional_t<prequantize_probabilities,
                                                    uint8_t, prob_buffer_t>;
        pv_prob_buffer_t* curr_prob_buffer =
            reinterpret_cast<pv_prob_buffer_t*>(logits_buffer);
        const int64_t prob_buffer_stride =
            prequantize_probabilities
                ? kv_tile_token_num
                : kv_tile_token_num *
                      (sizeof(logits_buffer_t) / sizeof(prob_buffer_t));
        partial_output_buffer_t* curr_partial_q_buffer = partial_q_buffer;
        bool accum_c = !is_first_iter;
        for (int32_t block_idx = start_block_idx; block_idx < end_block_idx;
             ++block_idx) {
          int32_t physical_block_idx = block_table[block_idx];
          kv_cache_t* v_cache_block_ptr =
              v_head_cache_ptr +
              physical_block_idx * kv_cache_num_blocks_stride +
              curr_group_offset;
          curr_group_num_in_block =
              std::min(remaining_group_num, curr_group_num_in_block);
          int32_t curr_token_num =
              curr_group_num_in_block * blocksize_alignment;

          for (int32_t head_dim_group_idx = 0;
               head_dim_group_idx < head_dim_group_num; ++head_dim_group_idx) {
            // output_tile = p_tile @ v_tile, [MaxQHeadNumPerIteration,
            // HeadDimAlignment] = [MaxQHeadNumPerIteration, block_size] @
            // [block_size, HeadDimAlignment]
            tile_gemm_t::template gemm<AttentionGemmPhase::PV, -1>(
                q_head_num, curr_prob_buffer, v_cache_block_ptr,
                curr_partial_q_buffer, prob_buffer_stride, head_dim, head_dim,
                block_size, curr_token_num, accum_c);

            // Update
            curr_partial_q_buffer += headdim_alignment;
            v_cache_block_ptr += v_cache_head_group_stride;
          }

          // Update
          remaining_group_num -= curr_group_num_in_block;
          curr_group_offset = 0;
          curr_group_num_in_block = token_group_num_per_block;
          curr_prob_buffer += curr_token_num * probability_token_stride;
          curr_partial_q_buffer = partial_q_buffer;
          accum_c = true;
        }
      }
    }

    void apply_mask(logits_buffer_t* __restrict__ logits_buffer,
                    const int64_t logits_buffer_stride,
                    const int32_t q_tile_start_pos, const int32_t kv_end_pos,
                    const int32_t kv_tile_start_pos,
                    const int32_t kv_tile_end_pos, const int32_t q_token_num,
                    const int32_t q_heads_per_kv,
                    const int32_t sliding_window_left,
                    const int32_t sliding_window_right) {
      // Apply mask
      constexpr logits_buffer_t neg_inf =
          -std::numeric_limits<logits_buffer_t>::infinity();
      logits_buffer_t* __restrict__ curr_logits_buffer = logits_buffer;
      int32_t curr_token_pos = q_tile_start_pos;
      for (int32_t token_idx = 0; token_idx < q_token_num; ++token_idx) {
        int32_t left_kv_pos = [&]() {
          int32_t pos = kv_tile_start_pos;
          if (sliding_window_left != -1) {
            pos = std::max(pos, curr_token_pos - sliding_window_left);
          }
          // Clamp to tile end to avoid OOB when window starts past the tile
          return std::min(pos, kv_tile_end_pos);
        }();

        int32_t right_kv_pos = [&]() {
          int32_t pos = kv_tile_end_pos;
          if (sliding_window_right != -1) {
            pos = std::min(pos,
                           std::max(kv_tile_start_pos,
                                    curr_token_pos + sliding_window_right + 1));
          }
          return std::min(pos, kv_end_pos);
        }();

        int32_t left_invalid_token_num = left_kv_pos - kv_tile_start_pos;
        int32_t right_invalid_token_num = kv_tile_end_pos - right_kv_pos;
        for (int32_t head_idx = 0; head_idx < q_heads_per_kv; ++head_idx) {
          logits_buffer_t* __restrict__ curr_logits_buffer_tail =
              curr_logits_buffer + right_kv_pos - kv_tile_start_pos;
          for (int32_t i = 0; i < left_invalid_token_num; ++i) {
            curr_logits_buffer[i] = neg_inf;
          }
          for (int32_t i = 0; i < right_invalid_token_num; ++i) {
            curr_logits_buffer_tail[i] = neg_inf;
          }

          curr_logits_buffer += logits_buffer_stride;
        }

        ++curr_token_pos;
      }
    }

    void apply_softmax(logits_buffer_t* __restrict__ logits_buffer,
                       float* __restrict__ partial_q_buffer,
                       float* __restrict__ max_buffer,
                       float* __restrict__ sum_buffer,
                       const int64_t logits_buffer_stride, int32_t q_head_num,
                       int32_t kv_tile_token_num, bool is_first_iter,
                       bool use_sink) {
#ifdef DEFINE_FAST_EXP
      DEFINE_FAST_EXP
      bool constexpr IsReducedPrecision =
          std::is_same_v<query_t, c10::BFloat16> ||
          std::is_same_v<query_t, c10::Half>;
#endif

      constexpr bool prequantize_probabilities = []() {
        if constexpr (requires { tile_gemm_t::prequantize_probabilities; }) {
          return tile_gemm_t::prequantize_probabilities;
        }
        return false;
      }();
      using softmax_prob_buffer_t =
          std::conditional_t<prequantize_probabilities, uint8_t, prob_buffer_t>;
      static_assert(sizeof(prob_buffer_t) <= sizeof(logits_buffer_t));

      logits_buffer_t* __restrict__ curr_logits_buffer = logits_buffer;
      softmax_prob_buffer_t* __restrict__ curr_prob_buffer =
          reinterpret_cast<softmax_prob_buffer_t*>(logits_buffer);
      float* __restrict__ curr_partial_q_buffer = partial_q_buffer;
      const int32_t vec_num = kv_tile_token_num / 16;
      const int32_t head_vec_num = head_dim / 16;
      for (int32_t i = 0; i < q_head_num; ++i) {
        float init_max_val = max_buffer[i];
        float init_sum_val = sum_buffer[i];

        // apply scale and compute max
        vec_op::FP32Vec16 max_vec(init_max_val);
        {
          logits_buffer_t* __restrict__ curr_logits_buffer_iter =
              curr_logits_buffer;
          for (int32_t j = 0; j < vec_num; ++j) {
            vec_op::FP32Vec16 vec(curr_logits_buffer_iter);
            max_vec = vec.max(max_vec);

            curr_logits_buffer_iter += 16;
          }
        }
        float new_max_val = max_vec.reduce_max();
        float rescale_factor = init_max_val - new_max_val;

        // use same rescale threshold with FA4.
        // https://github.com/Dao-AILab/flash-attention/blob/1b8e1e641c6a179be9a0538b7f40fd595050b735/flash_attn/cute/flash_fwd_sm100.py#L1271
        bool need_rescale = rescale_factor < -8.0;
        if (!need_rescale) {
          new_max_val = init_max_val;
        } else {
          max_buffer[i] = new_max_val;
        }

        // sub max, compute exp and sum
        max_vec = vec_op::FP32Vec16(new_max_val);
        vec_op::FP32Vec16 sum_vec(0.0);
        {
          logits_buffer_t* __restrict__ curr_logits_buffer_iter =
              curr_logits_buffer;
          softmax_prob_buffer_t* __restrict__ curr_prob_buffer_iter =
              curr_prob_buffer;
          for (int32_t j = 0; j < vec_num; ++j) {
            vec_op::FP32Vec16 vec(curr_logits_buffer_iter);
            vec = vec - max_vec;

            // compute exp

#if defined(DEFINE_FAST_EXP)
  #ifdef __aarch64__
            if constexpr (IsReducedPrecision) {
              vec = fast_exp_f16(vec);
            } else
  #endif
            {
              vec = fast_exp(vec);
            }

#else
            vec.save(curr_logits_buffer_iter);
            for (int32_t k = 0; k < 16; ++k) {
              curr_logits_buffer_iter[k] = std::exp(curr_logits_buffer_iter[k]);
            }
            vec = vec_op::FP32Vec16(curr_logits_buffer_iter);
#endif

            if constexpr (prequantize_probabilities) {
              vec.save(curr_logits_buffer_iter);
            } else {
              probability_token_store_t::store_probabilities(
                  curr_prob_buffer_iter, vec, i,
                  logits_buffer_stride * sizeof(logits_buffer_t) /
                      sizeof(prob_buffer_t));
            }

            sum_vec = sum_vec + vec;

            curr_logits_buffer_iter += 16;
            curr_prob_buffer_iter += 16 * probability_token_stride;
          }
          if constexpr (prequantize_probabilities) {
            tile_gemm_t::quantize_probability_row(
                curr_logits_buffer, curr_prob_buffer, kv_tile_token_num);
          }
        }
        float new_sum_val = sum_vec.reduce_sum();

        // rescale sum and partial outputs
        if (need_rescale) {
          // compute rescale factor
          rescale_factor = std::exp(rescale_factor);
          vec_op::FP32Vec16 rescale_factor_vec(rescale_factor);

          // rescale sum
          new_sum_val += rescale_factor * init_sum_val;

          // rescale output
          if (!is_first_iter) {
            float* __restrict__ curr_partial_q_buffer_iter =
                curr_partial_q_buffer;
            for (int32_t j = 0; j < head_vec_num; ++j) {
              vec_op::FP32Vec16 vec(curr_partial_q_buffer_iter);
              vec = vec * rescale_factor_vec;
              vec.save(curr_partial_q_buffer_iter);

              curr_partial_q_buffer_iter += 16;
            }
          }
        } else {
          new_sum_val += init_sum_val;
        }

        sum_buffer[i] = new_sum_val;

        curr_logits_buffer += logits_buffer_stride;
        if constexpr (prequantize_probabilities) {
          curr_prob_buffer += kv_tile_token_num;
        } else {
          curr_prob_buffer =
              reinterpret_cast<softmax_prob_buffer_t*>(curr_logits_buffer);
        }
        curr_partial_q_buffer += head_dim;
      }
    }

    void apply_softcap(logits_buffer_t* __restrict__ logits_buffer,
                       const int64_t logits_buffer_stride, int32_t q_head_num,
                       int32_t kv_tile_token_num, float softcap_scale) {
#ifdef DEFINE_FAST_EXP
      DEFINE_FAST_EXP
      bool constexpr IsReducedPrecision =
          std::is_same_v<query_t, c10::BFloat16> ||
          std::is_same_v<query_t, c10::Half>;
#endif

      float inv_softcap_scale = 1.0 / softcap_scale;
      vec_op::FP32Vec16 softcap_scale_vec(softcap_scale);
      vec_op::FP32Vec16 inv_softcap_scale_vec(inv_softcap_scale);
      vec_op::FP32Vec16 ones_vec(1.0);
      logits_buffer_t* __restrict__ curr_logits_buffer = logits_buffer;
      const int32_t vec_num = kv_tile_token_num / 16;
      for (int32_t i = 0; i < q_head_num; ++i) {
        logits_buffer_t* __restrict__ curr_logits_buffer_iter =
            curr_logits_buffer;
        for (int32_t j = 0; j < vec_num; ++j) {
          vec_op::FP32Vec16 vec(curr_logits_buffer_iter);
          vec = vec * inv_softcap_scale_vec;

#if defined(DEFINE_FAST_EXP)
  #ifdef __aarch64__
          if constexpr (IsReducedPrecision) {
            vec = fast_exp_f16(vec);
          } else
  #endif
          {
            vec = fast_exp(vec);
          }
          vec_op::FP32Vec16 inv_vec = ones_vec / vec;
          vec = (vec - inv_vec) / (vec + inv_vec);
#else
          vec.save(curr_logits_buffer_iter);
          for (int k = 0; k < 16; ++k) {
            curr_logits_buffer_iter[k] = std::tanh(curr_logits_buffer_iter[k]);
          }
          vec = vec_op::FP32Vec16(curr_logits_buffer_iter);
#endif
          vec = vec * softcap_scale_vec;
          vec.save(curr_logits_buffer_iter);

          curr_logits_buffer_iter += 16;
        }

        curr_logits_buffer += logits_buffer_stride;
      }
    }

    void apply_alibi_slopes(logits_buffer_t* __restrict__ logits_buffer,
                            const float* __restrict__ alibi_slopes,
                            const int64_t logits_buffer_stride,
                            const int32_t q_tile_start_pos,
                            const int32_t kv_tile_start_pos,
                            const int32_t q_token_num,
                            const int32_t kv_tile_token_num,
                            const int32_t q_heads_per_kv) {
      alignas(64) constexpr float initial_arange_vals[16] = {
          0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
          8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
      const int32_t vec_num = kv_tile_token_num / 16;

      vec_op::FP32Vec16 initial_arange_vals_vec(initial_arange_vals);
      initial_arange_vals_vec =
          initial_arange_vals_vec + vec_op::FP32Vec16((float)kv_tile_start_pos);
      vec_op::FP32Vec16 pos_offset_vec(16.0);
      logits_buffer_t* __restrict__ curr_logits_buffer = logits_buffer;
      for (int32_t i = 0; i < q_token_num; ++i) {
        vec_op::FP32Vec16 curr_q_pos_vec((float)(i + q_tile_start_pos));
        for (int32_t j = 0; j < q_heads_per_kv; ++j) {
          vec_op::FP32Vec16 alibi_scale_vec(alibi_slopes[j]);
          vec_op::FP32Vec16 curr_kv_pos_vec(initial_arange_vals_vec);
          logits_buffer_t* __restrict__ curr_logits_buffer_iter =
              curr_logits_buffer;
          for (int32_t k = 0; k < vec_num; ++k) {
            vec_op::FP32Vec16 alibi_bias_vec =
                alibi_scale_vec * (curr_kv_pos_vec - curr_q_pos_vec);
            vec_op::FP32Vec16 vec(curr_logits_buffer_iter);
            vec = vec + alibi_bias_vec;

            vec.save(curr_logits_buffer_iter);

            curr_kv_pos_vec = curr_kv_pos_vec + pos_offset_vec;
            curr_logits_buffer_iter += 16;
          }
          curr_logits_buffer += logits_buffer_stride;
        }
      }
    }
  };

 public:
  void operator()(const AttentionInput* input) {
    TORCH_CHECK_EQ(input->metadata->thread_num, cpu_utils::get_max_threads());
    alignas(64) std::atomic<int32_t> guard_counter(0);
    const auto run_thread = [&](const int thread_id, const int thread_num) {
      AttentionMetadata& metadata = *input->metadata;
      if (metadata.workitem_group_num == 0) {
        return;
      }

      attention_impl_t attn_impl;
      constexpr bool fp8_kv = std::is_same_v<kv_cache_t, c10::Float8_e4m3fn> ||
                              std::is_same_v<kv_cache_t, c10::Float8_e5m2>;
      float output_v_scale = 1.0f;
      if constexpr (fp8_kv) {
        attn_impl.init_from_input(input);
        output_v_scale = attn_impl.get_output_v_scale();
      }

      // general information
      const int32_t q_head_num = input->num_heads;
      AttentionWorkItemGroup* const workitem_groups =
          metadata.workitem_groups_ptr;
      ReductionWorkItemGroup* const reduction_items =
          metadata.reduction_items_ptr;
      const int64_t q_token_num_stride = input->query_num_tokens_stride;
      const int64_t q_head_num_stride = input->query_num_heads_stride;
      const int64_t kv_cache_head_num_stride = input->cache_num_kv_heads_stride;
      const int64_t kv_cache_block_num_stride = input->cache_num_blocks_stride;
      const int32_t sliding_window_size = input->sliding_window_size;
      const int32_t block_size = input->block_size;
      const float scale = input->scale;
      const float softcap_scale = input->softcap;
      const float* alibi_slopes = input->alibi_slopes;
      const void* s_aux = input->s_aux;
      const bool s_aux_is_bf16 = input->s_aux_is_bf16;
      const int32_t* dynamic_causal = input->dynamic_causal;
      const bool is_dynamic_causal = dynamic_causal != nullptr;

      const bool causal = input->causal;
      int32_t* const block_table = input->block_table;
      const int64_t block_table_stride = input->blt_num_tokens_stride;

      // init buffers
      void* scratchpad_ptr =
          cpu_utils::ScratchPadManager::get_scratchpad_manager()
              ->get_data<void>();
      AttentionScratchPad buffer_manager(thread_id, metadata, scratchpad_ptr);

      if (metadata.reduction_split_num > 0) {
        for (int32_t item_idx = thread_id;
             item_idx < metadata.reduction_item_num; item_idx += thread_num) {
          const ReductionWorkItemGroup& item = reduction_items[item_idx];
          buffer_manager.update(item.scratch_offset, item.split_num, head_dim,
                                item.q_token_id_num * item.q_head_num,
                                sizeof(partial_output_buffer_t));
          volatile bool* __restrict__ curr_flag_ptr =
              buffer_manager.get_reduce_flag_buffer();
          for (int32_t split_idx = 0; split_idx < item.split_num; ++split_idx) {
            curr_flag_ptr[split_idx] = false;
          }
        }
      }

      const int64_t available_cache_size = cpu_utils::get_available_l2_size();
      const int32_t default_tile_size =
          AttentionScheduler::calcu_default_tile_size(
              available_cache_size, head_dim, sizeof(kv_cache_t),
              sizeof(q_buffer_t), sizeof(logits_buffer_t),
              sizeof(partial_output_buffer_t), max_q_head_num_per_iter,
              max_q_head_num_per_iter);

      const int32_t reduction_item_num = metadata.reduction_item_num;
      const int32_t workitem_groups_counter_num = metadata.task_span_num;
      const int32_t reduction_items_counter_num = reduction_item_num;
      const int32_t total_counter_num =
          workitem_groups_counter_num + reduction_items_counter_num;

      if (metadata.reduction_split_num > 0) {
        guard_counter.fetch_add(1, std::memory_order_acq_rel);
        while (guard_counter.load(std::memory_order_acquire) != thread_num) {
#ifdef FAST_SPINNING
          FAST_SPINNING
#else
          std::this_thread::yield();
#endif
        }
      }

      // main loop
      for (;;) {
        int64_t task_idx = metadata.acquire_counter();

        if (task_idx >= total_counter_num) {
          // no more tasks, leave loop
          break;
        }

        if (task_idx < workitem_groups_counter_num) {
          const AttentionTaskSpan span = metadata.task_spans_ptr[task_idx];
          const int32_t span_start = span.start;
          const int32_t span_count = span.count;
          for (int32_t span_item = span_start;
               span_item < span_start + span_count; ++span_item) {
            AttentionWorkItemGroup* const current_workitem_group =
                workitem_groups + span_item;

            const int32_t current_group_idx = current_workitem_group->req_id;
            const int32_t current_group_causal =
                is_dynamic_causal ? dynamic_causal[current_group_idx] != 0
                                  : causal;
            auto [sliding_window_left, sliding_window_right] =
                AttentionScheduler::calcu_sliding_window_size(
                    sliding_window_size, current_group_causal);
            const int32_t kv_start_pos =
                current_workitem_group->kv_split_pos_start;
            const int32_t kv_end_pos = current_workitem_group->kv_split_pos_end;
            const int32_t curr_spilt_id = current_workitem_group->split_id;
            const int32_t q_token_id_start =
                current_workitem_group->q_token_id_start;
            const int32_t q_token_num = current_workitem_group->q_token_num;
            const int32_t curr_kv_head_idx =
                current_workitem_group->kv_head_idx;
            const int32_t curr_q_heads_per_kv =
                current_workitem_group->q_head_num;
            const int32_t curr_max_q_token_num_per_iter =
                max_q_head_num_per_iter / curr_q_heads_per_kv;
            const int32_t curr_default_q_tile_token_num =
                default_tile_size / curr_q_heads_per_kv;
            const int32_t q_head_start_idx =
                current_workitem_group->q_head_start;

            // taskgroup general information
            const int32_t q_end = input->query_start_loc[current_group_idx + 1];
            const int32_t q_start = input->query_start_loc[current_group_idx];
            const int32_t seq_len = input->seq_lens[current_group_idx];
            const int32_t q_start_pos = seq_len - (q_end - q_start);
            // Only apply sink for the first KV split
            bool use_sink = (s_aux != nullptr &&
                             current_workitem_group->local_split_id == 0);

            for (int32_t q_token_offset = 0; q_token_offset < q_token_num;
                 q_token_offset += curr_default_q_tile_token_num) {
              bool first_iter_flag[AttentionScheduler::MaxQTileIterNum];
              for (int32_t i = 0; i < AttentionScheduler::MaxQTileIterNum;
                   ++i) {
                first_iter_flag[i] = true;
              }

              const int32_t q_token_start_idx =
                  q_start + q_token_offset + q_token_id_start;
              const int32_t actual_q_token_num = std::min(
                  curr_default_q_tile_token_num, q_token_num - q_token_offset);
              const int32_t q_head_tile_size =
                  actual_q_token_num * curr_q_heads_per_kv;
              const int32_t rounded_q_head_tile_size =
                  ((q_head_tile_size + max_q_head_num_per_iter - 1) /
                   max_q_head_num_per_iter) *
                  max_q_head_num_per_iter;
              const int32_t kv_tile_size =
                  AttentionScheduler::calcu_tile_size_with_constant_q(
                      available_cache_size, head_dim, sizeof(kv_cache_t),
                      sizeof(q_buffer_t), sizeof(logits_buffer_t),
                      sizeof(partial_output_buffer_t), max_q_head_num_per_iter,
                      blocksize_alignment, rounded_q_head_tile_size,
                      rounded_q_head_tile_size <= max_q_head_num_per_iter);

              // update buffers
              buffer_manager.update(
                  head_dim, sizeof(q_buffer_t), sizeof(logits_buffer_t),
                  sizeof(partial_output_buffer_t), max_q_head_num_per_iter,
                  rounded_q_head_tile_size, kv_tile_size);
              q_buffer_t* q_buffer = buffer_manager.get_q_buffer<q_buffer_t>();
              float* logits_buffer = buffer_manager.get_logits_buffer();
              float* partial_q_buffer = buffer_manager.get_output_buffer();
              float* max_buffer = buffer_manager.get_max_buffer();
              float* sum_buffer = buffer_manager.get_sum_buffer();

              const int32_t q_tile_start_pos =
                  q_start_pos + q_token_offset + q_token_id_start;
              const int32_t q_tile_end_pos =
                  q_tile_start_pos + actual_q_token_num;
              const auto [kv_tile_start_pos, kv_tile_end_pos] =
                  AttentionScheduler::calcu_kv_tile_pos(
                      kv_start_pos, kv_end_pos, q_tile_start_pos,
                      q_tile_end_pos, sliding_window_size,
                      current_group_causal);
              const auto [rounded_kv_tile_start_pos, rounded_kv_tile_end_pos] =
                  AttentionScheduler::align_kv_tile_pos(
                      kv_tile_start_pos, kv_tile_end_pos, blocksize_alignment);

              // std::printf("thread_id: %d, req_id: %d, q_token_start: %d,
              // q_token_end: %d, q_head_start: %d, q_head_end: %d, kv_head_idx:
              // %d, kv_pos_start: %d, kv_pos_end: %d\n",
              //                 thread_id, current_group_idx,
              //                 q_token_start_idx, q_token_start_idx +
              //                 actual_q_token_num, q_head_start_idx,
              //                 q_head_start_idx + actual_q_heads_per_kv,
              //                 curr_kv_head_idx, kv_tile_start_pos,
              //                 kv_tile_end_pos);

              // move buffers
              kv_cache_t* curr_k_cache =
                  reinterpret_cast<kv_cache_t*>(input->key_cache) +
                  curr_kv_head_idx * kv_cache_head_num_stride;
              kv_cache_t* curr_v_cache =
                  reinterpret_cast<kv_cache_t*>(input->value_cache) +
                  curr_kv_head_idx * kv_cache_head_num_stride;
              query_t* const q_tile_ptr =
                  reinterpret_cast<query_t*>(input->query) +
                  q_token_start_idx * q_token_num_stride +
                  q_head_start_idx * q_head_num_stride;
              size_t output_buffer_offset =
                  q_token_start_idx * q_head_num * head_dim +
                  q_head_start_idx * head_dim;
              int32_t* curr_block_table =
                  block_table + current_group_idx * block_table_stride;
              const float* curr_alibi_slopes =
                  (alibi_slopes != nullptr ? alibi_slopes + q_head_start_idx
                                           : nullptr);
              // copy the Q tile to q_buffer, the logical layout of q_buffer is
              // [actual_q_token_num, curr_q_heads_per_kv, head_dim]
              {
                attn_impl.copy_q_heads_tile(
                    q_tile_ptr, q_buffer, actual_q_token_num,
                    curr_q_heads_per_kv, q_token_num_stride, q_head_num_stride,
                    scale);
              }

              if (use_sink) {
                alignas(64) float s_aux_fp32[max_q_head_num_per_iter];
                if (s_aux_is_bf16) {
                  const c10::BFloat16* curr_s_aux =
                      static_cast<const c10::BFloat16*>(s_aux) +
                      q_head_start_idx;
                  for (int32_t head_idx = 0; head_idx < curr_q_heads_per_kv;
                       ++head_idx) {
                    s_aux_fp32[head_idx] =
                        static_cast<float>(curr_s_aux[head_idx]);
                  }
                } else {
                  const float* curr_s_aux =
                      static_cast<const float*>(s_aux) + q_head_start_idx;
                  std::copy_n(curr_s_aux, curr_q_heads_per_kv, s_aux_fp32);
                }

                float* __restrict__ curr_sum_buffer = sum_buffer;
                float* __restrict__ curr_max_buffer = max_buffer;
                for (int32_t token_idx = 0; token_idx < actual_q_token_num;
                     ++token_idx) {
                  for (int32_t head_idx = 0; head_idx < curr_q_heads_per_kv;
                       ++head_idx) {
                    curr_sum_buffer[head_idx] = 1.0f;
                    curr_max_buffer[head_idx] = s_aux_fp32[head_idx];
                  }

                  curr_sum_buffer += curr_q_heads_per_kv;
                  curr_max_buffer += curr_q_heads_per_kv;
                }
              } else {
                float* __restrict__ curr_sum_buffer = sum_buffer;
                float* __restrict__ curr_max_buffer = max_buffer;
                for (int32_t token_idx = 0; token_idx < actual_q_token_num;
                     ++token_idx) {
                  for (int32_t head_idx = 0; head_idx < curr_q_heads_per_kv;
                       ++head_idx) {
                    curr_sum_buffer[head_idx] = 0.0f;
                    curr_max_buffer[head_idx] =
                        std::numeric_limits<float>::lowest();
                  }

                  curr_sum_buffer += curr_q_heads_per_kv;
                  curr_max_buffer += curr_q_heads_per_kv;
                }
              }

              // compute loop
              for (int32_t kv_tile_pos = rounded_kv_tile_start_pos;
                   kv_tile_pos < rounded_kv_tile_end_pos;
                   kv_tile_pos += kv_tile_size) {
                const int32_t kv_tile_pos_left = kv_tile_pos;
                const int32_t kv_tile_pos_right = std::min(
                    kv_tile_pos_left + kv_tile_size, rounded_kv_tile_end_pos);
                for (int32_t q_head_tile_token_offset = 0;
                     q_head_tile_token_offset < actual_q_token_num;
                     q_head_tile_token_offset +=
                     curr_max_q_token_num_per_iter) {
                  const int32_t q_tile_pos_left =
                      q_tile_start_pos + q_head_tile_token_offset;
                  const int32_t q_tile_token_num =
                      std::min(curr_max_q_token_num_per_iter,
                               actual_q_token_num - q_head_tile_token_offset);
                  const int32_t q_tile_head_offset =
                      q_head_tile_token_offset * curr_q_heads_per_kv;
                  const int32_t q_tile_head_num =
                      q_tile_token_num * curr_q_heads_per_kv;
                  const int32_t q_tile_pos_right =
                      q_tile_pos_left + q_tile_token_num;
                  const auto [actual_kv_tile_pos_left,
                              actual_kv_tile_pos_right] =
                      AttentionScheduler::calcu_kv_tile_pos(
                          kv_tile_pos_left, kv_tile_pos_right, q_tile_pos_left,
                          q_tile_pos_right, sliding_window_size,
                          current_group_causal);
                  const int32_t q_iter_idx =
                      q_head_tile_token_offset / curr_max_q_token_num_per_iter;

                  if (actual_kv_tile_pos_right <= actual_kv_tile_pos_left) {
                    continue;
                  }

                  // align kv_pos to blocksize_alignment
                  const auto [aligned_actual_kv_tile_pos_left,
                              aligned_actual_kv_tile_pos_right] =
                      AttentionScheduler::align_kv_tile_pos(
                          actual_kv_tile_pos_left, actual_kv_tile_pos_right,
                          blocksize_alignment);
                  const int32_t actual_kv_token_num =
                      aligned_actual_kv_tile_pos_right -
                      aligned_actual_kv_tile_pos_left;

                  //   std::printf("\tq_iter_idx: %d, q_token_start: %d,
                  //   q_token_end: %d, q_token_num: %d, q_head_num: %d,
                  //   q_pos_start: %d, q_pos_end: %d, kv_pos_start: %d,
                  //   kv_pos_end: %d\n",
                  //             q_iter_idx, q_token_start_idx +
                  //             q_head_tile_token_offset,  q_token_start_idx +
                  //             q_head_tile_token_offset + q_tile_token_num,
                  //             q_tile_token_num, q_tile_head_num,
                  //             q_tile_pos_left, q_tile_pos_right,
                  //             aligned_actual_kv_tile_pos_left,
                  //             aligned_actual_kv_tile_pos_right);

                  // Move buffers
                  q_buffer_t* curr_q_heads_buffer =
                      q_buffer + q_tile_head_offset * head_dim;
                  float* curr_partial_q_buffer =
                      partial_q_buffer + q_tile_head_offset * head_dim;
                  float* curr_max_buffer = max_buffer + q_tile_head_offset;
                  float* curr_sum_buffer = sum_buffer + q_tile_head_offset;

                  bool debug_info = false;
                  // if (debug_info) {
                  //   std::printf("\tq_iter_idx: %d, q_token_start: %d,"
                  //   "q_token_end: %d, q_token_num: %d, q_head_num: %d,"
                  //   "q_pos_start: %d, q_pos_end: %d, kv_pos_start: %d,"
                  //   "kv_pos_end: %d\n",
                  //             q_iter_idx, q_token_start_idx +
                  //             q_head_tile_token_offset,  q_token_start_idx
                  //             + q_head_tile_token_offset +
                  //             q_tile_token_num, q_tile_token_num,
                  //             q_tile_head_num, q_tile_pos_left,
                  //             q_tile_pos_right,
                  //             aligned_actual_kv_tile_pos_left,
                  //             aligned_actual_kv_tile_pos_right);
                  // }

                  attn_impl.template execute_attention<Attention>(
                      curr_q_heads_buffer, curr_k_cache, curr_v_cache,
                      logits_buffer, curr_partial_q_buffer, curr_max_buffer,
                      curr_sum_buffer, curr_block_table, kv_end_pos,
                      aligned_actual_kv_tile_pos_left,
                      aligned_actual_kv_tile_pos_right, actual_kv_token_num,
                      kv_cache_block_num_stride, q_tile_head_num,
                      q_tile_token_num, q_tile_pos_left, curr_q_heads_per_kv,
                      block_size, sliding_window_left, sliding_window_right,
                      scale, softcap_scale, curr_alibi_slopes,
                      first_iter_flag[q_iter_idx], use_sink, debug_info);
                  first_iter_flag[q_iter_idx] = false;
                }
              }

              // write back partial results to output buffer or reduction buffer
              {
                if (curr_spilt_id == -1) {
                  final_output(partial_q_buffer,
                               reinterpret_cast<query_t*>(input->output) +
                                   output_buffer_offset,
                               sum_buffer, curr_q_heads_per_kv,
                               actual_q_token_num, q_head_num, output_v_scale);
                } else {
                  const ReductionWorkItemGroup& reduction_item =
                      reduction_items[current_workitem_group->reduction_id];
                  const int32_t stride =
                      reduction_item.q_token_id_num * reduction_item.q_head_num;
                  const int32_t stats_stride =
                      AttentionScratchPad::reduction_stats_stride(stride);
                  buffer_manager.update(reduction_item.scratch_offset,
                                        reduction_item.split_num, head_dim,
                                        stride, sizeof(float));
                  volatile bool* split_flag_buffer =
                      buffer_manager.get_reduce_flag_buffer() + curr_spilt_id;
                  float* split_output_buffer =
                      buffer_manager.get_reduce_output_buffer() +
                      curr_spilt_id * stride * head_dim;
                  float* split_max_buffer =
                      buffer_manager.get_reduce_max_buffer() +
                      curr_spilt_id * stats_stride;
                  float* split_sum_buffer =
                      buffer_manager.get_reduce_sum_buffer() +
                      curr_spilt_id * stats_stride;

                  partial_output(partial_q_buffer, max_buffer, sum_buffer,
                                 q_head_tile_size, split_output_buffer,
                                 split_max_buffer, split_sum_buffer,
                                 split_flag_buffer);
                }
              }
            }
          }
        } else {
          task_idx -= workitem_groups_counter_num;
          ReductionWorkItemGroup* const curr_workitem_groups =
              reduction_items + task_idx;
          const int32_t curr_output_token_idx =
              curr_workitem_groups->q_token_id_start;
          const int32_t curr_output_token_num =
              curr_workitem_groups->q_token_id_num;
          const int32_t curr_split_num = curr_workitem_groups->split_num;
          const int32_t current_group_idx = curr_workitem_groups->req_id;
          const int32_t curr_q_heads_per_kv = curr_workitem_groups->q_head_num;
          const int32_t curr_output_head_num =
              curr_output_token_num * curr_q_heads_per_kv;

          const int32_t q_start = input->query_start_loc[current_group_idx];
          const int32_t q_token_start_idx = q_start + curr_output_token_idx;
          const int32_t q_head_start_idx = curr_workitem_groups->q_head_start;
          size_t output_buffer_offset =
              q_token_start_idx * q_head_num * head_dim +
              q_head_start_idx * head_dim;

          const int32_t stride = curr_output_token_num * curr_q_heads_per_kv;
          buffer_manager.update(curr_workitem_groups->scratch_offset,
                                curr_split_num, head_dim, stride,
                                sizeof(float));
          volatile bool* split_flag_buffer =
              buffer_manager.get_reduce_flag_buffer();
          float* split_output_buffer =
              buffer_manager.get_reduce_output_buffer();
          float* split_max_buffer = buffer_manager.get_reduce_max_buffer();
          float* split_sum_buffer = buffer_manager.get_reduce_sum_buffer();

          reduce_splits(split_output_buffer, split_max_buffer, split_sum_buffer,
                        split_flag_buffer, stride, curr_output_head_num,
                        curr_split_num);
          final_output(
              split_output_buffer,
              reinterpret_cast<query_t*>(input->output) + output_buffer_offset,
              split_sum_buffer, curr_q_heads_per_kv, curr_output_token_num,
              q_head_num, output_v_scale);
        }
      }
    };

#ifdef _OPENMP
  #pragma omp parallel
    {
      run_thread(omp_get_thread_num(), omp_get_num_threads());
    }
#else
    run_thread(0, 1);
#endif

    // Reset counter for next call
    input->metadata->reset_counter();
  }

  void reduce_splits(float* __restrict__ split_output_buffer,
                     float* __restrict__ split_max_buffer,
                     float* __restrict__ split_sum_buffer,
                     volatile bool* __restrict__ flags,
                     const int32_t head_num_per_split,
                     const int32_t curr_head_num, const int32_t split_num) {
#ifdef DEFINE_FAST_EXP
    DEFINE_FAST_EXP
#endif
    // restrict curr_head_num <= max_q_head_num_per_iter in the scheduler
    // elems in split_max_buffer, split_sum_buffer are not cache alignment, use
    // local buffers to reduce false-sharing
    alignas(64) float local_max[max_q_head_num_per_iter];
    alignas(64) float local_sum[max_q_head_num_per_iter];
    const int32_t stats_stride =
        AttentionScratchPad::reduction_stats_stride(head_num_per_split);

    float* __restrict__ curr_split_output_buffer = split_output_buffer;
    float* __restrict__ curr_split_max_buffer = split_max_buffer;
    float* __restrict__ curr_split_sum_buffer = split_sum_buffer;
    constexpr int32_t head_dim_group_num = head_dim / 16;
    for (int32_t split_idx = 0; split_idx < split_num; ++split_idx) {
      while (!flags[split_idx]) {
#ifdef FAST_SPINNING
        FAST_SPINNING
#else
        std::this_thread::yield();
#endif
      }
      std::atomic_thread_fence(std::memory_order_acquire);

      if (split_idx > 0) {
        float* __restrict__ curr_output_buffer = split_output_buffer;
        float* __restrict__ curr_split_output_buffer_iter =
            curr_split_output_buffer;
        for (int32_t head_idx = 0; head_idx < curr_head_num; ++head_idx) {
          float final_max = local_max[head_idx];
          float curr_max = curr_split_max_buffer[head_idx];
          float final_sum = local_sum[head_idx];
          float curr_sum = curr_split_sum_buffer[head_idx];
          float* __restrict__ non_scale_output_iter =
              final_max > curr_max ? curr_output_buffer
                                   : curr_split_output_buffer_iter;
          float* __restrict__ scale_output_iter =
              final_max > curr_max ? curr_split_output_buffer_iter
                                   : curr_output_buffer;
          float rescale_factor = final_max > curr_max ? curr_max - final_max
                                                      : final_max - curr_max;
          rescale_factor = std::exp(rescale_factor);
          vec_op::FP32Vec16 rescale_factor_vec(rescale_factor);

          local_sum[head_idx] = final_max > curr_max
                                    ? final_sum + rescale_factor * curr_sum
                                    : rescale_factor * final_sum + curr_sum;

          final_max = std::max(final_max, curr_max);
          local_max[head_idx] = final_max;
          for (int32_t i = 0; i < head_dim_group_num; ++i) {
            vec_op::FP32Vec16 non_scale_vec(non_scale_output_iter);
            vec_op::FP32Vec16 scale_vec(scale_output_iter);
            vec_op::FP32Vec16 final_vec =
                non_scale_vec + scale_vec * rescale_factor_vec;
            final_vec.save(curr_output_buffer);

            non_scale_output_iter += 16;
            scale_output_iter += 16;
            curr_output_buffer += 16;
          }
          curr_split_output_buffer_iter += head_dim;
        }
      } else {
        std::copy_n(split_max_buffer, curr_head_num, local_max);
        std::copy_n(split_sum_buffer, curr_head_num, local_sum);
      }

      curr_split_output_buffer += head_num_per_split * head_dim;
      curr_split_max_buffer += stats_stride;
      curr_split_sum_buffer += stats_stride;
    }
    // write back final max and sum
    for (int32_t i = 0; i < curr_head_num; ++i) {
      split_max_buffer[i] = local_max[i];
      split_sum_buffer[i] = local_sum[i];
    }
  }

  void partial_output(float* __restrict__ partial_output_buffer,
                      float* __restrict__ partial_max_buffer,
                      float* __restrict__ partial_sum_buffer,
                      int32_t curr_head_num,
                      float* __restrict__ split_output_buffer,
                      float* __restrict__ split_max_buffer,
                      float* __restrict__ split_sum_buffer,
                      volatile bool* __restrict__ flag) {
    float* __restrict__ curr_partial_output_buffer = partial_output_buffer;
    float* __restrict__ curr_split_output_buffer = split_output_buffer;
    constexpr int32_t head_dim_group_num = head_dim / 16;
    for (int32_t i = 0; i < curr_head_num; ++i) {
      split_max_buffer[i] = partial_max_buffer[i];
      split_sum_buffer[i] = partial_sum_buffer[i];
      for (int32_t j = 0; j < head_dim_group_num; ++j) {
        vec_op::FP32Vec16 vec(curr_partial_output_buffer);
        vec.save(curr_split_output_buffer);

        curr_partial_output_buffer += 16;
        curr_split_output_buffer += 16;
      }
    }
    std::atomic_thread_fence(std::memory_order_release);
    *flag = true;
  }

  void final_output(float* __restrict__ partial_q_buffer,
                    query_t* __restrict__ curr_output_buffer,
                    float* __restrict__ sum_buffer,
                    const int32_t q_heads_per_kv,
                    const int32_t actual_q_token_num, const int32_t q_head_num,
                    const float v_scale = 1.0f) {
    // final output
    using output_vec_t = typename VecTypeTrait<query_t>::vec_t;

    float* __restrict__ curr_partial_output_buffer = partial_q_buffer;
    float* __restrict__ curr_sum_buffer = sum_buffer;
    constexpr int32_t group_num_per_head = head_dim / 16;
    const int32_t partial_q_buffer_stride = q_heads_per_kv * head_dim;
    const int32_t output_buffer_stride = q_head_num * head_dim;
    for (int32_t token_idx = 0; token_idx < actual_q_token_num; ++token_idx) {
      float* __restrict__ curr_partial_output_buffer_iter =
          curr_partial_output_buffer;
      query_t* __restrict__ curr_output_buffer_iter = curr_output_buffer;
      for (int32_t head_idx = 0; head_idx < q_heads_per_kv; ++head_idx) {
        vec_op::FP32Vec16 inv_sum_scale_vec(v_scale / *curr_sum_buffer);

        for (int32_t i = 0; i < group_num_per_head; ++i) {
          vec_op::FP32Vec16 vec(curr_partial_output_buffer_iter);
          // divide the final sum val of softmax here
          vec = inv_sum_scale_vec * vec;

          // cast to query type
          output_vec_t output_vec(vec);
          output_vec.save(curr_output_buffer_iter);

          // update
          curr_partial_output_buffer_iter += 16;
          curr_output_buffer_iter += 16;
        }

        // update
        curr_sum_buffer += 1;
      }

      // update
      curr_partial_output_buffer += partial_q_buffer_stride;
      curr_output_buffer += output_buffer_stride;
    }
  }
};

}  // namespace cpu_attention

#endif
