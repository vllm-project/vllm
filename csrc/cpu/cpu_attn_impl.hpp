#ifndef CPU_ATTN_HPP
#define CPU_ATTN_HPP

#include <type_traits>
#include <cstddef>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#include "cpu/cpu_arch_macros.h"
#include "cpu/utils.hpp"

namespace cpu_attention {
enum class ISA { AMX, VEC, VEC16, NEON };

template <ISA isa, typename scalar_t, int64_t head_dim>
class AttentionImpl {};

struct AttentionWorkItemGroup {
  int32_t req_id;
  int32_t q_token_id_start;
  int32_t q_token_num;
  int32_t kv_split_pos_start;
  int32_t kv_split_pos_end;

  int64_t total_kv_len;
  int32_t split_id;
  int32_t local_split_id;

  AttentionWorkItemGroup(const int32_t req_id, const int32_t q_token_id_start,
                         const int32_t kv_split_pos_start,
                         const int32_t kv_split_pos_end)
      : req_id(req_id),
        q_token_id_start(q_token_id_start),
        q_token_num(0),
        kv_split_pos_start(kv_split_pos_start),
        kv_split_pos_end(kv_split_pos_end),
        total_kv_len(0),
        split_id(-1),
        local_split_id(0) {}

  std::string to_string() const {
    std::stringstream ss;
    ss << '[' << "req_id: " << req_id << ",\n";
    ss << "q_token_id_start: " << q_token_id_start << ",\n";
    ss << "q_token_num: " << q_token_num << ",\n";
    ss << "kv_split_pos_start: " << kv_split_pos_start << ",\n";
    ss << "kv_split_pos_end: " << kv_split_pos_end << ",\n";
    ss << "total_kv_len: " << total_kv_len << ",\n";
    ss << "split_id: " << split_id << ",\n";
    ss << "local_split_id: " << local_split_id << ",\n";
    ss << ']';

    return ss.str();
  }
};

struct ReductionWorkItemGroup {
  int32_t req_id;
  int32_t q_token_id_start;
  int32_t q_token_id_num;
  int32_t split_start_id;
  int32_t split_num;

  ReductionWorkItemGroup(const int32_t req_id, const int32_t q_token_id_start,
                         const int32_t q_token_id_num,
                         const int32_t split_start_id)
      : req_id(req_id),
        q_token_id_start(q_token_id_start),
        q_token_id_num(q_token_id_num),
        split_start_id(split_start_id),
        split_num(0) {}

  std::string to_string() const {
    std::stringstream ss;
    ss << '[' << "req_id: " << req_id << ",\n";
    ss << "q_token_id_start: " << q_token_id_start << ",\n";
    ss << "q_token_id_num: " << q_token_id_num << ",\n";
    ss << "split_start_id: " << split_start_id << ",\n";
    ss << "split_num: " << split_num << ",\n";
    ss << ']';

    return ss.str();
  }
};

struct AttentionMetadata {
  std::atomic_int64_t counter;
  char _padding1[56];
  ISA isa;
  int32_t workitem_group_num;
  int32_t reduction_item_num;
  int32_t reduction_split_num;
  int32_t thread_num;
  int32_t effective_thread_num;  // non-zero item num in workitem_num_per_thread
  int32_t split_kv_q_token_num_threshold;
  int64_t attention_scratchpad_size_per_thread;
  int64_t reduction_scratchpad_size_per_kv_head;
  AttentionWorkItemGroup* workitem_groups_ptr;
  ReductionWorkItemGroup* reduction_items_ptr;
  int32_t cu_workitem_num_per_thread[1025] = {
      0};  // prefix sum of workitem_num_per_thread
  char _padding2[56];

  AttentionMetadata(ISA isa, int32_t workitem_group_num,
                    int32_t reduction_item_num, int32_t reduction_split_num,
                    int32_t split_kv_q_token_num_threshold)
      : isa(isa),
        workitem_group_num(workitem_group_num),
        reduction_item_num(reduction_item_num),
        reduction_split_num(reduction_split_num),
        thread_num(omp_get_max_threads()),
        effective_thread_num(thread_num),
        split_kv_q_token_num_threshold(split_kv_q_token_num_threshold),
        attention_scratchpad_size_per_thread(0),
        reduction_scratchpad_size_per_kv_head(0),
        workitem_groups_ptr(
            (AttentionWorkItemGroup*)((char*)this + sizeof(AttentionMetadata))),
        reduction_items_ptr(
            (ReductionWorkItemGroup*)((char*)this + sizeof(AttentionMetadata) +
                                      workitem_group_num *
                                          sizeof(AttentionWorkItemGroup))),
        counter(0) {
    TORCH_CHECK_LE(thread_num, 1024);
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
      case ISA::VEC:
        ss << "VEC, ";
        break;
      case ISA::VEC16:
        ss << "VEC16, ";
        break;
      case ISA::NEON:
        ss << "NEON, ";
        break;
    }
    ss << "workitem_group_num: " << workitem_group_num
       << ", reduction_item_num: " << reduction_item_num
       << ", reduction_split_num: " << reduction_split_num
       << ", thread_num: " << thread_num
       << ", effective_thread_num: " << effective_thread_num
       << ", attention_scratchpad_size_per_thread: "
       << attention_scratchpad_size_per_thread
       << ", reduction_scratchpad_size_per_kv_head: "
       << reduction_scratchpad_size_per_kv_head << ", workitem groups:\n";
    for (int32_t i = 0; i < workitem_group_num; ++i) {
      ss << (workitem_groups_ptr + i)->to_string() << ",\n";
    }

    ss << "cu_workitem_num_per_thread: [";
    for (int32_t i = 0; i < thread_num + 1; ++i) {
      ss << cu_workitem_num_per_thread[i] << ", ";
    }
    ss << "]\n";

    ss << "reduction items: \n";

    for (int32_t i = 0; i < reduction_item_num; ++i) {
      ss << (reduction_items_ptr + i)->to_string() << ",\n";
    }

    std::printf("%s", ss.str().c_str());
  }
};

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
                attention_metadata.attention_scratchpad_size_per_thread),
        reduction_scratchpad_size_per_kv_head(
            attention_metadata.reduction_scratchpad_size_per_kv_head) {}

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

  // for reduction
  void update(const int32_t kv_head_idx, const int32_t total_split_num,
              const int64_t head_dim, const int64_t q_head_tile_size,
              const int64_t output_buffer_elem_size) {
    int64_t buffer_offset = kv_head_idx * reduction_scratchpad_size_per_kv_head;
    reduce_flag_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_reduce_flag_buffer_size(total_split_num);
    reduce_output_buffer_offset_ = buffer_offset;
    buffer_offset += calcu_reduce_output_buffer_size(
        total_split_num, q_head_tile_size, head_dim, output_buffer_elem_size);
    reduce_max_buffer_offset_ = buffer_offset;
    buffer_offset +=
        calcu_reduce_max_sum_buffer_size(total_split_num, q_head_tile_size);
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
  int64_t reduction_scratchpad_size_per_kv_head;
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
    int32_t left_sliding_window_size;
    int32_t right_sliding_window_size;
    bool casual;
    cpu_attention::ISA isa;
    int32_t max_num_q_per_iter;  // max Q head num can be hold in registers
    int32_t kv_block_alignment;  // context length alignment requirement
    bool enable_kv_split;
  };

  static constexpr int32_t MaxQTileIterNum = 128;

  AttentionScheduler()
      : available_cache_size_(cpu_utils::get_available_l2_size()) {}

  torch::Tensor schedule(const ScheduleInput& input) const {
    const bool casual = input.casual;
    const int32_t thread_num = omp_get_max_threads();
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    const int32_t max_num_q_per_iter = input.max_num_q_per_iter;
    const int32_t kv_len_alignment = input.kv_block_alignment;
    int32_t q_head_per_kv = input.num_heads_q / input.num_heads_kv;
    const bool use_gqa = (max_num_q_per_iter % q_head_per_kv == 0);
    if (!use_gqa) {
      q_head_per_kv = 1;  // fallback to MHA
    }
    const int32_t min_split_kv_len =
        ((max_num_q_per_iter * 4 + kv_len_alignment - 1) / kv_len_alignment) *
        kv_len_alignment;
    const int32_t max_num_q_token_per_iter = max_num_q_per_iter / q_head_per_kv;
    const int64_t default_tile_size = calcu_default_tile_size(
        cache_size, input.head_dim, input.elem_size, input.q_buffer_elem_size,
        input.logits_buffer_elem_size, input.output_buffer_elem_size,
        max_num_q_per_iter, max_num_q_per_iter);
    const int32_t default_tile_token_num = default_tile_size / q_head_per_kv;
    const int32_t split_kv_q_token_num_threshold =
        input.enable_kv_split ? 1 : 0;
    const int32_t left_sliding_window_size = input.left_sliding_window_size;
    const int32_t right_sliding_window_size = input.right_sliding_window_size;
    TORCH_CHECK_LE(split_kv_q_token_num_threshold * q_head_per_kv, 16);

    // get total kv len
    int64_t total_kv_len = 0;
    for (int32_t req_id = 0; req_id < input.num_reqs; ++req_id) {
      const int32_t seq_len = input.seq_lens[req_id];
      const int32_t q_token_num =
          input.query_start_loc[req_id + 1] - input.query_start_loc[req_id];
      const int32_t q_start_pos = (casual ? (seq_len - q_token_num) : 0);
      const int32_t kv_start_pos = 0;
      const int32_t kv_end_pos = seq_len;

      for (int32_t token_id = 0; token_id < q_token_num;
           token_id += max_num_q_token_per_iter) {
        const int32_t q_tile_token_num =
            std::min(max_num_q_token_per_iter, q_token_num - token_id);
        const int32_t q_tile_pos_left = q_start_pos + token_id;
        const int32_t q_tile_pos_right = q_tile_pos_left + q_tile_token_num;
        const auto [kv_tile_pos_left, kv_tile_pos_right] = calcu_kv_tile_pos(
            kv_start_pos, kv_end_pos, q_tile_pos_left, q_tile_pos_right,
            left_sliding_window_size, right_sliding_window_size);
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
        kv_len_alignment * (use_gqa ? input.num_heads_kv : input.num_heads_q);
    std::vector<AttentionWorkItemGroup> workitems;
    std::vector<ReductionWorkItemGroup> reduce_workitems;
    workitems.reserve(1024);
    reduce_workitems.reserve(1024);
    std::vector<int32_t> workitem_num_per_thread(thread_num, 0);

    // split tasks
    int32_t curr_thread_id = 0;
    int64_t remaining_kv_len = kv_len_per_thread;
    int32_t cum_split_num = 0;
    for (int32_t req_id = 0; req_id < input.num_reqs; ++req_id) {
      const int32_t seq_len = input.seq_lens[req_id];
      const int32_t q_token_num =
          input.query_start_loc[req_id + 1] - input.query_start_loc[req_id];
      const int32_t q_start_pos = (casual ? (seq_len - q_token_num) : 0);
      const int32_t kv_start_pos = 0;
      const int32_t kv_end_pos = seq_len;
      int32_t local_split_id = 0;

      AttentionWorkItemGroup curr_workitem(req_id, 0, 0, seq_len);
      for (int32_t token_id = 0; token_id < q_token_num;
           token_id += max_num_q_token_per_iter) {
        const int32_t q_tile_token_num =
            std::min(max_num_q_token_per_iter, q_token_num - token_id);
        const int32_t q_tile_pos_left = q_start_pos + token_id;
        const int32_t q_tile_pos_right = q_tile_pos_left + q_tile_token_num;
        const auto [kv_tile_pos_left, kv_tile_pos_right] = calcu_kv_tile_pos(
            kv_start_pos, kv_end_pos, q_tile_pos_left, q_tile_pos_right,
            left_sliding_window_size, right_sliding_window_size);
        const auto [aligned_kv_tile_pos_left, aligned_kv_tile_pos_right] =
            align_kv_tile_pos(kv_tile_pos_left, kv_tile_pos_right,
                              kv_len_alignment);
        int32_t curr_kv_len =
            aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
        int32_t kv_token_pos_start = aligned_kv_tile_pos_left;

        while (curr_kv_len > 0) {
          if (curr_kv_len <= (remaining_kv_len + min_split_kv_len) ||
              curr_thread_id == (thread_num - 1)) {
            curr_workitem.q_token_num += q_tile_token_num;
            curr_workitem.total_kv_len += curr_kv_len;
            remaining_kv_len -= curr_kv_len;
            curr_kv_len = 0;

            if (remaining_kv_len < 0) {
              // stop to accept more workitems
              remaining_kv_len -= min_split_kv_len;
            }

            if (curr_workitem.kv_split_pos_start != 0) {
              // got a partial kv spilt, need to create a single workitem
              curr_workitem.split_id = cum_split_num;
              curr_workitem.local_split_id = local_split_id;
              workitems.emplace_back(curr_workitem);
              ++workitem_num_per_thread[curr_thread_id];
              ++reduce_workitems.back().split_num;
              ++cum_split_num;

              curr_workitem = AttentionWorkItemGroup(
                  req_id, token_id + max_num_q_token_per_iter, 0, seq_len);
            }

            break;
          }

          if (remaining_kv_len < min_split_kv_len &&
              (curr_workitem.total_kv_len > 0 ||
               workitem_num_per_thread[curr_thread_id] > 0)) {
            // remaining_kv_len is too short, and have allocated workitems, just
            // leave to next thread
            if (curr_workitem.total_kv_len > 0) {
              workitems.emplace_back(curr_workitem);
              ++workitem_num_per_thread[curr_thread_id];
              curr_workitem =
                  AttentionWorkItemGroup(req_id, token_id, 0, seq_len);
            }

            // switch to next thread
            ++curr_thread_id;
            remaining_kv_len = kv_len_per_thread;

            // retry this iteration
            continue;
          }

          // only split tail splits with q_tile_token_num <=
          // split_kv_q_token_num_threshold
          if (token_id + max_num_q_token_per_iter < q_token_num ||
              q_tile_token_num > split_kv_q_token_num_threshold) {
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
                  AttentionWorkItemGroup(req_id, token_id, 0, seq_len);

              // switch to next thread
              ++curr_thread_id;
              remaining_kv_len = kv_len_per_thread;
            }

            curr_workitem.q_token_num += q_tile_token_num;
            curr_workitem.total_kv_len += curr_kv_len;
            remaining_kv_len -= curr_kv_len;
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
            reduce_workitems.emplace_back(ReductionWorkItemGroup(
                req_id, token_id, q_tile_token_num, cum_split_num));
          }

          int32_t spilt_size =
              std::min(std::max(remaining_kv_len, (int64_t)min_split_kv_len),
                       (int64_t)curr_kv_len);
          curr_workitem =
              AttentionWorkItemGroup(req_id, token_id, kv_token_pos_start,
                                     kv_token_pos_start + spilt_size);
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
          curr_workitem = AttentionWorkItemGroup(req_id, token_id,
                                                 kv_token_pos_start, seq_len);

          // switch to next thread
          ++curr_thread_id;
          remaining_kv_len = kv_len_per_thread;
        }
      }

      if (curr_workitem.total_kv_len > 0) {
        // write back curr workitem
        workitems.emplace_back(curr_workitem);
        ++workitem_num_per_thread[curr_thread_id];
      }
    }

    int64_t metadata_tensor_size =
        sizeof(AttentionMetadata) +
        workitems.size() * sizeof(AttentionWorkItemGroup) +
        reduce_workitems.size() * sizeof(ReductionWorkItemGroup);
    auto options =
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    torch::Tensor metadata_tensor =
        torch::empty({metadata_tensor_size}, options);
    AttentionMetadata* metadata_ptr = new (metadata_tensor.data_ptr())
        AttentionMetadata(input.isa, workitems.size(), reduce_workitems.size(),
                          cum_split_num, split_kv_q_token_num_threshold);
    AttentionWorkItemGroup* workitem_groups_ptr =
        metadata_ptr->workitem_groups_ptr;
    ReductionWorkItemGroup* reduction_items_ptr =
        metadata_ptr->reduction_items_ptr;
    std::memcpy(workitem_groups_ptr, workitems.data(),
                workitems.size() * sizeof(AttentionWorkItemGroup));
    std::memcpy(reduction_items_ptr, reduce_workitems.data(),
                reduce_workitems.size() * sizeof(ReductionWorkItemGroup));

    int32_t effective_thread_num = 0;
    for (; effective_thread_num < thread_num; ++effective_thread_num) {
      if (workitem_num_per_thread[effective_thread_num] == 0) {
        break;
      }
    }

    std::memcpy(metadata_ptr->cu_workitem_num_per_thread + 1,
                workitem_num_per_thread.data(),
                workitem_num_per_thread.size() * sizeof(int32_t));
    for (int32_t i = 1; i <= thread_num; ++i) {
      metadata_ptr->cu_workitem_num_per_thread[i] +=
          metadata_ptr->cu_workitem_num_per_thread[i - 1];
    }
    metadata_ptr->effective_thread_num = effective_thread_num;

    {
      // when q_tile_size = max_num_q_per_iter, requires max
      // attention_scratchpad_size
      AttentionScratchPad sc(0, *metadata_ptr, 0x0);
      int64_t n = AttentionScheduler::calcu_tile_size_with_constant_q(
          cache_size, input.head_dim, input.elem_size, input.q_buffer_elem_size,
          input.logits_buffer_elem_size, input.output_buffer_elem_size,
          max_num_q_per_iter, kv_len_alignment, max_num_q_per_iter, true);
      sc.update(input.head_dim, input.q_buffer_elem_size,
                input.logits_buffer_elem_size, input.output_buffer_elem_size,
                max_num_q_per_iter, max_num_q_per_iter, n);
      metadata_ptr->attention_scratchpad_size_per_thread =
          ((sc.get_thread_scratchpad_size() + 63) / 64) * 64;

      sc.update(0, metadata_ptr->reduction_split_num, input.head_dim,
                q_head_per_kv * split_kv_q_token_num_threshold,
                input.output_buffer_elem_size);
      metadata_ptr->reduction_scratchpad_size_per_kv_head =
          ((sc.get_reduction_scratchpad_size() + 63) / 64) * 64;
    }
    int64_t scratchpad_size =
        metadata_ptr->attention_scratchpad_size_per_thread *
            metadata_ptr->thread_num +
        metadata_ptr->reduction_scratchpad_size_per_kv_head *
            (use_gqa ? input.num_heads_kv : input.num_heads_q);
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

  FORCE_INLINE static std::pair<int32_t, int32_t> calcu_kv_tile_pos(
      int32_t kv_left_pos, int32_t kv_right_pos, int32_t q_left_pos,
      int32_t q_right_pos, int32_t sliding_window_left,
      int32_t sliding_window_right) {
    if (sliding_window_left != -1) {
      kv_left_pos = std::max(kv_left_pos, q_left_pos - sliding_window_left);
    }
    if (sliding_window_right != -1) {
      kv_right_pos = std::min(kv_right_pos, q_right_pos + sliding_window_right);
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

 private:
  int64_t available_cache_size_;
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
  c10::BFloat16* s_aux;
  float scale;
  bool causal;
  int32_t sliding_window_left;
  int32_t sliding_window_right;
  float softcap;
};

#define DEFINE_CPU_ATTENTION_PARAMS                                         \
  q_buffer_t *__restrict__ q_heads_buffer,                                  \
      kv_cache_t *__restrict__ k_head_cache_ptr,                            \
      kv_cache_t *__restrict__ v_head_cache_ptr,                            \
      logits_buffer_t *__restrict__ logits_buffer,                          \
      float *__restrict__ partial_q_buffer, float *__restrict__ max_buffer, \
      float *__restrict__ sum_buffer, int32_t *__restrict__ block_table,    \
      const int32_t kv_tile_start_pos, const int32_t kv_tile_end_pos,       \
      const int32_t kv_tile_token_num,                                      \
      const int64_t kv_cache_num_blocks_stride, const int32_t q_head_num,   \
      const int32_t q_token_num, const int32_t q_tile_start_pos,            \
      const int32_t q_heads_per_kv, const int32_t block_size,               \
      const int32_t left_window_size, const int32_t right_window_size,      \
      float scale, const float softcap_scale,                               \
      const float *__restrict__ alibi_slopes, const bool is_first_iter,     \
      const bool use_sink, const bool debug_info

#define CPU_ATTENTION_PARAMS                                                  \
  q_heads_buffer, k_head_cache_ptr, v_head_cache_ptr, logits_buffer,          \
      partial_q_buffer, max_buffer, sum_buffer, block_table,                  \
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

#if !defined(__powerpc__) && !defined(__s390x__)
template <>
struct VecTypeTrait<c10::Half> {
  using vec_t = vec_op::FP16Vec16;
};
#endif

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

  static constexpr int64_t max_q_head_num_per_iter =
      attention_impl_t::MaxQHeadNumPerIteration;
  static constexpr int64_t blocksize_alignment =
      attention_impl_t::BlockSizeAlignment;
  static constexpr int64_t headdim_alignment =
      attention_impl_t::HeadDimAlignment;
  static constexpr int64_t head_dim = attention_impl_t::HeadDim;
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
        // if (debug_info){
        //     print_logits("raw logits", logits_buffer, q_head_num,
        //     kv_tile_token_num, kv_tile_token_num);
        // }

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
                   kv_tile_start_pos, kv_tile_end_pos, q_token_num,
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

        // if (debug_info){
        //     print_logits("softmax logits",
        //     reinterpret_cast<prob_buffer_t*>(logits_buffer), q_head_num,
        //     kv_tile_token_num, kv_tile_token_num * sizeof(logits_buffer_t) /
        //     sizeof(prob_buffer_t));
        //     print_logits("new_max", max_buffer, 1, q_head_num, q_head_num);
        //     print_logits("new_sum", sum_buffer, 1, q_head_num, q_head_num);
        // }
      }

      // compute P@V
      {
        int32_t curr_group_offset =
            start_block_group_offset * v_cache_token_group_stride;
        int32_t curr_group_num_in_block =
            token_group_num_per_block - start_block_group_offset;
        int32_t remaining_group_num = token_group_num;
        int32_t head_dim_group_num = head_dim / headdim_alignment;
        prob_buffer_t* curr_prob_buffer =
            reinterpret_cast<prob_buffer_t*>(logits_buffer);
        int64_t prob_buffer_stride =
            kv_tile_token_num *
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
          curr_prob_buffer += curr_token_num;
          curr_partial_q_buffer = partial_q_buffer;
          accum_c = true;
        }
      }
      //   if (debug_info) {
      //     print_logits("output", partial_q_buffer, q_head_num, head_dim,
      //     head_dim);
      //   }
    }

    void apply_mask(logits_buffer_t* __restrict__ logits_buffer,
                    const int64_t logits_buffer_stride,
                    const int32_t q_tile_start_pos,
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
          return pos;
        }();

        int32_t right_kv_pos = [&]() {
          int32_t pos = kv_tile_end_pos;
          if (sliding_window_right != -1) {
            pos = std::min(pos,
                           std::max(kv_tile_start_pos,
                                    curr_token_pos + sliding_window_right + 1));
          }
          return pos;
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
#endif
      using prob_buffer_vec_t = typename VecTypeTrait<prob_buffer_t>::vec_t;
      static_assert(sizeof(prob_buffer_t) <= sizeof(logits_buffer_t));

      logits_buffer_t* __restrict__ curr_logits_buffer = logits_buffer;
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
          prob_buffer_t* __restrict__ curr_prob_buffer_iter =
              reinterpret_cast<prob_buffer_t*>(curr_logits_buffer);
          for (int32_t j = 0; j < vec_num; ++j) {
            vec_op::FP32Vec16 vec(curr_logits_buffer_iter);
            vec = vec - max_vec;

            // compute exp
#ifdef DEFINE_FAST_EXP
            vec = fast_exp(vec);
            prob_buffer_vec_t output_vec(vec);
            output_vec.save(curr_prob_buffer_iter);
#else
            vec.save(curr_logits_buffer_iter);
            for (int32_t k = 0; k < 16; ++k) {
              curr_logits_buffer_iter[k] = std::exp(curr_logits_buffer_iter[k]);
            }
            vec = vec_op::FP32Vec16(curr_logits_buffer_iter);
#endif

            sum_vec = sum_vec + vec;

            curr_logits_buffer_iter += 16;
            curr_prob_buffer_iter += 16;
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
        curr_partial_q_buffer += head_dim;
      }
    }

    void apply_softcap(logits_buffer_t* __restrict__ logits_buffer,
                       const int64_t logits_buffer_stride, int32_t q_head_num,
                       int32_t kv_tile_token_num, float softcap_scale) {
#ifdef DEFINE_FAST_EXP
      DEFINE_FAST_EXP
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

#ifdef DEFINE_FAST_EXP
          vec = fast_exp(vec);
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
    const int thread_num = omp_get_max_threads();
    TORCH_CHECK_EQ(input->metadata->thread_num, thread_num);
    std::atomic<int32_t> guard_counter(0);
    std::atomic<int32_t>* guard_counter_ptr = &guard_counter;

#pragma omp parallel for schedule(static, 1)
    for (int thread_id = 0; thread_id < thread_num; ++thread_id) {
      AttentionMetadata& metadata = *input->metadata;
      if (metadata.workitem_group_num == 0) {
        continue;
      }

      attention_impl_t attn_impl;

      // general information
      const int32_t q_head_num = input->num_heads;
      const int32_t kv_head_num = input->num_kv_heads;
      const int32_t q_heads_per_kv = q_head_num / kv_head_num;
      const bool use_gqa =
          (max_q_head_num_per_iter % q_heads_per_kv == 0) ? true : false;
      const int32_t actual_kv_head_num = use_gqa ? kv_head_num : q_head_num;
      const int32_t actual_q_heads_per_kv = use_gqa ? q_heads_per_kv : 1;
      TORCH_CHECK_LE(actual_q_heads_per_kv, max_q_head_num_per_iter);
      const int32_t max_q_token_num_per_iter =
          max_q_head_num_per_iter / actual_q_heads_per_kv;
      const int64_t q_token_num_stride = input->query_num_tokens_stride;
      const int64_t q_head_num_stride = input->query_num_heads_stride;
      const int64_t kv_cache_head_num_stride = input->cache_num_kv_heads_stride;
      const int64_t kv_cache_block_num_stride = input->cache_num_blocks_stride;
      const int32_t sliding_window_left = input->sliding_window_left;
      const int32_t sliding_window_right = input->sliding_window_right;
      const int32_t block_size = input->block_size;
      const float scale = input->scale;
      const float softcap_scale = input->softcap;
      const float* alibi_slopes = input->alibi_slopes;
      const c10::BFloat16* s_aux = input->s_aux;

      const bool casual = input->causal;
      int32_t* const block_table = input->block_table;
      const int64_t block_table_stride = input->blt_num_tokens_stride;

      // init buffers
      void* scratchpad_ptr =
          cpu_utils::ScratchPadManager::get_scratchpad_manager()
              ->get_data<void>();
      AttentionScratchPad buffer_manager(thread_id, metadata, scratchpad_ptr);

      const int32_t total_reduction_split_num = metadata.reduction_split_num;
      if (metadata.reduction_split_num > 0) {
        // reset split flag
        for (int32_t head_idx = thread_id; head_idx < actual_kv_head_num;
             head_idx += thread_num) {
          buffer_manager.update(head_idx, total_reduction_split_num, head_dim,
                                0, sizeof(partial_output_buffer_t));
          volatile bool* __restrict__ curr_flag_ptr =
              buffer_manager.get_reduce_flag_buffer();
          for (int32_t split_idx = 0; split_idx < total_reduction_split_num;
               ++split_idx) {
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
      const int32_t default_q_tile_token_num =
          default_tile_size / actual_q_heads_per_kv;

      AttentionWorkItemGroup* const workitem_groups =
          metadata.workitem_groups_ptr;
      const int32_t* cu_workitem_num_per_thread =
          metadata.cu_workitem_num_per_thread;
      ReductionWorkItemGroup* const reduction_items =
          metadata.reduction_items_ptr;

      const int32_t effective_thread_num = metadata.effective_thread_num;
      const int32_t reduction_item_num = metadata.reduction_item_num;
      const int32_t split_kv_q_token_num_threshold =
          metadata.split_kv_q_token_num_threshold;
      const int32_t workitem_groups_counter_num =
          actual_kv_head_num * effective_thread_num;
      const int32_t reduction_items_counter_num =
          actual_kv_head_num * reduction_item_num;
      const int32_t total_counter_num =
          workitem_groups_counter_num + reduction_items_counter_num;

      if (metadata.reduction_split_num > 0) {
        ++(*guard_counter_ptr);
        while (guard_counter_ptr->load() != thread_num) {
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
          // attention task
          // map task_idx to workitem_groups
          const int32_t kv_head_idx = task_idx / effective_thread_num;
          const int32_t thread_offset = task_idx % effective_thread_num;
          AttentionWorkItemGroup* const curr_workitem_groups =
              workitem_groups + cu_workitem_num_per_thread[thread_offset];
          const int32_t curr_workitem_groups_num =
              cu_workitem_num_per_thread[thread_offset + 1] -
              cu_workitem_num_per_thread[thread_offset];

          const int32_t q_head_start_idx = kv_head_idx * actual_q_heads_per_kv;

          for (int32_t workitem_group_idx = 0;
               workitem_group_idx < curr_workitem_groups_num;
               ++workitem_group_idx) {
            AttentionWorkItemGroup* const current_workitem_group =
                &curr_workitem_groups[workitem_group_idx];

            const int32_t current_group_idx = current_workitem_group->req_id;
            const int32_t kv_start_pos =
                current_workitem_group->kv_split_pos_start;
            const int32_t kv_end_pos = current_workitem_group->kv_split_pos_end;
            const int32_t curr_spilt_id = current_workitem_group->split_id;
            const int32_t q_token_id_start =
                current_workitem_group->q_token_id_start;
            const int32_t q_token_num = current_workitem_group->q_token_num;

            // taskgroup general information
            const int32_t q_end = input->query_start_loc[current_group_idx + 1];
            const int32_t q_start = input->query_start_loc[current_group_idx];
            const int32_t seq_len = input->seq_lens[current_group_idx];
            const int32_t q_start_pos =
                (casual ? seq_len - (q_end - q_start) : 0);
            const int32_t block_num = (seq_len + block_size - 1) / block_size;
            // Only apply sink for the first KV split
            bool use_sink = (s_aux != nullptr &&
                             current_workitem_group->local_split_id == 0);

            for (int32_t q_token_offset = 0; q_token_offset < q_token_num;
                 q_token_offset += default_q_tile_token_num) {
              bool first_iter_flag[AttentionScheduler::MaxQTileIterNum];
              for (int32_t i = 0; i < AttentionScheduler::MaxQTileIterNum;
                   ++i) {
                first_iter_flag[i] = true;
              }

              const int32_t q_token_start_idx =
                  q_start + q_token_offset + q_token_id_start;
              const int32_t actual_q_token_num = std::min(
                  default_q_tile_token_num, q_token_num - q_token_offset);
              const int32_t q_head_tile_size =
                  actual_q_token_num * actual_q_heads_per_kv;
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
                      q_tile_end_pos, sliding_window_left,
                      sliding_window_right);
              const auto [rounded_kv_tile_start_pos, rounded_kv_tile_end_pos] =
                  AttentionScheduler::align_kv_tile_pos(
                      kv_tile_start_pos, kv_tile_end_pos, blocksize_alignment);

              int32_t curr_kv_head_idx =
                  use_gqa ? kv_head_idx
                          : (kv_head_idx /
                             q_heads_per_kv);  // for GQA disabled case

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
              const c10::BFloat16* curr_s_aux =
                  (s_aux != nullptr ? s_aux + q_head_start_idx : nullptr);

              // copy the Q tile to q_buffer, the logical layout of q_buffer is
              // [actual_q_token_num, actual_q_heads_per_kv, head_dim]
              {
                attn_impl.copy_q_heads_tile(
                    q_tile_ptr, q_buffer, actual_q_token_num,
                    actual_q_heads_per_kv, q_token_num_stride,
                    q_head_num_stride, scale);
              }

              if (use_sink) {
                alignas(64) float s_aux_fp32[16];
                // All other platforms have BF16Vec16 available
                vec_op::BF16Vec16 vec_bf16(curr_s_aux);
                vec_op::FP32Vec16 vec_fp32(vec_bf16);
                vec_fp32.save(s_aux_fp32);

                float* __restrict__ curr_sum_buffer = sum_buffer;
                float* __restrict__ curr_max_buffer = max_buffer;
                for (int32_t token_idx = 0; token_idx < actual_q_token_num;
                     ++token_idx) {
                  for (int32_t head_idx = 0; head_idx < actual_q_heads_per_kv;
                       ++head_idx) {
                    curr_sum_buffer[head_idx] = 1.0f;
                    curr_max_buffer[head_idx] = s_aux_fp32[head_idx];
                  }

                  curr_sum_buffer += actual_q_heads_per_kv;
                  curr_max_buffer += actual_q_heads_per_kv;
                }
              } else {
                float* __restrict__ curr_sum_buffer = sum_buffer;
                float* __restrict__ curr_max_buffer = max_buffer;
                for (int32_t token_idx = 0; token_idx < actual_q_token_num;
                     ++token_idx) {
                  for (int32_t head_idx = 0; head_idx < actual_q_heads_per_kv;
                       ++head_idx) {
                    curr_sum_buffer[head_idx] = 0.0f;
                    curr_max_buffer[head_idx] =
                        std::numeric_limits<float>::lowest();
                  }

                  curr_sum_buffer += actual_q_heads_per_kv;
                  curr_max_buffer += actual_q_heads_per_kv;
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
                     q_head_tile_token_offset += max_q_token_num_per_iter) {
                  const int32_t q_tile_pos_left =
                      q_tile_start_pos + q_head_tile_token_offset;
                  const int32_t q_tile_token_num =
                      std::min(max_q_token_num_per_iter,
                               actual_q_token_num - q_head_tile_token_offset);
                  const int32_t q_tile_head_offset =
                      q_head_tile_token_offset * actual_q_heads_per_kv;
                  const int32_t q_tile_head_num =
                      q_tile_token_num * actual_q_heads_per_kv;
                  const int32_t q_tile_pos_right =
                      q_tile_pos_left + q_tile_token_num;
                  const auto [actual_kv_tile_pos_left,
                              actual_kv_tile_pos_right] =
                      AttentionScheduler::calcu_kv_tile_pos(
                          kv_tile_pos_left, kv_tile_pos_right, q_tile_pos_left,
                          q_tile_pos_right, sliding_window_left,
                          sliding_window_right);
                  const int32_t q_iter_idx =
                      q_head_tile_token_offset / max_q_token_num_per_iter;

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
                  //   bool debug_info = (
                  //     q_head_start_idx == 4 &&
                  //     (q_token_start_idx + q_head_tile_token_offset) <=
                  //     4
                  //     && (q_token_start_idx + q_head_tile_token_offset +
                  //     q_tile_token_num) > 4
                  //   );
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
                      curr_sum_buffer, curr_block_table,
                      aligned_actual_kv_tile_pos_left,
                      aligned_actual_kv_tile_pos_right, actual_kv_token_num,
                      kv_cache_block_num_stride, q_tile_head_num,
                      q_tile_token_num, q_tile_pos_left, actual_q_heads_per_kv,
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
                               sum_buffer, actual_q_heads_per_kv,
                               actual_q_token_num, q_head_num);
                } else {
                  const int32_t stride =
                      actual_q_heads_per_kv * split_kv_q_token_num_threshold;
                  buffer_manager.update(kv_head_idx, total_reduction_split_num,
                                        head_dim, stride, sizeof(float));
                  volatile bool* split_flag_buffer =
                      buffer_manager.get_reduce_flag_buffer() + curr_spilt_id;
                  float* split_output_buffer =
                      buffer_manager.get_reduce_output_buffer() +
                      curr_spilt_id * stride * head_dim;
                  float* split_max_buffer =
                      buffer_manager.get_reduce_max_buffer() +
                      curr_spilt_id * stride;
                  float* split_sum_buffer =
                      buffer_manager.get_reduce_sum_buffer() +
                      curr_spilt_id * stride;

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
          const int32_t kv_head_idx = task_idx / reduction_item_num;
          const int32_t item_offset = task_idx % reduction_item_num;
          ReductionWorkItemGroup* const curr_workitem_groups =
              reduction_items + item_offset;
          const int32_t curr_output_token_idx =
              curr_workitem_groups->q_token_id_start;
          const int32_t curr_output_token_num =
              curr_workitem_groups->q_token_id_num;
          const int32_t curr_split_id = curr_workitem_groups->split_start_id;
          const int32_t curr_split_num = curr_workitem_groups->split_num;
          const int32_t current_group_idx = curr_workitem_groups->req_id;
          const int32_t curr_output_head_num =
              curr_output_token_num * actual_q_heads_per_kv;

          const int32_t q_start = input->query_start_loc[current_group_idx];
          const int32_t q_token_start_idx = q_start + curr_output_token_idx;
          const int32_t q_head_start_idx = kv_head_idx * actual_q_heads_per_kv;
          size_t output_buffer_offset =
              q_token_start_idx * q_head_num * head_dim +
              q_head_start_idx * head_dim;

          const int32_t stride =
              actual_q_heads_per_kv * split_kv_q_token_num_threshold;
          buffer_manager.update(kv_head_idx, total_reduction_split_num,
                                head_dim, stride, sizeof(float));
          volatile bool* split_flag_buffer =
              buffer_manager.get_reduce_flag_buffer() + curr_split_id;
          float* split_output_buffer =
              buffer_manager.get_reduce_output_buffer() +
              curr_split_id * stride * head_dim;
          float* split_max_buffer =
              buffer_manager.get_reduce_max_buffer() + curr_split_id * stride;
          float* split_sum_buffer =
              buffer_manager.get_reduce_sum_buffer() + curr_split_id * stride;

          reduce_splits(split_output_buffer, split_max_buffer, split_sum_buffer,
                        split_flag_buffer, stride, curr_output_head_num,
                        curr_split_num);
          final_output(
              split_output_buffer,
              reinterpret_cast<query_t*>(input->output) + output_buffer_offset,
              split_sum_buffer, actual_q_heads_per_kv, curr_output_token_num,
              q_head_num);
        }
      }
    }
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
    // restrict curr_head_num <= 16 in the scheduler
    // elems in split_max_buffer, split_sum_buffer are not cache alignment, use
    // local buffers to reduce false-sharing
    alignas(64) float local_max[16];
    alignas(64) float local_sum[16];

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
        vec_op::FP32Vec16 final_max(split_max_buffer);
        final_max.save(local_max);
        vec_op::FP32Vec16 final_sum(split_sum_buffer);
        final_sum.save(local_sum);
      }

      curr_split_output_buffer += head_num_per_split * head_dim;
      curr_split_max_buffer += head_num_per_split;
      curr_split_sum_buffer += head_num_per_split;
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
                    const int32_t actual_q_token_num,
                    const int32_t q_head_num) {
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
        vec_op::FP32Vec16 inv_sum_scale_vec(1.0 / *curr_sum_buffer);

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
