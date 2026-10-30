#pragma once

#include <cstdint>

template<
    typename ValueT_, typename OutIdxT_,
    bool sorted_value_, bool sorted_index_, bool return_value_,
    uint32_t max_topk_,
    uint32_t num_threads_, uint32_t target_occupancy_,
    uint32_t elements_per_round_, uint32_t reconstruct_threshold_, uint32_t tma_buffer_depth_, uint32_t elements_per_segment_ = 512,
    uint32_t cluster_size_ = 1,
    bool page_transform_ = false    // [SGLang] map indices through `TopkSelectArgs::page_table` in the epilogue
>
struct TopkSelectConfig {
    using ValueT = ValueT_;
    using OutIdxT = OutIdxT_;
    static constexpr bool sorted_value = sorted_value_;
    static constexpr bool sorted_index = sorted_index_;
    static constexpr bool return_value = return_value_;
    static constexpr uint32_t max_topk = max_topk_;
    static constexpr uint32_t num_threads = num_threads_;
    static constexpr uint32_t target_occupancy = target_occupancy_; // Number of CTAs per SM
    static constexpr uint32_t elements_per_round = elements_per_round_;
    static constexpr uint32_t reconstruct_threshold = reconstruct_threshold_;
    static constexpr uint32_t tma_buffer_depth = tma_buffer_depth_;
    static constexpr uint32_t elements_per_segment = elements_per_segment_;
    static constexpr uint32_t cluster_size = cluster_size_; // (1 = not a cluster variant)
    static constexpr bool page_transform = page_transform_;
};
