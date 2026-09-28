/* Cooperative TopK exact overflow recovery and large-tie helpers.
 * Included after the shared-memory layouts inside vllm::cooperative.
 */
#ifndef COOPERATIVE_TOPK_OVERFLOW_CUH_
#define COOPERATIVE_TOPK_OVERFLOW_CUH_

// Records how far the cheap overflow probe advanced exact selection.
enum class OverflowProbeStatus : uint32_t {
  kFullRescan = 0,
  kKnownPivot = 1,
  kFirstDigitReady = 2,
};

struct OverflowProbeState {
  OverflowProbeStatus status;
  uint32_t pivot_or_prefix;
  uint32_t remaining;
};

struct OverflowProbeSummary {
  uint32_t min_key;
  uint32_t max_key;
  uint32_t flags;
};

__device__ __forceinline__ bool coarse_bin_needs_full_fp32_radix(
    uint32_t coarse_bin) {
  const uint16_t key =
      static_cast<uint16_t>(coarse_bin << (16 - kHistBits));
  const uint16_t bits = (key & 0x8000u)
                            ? static_cast<uint16_t>(key & 0x7FFFu)
                            : static_cast<uint16_t>(~key);
  return (bits & 0x7C00u) == 0;
}

// Reconstruct exact ordered-FP32 boundaries from reduced-precision keys.
__device__ __forceinline__ uint32_t ordered_fp32_from_fp16_key(
    uint16_t key) {
  const uint16_t bits = (key & 0x8000u)
                            ? static_cast<uint16_t>(key & 0x7FFFu)
                            : static_cast<uint16_t>(~key);
  return hist4096::convert_to_uint32_v2(
      __half2float(__ushort_as_half(bits)));
}

__device__ __forceinline__ uint32_t ordered_fp32_from_bf16_key(
    uint32_t key) {
  return (key << 16) | ((key & 0x8000u) ? 0u : 0xFFFFu);
}

// Locate the target digit in a 2,048-bin exact-radix histogram.
__device__ __forceinline__ void find_threshold_exact(
    uint32_t* histogram, uint32_t* warp_sum, uint32_t target,
    hist4096::MatchBin* match) {
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % hist4096::kWarpSize;
  const uint32_t warp = tx / hist4096::kWarpSize;
  const uint32_t bin0 = 2 * tx;
  const uint32_t bin1 = bin0 + 1;
  const uint32_t count0 = histogram[bin0];
  const uint32_t count1 = histogram[bin1];
  const uint32_t local = count0 + count1;
  const uint32_t warp_inclusive =
      hist4096::warp_inclusive_sum(lane, local);
  if (lane == hist4096::kWarpSize - 1) {
    warp_sum[warp] = warp_inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    warp_sum[lane] =
        hist4096::warp_inclusive_sum(lane, warp_sum[lane]);
  }
  __syncthreads();

  const uint32_t total = warp_sum[hist4096::kNumWarps - 1];
  const uint32_t before_warp = warp == 0 ? 0 : warp_sum[warp - 1];
  const uint32_t before_thread = before_warp + warp_inclusive - local;
  uint32_t above = total - before_thread - local;
  if (above < target && above + count1 >= target) {
    *match = {.bin = bin1, .above_count = above, .equal_count = count1};
  } else {
    above += count1;
    if (above < target && above + count0 >= target) {
      *match = {.bin = bin0, .above_count = above, .equal_count = count0};
    }
  }
  __syncthreads();
}

// Build one radix digit when the coarse bin cannot bound the FP32 keys.
__device__ void build_exact_histogram(const float* __restrict__ scores,
                                      uint32_t length, uint32_t prefix,
                                      uint32_t round, uint32_t* histogram) {
  const uint32_t shift = ExactRadix::shift(round);
  const uint32_t digit_mask = ExactRadix::digit_mask(round);
  const uint32_t prefix_mask = ExactRadix::prefix_mask(round);
  hist4096::scan_scores<hist4096::kBlockSize, 4>(
      scores, length, [&](uint32_t, float score) {
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        if ((ordered & prefix_mask) == prefix) {
          atomicAdd(&histogram[(ordered >> shift) & digit_mask], 1);
        }
      });
}

// Reuse resident TMA data when possible; otherwise rescan global memory.
template <bool UseResident, typename SmemType, typename Visit>
__device__ __forceinline__ void for_each_partition_score(
    const float* __restrict__ scores, uint32_t length, SmemType* smem,
    Visit visit) {
  if constexpr (UseResident) {
    const uint32_t tx = threadIdx.x;
    const uint32_t stages = (length + kSizePerStage - 1) / kSizePerStage;
    for (uint32_t stage = 0; stage < stages; ++stage) {
      const uint32_t offset = stage * kSizePerStage;
      const uint32_t size = min(kSizePerStage, length - offset);
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; ++i) {
        const uint32_t local = tx + i * hist4096::kBlockSize;
        if (local >= size) break;
        visit(offset + local, smem->score_buffer[stage][local]);
      }
    }
  } else {
    hist4096::scan_scores<hist4096::kBlockSize, 4>(scores, length, visit);
  }
}

// Bounded non-subnormal coarse bins fit in two ordered-FP32 radix passes.
template <bool UseResident, typename SmemType>
__device__ void build_coarse_refine_histogram(
    const float* __restrict__ scores, uint32_t length, uint32_t coarse_bin,
    uint32_t base_key, uint32_t prefix, uint32_t round, SmemType* smem,
    uint32_t* histogram) {
  const uint32_t shift = round == 0 ? 11 : 0;
  const uint32_t prefix_mask = round == 0 ? 0u : 0xFFFFF800u;
  for_each_partition_score<UseResident>(
      scores, length, smem, [&](uint32_t, float score) {
        if (extract_coarse_bin(score) != coarse_bin) return;
        const uint32_t delta =
            hist4096::convert_to_uint32_v2(score) - base_key;
        if (round == 0 || (delta & prefix_mask) == prefix) {
          atomicAdd(&histogram[(delta >> shift) & 0x7FFu], 1);
        }
      });
}

template <bool UseResident, typename SmemType>
__device__ void collect_exact_candidates(
    const float* __restrict__ scores, uint32_t length, uint32_t pivot,
    uint32_t equal_limit, SmemType* smem, uint32_t* above_count,
    uint32_t* equal_count, int32_t* above_indices, int32_t* equal_indices) {
  for_each_partition_score<UseResident>(
      scores, length, smem, [&](uint32_t idx, float score) {
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        if (ordered > pivot) {
          above_indices[atomicAdd(above_count, 1)] =
              static_cast<int32_t>(idx);
        } else if (ordered == pivot) {
          const uint32_t pos = atomicAdd(equal_count, 1);
          if (pos < equal_limit) {
            equal_indices[pos] = static_cast<int32_t>(idx);
          }
        }
      });
}

struct ClusterCandidateOffsets {
  uint32_t above;
  uint32_t equal;
  uint32_t total_above;
};

template <uint32_t CS>
__device__ __forceinline__ ClusterCandidateOffsets candidate_offsets(
    uint32_t local_above, uint32_t local_equal) {
  constexpr uint32_t kCountBits = 16;
  constexpr uint32_t kCountMask = (1u << kCountBits) - 1;
  const uint32_t tx = threadIdx.x;
  const uint32_t rank = blockIdx.y;
  auto cluster = cooperative_groups::this_cluster();
  __shared__ uint32_t counts[CS];
  __shared__ uint32_t packed_offset;
  __shared__ uint32_t total_above;

  if (tx < CS) {
    auto* dst = cluster.map_shared_rank(counts, tx);
    dst[rank] = (local_equal << kCountBits) | local_above;
  }
  cluster.sync();

  if (tx == 0) {
    uint32_t above = 0;
    uint32_t equal = 0;
    for (uint32_t i = 0; i < CS; ++i) {
      if (i == rank) packed_offset = (equal << kCountBits) | above;
      above += counts[i] & kCountMask;
      equal += counts[i] >> kCountBits;
    }
    total_above = above;
  }
  __syncthreads();

  return {
      .above = packed_offset & kCountMask,
      .equal = packed_offset >> kCountBits,
      .total_above = total_above,
  };
}

// Build the first exact radix digit while probing an overflowing coarse bin.
// The probe either resolves an exact FP16-grid pivot or leaves enough state
// for recovery to resume at the second radix digit.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ OverflowProbeStatus probe_reduced_precision_overflow(
    const float* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
    uint32_t coarse_bin, uint32_t coarse_above, SmemType* smem,
    int32_t* scratch) {
  constexpr uint32_t kBf16Bins = 256;
  constexpr uint32_t kDigitBits = 11;
  constexpr uint32_t kDeltaLimit = 1u << (2 * kDigitBits);
  constexpr uint32_t kValid = 1;
  constexpr uint32_t kHasNonBf16 = 2;
  constexpr uint32_t kHasNonFp16 = 4;
  const uint32_t tx = threadIdx.x;
  const uint32_t rank = blockIdx.y;
  const uint32_t needed = TopK - coarse_above;
  auto cluster = cooperative_groups::this_cluster();
  __shared__ uint32_t rank_min[CS];
  __shared__ uint32_t rank_max[CS];
  __shared__ uint32_t rank_flags[CS];
  __shared__ OverflowProbeSummary summary;
  auto* exact_histogram = reinterpret_cast<uint32_t*>(scratch);

  const auto range = hist4096::coarse_refine_range<kHistBits>(coarse_bin);

  for (uint32_t bin = tx; bin < kBf16Bins;
       bin += hist4096::kBlockSize) {
    smem->histogram[bin] = 0;
  }
  for (uint32_t bin = tx; bin < kExactHistBins;
       bin += hist4096::kBlockSize) {
    exact_histogram[bin] = 0;
  }
  if (tx == 0) {
    smem->warp_sum[0] = 0xFFFFFFFFu;
    smem->warp_sum[1] = 0;
    smem->warp_sum[2] = range.finite;
    smem->warp_sum[3] = 0;
  }
  __syncthreads();

  uint32_t thread_min = 0xFFFFFFFFu;
  uint32_t thread_max = 0;
  uint32_t thread_invalid = 0;
  uint32_t thread_non_bf16 = 0;
  uint32_t thread_non_fp16 = 0;
  const auto visit = [&](uint32_t, float score) {
    const uint32_t bin = extract_coarse_bin(score);
    if (bin != coarse_bin) return;

    const uint32_t raw = __float_as_uint(score);
    const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
    thread_min = min(thread_min, ordered);
    thread_max = max(thread_max, ordered);
    const bool is_bf16 = (raw & 0xFFFFu) == 0;
    if (is_bf16) {
      atomicAdd(&smem->histogram[(ordered >> 16) & 0xFFu], 1);
    } else {
      thread_non_bf16 = 1;
    }
    const __half half_score = __float2half_rn(score);
    thread_non_fp16 |=
        raw ^ __float_as_uint(__half2float(half_score));
    const uint32_t delta = ordered - range.base_key;
    const bool valid = ordered >= range.base_key && delta < kDeltaLimit;
    thread_invalid |= !valid;
    if (valid && !is_bf16) {
      atomicAdd(&exact_histogram[delta >> kDigitBits], 1);
    }
  };

  for_each_partition_score<UseResident>(
      row_input + my_start, my_len, smem, visit);
  atomicMin(&smem->warp_sum[0], thread_min);
  atomicMax(&smem->warp_sum[1], thread_max);
  atomicAnd(&smem->warp_sum[2], thread_invalid == 0);
  atomicOr(&smem->warp_sum[3],
           (thread_non_bf16 ? kHasNonBf16 : 0) |
               (thread_non_fp16 ? kHasNonFp16 : 0));
  __syncthreads();

  if (tx < CS) {
    auto* min_dst = cluster.map_shared_rank(rank_min, tx);
    auto* max_dst = cluster.map_shared_rank(rank_max, tx);
    auto* flags_dst = cluster.map_shared_rank(rank_flags, tx);
    min_dst[rank] = smem->warp_sum[0];
    max_dst[rank] = smem->warp_sum[1];
    flags_dst[rank] = smem->warp_sum[2] |
                      smem->warp_sum[3];
  }
  cluster.sync();

  if (tx == 0) {
    summary = {.min_key = 0xFFFFFFFFu, .max_key = 0, .flags = 0};
    uint32_t valid = kValid;
    uint32_t features = 0;
    for (uint32_t i = 0; i < CS; ++i) {
      summary.min_key = min(summary.min_key, rank_min[i]);
      summary.max_key = max(summary.max_key, rank_max[i]);
      valid &= rank_flags[i] & kValid;
      features |= rank_flags[i] & ~kValid;
    }
    summary.flags = valid | features;
  }
  __syncthreads();

  if (summary.min_key == summary.max_key) {
    if (tx == 0) {
      smem->match = {.bin = summary.min_key,
                     .above_count = needed,
                     .equal_count = static_cast<uint32_t>(
                         OverflowProbeStatus::kKnownPivot)};
    }
    __syncthreads();
    return OverflowProbeStatus::kKnownPivot;
  }

  const uint32_t min_bf16 = summary.min_key >> 16;
  const uint32_t max_bf16 = summary.max_key >> 16;
  if ((summary.flags & kHasNonBf16) == 0 &&
      max_bf16 - min_bf16 < kBf16Bins) {
    dsmem_hist_reduce<CS, kBf16Bins>(smem->histogram);
    if (tx == 0) {
      uint32_t above = 0;
      for (uint32_t key = max_bf16;; --key) {
        const uint32_t count = smem->histogram[key & 0xFFu];
        if (above < needed && above + count >= needed) {
          const uint32_t pivot = ordered_fp32_from_bf16_key(key);
          smem->match = {.bin = pivot,
                         .above_count = needed - above,
                         .equal_count = static_cast<uint32_t>(
                             OverflowProbeStatus::kKnownPivot)};
          break;
        }
        above += count;
        if (key == min_bf16) break;
      }
    }
    __syncthreads();
    return OverflowProbeStatus::kKnownPivot;
  }

  const bool bf16_keys_unambiguous = max_bf16 - min_bf16 < kBf16Bins;
  if ((summary.flags & kValid) != 0 && bf16_keys_unambiguous) {
    for (uint32_t key = min_bf16 + tx; key <= max_bf16;
         key += hist4096::kBlockSize) {
      const uint32_t count = smem->histogram[key & 0xFFu];
      if (count == 0) continue;
      const uint32_t ordered = ordered_fp32_from_bf16_key(key);
      const uint32_t delta = ordered - range.base_key;
      atomicAdd(&exact_histogram[delta >> kDigitBits], count);
    }
    __syncthreads();
    dsmem_hist_reduce<CS, kExactHistBins>(exact_histogram);
    find_threshold_exact(exact_histogram, smem->warp_sum, needed,
                         &smem->match);
  }

  if (tx == 0) {
    uint32_t status = 0;
    uint32_t pivot = 0;
    uint32_t remaining = 0;
    if ((summary.flags & kValid) != 0 && bf16_keys_unambiguous) {
      const uint32_t first_prefix = smem->match.bin << kDigitBits;
      remaining = needed - smem->match.above_count;
      if ((summary.flags & kHasNonFp16) == 0) {
        const uint32_t first_lower = range.base_key + first_prefix;
        const uint32_t first_upper = first_lower + (1u << kDigitBits) - 1;
        for (uint32_t low = 0; low < (1u << (16 - kHistBits)); ++low) {
          const uint16_t half_key =
              static_cast<uint16_t>((coarse_bin << (16 - kHistBits)) | low);
          const uint32_t candidate = ordered_fp32_from_fp16_key(half_key);
          if (candidate >= first_lower && candidate <= first_upper) {
            status = static_cast<uint32_t>(
                OverflowProbeStatus::kKnownPivot);
            pivot = candidate;
            break;
          }
        }
      }
      if (status == 0) {
        status = static_cast<uint32_t>(
            OverflowProbeStatus::kFirstDigitReady);
        pivot = first_prefix;
      }
    }
    smem->match = {
        .bin = pivot, .above_count = remaining, .equal_count = status};
  }
  __syncthreads();
  return static_cast<OverflowProbeStatus>(smem->match.equal_count);
}

// Arbitrary-FP32 overflow path: build the first exact radix digit directly.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ OverflowProbeStatus probe_arbitrary_fp32_overflow(
    const float* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
    uint32_t coarse_bin, uint32_t coarse_above, SmemType* smem,
    int32_t* scratch) {
  constexpr uint32_t kDigitBits = 11;
  const uint32_t tx = threadIdx.x;
  const uint32_t needed = TopK - coarse_above;
  auto* exact_histogram = reinterpret_cast<uint32_t*>(scratch);

  const auto range = hist4096::coarse_refine_range<kHistBits>(coarse_bin);
  if (!range.finite || coarse_bin_needs_full_fp32_radix(coarse_bin)) {
    return OverflowProbeStatus::kFullRescan;
  }

  for (uint32_t bin = tx; bin < kExactHistBins;
       bin += hist4096::kBlockSize) {
    exact_histogram[bin] = 0;
  }
  __syncthreads();

  // The caller excludes bins that do not fit in two radix digits.
  for_each_partition_score<UseResident>(
      row_input + my_start, my_len, smem, [&](uint32_t, float score) {
        if (extract_coarse_bin(score) != coarse_bin) return;
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        const uint32_t delta = ordered - range.base_key;
        atomicAdd(&exact_histogram[delta >> kDigitBits], 1);
      });
  __syncthreads();

  dsmem_hist_reduce<CS, kExactHistBins>(exact_histogram);
  find_threshold_exact(exact_histogram, smem->warp_sum, needed, &smem->match);

  if (tx == 0) {
    const uint32_t first_prefix = smem->match.bin << kDigitBits;
    const uint32_t remaining = needed - smem->match.above_count;
    smem->match = {
        .bin = first_prefix,
        .above_count = remaining,
        .equal_count =
            static_cast<uint32_t>(OverflowProbeStatus::kFirstDigitReady)};
  }
  __syncthreads();
  return OverflowProbeStatus::kFirstDigitReady;
}

// Every CTA reads the same sample, making the choice cluster-uniform without a
// cluster synchronization. A missed arbitrary-FP32 input only takes the slower
// general probe; both paths remain exact.
__device__ __forceinline__ bool prefer_direct_fp32_probe(
    const float* __restrict__ row_input) {
  const uint32_t tx = threadIdx.x;
  __shared__ uint32_t use_direct;
  if (tx < hist4096::kWarpSize) {
    const float score = row_input[tx];
    const uint32_t raw = __float_as_uint(score);
    const bool is_bf16 = (raw & 0xFFFFu) == 0;
    const bool is_fp16 =
        raw == __float_as_uint(__half2float(__float2half_rn(score)));
    const uint32_t arbitrary_mask =
        __ballot_sync(0xFFFFFFFFu, !is_bf16 && !is_fp16);
    const uint32_t unequal_mask =
        __ballot_sync(0xFFFFFFFFu,
                      raw != __float_as_uint(row_input[0]));
    if (tx == 0) {
      use_direct = arbitrary_mask != 0 && unequal_mask != 0;
    }
  }
  __syncthreads();
  return use_direct != 0;
}

template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ OverflowProbeStatus probe_coarse_overflow(
    const float* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
    uint32_t coarse_bin, uint32_t coarse_above, SmemType* smem,
    int32_t* scratch) {
  if (prefer_direct_fp32_probe(row_input)) {
    return probe_arbitrary_fp32_overflow<TopK, CS, UseResident>(
        row_input, my_start, my_len, coarse_bin, coarse_above, smem, scratch);
  }
  const auto status = probe_reduced_precision_overflow<TopK, CS, UseResident>(
      row_input, my_start, my_len, coarse_bin, coarse_above, smem, scratch);
  if (status == OverflowProbeStatus::kFirstDigitReady &&
      coarse_bin_needs_full_fp32_radix(coarse_bin)) {
    return OverflowProbeStatus::kFullRescan;
  }
  return status;
}

template <uint32_t TopK, uint32_t CS, typename SmemType>
__device__ bool emit_staged_candidates(
    const float* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t my_start, uint32_t pivot, uint32_t remaining,
    uint32_t staged_above, uint32_t staged_candidates, SmemType* smem,
    int32_t* scratch) {
  const uint32_t tx = threadIdx.x;
  __shared__ uint32_t rank_staging_ok[CS];
  __shared__ uint32_t staging_ok;
  auto cluster = cooperative_groups::this_cluster();

  if (tx < CS) {
    auto* dst = cluster.map_shared_rank(rank_staging_ok, tx);
    dst[blockIdx.y] = staged_above + staged_candidates <= kMaxTopK;
  }
  cluster.sync();
  if (tx == 0) {
    staging_ok = 1;
    for (uint32_t i = 0; i < CS; ++i) {
      staging_ok &= rank_staging_ok[i];
    }
  }
  __syncthreads();
  if (staging_ok == 0) return false;

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();
  for (uint32_t i = tx; i < staged_candidates;
       i += hist4096::kBlockSize) {
    const int32_t idx = scratch[kMaxTopK - 1 - i];
    const uint32_t ordered = hist4096::convert_to_uint32_v2(
        row_input[my_start + idx]);
    if (ordered > pivot) {
      atomicAdd(&smem->counter_gt, 1);
    } else if (ordered == pivot) {
      atomicAdd(&smem->counter_eq, 1);
    }
  }
  __syncthreads();

  const auto offsets = candidate_offsets<CS>(
      staged_above + smem->counter_gt, smem->counter_eq);
  for (uint32_t i = tx; i < staged_above; i += hist4096::kBlockSize) {
    row_output[offsets.above + i] = scratch[i] + my_start;
  }

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();
  for (uint32_t i = tx; i < staged_candidates;
       i += hist4096::kBlockSize) {
    const int32_t idx = scratch[kMaxTopK - 1 - i];
    const uint32_t ordered = hist4096::convert_to_uint32_v2(
        row_input[my_start + idx]);
    if (ordered > pivot) {
      const uint32_t pos = atomicAdd(&smem->counter_gt, 1);
      row_output[offsets.above + staged_above + pos] = idx + my_start;
    } else if (ordered == pivot) {
      const uint32_t pos = atomicAdd(&smem->counter_eq, 1);
      if (offsets.equal + pos < remaining) {
        row_output[offsets.total_above + offsets.equal + pos] =
            idx + my_start;
      }
    }
  }
  return true;
}

template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ void exact_topk_rescan_cluster(
    const float* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t seq_len, uint32_t my_start, uint32_t my_len, uint32_t coarse_bin,
    uint32_t coarse_above, OverflowProbeState probe, SmemType* smem,
    int32_t* s_topk) {
  const uint32_t tx = threadIdx.x;
  const auto range = hist4096::coarse_refine_range<kHistBits>(coarse_bin);
  const bool bounded =
      range.finite && !coarse_bin_needs_full_fp32_radix(coarse_bin);

  const bool has_known_pivot = probe.status == OverflowProbeStatus::kKnownPivot;
  const uint32_t start_round =
      probe.status == OverflowProbeStatus::kFirstDigitReady ? 1 : 0;
  uint32_t prefix = start_round == 0 ? 0 : probe.pivot_or_prefix;
  uint32_t remaining = probe.status == OverflowProbeStatus::kFullRescan
                           ? (bounded ? TopK - coarse_above : TopK)
                           : probe.remaining;
  const uint32_t rounds = bounded ? 2 : ExactRadix::kRounds;
  const bool stage_final_candidates =
      bounded && start_round == 1 && seq_len >= CS * kSizePerStage;
  uint32_t staged_above = 0;
  uint32_t staged_candidates = 0;
  for (uint32_t round = start_round;
       round < (has_known_pivot ? start_round : rounds); ++round) {
    for (uint32_t bin = tx; bin < kExactHistBins;
         bin += hist4096::kBlockSize) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    if (bounded) {
      if (stage_final_candidates && round == 1) {
        if (tx == 0) {
          smem->counter_gt = 0;
          smem->counter_eq = 0;
        }
        __syncthreads();
        const uint32_t interval_lower = range.base_key + prefix;
        const uint32_t interval_upper = interval_lower + 0x7FFu;
        for_each_partition_score<UseResident>(
            row_input + my_start, my_len, smem,
            [&](uint32_t idx, float score) {
              const uint32_t ordered =
                  hist4096::convert_to_uint32_v2(score);
              if (ordered > interval_upper) {
                const uint32_t pos = atomicAdd(&smem->counter_gt, 1);
                if (pos < kMaxTopK) {
                  s_topk[pos] = static_cast<int32_t>(idx);
                }
              } else if (ordered >= interval_lower) {
                const uint32_t pos = atomicAdd(&smem->counter_eq, 1);
                if (pos < kMaxTopK) {
                  s_topk[kMaxTopK - 1 - pos] = static_cast<int32_t>(idx);
                }
                atomicAdd(&smem->histogram[ordered - interval_lower], 1);
              }
            });
        __syncthreads();
        staged_above = smem->counter_gt;
        staged_candidates = smem->counter_eq;
      } else {
        build_coarse_refine_histogram<UseResident>(
            row_input + my_start, my_len, coarse_bin, range.base_key, prefix,
            round, smem, smem->histogram);
      }
    } else {
      build_exact_histogram(row_input + my_start, my_len, prefix, round,
                            smem->histogram);
    }
    __syncthreads();

    dsmem_hist_reduce<CS, kExactHistBins>(smem->histogram);
    find_threshold_exact(smem->histogram, smem->warp_sum, remaining,
                         &smem->match);
    const uint32_t shift = bounded ? (round == 0 ? 11 : 0)
                                 : ExactRadix::shift(round);
    prefix |= smem->match.bin << shift;
    remaining -= smem->match.above_count;
  }

  const uint32_t pivot = has_known_pivot
                             ? probe.pivot_or_prefix
                             : (bounded ? range.base_key + prefix
                                        : prefix);

  if (stage_final_candidates &&
      emit_staged_candidates<TopK, CS>(
          row_input, row_output, my_start, pivot, remaining, staged_above,
          staged_candidates, smem, s_topk)) {
    return;
  }

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();

  int32_t* equal_indices = reinterpret_cast<int32_t*>(smem->tie_buffer);
  collect_exact_candidates<UseResident>(
      row_input + my_start, my_len, pivot, remaining, smem,
      &smem->counter_gt, &smem->counter_eq, s_topk, equal_indices);
  __syncthreads();

  const uint32_t local_above = smem->counter_gt;
  const uint32_t local_equal = min(smem->counter_eq, remaining);
  const auto offsets = candidate_offsets<CS>(local_above, local_equal);
  for (uint32_t i = tx; i < local_above; i += hist4096::kBlockSize) {
    row_output[offsets.above + i] = s_topk[i] + my_start;
  }
  const uint32_t equal_to_write = offsets.equal < remaining
                                      ? min(local_equal,
                                            remaining - offsets.equal)
                                      : 0;
  for (uint32_t i = tx; i < equal_to_write; i += hist4096::kBlockSize) {
    row_output[offsets.total_above + offsets.equal + i] =
        equal_indices[i] + my_start;
  }
}

// Keep all exceptional-path state behind one call boundary so the healthy
// large kernel retains the same live ranges as the original implementation.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ void recover_coarse_overflow(
    const float* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t seq_len, uint32_t my_start, uint32_t my_len, SmemType* smem,
    int32_t* s_topk) {
  const uint32_t coarse_bin = smem->match.bin;
  const uint32_t coarse_above = smem->match.above_count;
  const auto probe_status = probe_coarse_overflow<TopK, CS, UseResident>(
      row_input, my_start, my_len, coarse_bin, coarse_above, smem, s_topk);
  const OverflowProbeState probe = {
      .status = probe_status,
      .pivot_or_prefix = smem->match.bin,
      .remaining = smem->match.above_count,
  };

  exact_topk_rescan_cluster<TopK, CS, UseResident>(
      row_input, row_output, seq_len, my_start, my_len, coarse_bin, coarse_above,
      probe, smem, s_topk);
}

// Keep the two-candidate-per-thread refinement out of the common kernel body.
// Inlining it increases register pressure even when the coarse bin contains at
// most 1,024 candidates and this branch is never executed.
template <uint32_t TopK>
__device__ __noinline__ void refine_large_coarse_ties(
    const hist4096::Tie* ties, uint32_t num_ties, uint32_t num_above,
    int32_t* output, void* smem) {
  static_assert(TopK <= hist4096::kBlockSize);
  hist4096::tie_handle_large<TopK, kCoarseTieCapacity>(
      ties, num_ties, num_above, output, smem);
}

// Keep the second candidate copy out of the one-candidate-per-thread path.
template <uint32_t TopK>
__device__ __noinline__ void copy_extra_coarse_ties(
    const hist4096::Tie* tie_buffer, uint32_t num_ties,
    uint32_t prefix_equal, uint32_t total_above, uint32_t row_start,
    hist4096::Tie* tie_ws, int32_t* row_output) {
  const uint32_t i = hist4096::kBlockSize + threadIdx.x;
  if (i >= num_ties) {
    return;
  }

  const auto tie = tie_buffer[i];
  const uint32_t output_pos = total_above + prefix_equal + i;
  if (output_pos < TopK) {
    row_output[output_pos] = tie.idx + row_start;
  }
  const uint32_t tie_pos = prefix_equal + i;
  if (tie_pos < kCoarseTieCapacity) {
    tie_ws[tie_pos] = {tie.idx + row_start, tie.score};
  }
}

#endif  // COOPERATIVE_TOPK_OVERFLOW_CUH_
