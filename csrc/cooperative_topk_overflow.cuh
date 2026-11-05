/* Cooperative TopK exact overflow recovery and large-tie helpers.
 * Included after the shared-memory layouts inside vllm::cooperative.
 */
#ifndef COOPERATIVE_TOPK_OVERFLOW_CUH_
#define COOPERATIVE_TOPK_OVERFLOW_CUH_

// Records whether the first exact radix digit is already known.
enum class OverflowProbeStatus : uint32_t {
  kFullRescan = 0,
  kFirstDigitReady = 1,
};

struct OverflowProbeState {
  OverflowProbeStatus status;
  uint32_t prefix;
  uint32_t remaining;
};

// Zero and subnormal FP16 bins require the general FP32 radix path.
__device__ __forceinline__ bool coarse_bin_needs_full_fp32_radix(
    uint32_t coarse_bin) {
  const uint16_t key = static_cast<uint16_t>(coarse_bin << (16 - kHistBits));
  const uint16_t bits = (key & 0x8000u) ? static_cast<uint16_t>(key & 0x7FFFu)
                                        : static_cast<uint16_t>(~key);
  return (bits & 0x7C00u) == 0;
}

// Coarse bin of an ordered FP32 key (inverse of convert_to_uint32_v2).
__device__ __forceinline__ uint32_t coarse_bin_of_key(uint32_t ordered) {
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered & 0x7FFFFFFFu) : ~ordered;
  return extract_coarse_bin(__uint_as_float(bits));
}

// Locate the target digit in a 2,048-bin exact-radix histogram.
__device__ __forceinline__ void find_threshold_exact(
    uint32_t* histogram, uint32_t* warp_sum, uint32_t target,
    hist4096::MatchBin* match) {
  static_assert(kExactHistBins == 2 * hist4096::kBlockSize,
                "find_threshold_exact scans two bins per thread");
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % hist4096::kWarpSize;
  const uint32_t warp = tx / hist4096::kWarpSize;
  const uint32_t bin0 = 2 * tx;
  const uint32_t bin1 = bin0 + 1;
  const uint32_t count0 = histogram[bin0];
  const uint32_t count1 = histogram[bin1];
  const uint32_t local = count0 + count1;
  const uint32_t warp_inclusive = hist4096::warp_inclusive_sum(lane, local);
  if (lane == hist4096::kWarpSize - 1) {
    warp_sum[warp] = warp_inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    warp_sum[lane] = hist4096::warp_inclusive_sum(lane, warp_sum[lane]);
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
template <typename InT>
__device__ void build_exact_histogram(const InT* __restrict__ scores,
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
template <bool UseResident, typename InT, typename SmemType, typename Visit>
__device__ __forceinline__ void for_each_partition_score(
    const InT* __restrict__ scores, uint32_t length, SmemType* smem,
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
        visit(offset + local,
              hist4096::to_float(smem->score_buffer[stage][local]));
      }
    }
  } else {
    hist4096::scan_scores<hist4096::kBlockSize, 4>(scores, length, visit);
  }
}

// Bounded non-subnormal coarse bins fit in two ordered-FP32 radix passes.
template <bool UseResident, typename SmemType, typename InT>
__device__ void build_coarse_refine_histogram(
    const InT* __restrict__ scores, uint32_t length, uint32_t coarse_bin,
    uint32_t base_key, uint32_t prefix, uint32_t round, SmemType* smem,
    uint32_t* histogram) {
  const uint32_t shift = round == 0 ? 11 : 0;
  const uint32_t prefix_mask = round == 0 ? 0u : 0xFFFFF800u;
  for_each_partition_score<UseResident>(
      scores, length, smem, [&](uint32_t, float score) {
        if (extract_coarse_bin(score) != coarse_bin) return;
        const uint32_t delta = hist4096::convert_to_uint32_v2(score) - base_key;
        if (round == 0 || (delta & prefix_mask) == prefix) {
          atomicAdd(&histogram[(delta >> shift) & 0x7FFu], 1);
        }
      });
}

// Gather scores above the exact pivot and bounded equal-pivot candidates.
template <bool UseResident, typename SmemType, typename InT>
__device__ void collect_exact_candidates(
    const InT* __restrict__ scores, uint32_t length, uint32_t pivot,
    uint32_t equal_limit, SmemType* smem, uint32_t* above_count,
    uint32_t* equal_count, int32_t* above_indices, int32_t* equal_indices) {
  for_each_partition_score<UseResident>(
      scores, length, smem, [&](uint32_t idx, float score) {
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        if (ordered > pivot) {
          above_indices[atomicAdd(above_count, 1)] = static_cast<int32_t>(idx);
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

// Place each CTA's above-pivot candidates before all equal-pivot candidates.
template <uint32_t CS>
__device__ __forceinline__ ClusterCandidateOffsets
candidate_offsets(uint32_t local_above, uint32_t local_equal) {
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

struct ClusterKeyRange {
  uint32_t lo;
  uint32_t hi;
};

// Cluster-wide min/max of the threshold bin's ordered keys. CTAs without
// in-bin scores pass (~0u, 0). `scratch` is idle shared memory at the same
// offset in every CTA, so ranks can exchange through DSMEM.
template <uint32_t CS>
__device__ __forceinline__ ClusterKeyRange
cluster_key_range(uint32_t lo, uint32_t hi, uint32_t* scratch) {
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % hist4096::kWarpSize;
  const uint32_t warp = tx / hist4096::kWarpSize;
  auto cluster = cooperative_groups::this_cluster();
  uint32_t* warp_lo = scratch;
  uint32_t* warp_hi = warp_lo + hist4096::kNumWarps;
  uint32_t* rank_lo = warp_hi + hist4096::kNumWarps;
  uint32_t* rank_hi = rank_lo + CS;

#pragma unroll
  for (uint32_t offset = hist4096::kWarpSize / 2; offset > 0; offset /= 2) {
    lo = min(lo, __shfl_xor_sync(0xFFFFFFFF, lo, offset));
    hi = max(hi, __shfl_xor_sync(0xFFFFFFFF, hi, offset));
  }
  if (lane == 0) {
    warp_lo[warp] = lo;
    warp_hi[warp] = hi;
  }
  __syncthreads();
  if (warp == 0) {
    lo = warp_lo[lane];
    hi = warp_hi[lane];
#pragma unroll
    for (uint32_t offset = hist4096::kWarpSize / 2; offset > 0; offset /= 2) {
      lo = min(lo, __shfl_xor_sync(0xFFFFFFFF, lo, offset));
      hi = max(hi, __shfl_xor_sync(0xFFFFFFFF, hi, offset));
    }
    if (lane < CS) {
      cluster.map_shared_rank(rank_lo, lane)[blockIdx.y] = lo;
      cluster.map_shared_rank(rank_hi, lane)[blockIdx.y] = hi;
    }
  }
  cluster.sync();

  lo = ~0u;
  hi = 0;
#pragma unroll
  for (uint32_t i = 0; i < CS; ++i) {
    lo = min(lo, rank_lo[i]);
    hi = max(hi, rank_hi[i]);
  }
  return {.lo = lo, .hi = hi};
}

// Build the first exact FP32 radix digit for a bounded coarse bin.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType,
          typename InT>
__device__ __noinline__ OverflowProbeStatus probe_arbitrary_fp32_overflow(
    const InT* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
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

  for (uint32_t bin = tx; bin < kExactHistBins; bin += hist4096::kBlockSize) {
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
    smem->match = {.bin = first_prefix,
                   .above_count = remaining,
                   .equal_count = static_cast<uint32_t>(
                       OverflowProbeStatus::kFirstDigitReady)};
  }
  __syncthreads();
  return OverflowProbeStatus::kFirstDigitReady;
}

// Emit candidates saved in the last radix pass if every CTA's staging fits.
// `stage` holds above-window indices from the front and window candidates from
// the back of `capacity` slots.
template <uint32_t TopK, uint32_t CS, typename SmemType, typename InT>
__device__ bool emit_staged_candidates(
    const InT* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t my_start, uint32_t pivot, uint32_t remaining,
    uint32_t staged_above, uint32_t staged_candidates, SmemType* smem,
    const int32_t* stage, uint32_t capacity) {
  const uint32_t tx = threadIdx.x;
  __shared__ uint32_t rank_staging_ok[CS];
  __shared__ uint32_t staging_ok;
  auto cluster = cooperative_groups::this_cluster();

  if (tx < CS) {
    auto* dst = cluster.map_shared_rank(rank_staging_ok, tx);
    dst[blockIdx.y] = staged_above + staged_candidates <= capacity;
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
  for (uint32_t i = tx; i < staged_candidates; i += hist4096::kBlockSize) {
    const int32_t idx = stage[capacity - 1 - i];
    const uint32_t ordered = hist4096::convert_to_uint32_v2(
        hist4096::to_float(row_input[my_start + idx]));
    if (ordered > pivot) {
      atomicAdd(&smem->counter_gt, 1);
    } else if (ordered == pivot) {
      atomicAdd(&smem->counter_eq, 1);
    }
  }
  __syncthreads();

  // Only `remaining` equal entries are written, which also keeps the packed
  // per-CTA counts within 16 bits.
  const auto offsets = candidate_offsets<CS>(staged_above + smem->counter_gt,
                                             min(smem->counter_eq, remaining));
  for (uint32_t i = tx; i < staged_above; i += hist4096::kBlockSize) {
    row_output[offsets.above + i] = stage[i] + my_start;
  }

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();
  for (uint32_t i = tx; i < staged_candidates; i += hist4096::kBlockSize) {
    const int32_t idx = stage[capacity - 1 - i];
    const uint32_t ordered = hist4096::convert_to_uint32_v2(
        hist4096::to_float(row_input[my_start + idx]));
    if (ordered > pivot) {
      const uint32_t pos = atomicAdd(&smem->counter_gt, 1);
      row_output[offsets.above + staged_above + pos] = idx + my_start;
    } else if (ordered == pivot) {
      const uint32_t pos = atomicAdd(&smem->counter_eq, 1);
      if (offsets.equal + pos < remaining) {
        row_output[offsets.total_above + offsets.equal + pos] = idx + my_start;
      }
    }
  }
  return true;
}

// Write every score above the exact pivot and the first `remaining` equal to
// it.
template <uint32_t CS, bool UseResident, typename SmemType, typename InT>
__device__ void emit_exact_candidates(const InT* __restrict__ row_input,
                                      int32_t* __restrict__ row_output,
                                      uint32_t my_start, uint32_t my_len,
                                      uint32_t pivot, uint32_t remaining,
                                      SmemType* smem, int32_t* s_topk) {
  const uint32_t tx = threadIdx.x;
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();

  int32_t* equal_indices = reinterpret_cast<int32_t*>(smem->tie_buffer);
  collect_exact_candidates<UseResident>(
      row_input + my_start, my_len, pivot, remaining, smem, &smem->counter_gt,
      &smem->counter_eq, s_topk, equal_indices);
  __syncthreads();

  const uint32_t local_above = smem->counter_gt;
  const uint32_t local_equal = min(smem->counter_eq, remaining);
  const auto offsets = candidate_offsets<CS>(local_above, local_equal);
  for (uint32_t i = tx; i < local_above; i += hist4096::kBlockSize) {
    row_output[offsets.above + i] = s_topk[i] + my_start;
  }
  const uint32_t equal_to_write =
      offsets.equal < remaining ? min(local_equal, remaining - offsets.equal)
                                : 0;
  for (uint32_t i = tx; i < equal_to_write; i += hist4096::kBlockSize) {
    row_output[offsets.total_above + offsets.equal + i] =
        equal_indices[i] + my_start;
  }
}

// First full-radix pass for zero/subnormal/infinite threshold bins. Builds the
// round-0 histogram and, when every score in the bin shares one key (e.g. many
// 0.0 or -inf), emits the result directly. Kept out of line so the common
// rescan keeps its register budget.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType,
          typename InT>
__device__ __noinline__ bool unbounded_round0_single_key(
    const InT* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t my_start, uint32_t my_len, uint32_t coarse_bin,
    uint32_t coarse_above, SmemType* smem, int32_t* s_topk) {
  const uint32_t tx = threadIdx.x;
  for (uint32_t bin = tx; bin < kExactHistBins; bin += hist4096::kBlockSize) {
    smem->histogram[bin] = 0;
  }
  __syncthreads();

  uint32_t key_lo = ~0u;
  uint32_t key_hi = 0;
  hist4096::scan_scores<hist4096::kBlockSize, 4>(
      row_input + my_start, my_len, [&](uint32_t, float score) {
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        atomicAdd(&smem->histogram[(ordered >> ExactRadix::shift(0)) &
                                   ExactRadix::digit_mask(0)],
                  1);
        if (extract_coarse_bin(score) == coarse_bin) {
          key_lo = min(key_lo, ordered);
          key_hi = max(key_hi, ordered);
        }
      });
  __syncthreads();

  const auto keys = cluster_key_range<CS>(key_lo, key_hi,
                                          reinterpret_cast<uint32_t*>(s_topk));
  if (keys.lo != keys.hi) {
    return false;
  }
  emit_exact_candidates<CS, UseResident>(row_input, row_output, my_start,
                                         my_len, keys.lo, TopK - coarse_above,
                                         smem, s_topk);
  return true;
}

// Resume exact radix selection from the probe state and write selected indices.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType,
          typename InT>
__device__ __noinline__ void exact_topk_rescan_cluster(
    const InT* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t seq_len, uint32_t my_start, uint32_t my_len, uint32_t coarse_bin,
    uint32_t coarse_above, OverflowProbeState probe, SmemType* smem,
    int32_t* s_topk) {
  const uint32_t tx = threadIdx.x;
  const auto range = hist4096::coarse_refine_range<kHistBits>(coarse_bin);
  const bool bounded =
      range.finite && !coarse_bin_needs_full_fp32_radix(coarse_bin);

  const uint32_t start_round =
      probe.status == OverflowProbeStatus::kFirstDigitReady ? 1 : 0;
  uint32_t prefix = start_round == 0 ? 0 : probe.prefix;
  uint32_t remaining = probe.status == OverflowProbeStatus::kFullRescan
                           ? (bounded ? TopK - coarse_above : TopK)
                           : probe.remaining;
  const uint32_t rounds = bounded ? 2 : ExactRadix::kRounds;
  // Streaming recovery rescans global memory, leaving the TMA staging buffer
  // free to hold far more final-window candidates than s_topk.
  constexpr uint32_t kStageCapacity =
      UseResident
          ? kMaxTopK
          : static_cast<uint32_t>(sizeof(smem->score_buffer) / sizeof(int32_t));
  int32_t* stage =
      UseResident ? s_topk : reinterpret_cast<int32_t*>(smem->score_buffer);
  const bool stage_final_candidates =
      bounded && start_round == 1 && seq_len >= CS * kSizePerStage;
  uint32_t staged_above = 0;
  uint32_t staged_candidates = 0;
  // Unbounded bins start from the round-0 histogram built while checking for
  // a single-key bin; that check emits the result itself when it succeeds.
  if (!bounded && start_round == 0 &&
      unbounded_round0_single_key<TopK, CS, UseResident>(
          row_input, row_output, my_start, my_len, coarse_bin, coarse_above,
          smem, s_topk)) {
    return;
  }
  for (uint32_t round = start_round; round < rounds; ++round) {
    const bool prebuilt = !bounded && round == 0;
    if (!prebuilt) {
      for (uint32_t bin = tx; bin < kExactHistBins;
           bin += hist4096::kBlockSize) {
        smem->histogram[bin] = 0;
      }
      __syncthreads();
    }

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
            row_input + my_start, my_len, smem, [&](uint32_t idx, float score) {
              const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
              if (ordered > interval_upper) {
                const uint32_t pos = atomicAdd(&smem->counter_gt, 1);
                if (pos < kStageCapacity) {
                  stage[pos] = static_cast<int32_t>(idx);
                }
              } else if (ordered >= interval_lower) {
                const uint32_t pos = atomicAdd(&smem->counter_eq, 1);
                if (pos < kStageCapacity) {
                  stage[kStageCapacity - 1 - pos] = static_cast<int32_t>(idx);
                }
                atomicAdd(&smem->histogram[ordered - interval_lower], 1);
              }
            });
        __syncthreads();
        // One slot per key: a negative bin's window can hold neighbouring-bin
        // keys, which stay staged for exact emission but must not be counted.
        for (uint32_t key = tx; key < kExactHistBins;
             key += hist4096::kBlockSize) {
          if (coarse_bin_of_key(interval_lower + key) != coarse_bin) {
            smem->histogram[key] = 0;
          }
        }
        staged_above = smem->counter_gt;
        staged_candidates = smem->counter_eq;
      } else {
        build_coarse_refine_histogram<UseResident>(
            row_input + my_start, my_len, coarse_bin, range.base_key, prefix,
            round, smem, smem->histogram);
      }
    } else if (!prebuilt) {
      build_exact_histogram(row_input + my_start, my_len, prefix, round,
                            smem->histogram);
    }
    __syncthreads();

    dsmem_hist_reduce<CS, kExactHistBins>(smem->histogram);
    find_threshold_exact(smem->histogram, smem->warp_sum, remaining,
                         &smem->match);
    const uint32_t shift =
        bounded ? (round == 0 ? 11 : 0) : ExactRadix::shift(round);
    prefix |= smem->match.bin << shift;
    remaining -= smem->match.above_count;
  }

  const uint32_t pivot = bounded ? range.base_key + prefix : prefix;

  if (stage_final_candidates &&
      emit_staged_candidates<TopK, CS>(
          row_input, row_output, my_start, pivot, remaining, staged_above,
          staged_candidates, smem, stage, kStageCapacity)) {
    return;
  }

  emit_exact_candidates<CS, UseResident>(
      row_input, row_output, my_start, my_len, pivot, remaining, smem, s_topk);
}

// Recover exact TopK from an overflowing threshold bin.
// Keep exceptional-path state behind this call boundary to preserve the
// common kernel's live ranges.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType,
          typename InT>
__device__ __noinline__ void recover_coarse_overflow(
    const InT* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t seq_len, uint32_t my_start, uint32_t my_len, SmemType* smem,
    int32_t* s_topk) {
  const uint32_t coarse_bin = smem->match.bin;
  const uint32_t coarse_above = smem->match.above_count;
  const auto probe_status =
      probe_arbitrary_fp32_overflow<TopK, CS, UseResident>(
          row_input, my_start, my_len, coarse_bin, coarse_above, smem, s_topk);
  const OverflowProbeState probe = {
      .status = probe_status,
      .prefix = smem->match.bin,
      .remaining = smem->match.above_count,
  };

  exact_topk_rescan_cluster<TopK, CS, UseResident>(
      row_input, row_output, seq_len, my_start, my_len, coarse_bin,
      coarse_above, probe, smem, s_topk);
}

#endif  // COOPERATIVE_TOPK_OVERFLOW_CUH_
