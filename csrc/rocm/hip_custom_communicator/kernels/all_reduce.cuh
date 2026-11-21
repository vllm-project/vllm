#pragma once

// All-reduce kernels, one function per algorithm.

#include "all_gather.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

enum class AllReduceAlgorithm {
  one_shot_pull,
  one_shot_push,
  two_shot_pull,
  two_shot_push,
  one_shot_push_staged,
};

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_one_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;

  if constexpr (QUANTIZE_INT4) {
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
      const int clip_n =
          kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
      DTYPE mine[kRows][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
          &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
          input_size_m, input_size_n, input_size_m, clip_n, mine, offs_m,
          offs_n, 1);
      store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::tile,
            TAIL>(mine, staging_ptr_array, staging_ptr_array_size, rank, 0, 0,
                  input_size_n, 1, 1, input_size_m, input_size_n, input_size_m,
                  clip_n, offs_m, offs_n, 1);
    }
  }

  std::conditional_t<QUANTIZE_INT4, char*, const DTYPE*> entries[WORLD];
#pragma unroll
  for (int r = 0; r < WORLD; ++r) {
    if constexpr (QUANTIZE_INT4)
      entries[r] = load(staging_ptr_array, staging_ptr_array_size, r);
    else
      entries[r] = load(input_ptr_array, input_ptr_array_size, r);
  }

  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_pull, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, TAIL>(
        entries,
        QUANTIZE_INT4 ? int64_t{input_size_n} : input_ptr_array_stride_m,
        QUANTIZE_INT4 ? int64_t{1} : input_ptr_array_stride_n, input_size_m,
        input_size_n, clip_n, offs_m, offs_n, copies);
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
  }

  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_two_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  reduce_scatter<ReduceScatterAlgorithm::flat_pull, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                 CONTIGUOUS, TAIL>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
      input_ptr_array_stride_n, staging_ptr_array, staging_ptr_array_size,
      staging_stride_m, staging_stride_n, staging_size_m, staging_size_n,
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  all_gather<AllGatherAlgorithm::flat, QUANTIZE_INT4, DTYPE, WORLD, TILE_N,
             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CONTIGUOUS, TAIL>(
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, output_ptr,
      output_stride_m, output_stride_n, input_size_m, input_size_n, 0, 0, 0,
      rank);
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_one_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, int32_t* unfenced_epochs_ptr,
    int64_t unfenced_epochs_stride_block, int64_t unfenced_epochs_stride_n,
    int unfenced_epochs_size_block, int unfenced_epochs_size_n,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;

  constexpr int kPackElems =
      QUANTIZE_INT4
          ? traits<DTYPE, PACK_BYTES>::N
          : (PACK_BYTES < kLinePayloadBytes ? PACK_BYTES : kLinePayloadBytes) /
                static_cast<int>(sizeof(DTYPE));

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;

  const int first_row = static_cast<int>(blockIdx.x) * TILE_M;
  if (HIP_CUSTOM_COMMUNICATOR_DEBUG &&
      static_cast<int>(gridDim.x) * TILE_M > unfenced_size_m / 2)
    __builtin_trap();

  int32_t last[kRows][1][kLineBytes / static_cast<int>(sizeof(int32_t))];
  load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
      &unfenced_epochs_ptr, 1, 0, 0, 0, unfenced_epochs_stride_block,
      unfenced_epochs_stride_n, 1, unfenced_epochs_size_block,
      unfenced_epochs_size_n, unfenced_epochs_size_block,
      unfenced_epochs_size_n, last, first_row, 0, 1);
  uint32_t epochs[kRows] = {};
  bool ran = false;

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;

    const int clip_m = min(TILE_M, input_size_m - offs_m);
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE mine[kRows][kPacks][kPackElems];
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
        input_size_n, input_size_m, clip_n, mine, offs_m, offs_n, 1);

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const uint32_t before = ran ? epochs[i]
                              : kThreadsM == 1
                                  ? __builtin_amdgcn_readfirstlane(
                                        static_cast<uint32_t>(last[i][0][0]))
                                  : static_cast<uint32_t>(last[i][0][0]);
      const bool written =
          (kThreadsM == 1 ? i
                          : static_cast<int>(threadIdx.x) / kThreadsN +
                                i * kThreadsM) < clip_m;
      epochs[i] = !written ? before : before + 1u == 0u ? 2u : before + 1u;
    }
    ran = true;
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_push, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, THREADS_PER_BLOCK, WAVE_SIZE>(
        mine, epochs, unfenced_ptr_array, unfenced_ptr_array_size,
        unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        first_row + clip_m, min(kTileN, input_size_n - offs_n), first_row, rank,
        status_ptr, status_stride_field, status_size_field, timeout_ticks,
        copies);
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);

    __syncthreads();
  }

  int32_t next[kRows][1][kLineBytes / static_cast<int>(sizeof(int32_t))] = {};
#pragma unroll
  for (int i = 0; i < kRows; ++i)
    next[i][0][0] = ran ? static_cast<int32_t>(epochs[i]) : last[i][0][0];
  store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
      next, &unfenced_epochs_ptr, 1, 0, 0, 0, unfenced_epochs_stride_block,
      unfenced_epochs_stride_n, 1, unfenced_epochs_size_block,
      unfenced_epochs_size_n, unfenced_epochs_size_block,
      unfenced_epochs_size_n, first_row, 0, 1);
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_one_shot_push_staged(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* staging_ptr_array, int staging_ptr_array_size,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  constexpr int kThreadsM = THREADS_PER_BLOCK % TILE_M == 0 ? TILE_M : 1;
  constexpr int kThreadsN = THREADS_PER_BLOCK / kThreadsM;
  constexpr int kRows = TILE_M / kThreadsM;
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  constexpr int kTileN = (TILE_N + kUnit - 1) / kUnit * kUnit;
  constexpr int kPacks =
      (kTileN + kPackElems * kThreadsN - 1) / (kPackElems * kThreadsN);
  constexpr bool kWhole = kPacks * kPackElems * kThreadsN == kTileN;
  const int tiles_n = (input_size_n + kTileN - 1) / kTileN;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M * tiles_n;

  const int64_t slot = QUANTIZE_INT4
                           ? (int4_payload_bytes(input_size_m, input_size_n) +
                              kInt4ScalesAlign - 1) /
                                 kInt4ScalesAlign * kInt4ScalesAlign
                           : int64_t{input_size_m} * input_size_n;

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE mine[kRows][kPacks][kPackElems];
    load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, input_size_m,
        input_size_n, input_size_m, clip_n, mine, offs_m, offs_n, 1);
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
           Layout::tile, TAIL>(mine, staging_ptr_array, staging_ptr_array_size,
                               r, rank, slot, input_size_n, 1, WORLD,
                               input_size_m, input_size_n, input_size_m, clip_n,
                               offs_m, offs_n, 1);
  }

  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::launched, 0);

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int offs_m = t / tiles_n * TILE_M, offs_n = t % tiles_n * kTileN;
    const int clip_n =
        kWhole ? input_size_n : min(input_size_n, offs_n + kTileN);
    DTYPE copies[WORLD][kRows][kPacks][kPackElems];
    all_gather<AllGatherAlgorithm::tile_staged, QUANTIZE_INT4, DTYPE, WORLD,
               TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, TAIL>(
        staging_ptr_array, staging_ptr_array_size, slot, input_size_m,
        input_size_n, clip_n, offs_m, offs_n, rank, copies);
    DTYPE rounded[kRows][kPacks][kPackElems];
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
#pragma unroll
        for (int j = 0; j < kPackElems; ++j) {
          float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
          for (int r = 1; r < WORLD; ++r)
            acc += static_cast<float>(copies[r][i][k][j]);
          rounded[i][k][j] = static_cast<DTYPE>(acc);
        }
    store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
        rounded, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
        input_size_m, input_size_n, input_size_m, clip_n, offs_m, offs_n, 1);
  }

  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL>
DINLINE void all_reduce_two_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, DTYPE* output_ptr,
    int64_t output_stride_m, int64_t output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks) {
  reduce_scatter<ReduceScatterAlgorithm::flat_push, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                 CONTIGUOUS, TAIL>(
      input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
      static_cast<const DTYPE* const*>(nullptr), 0, 0, 1, staging_ptr_array,
      staging_ptr_array_size, staging_stride_m, staging_stride_n,
      staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);
  all_gather<AllGatherAlgorithm::flat, QUANTIZE_INT4, DTYPE, WORLD, TILE_N,
             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, CONTIGUOUS, TAIL>(
      partials_ptr_array, partials_ptr_array_size, partials_stride_m,
      partials_stride_n, partials_size_m, partials_size_n, output_ptr,
      output_stride_m, output_stride_n, input_size_m, input_size_n, 0, 0, 0,
      rank);
}

template <AllReduceAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int TILE_N, int THREADS_PER_BLOCK,
          int WAVE_SIZE, int PACK_BYTES, int WAVES_PER_EU, bool CONTIGUOUS,
          bool TAIL>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU) all_reduce(
    const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* const* __restrict__ input_ptr_array, int input_ptr_array_size,
    int64_t input_ptr_array_stride_m, int64_t input_ptr_array_stride_n,
    DTYPE* __restrict__ output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, int output_size_m, int output_size_n,
    char* const* __restrict__ partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* __restrict__ staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* __restrict__ unfenced_ptr_array, int unfenced_ptr_array_size,
    int64_t unfenced_stride_slot, int64_t unfenced_stride_m,
    int64_t unfenced_stride_n, int unfenced_size_slot, int unfenced_size_m,
    int unfenced_size_n, int32_t* __restrict__ unfenced_epochs_ptr,
    int64_t unfenced_epochs_stride_block, int64_t unfenced_epochs_stride_n,
    int unfenced_epochs_size_block, int unfenced_epochs_size_n,
    uint32_t* const* __restrict__ signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer,
    uint32_t* __restrict__ status_ptr, int64_t status_stride_field,
    int status_size_field, int rank, uint64_t timeout_ticks) {
  if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_pull)
    all_reduce_one_shot_pull<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, output_ptr, output_stride_m, output_stride_n,
        staging_ptr_array, staging_ptr_array_size, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::two_shot_pull)
    all_reduce_two_shot_pull<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, output_ptr, output_stride_m, output_stride_n,
        partials_ptr_array, partials_ptr_array_size, partials_stride_m,
        partials_stride_n, partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_push)
    all_reduce_one_shot_push<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, unfenced_ptr_array,
        unfenced_ptr_array_size, unfenced_stride_slot, unfenced_stride_m,
        unfenced_stride_n, unfenced_size_slot, unfenced_size_m, unfenced_size_n,
        unfenced_epochs_ptr, unfenced_epochs_stride_block,
        unfenced_epochs_stride_n, unfenced_epochs_size_block,
        unfenced_epochs_size_n, status_ptr, status_stride_field,
        status_size_field, rank, timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::one_shot_push_staged)
    all_reduce_one_shot_push_staged<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                                    THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                    CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, staging_ptr_array,
        staging_ptr_array_size, signals_ptr_array, signals_ptr_array_size,
        signals_stride_parity, signals_stride_block, signals_stride_writer,
        signals_size_parity, signals_size_block, signals_size_writer,
        status_ptr, status_stride_field, status_size_field, rank,
        timeout_ticks);
  else if constexpr (ALGORITHM == AllReduceAlgorithm::two_shot_push)
    all_reduce_two_shot_push<QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N,
                             THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                             CONTIGUOUS, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        output_ptr, output_stride_m, output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
}

}  // namespace hip_custom_communicator
