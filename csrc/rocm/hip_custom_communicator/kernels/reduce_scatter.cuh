#pragma once

// Reduce-scatter stages: each rank sums its slice of every rank's input.

#include "common.cuh"

namespace hip_custom_communicator {

enum class ReduceScatterAlgorithm {
  flat_pull,
  flat_push,
  rows_pull,
  columns_pull,
  columns_push
};

namespace internal {

DINLINE int flat_share_len(int r, int slice, int64_t total) {
  return static_cast<int>(
      max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
}

template <int THREADS_PER_BLOCK, int WAVE_SIZE, int SLICE_WAVES>
constexpr int columns_threads_n() {
  for (int t = SLICE_WAVES < THREADS_PER_BLOCK ? SLICE_WAVES
                                               : THREADS_PER_BLOCK;
       t < THREADS_PER_BLOCK; t += WAVE_SIZE)
    if (THREADS_PER_BLOCK % t == 0) return t;
  return THREADS_PER_BLOCK;
}

template <bool QUANTIZE_INT4, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL, typename DTYPE>
DINLINE void reduce_scatter_flat_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  const int64_t total = int64_t{input_size_m} * input_size_n;
  const int64_t units = TAIL ? (total + kUnit - 1) / kUnit : total / kUnit;
  const int slice = static_cast<int>((units + WORLD - 1) / WORLD * kUnit);
  if constexpr (QUANTIZE_INT4) {
    constexpr Layout kInput =
        CONTIGUOUS ? Layout::group_run : Layout::group_flat;

    const int groups_all = slice / kInt4Group;
    const int grid = static_cast<int>(gridDim.x);
    const int per_block = (groups_all + grid - 1) / grid;
    const int g_first = static_cast<int>(blockIdx.x) * per_block;
    const int g_end = min(g_first + per_block, groups_all);
    constexpr int kRankLanes = THREADS_PER_BLOCK / WORLD;
    static_assert(kRankLanes * WORLD == THREADS_PER_BLOCK,
                  "a block is a row of threads a rank");
    const int r = (rank + static_cast<int>(threadIdx.x) / kRankLanes) % WORLD;
    const int len = flat_share_len(r, slice, total);

    if (TAIL ? len != 0 : len / kInt4Group != 0)
      for (int g0 = g_first; g0 < g_end; g0 += kRankLanes) {
        DTYPE x[1][kInt4GroupPacks][kPackElems];
        load<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
            &input_ptr, 1, 0, r, slice, input_stride_m, input_stride_n, WORLD,
            input_size_m, input_size_n, input_size_m, len, x, 0,
            g0 * kInt4Group, 1);

        uint32_t payload[1][kInt4GroupPacks][2];
        quantize(x[0], payload[0]);
        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, true,
              Layout::group_run, TAIL>(
            payload, staging_ptr_array, staging_ptr_array_size, rank, r, slice,
            staging_stride_m, staging_stride_n, WORLD, staging_size_m,
            staging_size_n, 1,
            TAIL ? min(g_end * kInt4Group, len)
                 : min(g_end, len / kInt4Group) * kInt4Group,
            0, g0 * kInt4Group, 1);
      }

    char* staged[WORLD];
#pragma unroll
    for (int i = 0; i < WORLD; ++i)
      staged[i] =
          load(staging_ptr_array, staging_ptr_array_size, (rank + i) % WORLD);
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::launched, 0);

    const int own = flat_share_len(rank, slice, total);
    if (TAIL ? own != 0 : own / kInt4Group != 0)
      for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
        uint32_t words[WORLD][1][kInt4GroupPacks][2];
#pragma unroll
        for (int i = 0; i < WORLD; ++i)
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run,
               TAIL>(&staged[i], 1, 0, rank, slice, staging_stride_m,
                     staging_stride_n, WORLD, staging_size_m, staging_size_n, 1,
                     own, words[i], 0, g0 * kInt4Group, 1);

        float sum[1][kInt4GroupPacks][kPackElems];
        dequantize<DTYPE>(words, sum);
        store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run,
              TAIL>(sum, partials_ptr_array, partials_ptr_array_size, rank, 0,
                    0, partials_stride_m, partials_stride_n, 1, partials_size_m,
                    partials_size_n, 1,
                    TAIL ? min(g_end * kInt4Group, own)
                         : min(g_end, own / kInt4Group) * kInt4Group,
                    0, g0 * kInt4Group, 1);
      }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::visible, 1);
  } else {
    constexpr Layout kInput = CONTIGUOUS ? Layout::run : Layout::flat;
    constexpr int kLanes = THREADS_PER_BLOCK / WORLD;

    constexpr int kChunkN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
    constexpr int kPacks =
        (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
    static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                  "a block is a group of threads a rank");

    constexpr bool kWhole = kPacks * kPackElems * kLanes == kChunkN;
    const int chunks = (slice + kChunkN - 1) / kChunkN;

    const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
    constexpr bool kWaves = kLanes % WAVE_SIZE == 0;
    [[maybe_unused]] const DTYPE* input_entry = nullptr;
    if constexpr (kWaves)
      input_entry = load(input_ptr_array, input_ptr_array_size, w);
    char* const partials_entry =
        load(partials_ptr_array, partials_ptr_array_size, rank);

    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::started, 0);

    const int len = flat_share_len(rank, slice, total);
    if (len != 0) {
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        const int end = kWhole ? len : min(len, chunk * kChunkN + kChunkN);
        DTYPE got[1][kPacks][kPackElems];
        if constexpr (kWaves)
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
              &input_entry, 1, 0, rank, slice, input_ptr_array_stride_m,
              input_ptr_array_stride_n, WORLD, input_size_m, input_size_n,
              input_size_m, end, got, 0, chunk * kChunkN, 1);
        else
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
              input_ptr_array, input_ptr_array_size, w, rank, slice,
              input_ptr_array_stride_m, input_ptr_array_stride_n, WORLD,
              input_size_m, input_size_n, input_size_m, end, got, 0,
              chunk * kChunkN, 1);
        DTYPE rounded[1][kPacks][kPackElems];
        reduce<WORLD, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::m>(got[0],
                                                                   rounded[0]);
        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            rounded, &partials_entry, 1, 0, 0, 0, 0, 1, 1, 1, len, 1, end, 0,
            chunk * kChunkN, 1);
      }
    }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::visible, 1);
  }
}

template <bool QUANTIZE_INT4, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL, typename DTYPE>
DINLINE void reduce_scatter_flat_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  constexpr int kUnit = QUANTIZE_INT4 ? kInt4Group : kPackElems;
  const int64_t total = int64_t{input_size_m} * input_size_n;
  const int64_t units = TAIL ? (total + kUnit - 1) / kUnit : total / kUnit;
  const int slice = static_cast<int>((units + WORLD - 1) / WORLD * kUnit);
  if constexpr (QUANTIZE_INT4) {
    constexpr Layout kInput =
        CONTIGUOUS ? Layout::group_run : Layout::group_flat;

    const int groups_all = slice / kInt4Group;
    const int grid = static_cast<int>(gridDim.x);
    const int per_block = (groups_all + grid - 1) / grid;
    const int g_first = static_cast<int>(blockIdx.x) * per_block;
    const int g_end = min(g_first + per_block, groups_all);
    constexpr int kRankLanes = THREADS_PER_BLOCK / WORLD;
    static_assert(kRankLanes * WORLD == THREADS_PER_BLOCK,
                  "a block is a row of threads a rank");
    const int r = (rank + static_cast<int>(threadIdx.x) / kRankLanes) % WORLD;
    const int len = flat_share_len(r, slice, total);

    if (TAIL ? len != 0 : len / kInt4Group != 0)
      for (int g0 = g_first; g0 < g_end; g0 += kRankLanes) {
        DTYPE x[1][kInt4GroupPacks][kPackElems];
        load<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
            &input_ptr, 1, 0, r, slice, input_stride_m, input_stride_n, WORLD,
            input_size_m, input_size_n, input_size_m, len, x, 0,
            g0 * kInt4Group, 1);

        uint32_t payload[1][kInt4GroupPacks][2];
        quantize(x[0], payload[0]);
        send<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, true,
             Layout::group_run, TAIL>(
            payload, staging_ptr_array, staging_ptr_array_size, r, rank, slice,
            staging_stride_m, staging_stride_n, WORLD, staging_size_m,
            staging_size_n, 1,
            TAIL ? min(g_end * kInt4Group, len)
                 : min(g_end, len / kInt4Group) * kInt4Group,
            0, g0 * kInt4Group, 1);
      }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::launched, 0);

    const int own = flat_share_len(rank, slice, total);
    if (TAIL ? own != 0 : own / kInt4Group != 0)
      for (int g0 = g_first; g0 < g_end; g0 += THREADS_PER_BLOCK) {
        uint32_t words[WORLD][1][kInt4GroupPacks][2];
        load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run,
             TAIL>(staging_ptr_array, staging_ptr_array_size, rank, rank, slice,
                   staging_stride_m, staging_stride_n, WORLD, staging_size_m,
                   staging_size_n, 1, own, words, 0, g0 * kInt4Group, 1);

        float sum[1][kInt4GroupPacks][kPackElems];
        dequantize<DTYPE>(words, sum);
        store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::group_run,
              TAIL>(sum, partials_ptr_array, partials_ptr_array_size, rank, 0,
                    0, partials_stride_m, partials_stride_n, 1, partials_size_m,
                    partials_size_n, 1,
                    TAIL ? min(g_end * kInt4Group, own)
                         : min(g_end, own / kInt4Group) * kInt4Group,
                    0, g0 * kInt4Group, 1);
      }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::visible, 1);
  } else {
    constexpr Layout kInput = CONTIGUOUS ? Layout::run : Layout::flat;
    constexpr int kLanes = THREADS_PER_BLOCK / WORLD;

    constexpr int kChunkN = (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
    constexpr int kPacks =
        (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
    static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                  "a block is a group of threads a rank");

    constexpr bool kWhole = kPacks * kPackElems * kLanes == kChunkN;
    const int chunks = (slice + kChunkN - 1) / kChunkN;

    const int w = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
    char* const partials_entry =
        load(partials_ptr_array, partials_ptr_array_size, rank);

    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
      const int len = flat_share_len(w, slice, total);
      if (len == 0) continue;

      const int end = kWhole ? len : min(len, chunk * kChunkN + kChunkN);
      DTYPE part[1][kPacks][kPackElems];
      load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kInput, TAIL>(
          &input_ptr, 1, 0, w, slice, input_stride_m, input_stride_n, WORLD,
          input_size_m, input_size_n, input_size_m, end, part, 0,
          chunk * kChunkN, 1);
      send<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::run>(
          part, staging_ptr_array, staging_ptr_array_size, w, rank,
          staging_stride_m, staging_stride_m, staging_stride_n, WORLD,
          staging_size_m, staging_size_n, 1, end, 0, chunk * kChunkN, 1);
    }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::launched, 0);

    const int len = flat_share_len(rank, slice, total);
    if (len != 0) {
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        const int end = kWhole ? len : min(len, chunk * kChunkN + kChunkN);
        DTYPE got[1][kPacks][kPackElems];
        load<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::run>(
            staging_ptr_array, staging_ptr_array_size, rank, w,
            staging_stride_m, staging_stride_m, staging_stride_n, WORLD,
            staging_size_m, staging_size_n, 1, end, got, 0, chunk * kChunkN, 1);
        DTYPE rounded[1][kPacks][kPackElems];
        reduce<WORLD, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::m>(got[0],
                                                                   rounded[0]);
        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            rounded, &partials_entry, 1, 0, 0, 0, 0, 1, 1, 1, len, 1, end, 0,
            chunk * kChunkN, 1);
      }
    }
    barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                   signals_stride_parity, signals_stride_block,
                   signals_stride_writer, signals_size_parity,
                   signals_size_block, signals_size_writer, status_ptr,
                   status_stride_field, status_size_field, rank, timeout_ticks,
                   Until::visible, 1);
  }
}

template <bool QUANTIZE_INT4, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL, typename DTYPE>
DINLINE void reduce_scatter_columns_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  const int packs = input_size_n / kPackElems;

  constexpr int kSliceUnit = TAIL ? kInt4GroupPacks : 1;
  const int slice =
      ((packs + kSliceUnit - 1) / kSliceUnit + WORLD - 1) / WORLD * kSliceUnit;
  const int col0 = min(rank * slice, packs);
  const int own_packs = max(0, min(slice, packs - col0));
  const int end = (col0 + own_packs) * kPackElems;

  if constexpr (QUANTIZE_INT4) {
    constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
    const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
    for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
      for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * kPackElems) {
        DTYPE x[kTiles][TILE_M][1][kPackElems];
#pragma unroll
        for (int s = 0; s < kTiles; ++s)
          load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false,
               Layout::tile>(
              &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
              input_size_m, input_size_n, input_size_m, input_size_n, x[s],
              (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
        for (int s = 0; s < kTiles; ++s)
          store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
                Layout::tile, TAIL>(
              x[s], staging_ptr_array, staging_ptr_array_size, rank, 0, 0,
              staging_stride_m, staging_stride_n, 1, staging_size_m,
              staging_size_n, input_size_m, input_size_n,
              (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
      }
  }

  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);
  constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
  static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                "a block is a row of threads a rank");
  using Summed = std::conditional_t<QUANTIZE_INT4, float, DTYPE>;
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    for (int row = t * TILE_M; row < min(t * TILE_M + TILE_M, input_size_m);
         ++row)
      for (int c = col0 * kPackElems; c < end; c += kLanes * kPackElems) {
        const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
        DTYPE got[1][1][kPackElems];
        if constexpr (!QUANTIZE_INT4)
          recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
              input_ptr_array, input_ptr_array_size, r, 0, 0,
              input_ptr_array_stride_m, input_ptr_array_stride_n, 1,
              input_size_m, input_size_n, input_size_m, end, got, row, c, 0);
        else
          recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Layout::tile, TAIL>(staging_ptr_array, staging_ptr_array_size, r,
                                   0, 0, staging_stride_m, staging_stride_n, 1,
                                   staging_size_m, staging_size_n, input_size_m,
                                   end, got, row, c, 0);

        Summed rounded[1][1][kPackElems];
        reduce<WORLD, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::m>(got[0],
                                                                   rounded[0]);

        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
              Layout::tile, TAIL>(
            rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, row + 1, end, row, c, 1);
      }
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::visible, 1);
}

template <bool QUANTIZE_INT4, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL, typename DTYPE>
DINLINE void reduce_scatter_rows_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  const int slice = (input_size_m + WORLD - 1) / WORLD;
  const int row0 = min(rank * slice, input_size_m);
  const int row_end = min(row0 + slice, input_size_m);
  const int end = input_size_n;

  if constexpr (QUANTIZE_INT4) {
    constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
    const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
    for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
      for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * kPackElems) {
        DTYPE x[kTiles][TILE_M][1][kPackElems];
#pragma unroll
        for (int s = 0; s < kTiles; ++s)
          load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false,
               Layout::tile>(
              &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
              input_size_m, input_size_n, input_size_m, input_size_n, x[s],
              (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
        for (int s = 0; s < kTiles; ++s)
          store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
                Layout::tile, TAIL>(
              x[s], staging_ptr_array, staging_ptr_array_size, rank, 0, 0,
              staging_stride_m, staging_stride_n, 1, staging_size_m,
              staging_size_n, input_size_m, input_size_n,
              (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
      }
  }

  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);
  constexpr int kLanes = THREADS_PER_BLOCK / WORLD;
  static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                "a block is a row of threads a rank");
  using Summed = std::conditional_t<QUANTIZE_INT4, float, DTYPE>;

  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    for (int row = max(t * TILE_M, row0);
         row < min(t * TILE_M + TILE_M, row_end); ++row)
      for (int c = 0; c < end; c += kLanes * kPackElems) {
        const int r = (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
        DTYPE got[1][1][kPackElems];
        if constexpr (!QUANTIZE_INT4)
          recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
              input_ptr_array, input_ptr_array_size, r, 0, 0,
              input_ptr_array_stride_m, input_ptr_array_stride_n, 1,
              input_size_m, input_size_n, input_size_m, end, got, row, c, 0);
        else
          recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Layout::tile, TAIL>(staging_ptr_array, staging_ptr_array_size, r,
                                   0, 0, staging_stride_m, staging_stride_n, 1,
                                   staging_size_m, staging_size_n, input_size_m,
                                   end, got, row, c, 0);

        Summed rounded[1][1][kPackElems];
        reduce<WORLD, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::m>(got[0],
                                                                   rounded[0]);

        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
              Layout::tile, TAIL>(
            rounded, partials_ptr_array, partials_ptr_array_size, rank, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, row + 1, end, row, c, 1);
      }
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::visible, 1);
}

template <bool QUANTIZE_INT4, int WORLD, int TILE_M, int TILE_N,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CONTIGUOUS,
          bool TAIL, typename DTYPE>
DINLINE void reduce_scatter_columns_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;

  const int packs = input_size_n / kPackElems;

  constexpr int kSliceUnit = TAIL ? kInt4GroupPacks : 1;
  const int slice =
      ((packs + kSliceUnit - 1) / kSliceUnit + WORLD - 1) / WORLD * kSliceUnit;
  const int col0 = min(rank * slice, packs);
  const int own_packs = max(0, min(slice, packs - col0));

  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;

  if constexpr (QUANTIZE_INT4) {
    constexpr int kTiles = TILE_M < 4 ? 4 / TILE_M : 1;
    const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
    for (int t = blockIdx.x; t < tiles; t += kTiles * gridDim.x)
      for (int c = 0; c < input_size_n; c += THREADS_PER_BLOCK * kPackElems) {
        DTYPE x[kTiles][TILE_M][1][kPackElems];
#pragma unroll
        for (int s = 0; s < kTiles; ++s)
          load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false,
               Layout::tile>(
              &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1,
              input_size_m, input_size_n, input_size_m, input_size_n, x[s],
              (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
#pragma unroll
        for (int s = 0; s < kTiles; ++s)
          store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
                Layout::tile, TAIL>(
              x[s], staging_ptr_array, staging_ptr_array_size, rank, 0, 0,
              staging_stride_m, staging_stride_n, 1, staging_size_m,
              staging_size_n, input_size_m, input_size_n,
              (t + s * static_cast<int>(gridDim.x)) * TILE_M, c, 1);
      }
  }

  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);
  constexpr int kSliceWaves =
      (TILE_N / WORLD / kPackElems + WAVE_SIZE - 1) / WAVE_SIZE * WAVE_SIZE;

  constexpr int kThreadsN =
      columns_threads_n<THREADS_PER_BLOCK, WAVE_SIZE, kSliceWaves>();
  constexpr int kThreadsM = THREADS_PER_BLOCK / kThreadsN;
  using Summed = std::conditional_t<QUANTIZE_INT4, float, DTYPE>;
  const int offs_n = col0 * kPackElems;
  const int size_n = own_packs * kPackElems;
  if (size_n > 0)
    for (int t = blockIdx.x; t < tiles; t += gridDim.x)
      for (int q = 0; q < TILE_M; q += kThreadsM)
        for (int c = offs_n; c < offs_n + size_n; c += kThreadsN * kPackElems) {
          const int row = t * TILE_M + q;
          const int tile_end = min(input_size_m, t * TILE_M + TILE_M);
          Summed rounded[1][1][kPackElems];
          if constexpr (!QUANTIZE_INT4) {
            DTYPE peers[WORLD][1][1][kPackElems];
            for (int r = 0; r < WORLD; ++r)
              recv<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE, false,
                   Layout::tile>(input_ptr_array, input_ptr_array_size, r, 0, 0,
                                 input_ptr_array_stride_m,
                                 input_ptr_array_stride_n, 1, input_size_m,
                                 input_size_n, tile_end, offs_n + size_n,
                                 peers[r], row, c, 1);

#pragma unroll
            for (int j = 0; j < kPackElems; ++j) {
              float acc = static_cast<float>(peers[0][0][0][j]);
#pragma unroll
              for (int r = 1; r < WORLD; ++r)
                acc += static_cast<float>(peers[r][0][0][j]);
              rounded[0][0][j] = static_cast<Summed>(acc);
            }
          } else {
            uint32_t raw[WORLD][1][1][2];
            for (int r = 0; r < WORLD; ++r)
              recv<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE,
                   QUANTIZE_INT4, Layout::tile, TAIL>(
                  staging_ptr_array, staging_ptr_array_size, r, 0, 0,
                  staging_stride_m, staging_stride_n, 1, staging_size_m,
                  staging_size_n, tile_end, offs_n + size_n, raw[r], row, c, 1);

            float sum[kPackElems];
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) sum[j] = 0.0f;
            int w = 0;
            if constexpr (kBf16<DTYPE>) {
              const bf16x2 ones = __builtin_bit_cast(bf16x2, 0x3f803f80u);
#pragma unroll
              for (; w + 1 < WORLD; w += 2) {
                float a[kPackElems], b[kPackElems];
                dequantize(raw[w][0][0], a);
                dequantize(raw[w + 1][0][0], b);
#pragma unroll
                for (int j = 0; j < kPackElems; ++j)
                  sum[j] = __builtin_amdgcn_fdot2_f32_bf16(
                      __builtin_convertvector(f32x2{a[j], b[j]}, bf16x2), ones,
                      sum[j], false);
#pragma unroll
                for (int j = 0; j < kPackElems; ++j)
                  asm volatile("" ::"v"(sum[j]));
              }
            }
#pragma unroll
            for (; w < WORLD; ++w) {
              float a[kPackElems];
              dequantize(raw[w][0][0], a);
#pragma unroll
              for (int j = 0; j < kPackElems; ++j)
                sum[j] += static_cast<float>(static_cast<DTYPE>(a[j]));
#pragma unroll
              for (int j = 0; j < kPackElems; ++j)
                asm volatile("" ::"v"(sum[j]));
            }
#pragma unroll
            for (int j = 0; j < kPackElems; ++j) rounded[0][0][j] = sum[j];
          }

          [[maybe_unused]] uint32_t payload[1][1][2];
          if constexpr (QUANTIZE_INT4) {
            quantize(rounded[0], payload[0]);
          }
          for (int r = 0; r < WORLD; ++r) {
            if constexpr (QUANTIZE_INT4)
              send<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE,
                   QUANTIZE_INT4, Layout::tile, TAIL>(
                  payload, partials_ptr_array, partials_ptr_array_size, r, 0, 0,
                  partials_stride_m, partials_stride_n, 1, partials_size_m,
                  partials_size_n, tile_end, offs_n + size_n, row, c, 1);
            else
              send<WORLD, kThreadsM, THREADS_PER_BLOCK, WAVE_SIZE,
                   QUANTIZE_INT4, Layout::tile, TAIL>(
                  rounded, partials_ptr_array, partials_ptr_array_size, r, 0, 0,
                  partials_stride_m, partials_stride_n, 1, partials_size_m,
                  partials_size_n, tile_end, offs_n + size_n, row, c, 1);
          }
        }
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::visible, 1);
}

}  // namespace internal

template <ReduceScatterAlgorithm ALGORITHM, bool QUANTIZE_INT4, int WORLD,
          int TILE_M, int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL, typename DTYPE>
DINLINE void reduce_scatter(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, uint32_t* const* signals_ptr_array,
    int signals_ptr_array_size, int64_t signals_stride_parity,
    int64_t signals_stride_block, int64_t signals_stride_writer,
    int signals_size_parity, int signals_size_block, int signals_size_writer,
    uint32_t* status_ptr, int64_t status_stride_field, int status_size_field,
    int rank, uint64_t timeout_ticks) {
  if constexpr (ALGORITHM == ReduceScatterAlgorithm::flat_pull)
    internal::reduce_scatter_flat_pull<QUANTIZE_INT4, WORLD, TILE_M, TILE_N,
                                       THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                       CONTIGUOUS, TAIL, DTYPE>(
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
  else if constexpr (ALGORITHM == ReduceScatterAlgorithm::flat_push)
    internal::reduce_scatter_flat_push<QUANTIZE_INT4, WORLD, TILE_M, TILE_N,
                                       THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                       CONTIGUOUS, TAIL, DTYPE>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        staging_ptr_array, staging_ptr_array_size, staging_stride_m,
        staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks);
  else if constexpr (ALGORITHM == ReduceScatterAlgorithm::rows_pull)
    internal::reduce_scatter_rows_pull<QUANTIZE_INT4, WORLD, TILE_M, TILE_N,
                                       THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                       CONTIGUOUS, TAIL, DTYPE>(
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
  else if constexpr (ALGORITHM == ReduceScatterAlgorithm::columns_pull)
    internal::reduce_scatter_columns_pull<QUANTIZE_INT4, WORLD, TILE_M, TILE_N,
                                          THREADS_PER_BLOCK, WAVE_SIZE,
                                          PACK_BYTES, CONTIGUOUS, TAIL, DTYPE>(
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
  else if constexpr (ALGORITHM == ReduceScatterAlgorithm::columns_push)
    internal::reduce_scatter_columns_push<QUANTIZE_INT4, WORLD, TILE_M, TILE_N,
                                          THREADS_PER_BLOCK, WAVE_SIZE,
                                          PACK_BYTES, CONTIGUOUS, TAIL, DTYPE>(
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
}

}  // namespace hip_custom_communicator
