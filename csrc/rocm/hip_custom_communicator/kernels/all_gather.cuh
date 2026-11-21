#pragma once

// All-gather stages: each rank reads every rank's reduced slice into its
// output.

#include <type_traits>

#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

enum class AllGatherAlgorithm {
  flat,
  rows,
  columns,
  tile_pull,
  tile_push,
  tile_staged
};

template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_N, int THREADS_PER_BLOCK, int WAVE_SIZE,
          int PACK_BYTES, bool CONTIGUOUS, bool TAIL, typename OUTPUT>
DINLINE void all_gather(char* const* partials_ptr_array,
                        int partials_ptr_array_size, int64_t partials_stride_m,
                        int64_t partials_stride_n, int partials_size_m,
                        int partials_size_n, OUTPUT&& output,
                        int64_t output_stride_m, int64_t output_stride_n,
                        int output_size_m, int output_size_n, int offs_m,
                        int offs_n, int slice, int rank) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr bool kInt4 = QUANTIZE_INT4;
  if constexpr (ALGORITHM == AllGatherAlgorithm::flat) {
    DTYPE* const output_ptr = output;
    constexpr int kUnit = kInt4 ? kInt4Group : kPackElems;
    const int64_t total = int64_t{output_size_m} * output_size_n;

    const int64_t units = TAIL ? (total + kUnit - 1) / kUnit : total / kUnit;
    const int slice = static_cast<int>((units + WORLD - 1) / WORLD * kUnit);
    const auto share_len = [&](int r) {
      return static_cast<int>(
          max(int64_t{0}, min(int64_t{slice}, total - int64_t{r} * slice)));
    };
    if constexpr (kInt4) {
      constexpr Layout kOutput =
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
      const int end = TAIL ? min(g_end * kInt4Group, share_len(r))
                           : min(g_end, share_len(r) / kInt4Group) * kInt4Group;
      for (int g0 = g_first; g0 < g_end; g0 += kRankLanes) {
        uint32_t words[1][kInt4GroupPacks][2];
        recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Layout::group_run, TAIL>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, 1, slice, words, 0, g0 * kInt4Group, 1);

        DTYPE x[1][kInt4GroupPacks][kPackElems];
#pragma unroll
        for (int k = 0; k < kInt4GroupPacks; ++k)
          dequantize(words[0][k], x[0][k]);
        store<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, kOutput, TAIL>(
            x, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n,
            WORLD, output_size_m, output_size_n, 1, end, 0, g0 * kInt4Group, 1);
      }
    } else {
      constexpr Layout kOutput = CONTIGUOUS ? Layout::run : Layout::flat;
      constexpr int kLanes = THREADS_PER_BLOCK / WORLD;

      constexpr int kChunkN =
          (TILE_N + kPackElems - 1) / kPackElems * kPackElems;
      constexpr int kPacks =
          (kChunkN + kPackElems * kLanes - 1) / (kPackElems * kLanes);
      static_assert(kLanes * WORLD == THREADS_PER_BLOCK,
                    "a block is a group of threads a rank");

      constexpr bool kWaves = kLanes % WAVE_SIZE == 0;
      constexpr bool kWhole = kPacks * kPackElems * kLanes == kChunkN;
      const int chunks = (slice + kChunkN - 1) / kChunkN;

      const int r =
          kWaves ? __builtin_amdgcn_readfirstlane(
                       (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD)
                 : (rank + static_cast<int>(threadIdx.x) / kLanes) % WORLD;
      for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        DTYPE got[1][kPacks][kPackElems];

        recv<WORLD, WORLD, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            partials_ptr_array, partials_ptr_array_size, r, 0, 0, 0, 1, 1, 1,
            slice, 1, kWhole ? slice : min(slice, chunk * kChunkN + kChunkN),
            got, 0, chunk * kChunkN, 0);

        store<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, kOutput, TAIL>(
            got, &output_ptr, 1, 0, r, slice, output_stride_m, output_stride_n,
            WORLD, output_size_m, output_size_n, 1,
            kWhole ? share_len(r)
                   : min(share_len(r), chunk * kChunkN + kChunkN),
            0, chunk * kChunkN, 1);
      }
    }
  } else if constexpr (ALGORITHM == AllGatherAlgorithm::rows) {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems,
                  "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int owner = min((offs_m + i) / slice, WORLD - 1);
      if constexpr (!kInt4) {
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
          DTYPE pack[1][1][kPackElems];
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Layout::tile, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m,
              partials_size_n, output_size_m, output_size_n, pack, offs_m + i,
              offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[0][0][j];
        }
      } else {
        uint32_t raw[kPacks][1][1][2];
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
          recv<WORLD, 1, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
               Layout::tile, TAIL>(
              partials_ptr_array, partials_ptr_array_size, owner, 0, 0,
              partials_stride_m, partials_stride_n, 1, partials_size_m,
              partials_size_n, output_size_m, output_size_n, raw[k], offs_m + i,
              offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int k = 0; k < kPacks; ++k) dequantize(raw[k][0][0], output[i][k]);
      }
    }
  } else {
    using Tile = std::remove_reference_t<OUTPUT>;
    constexpr int kRows = std::extent_v<Tile, 0>;
    constexpr int kPacks = std::extent_v<Tile, 1>;
    static_assert(std::extent_v<Tile, 2> == kPackElems,
                  "a tile's pack is the dtype's");
    (void)output_stride_m;
    (void)output_stride_n;

    const auto owner_of = [&](int k) {
      const int n = offs_n / kPackElems +
                    static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK +
                    k * THREADS_PER_BLOCK;
      const int col =
          n < output_size_n / kPackElems ? n : output_size_n / kPackElems - 1;
      return min(col / (slice / kPackElems), WORLD - 1);
    };
    if constexpr (!kInt4) {
#pragma unroll
      for (int k = 0; k < kPacks; ++k) {
        DTYPE pack[kRows][1][kPackElems];
        recv<WORLD, kRows, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Layout::tile, TAIL>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, output_size_m, output_size_n, pack, offs_m,
            offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
        for (int i = 0; i < kRows; ++i)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) output[i][k][j] = pack[i][0][j];
      }
    } else {
      uint32_t raw[kPacks][kRows][1][2];
#pragma unroll
      for (int k = 0; k < kPacks; ++k)
        recv<WORLD, kRows, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
             Layout::tile, TAIL>(
            partials_ptr_array, partials_ptr_array_size, owner_of(k), 0, 0,
            partials_stride_m, partials_stride_n, 1, partials_size_m,
            partials_size_n, output_size_m, output_size_n, raw[k], offs_m,
            offs_n + k * THREADS_PER_BLOCK * kPackElems, 1);
#pragma unroll
      for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k) dequantize(raw[k][i][0], output[i][k]);
    }
  }
}

template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool TAIL, typename ENTRY, int ROWS, int PACKS, int ELEMS>
DINLINE void all_gather(ENTRY (&entries)[WORLD], int64_t stride_m,
                        int64_t stride_n, int size_m, int size_n, int clip_n,
                        int offs_m, int offs_n,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_pull,
                "a tile from where each rank published it");
#pragma unroll
  for (int r = 0; r < WORLD; ++r)
    recv<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
         Layout::tile, TAIL>(&entries[r], 1, 0, 0, 0, stride_m, stride_n, 1,
                             size_m, size_n, size_m, clip_n, copies[r], offs_m,
                             offs_n, 1);
}

template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE, int ROWS,
          int PACKS, int ELEMS>
DINLINE void all_gather(const DTYPE (&mine)[ROWS][PACKS][ELEMS],
                        const uint32_t (&epochs)[ROWS],
                        char* const* inbox_ptr_array, int inbox_ptr_array_size,
                        int64_t inbox_stride_slot, int64_t inbox_stride_m,
                        int64_t inbox_stride_n, int inbox_size_slot,
                        int inbox_size_m, int inbox_size_n, int clip_m,
                        int clip_n, int offs_m, int rank, uint32_t* status_ptr,
                        int64_t status_stride_field, int status_size_field,
                        uint64_t timeout_ticks,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_push,
                "a tile sent and polled as lines");

  if constexpr (QUANTIZE_INT4) {
    uint32_t payload[ROWS][PACKS][2];
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < PACKS; ++k)
        quantize(reinterpret_cast<const DTYPE(&)[1][ELEMS]>(mine[i][k]),
                 reinterpret_cast<uint32_t (&)[1][2]>(payload[i][k]));
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::line>(
          payload, inbox_ptr_array, inbox_ptr_array_size, r, rank,
          inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
          inbox_size_m, inbox_size_n, clip_m, clip_n, offs_m, 0, 1, epochs);
  } else {
#pragma unroll
    for (int r = 0; r < WORLD; ++r)
      send<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::line>(
          mine, inbox_ptr_array, inbox_ptr_array_size, r, rank,
          inbox_stride_slot, inbox_stride_m, inbox_stride_n, inbox_size_slot,
          inbox_size_m, inbox_size_n, clip_m, clip_n, offs_m, 0, 1, epochs);
  }
  load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
       Layout::line>(inbox_ptr_array, inbox_ptr_array_size, rank, 0,
                     inbox_stride_slot, inbox_stride_m, inbox_stride_n,
                     inbox_size_slot, inbox_size_m, inbox_size_n, clip_m,
                     clip_n, copies, offs_m, 0, 1, epochs, status_ptr,
                     status_stride_field, status_size_field, timeout_ticks);
}

template <AllGatherAlgorithm ALGORITHM, bool QUANTIZE_INT4, typename DTYPE,
          int WORLD, int TILE_M, int THREADS_PER_BLOCK, int WAVE_SIZE,
          bool TAIL, int ROWS, int PACKS, int ELEMS>
DINLINE void all_gather(char* const* inbox_ptr_array, int inbox_ptr_array_size,
                        int64_t inbox_stride_slot, int size_m, int size_n,
                        int clip_n, int offs_m, int offs_n, int rank,
                        DTYPE (&copies)[WORLD][ROWS][PACKS][ELEMS]) {
  static_assert(ALGORITHM == AllGatherAlgorithm::tile_staged,
                "a tile out of the staged inbox");
  load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4, Layout::tile,
       TAIL>(inbox_ptr_array, inbox_ptr_array_size, rank, 0, inbox_stride_slot,
             size_n, 1, WORLD, size_m, size_n, size_m, clip_n, copies, offs_m,
             offs_n, 1);
}

}  // namespace hip_custom_communicator
