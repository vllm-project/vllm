#pragma once

#include <hip/hip_runtime.h>
#include "base.h"

namespace quickreduce {
// ============================================================
// Oneshot
// ============================================================
// MARK: Oneshot All Reduce
template <typename T>
struct AllReduceOneshot {
  // Fixed magic implementation.
  // We will use a workgroup of 256 threads (standard kBlock) across 8 atoms of
  // work.
  static_assert(sizeof(T) == 2);
  static int constexpr kAtoms = 8;

  // Size and atom stride of data that the workgroup will process.
  static int constexpr kTileSize = 256 * kAtoms * sizeof(int32x4_t);
  static int constexpr kAtomStride = 256;

  __device__ static void run(
      T const* __restrict__ A,             // input
      T* __restrict__ B,                   // output
      int const N,                         // number of elements
      int const block,                     // this block's index
      int const num_blocks,                // total number of blocks
      int const world_size,                // total number of ranks
      int const rank,                      // this rank's index
      uint8_t** __restrict__ buffer_list,  // communication buffers
      long const data_offset,              // offset to start of the data buffer
      int flag_color                       // Flag color for the network barrier
  ) {
    // Topology
    int thread = threadIdx.x + threadIdx.y * kWavefront;

    long data_stride = num_blocks * kTileSize;
    long flags_stride = num_blocks * sizeof(int);

    uint8_t* rank_buffer = buffer_list[rank];

    // --------------------------------------------------------
    // Read A into registers
    int32x4_t tA[kAtoms];

    BufferResource src_buffer(const_cast<T*>(A), N * sizeof(T));

    int src_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      src_offset += kAtomStride * sizeof(int32x4_t);
    }

    // --------------------------------------------------------
    // Write rank data into this rank segment of every rank's communication
    // buffer.
    long comm_data_offset =
        data_offset + rank * data_stride + block * kTileSize;
    long comm_flags_offset = rank * flags_stride + block * sizeof(int);

    if (thread < world_size) {
      int r = thread;
      int* flag_ptr =
          reinterpret_cast<int*>(buffer_list[r] + comm_flags_offset);
      while (__atomic_load_n(flag_ptr, __ATOMIC_RELAXED) != flag_color - 1) {
      }
    }
    __syncthreads();

    for (int r = 0; r < world_size; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data_offset);
      for (int i = 0; i < kAtoms; i++) {
        __builtin_nontemporal_store(tA[i], send_buffer + thread);
        send_buffer += kAtomStride;
      }
    }

    // Inform the other ranks that th data has been posted.
    __syncthreads();
    if (thread < world_size) {
      int r = thread;
      int* flag_ptr =
          reinterpret_cast<int*>(buffer_list[r] + comm_flags_offset);
      __atomic_store_n(flag_ptr, flag_color, __ATOMIC_RELEASE);
    }

    // --------------------------------------------------------
    // Read and reduce the data from this rank's communication buffer.
    int32x4_t tB[kAtoms];

    {
      int r = 0;

      // Wait for the flags to be set.
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + r * flags_stride +
                                             block * sizeof(int));
      if (thread == 0) {
        while (__atomic_load_n(flag_ptr, __ATOMIC_RELAXED) != flag_color) {
        }
      }
      __syncthreads();

      // Read posted data from the rank's communication buffer.
      int32x4_t* recv_buffer = reinterpret_cast<int32x4_t*>(
          rank_buffer + data_offset + r * data_stride + block * kTileSize);

      for (int i = 0; i < kAtoms; i++) {
        tB[i] = __builtin_nontemporal_load(recv_buffer + thread);
        recv_buffer += kAtomStride;
      }
    }

    for (int r = 1; r < world_size; r++) {
      // Wait for the flags to be set.
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + r * flags_stride +
                                             block * sizeof(int));
      if (thread == 0) {
        while (__atomic_load_n(flag_ptr, __ATOMIC_RELAXED) != flag_color) {
        }
      }
      __syncthreads();

      // Read posted data from the rank's communication buffer.
      int32x4_t* recv_buffer = reinterpret_cast<int32x4_t*>(
          rank_buffer + data_offset + r * data_stride + block * kTileSize);

      for (int i = 0; i < kAtoms; i++) {
        tA[i] = __builtin_nontemporal_load(recv_buffer + thread);
        recv_buffer += kAtomStride;
      }

      // Reduce.
      for (int i = 0; i < kAtoms; i++) {
        packed_assign_add<T>(&tB[i], &tA[i]);
      }
    }

    __syncthreads();
    if (thread < world_size) {
      int r = thread;
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + r * flags_stride +
                                             block * sizeof(int));
      __atomic_store_n(flag_ptr, flag_color, __ATOMIC_RELAXED);
    }

    // --------------------------------------------------------
    // Write the result to B.
    BufferResource dst_buffer(B, N * sizeof(T));
    int dst_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      buffer_store_dwordx4(tB[i], dst_buffer.descriptor, dst_offset, 0, 0);
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

// ============================================================
// Twoshot
// ============================================================
// MARK: FP16 Line Codec
template <typename T, int world_size>
struct TwoshotFP16LineCodec {
  /*
      Default FP16 line codec for Twoshot collectives.
      No actual compression is involved.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each thread processes atoms of fp16x8_t (16B).
  static int constexpr kRankAtoms = kAtoms / kWorldSize;
  static int constexpr kRankTileSize = 256 * kRankAtoms * sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  int const thread;
  int const rank;

  __device_inline__ TwoshotFP16LineCodec(int thread, int rank)
      : thread(thread), rank(rank) {
    static_assert(kRankTileSize % 16 == 0,
                  "kRankTileSize must be 16B aligned.");
  }

  __device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                              int32x4_t const* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      __builtin_nontemporal_store(data[i], send_buffer + thread);
      send_buffer += kAtomStride;
    }
  }

  __device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                              int32x4_t* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      data[i] = __builtin_nontemporal_load(*recv_buffer + thread);
      *recv_buffer += kAtomStride;
    }
  }
};

// MARK: Q4 Line Codec
template <typename T, int world_size>
struct TwoshotQ4LineCodec {
  /*
      Int4-blocking Line codec for Twoshot collectives.
      We quantize the FP16 data to block-scaled Int4 in blocks of 32.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int4x8_t (4B) and a fp16 scale shared among 32 values.
  static int constexpr kRankAtoms = kAtoms / kWorldSize;
  static int constexpr kRankTileStride = 1152;
  static int constexpr kRankTileScaleOffset = 1024;
  static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

  static int constexpr kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  // Constants configuration

  // {-1/8.0h, -1/8.0h}, f16x2_t
  static int constexpr kScaleFactor =
      std::is_same<T, half>::value ? 0xB000B000 : 0xBE00BE00;

  // {1e-7, 1e-7}, f16x2_t
  static int constexpr kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {-8, -8}, f16x2_t
  static int constexpr kRangeMin =
      std::is_same<T, half>::value ? 0xC800C800 : 0xC100C100;

  // {+7, +7}, f16x2_t
  static int constexpr kRangeMax =
      std::is_same<T, half>::value ? 0x47004700 : 0x40E040E0;

  // {+8, +8}, int16x2_t
  static int constexpr kRangeBias = 0x00080008;

  int const thread;
  int const rank;
  int const group_leader;

  __device_inline__ TwoshotQ4LineCodec(int thread, int rank)
      : thread(thread), rank(rank), group_leader((threadIdx.x / 8) * 8) {
    static_assert(kRankTileSize % 16 == 0,
                  "kRankTileSize must be 16B aligned.");
    set_fp16_ovfl(true);
  }

  __device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                              int32x4_t const* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // max(w), min(w)
      int wmax, wmin, wblockmax;
      {
        int a, b;
        a = packed_max<T>(atom[0], atom[1]);
        b = packed_max<T>(atom[2], atom[3]);
        wmax = packed_max<T>(a, b);

        a = packed_min<T>(atom[0], atom[1]);
        b = packed_min<T>(atom[2], atom[3]);
        wmin = packed_min<T>(a, b);

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          wmax = packed_max<T>(wmax, x);

          int y = __shfl_down(wmin, i);
          wmin = packed_min<T>(wmin, y);
        }
        wblockmax = packed_abs_max<T>(wmax, wmin);

        // Share with the cohort
        wblockmax = __shfl(wblockmax, group_leader);
      }

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast<T>(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q4 into int32_t
      int qw = q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12);

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                              int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      int32_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack q4 into fp16x8_t
      int32x4_t w;
      {
        static uint constexpr kMask000F = 0x000F000F;
        // {1024.0, 1024.0}, f16x2_t
        static uint constexpr kF162_1024 =
            std::is_same<T, half>::value ? 0x64006400 : 0x44804480;
        // {-1032.0, -1032.0}, f16x2_t
        static uint constexpr kF162_1032 =
            std::is_same<T, half>::value ? 0xE408E408 : 0xC481C481;

        for (int i = 0; i < 4; i++) {
          int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kF162_1024;
          w[i] = packed_add<T>(q4, kF162_1032);
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

template <typename T, int world_size>
struct TwoshotMaxMinQ4LineCodec {
  /*
      Int4-blocking Line codec for Twoshot collectives.
      We quantize the FP16/BF16 data to block-scaled Int4 in blocks of 32.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of f16x8_t (16B),
  // into a int4x8_t (4B) and a 2 f16 scale shared among 32 values.
  static int constexpr kRankAtoms = kAtoms / kWorldSize;

  // 1024 + 128 + 128
  static int constexpr kRankTileStride = 1280;
  static int constexpr kRankTileScaleOffset = 1024;
  static int constexpr kRankTileZeroOffset = 1152;
  static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

  static int constexpr kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  // Constants configuration

  // {-1/16.0h, -1/16.0h}, f16x2_t
  static int constexpr kScaleFactor =
      std::is_same<T, half>::value ? 0xAC00AC00 : 0xBD80BD80;

  // {1e-7, 1e-7}, f16x2_t
  static int constexpr kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {0, 0}, f16x2_t
  static int constexpr kRangeMin = 0x00000000;

  // {+15, +15}, f16x2_t
  static int constexpr kRangeMax =
      std::is_same<T, half>::value ? 0x4B804B80 : 0x41704170;

  static unsigned char constexpr kMask0F = 0x0F;

  int const thread;
  int const rank;
  int const group_leader;

  __device_inline__ TwoshotMaxMinQ4LineCodec(int thread, int rank)
      : thread(thread), rank(rank), group_leader((threadIdx.x / 8) * 8) {
    static_assert(kRankTileSize % 16 == 0,
                  "kRankTileSize must be 16B aligned.");
    set_fp16_ovfl(true);
  }

  __device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                              int32x4_t const* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // max(w), min(w)
      int wmax, wmin, wblockmax, wblockmin;
      {
        int a, b;
        a = packed_max<T>(atom[0], atom[1]);
        b = packed_max<T>(atom[2], atom[3]);
        wmax = packed_max<T>(a, b);

        a = packed_min<T>(atom[0], atom[1]);
        b = packed_min<T>(atom[2], atom[3]);
        wmin = packed_min<T>(a, b);

        // Reduce the max and min among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          wmax = packed_max<T>(wmax, x);

          int y = __shfl_down(wmin, i);
          wmin = packed_min<T>(wmin, y);
        }

        // Share with the cohort
        wblockmax = __shfl(wmax, group_leader);
        wblockmin = __shfl(wmin, group_leader);
      }

      // Derive zeros and scales
      int decoding_zero = wblockmin;
      int decoding_scale;
      int encoding_scale;

      decoding_scale = packed_sub<T>(wblockmax, decoding_zero);
      decoding_scale = packed_mul<T>(decoding_scale, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_sub<T>(atom[i], decoding_zero);
        w[i] = packed_mul<T>(w[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32_t qw = 0;
      {
        unsigned char* qi = reinterpret_cast<unsigned char*>(&qw);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++) {
          auto val = (unsigned char)T2float_cast<T>(wh[i]) & kMask0F;
          qi[i / 2] |= val << (4 * (i & 1));
        }
      }

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);
      int* qz_ptr =
          reinterpret_cast<int*>(atom_ptr + kRankTileZeroOffset) + (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
        __builtin_nontemporal_store(decoding_zero, qz_ptr);
      }
    }
  }

  __device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                              int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32_t* qw_ptr = reinterpret_cast<int32_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);
      int* qz_ptr =
          reinterpret_cast<int*>(atom_ptr + kRankTileZeroOffset) + (thread / 8);

      int32_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);
      int qz = __builtin_nontemporal_load(qz_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack 8xq4 into f16x8_t
      int32x4_t w;
      {
        T* wh = reinterpret_cast<T*>(&w);
        unsigned char* qi = reinterpret_cast<unsigned char*>(&qw);

#pragma unroll
        for (int i = 0; i < 8; i++) {
          auto val = (qi[i / 2] >> (4 * (i & 1))) & kMask0F;
          wh[i] = float2T_cast<T>((float)val);
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
        w[i] = packed_add<T>(w[i], qz);
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

// MARK: Q8 Line Codec
template <typename T, int world_size>
struct TwoshotQ8LineCodec {
  /*
      Int8-blocking Line codec for Twoshot collectives.
      We quantize the FP16/BF16 data to block-scaled Int8 in blocks of 32.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of f16x8_t (16B),
  // into a int8x8_t (8B) and a f16 scale shared among 32 values.
  static int constexpr kRankAtoms = kAtoms / kWorldSize;
  static int constexpr kRankTileStride = 2176;
  static int constexpr kRankTileScaleOffset = 2048;
  static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

  static int constexpr kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  // Constants configuration

  // {-1/128.0h, -1/128.0h}, f16x2_t
  static int constexpr kScaleFactor =
      std::is_same<T, half>::value ? 0xA000A000 : 0xBC00BC00;

  // {1e-7, 1e-7}, f16x2_t
  static int constexpr kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {-128, -128}, f16x2_t
  static int constexpr kRangeMin =
      std::is_same<T, half>::value ? 0xD800D800 : 0xC300C300;
  // {+127, +127}, f16x2_t
  static int constexpr kRangeMax =
      std::is_same<T, half>::value ? 0x57F057F0 : 0x42FE42FE;

  // {+128, +128}, int16x2_t
  static int constexpr kRangeBias = 0x00800080;

  int const thread;
  int const rank;
  int const group_leader;

  __device_inline__ TwoshotQ8LineCodec(int thread, int rank)
      : thread(thread), rank(rank), group_leader((threadIdx.x / 8) * 8) {
    static_assert(kRankTileSize % 16 == 0,
                  "kRankTileSize must be 16B aligned.");
    set_fp16_ovfl(true);
  }

  __device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                              int32x4_t const* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // max(w), min(w)
      int wmax, wmin, wblockmax;
      {
        int a, b;
        a = packed_max<T>(atom[0], atom[1]);
        b = packed_max<T>(atom[2], atom[3]);

        wmax = packed_max<T>(a, b);

        a = packed_min<T>(atom[0], atom[1]);
        b = packed_min<T>(atom[2], atom[3]);

        wmin = packed_min<T>(a, b);

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          wmax = packed_max<T>(wmax, x);

          int y = __shfl_down(wmin, i);
          wmin = packed_min<T>(wmin, y);
        }
        wblockmax = packed_abs_max<T>(wmax, wmin);

        // Share with the cohort
        wblockmax = __shfl(wblockmax, group_leader);
      }

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<T>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(atom[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from f16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (int16_t)rintf(T2float_cast<T>(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      // Pack 8 x q8 into int32x2_t
      int32x2_t qw;
      qw[0] = q[0] | (q[1] << 8);
      qw[1] = q[2] | (q[3] << 8);

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
      }
    }
  }

  __device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                              int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      int32x2_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack q8 into fp16x8_t
      int32x4_t w;
      {
        static uint constexpr kMask00FF = 0x00FF00FF;

        // {1024.0, 1024.0}, f16x2_t
        static uint constexpr kF162_1024 =
            std::is_same<T, half>::value ? 0x64006400 : 0x44804480;

        // {-1152.0, -1152.0}, f16x2_t
        static uint constexpr kF162_1152 =
            std::is_same<T, half>::value ? 0xE480E480 : 0xC490C490;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int32_t q8 = ((qw[i / 2] >> ((i % 2) * 8)) & kMask00FF) | kF162_1024;
          w[i] = packed_add<T>(q8, kF162_1152);
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

template <typename T, int world_size>
struct TwoshotMaxMinQ8LineCodec {
  /*
      Int8-blocking Line codec for Twoshot collectives.
      We quantize the FP16 data to block-scaled Int8 in blocks of 32.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each thread processes a fragment of fp16x8_t (16B),
  // into a int8x8_t (8B) and a fp16 zero and a fp16 scale shared among 32
  // values.
  static int constexpr kRankAtoms = kAtoms / kWorldSize;
  // 2048 + 128 + 128
  static int constexpr kRankTileStride = 2304;
  static int constexpr kRankTileScaleOffset = 2048;
  static int constexpr kRankTileZeroOffset = 2176;
  static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

  static int constexpr kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  // Constants configuration
  // {1/255.0h, 1/255.0h}, f16x2_t
  static int constexpr kScaleFactor =
      std::is_same<T, half>::value ? 0x1C041C04 : 0x3B813B81;

  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {0, 0}, f16x2_t
  static int constexpr kRangeMin = 0x00000000;

  // {+255, +255}, f16x2_t
  static int constexpr kRangeMax =
      std::is_same<T, half>::value ? 0x5BF85BF8 : 0x437F437F;

  // {+128, +128}, int16x2_t
  static int constexpr kRangeBias = 0x00800080;

  int const thread;
  int const rank;
  int const group_leader;

  __device_inline__ TwoshotMaxMinQ8LineCodec(int thread, int rank)
      : thread(thread), rank(rank), group_leader((threadIdx.x / 8) * 8) {
    static_assert(kRankTileSize % 16 == 0,
                  "kRankTileSize must be 16B aligned.");
    set_fp16_ovfl(true);
  }

  __device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                              int32x4_t const* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];
      // max(w), min(w)
      int wmax, wmin, wblockmax, wblockmin;
      {
        int a, b;
        a = packed_max<T>(atom[0], atom[1]);
        b = packed_max<T>(atom[2], atom[3]);
        wmax = packed_max<T>(a, b);

        a = packed_min<T>(atom[0], atom[1]);
        b = packed_min<T>(atom[2], atom[3]);
        wmin = packed_min<T>(a, b);

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          wmax = packed_max<T>(wmax, x);

          int y = __shfl_down(wmin, i);
          wmin = packed_min<T>(wmin, y);
        }

        // Share with the cohort
        wblockmax = __shfl(wmax, group_leader);
        wblockmin = __shfl(wmin, group_leader);
      }

      // Derive zeros and scales
      int decoding_zero = wblockmin;
      int decoding_scale;
      int encoding_scale;

      decoding_scale = packed_sub<T>(wblockmax, decoding_zero);
      decoding_scale = packed_mul<T>(decoding_scale, kScaleFactor);
      encoding_scale = packed_add<T>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<T>(encoding_scale);

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_sub<T>(atom[i], decoding_zero);
        w[i] = packed_mul<T>(w[i], encoding_scale);
        w[i] = packed_max<T>(w[i], kRangeMin);
        w[i] = packed_min<T>(w[i], kRangeMax);
      }

      // Convert from fp16x8_t to uint8x8_t and pack into int32x2_t
      int32x2_t qw;
      {
        unsigned char* qi = reinterpret_cast<unsigned char*>(&qw);
        T* wh = reinterpret_cast<T*>(&w);
        for (int i = 0; i < 8; i++)
          qi[i] = (unsigned char)T2float_cast<T>(wh[i]);
      }

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);
      int* qz_ptr =
          reinterpret_cast<int*>(atom_ptr + kRankTileZeroOffset) + (thread / 8);

      __builtin_nontemporal_store(qw, qw_ptr);
      if (threadIdx.x == group_leader) {
        __builtin_nontemporal_store(decoding_scale, qs_ptr);
        __builtin_nontemporal_store(decoding_zero, qz_ptr);
      }
    }
  }

  __device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                              int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      // Directly read quantized atom from recv_buffer
      uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
      int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);
      int* qz_ptr =
          reinterpret_cast<int*>(atom_ptr + kRankTileZeroOffset) + (thread / 8);

      int32x2_t qw = __builtin_nontemporal_load(qw_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);
      int qz = __builtin_nontemporal_load(qz_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack uint8x8_t into fp16x8_t
      int32x4_t w;
      {
        T* wh = reinterpret_cast<T*>(&w);
        unsigned char* qi = reinterpret_cast<unsigned char*>(&qw);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          wh[i] = float2T_cast<T>((float)qi[i]);
        }
      }

      // Apply decoding scales and zeros
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
        w[i] = packed_add<T>(w[i], qz);
      }

      data[k] = w;
    }
  }
};

// MARK: Twoshot All Reduce
template <typename T, class LineCodec>
struct AllReduceTwoshot {
  // Fixed magic implementation.
  // We will use a workgroup of 256 threads (standard kBlock) across 8 atoms of
  // work.
  static int constexpr kAtoms = 8;

  // Size and atom stride of source/destination data that the workgroup will
  // process.
  static int constexpr kTileSize = 256 * kAtoms * sizeof(int32x4_t);
  static int constexpr kAtomStride = 256;

  static int constexpr kWorldSize = LineCodec::kWorldSize;

  __device__ static void run(
      T const* __restrict__ A,  // input
      T* __restrict__ B,        // output
      int const N,              // number of elements
      int const block,          // block index
      int const num_blocks,     // number of blocks
      int const world_size,     // unused - only kept around for API consistency
      int const rank,           // rank index
      uint8_t** __restrict__ buffer_list,  // communication buffers
      long const data_offset,              // offset to start of the data buffer
      int flag_color) {
    // Topology
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    uint8_t* rank_buffer = buffer_list[rank];
    LineCodec codec(thread, rank);

    // --------------------------------------------------------
    // Read A into registers
    int32x4_t tA[kAtoms];

    BufferResource src_buffer(const_cast<T*>(A), N * sizeof(T));
    int src_offset = block * kTileSize + thread * sizeof(int32x4_t);
    int32x4_t* src = reinterpret_cast<int32x4_t*>(const_cast<T*>(A));

    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      src_offset += kAtomStride * sizeof(int32x4_t);
    }

    // --------------------------------------------------------
    // Phase-1A: Write segment data into the communication buffer of the target
    // rank responsible for this segment.
    long comm_data0_offset = data_offset + block * LineCodec::kTileSize;
    long comm_data1_offset =
        num_blocks * LineCodec::kTileSize + comm_data0_offset;

    long comm_flags0_offset = block * (kWorldSize * sizeof(int));
    long comm_flags1_offset =
        num_blocks * (kWorldSize * sizeof(int)) + comm_flags0_offset;

    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
          buffer_list[r] + comm_data0_offset + rank * LineCodec::kRankTileSize);
      codec.send(send_buffer, &tA[r * LineCodec::kRankAtoms]);
    }

    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      int* flag_ptr = reinterpret_cast<int*>(
          buffer_list[r] + comm_flags0_offset + rank * sizeof(int));
      __atomic_store_n(flag_ptr, flag_color, __ATOMIC_RELEASE);
    }
    // --------------------------------------------------------
    // Phase-1B: Reduce the segment data from the communication buffers.
    int32x4_t tR[LineCodec::kRankAtoms] = {};
    {
      // Read the data from the communication buffer.
      int32x4_t* recv_buffer =
          reinterpret_cast<int32x4_t*>(rank_buffer + comm_data0_offset);
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + comm_flags0_offset);

      for (int r = 0; r < kWorldSize; r++) {
        // Wait for the flags to be set.
        if (thread == 0) {
          while (__atomic_load_n(&flag_ptr[r], __ATOMIC_RELAXED) !=
                 flag_color) {
          }
        }
        __syncthreads();

        // note: we reuse tA as temp buffer here
        codec.recv(&recv_buffer, tA);

        for (int i = 0; i < LineCodec::kRankAtoms; i++) {
          packed_assign_add<T>(&tR[i], &tA[i]);
        }
      }
    }

    // --------------------------------------------------------
    // Phase-2: Write the reduced segment to every other rank
    // This is basically an all-gather.
    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
          buffer_list[r] + comm_data1_offset + rank * LineCodec::kRankTileSize);
      codec.send(send_buffer, tR);
    }

    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      int* flag_ptr = reinterpret_cast<int*>(
          buffer_list[r] + comm_flags1_offset + rank * sizeof(int));
      __atomic_store_n(flag_ptr, flag_color, __ATOMIC_RELEASE);
    }

    // --------------------------------------------------------
    // Phase-2: Read the gather segments from the rank's communication buffer.
    {
      // Read the data from the communication buffer.
      int32x4_t* recv_buffer =
          reinterpret_cast<int32x4_t*>(rank_buffer + comm_data1_offset);
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + comm_flags1_offset);

      for (int r = 0; r < kWorldSize; r++) {
        // Wait for the flags to be set.
        if (thread == 0) {
          while (__atomic_load_n(&flag_ptr[r], __ATOMIC_RELAXED) !=
                 flag_color) {
          }
        }
        __syncthreads();

        // Gather all reduced and final rank segments into tA.
        codec.recv(&recv_buffer, &tA[r * LineCodec::kRankAtoms]);
      }
    }

    // --------------------------------------------------------
    // Write the result to B.
    BufferResource dst_buffer(B, N * sizeof(T));
    int dst_offset = block * kTileSize + thread * sizeof(int32x4_t);
    int32x4_t* dst = reinterpret_cast<int32x4_t*>(B);

    for (int i = 0; i < kAtoms; i++) {
      buffer_store_dwordx4(tA[i], dst_buffer.descriptor, dst_offset, 0, 0);
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

}  // namespace quickreduce