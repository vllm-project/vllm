#pragma once

#include <hip/hip_runtime.h>
#include "base.h"

// ============================================================
// Oneshot
// ============================================================
// MARK: Oneshot All Reduce
struct AllReduceOneshot {
  // Fixed magic implementation.
  // We will use a workgroup of 256 threads (standard kBlock) across 8 atoms of
  // work.
  static int constexpr kAtoms = 8;

  // Size and atom stride of data that the workgroup will process.
  static int constexpr kTileSize = 256 * kAtoms * sizeof(int32x4_t);
  static int constexpr kAtomStride = 256;

  __device__ static void run(
      half const* __restrict__ A,          // input
      half* __restrict__ B,                // output
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

    BufferResource src_buffer(const_cast<half*>(A), N * sizeof(half));
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
        int32x4_t& tA_fragment = tA[i];
        int32x4_t& tB_fragment = tB[i];
        asm volatile("v_pk_add_f16 %0, %1, %2"
                     : "=v"(tB_fragment[0])
                     : "v"(tB_fragment[0]), "v"(tA_fragment[0]));
        asm volatile("v_pk_add_f16 %0, %1, %2"
                     : "=v"(tB_fragment[1])
                     : "v"(tB_fragment[1]), "v"(tA_fragment[1]));
        asm volatile("v_pk_add_f16 %0, %1, %2"
                     : "=v"(tB_fragment[2])
                     : "v"(tB_fragment[2]), "v"(tA_fragment[2]));
        asm volatile("v_pk_add_f16 %0, %1, %2"
                     : "=v"(tB_fragment[3])
                     : "v"(tB_fragment[3]), "v"(tA_fragment[3]));
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
    BufferResource dst_buffer(B, N * sizeof(half));
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
template <int world_size>
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

// MARK: FP8 Line Codec
// template <int world_size>
// struct TwoshotFP8LineCodec {
//   /*
//       FP8 Line codec for Twoshot collectives.
//       We quantize the FP16 data to block-scaled FP8 in blocks of 32.
//   */

//   static int constexpr kAtoms = 8;
//   static int constexpr kAtomStride = 256;
//   static int constexpr kWorldSize = world_size;

//   // Codec tile size process by this workgroup.
//   // Each threads processes a fragment of fp16x8_t (16B),
//   // into a fp8x8_t (8B) and a fp16 scale shared among 32 values.
//   static int constexpr kRankAtoms = kAtoms / kWorldSize;
//   static int constexpr kRankTileStride = 2176;
//   static int constexpr kRankTileScaleOffset = 2048;
//   static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

//   static int constexpr kRankBufferTileStride =
//       kRankTileStride / sizeof(int32x4_t);

//   // Total tile size for the collective communication.
//   static int constexpr kTileSize = kRankTileSize * kWorldSize;

//   // FP8 Maximum value (on AMD Instinct MI300X - float8_e4m3fnuz)
//   static float constexpr kFP8Max = 240.0f;
//   static int constexpr kScaleFactor = 0x1C441C44;   // {1/240.0h, 1/240.0h}
//   static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}

//   int const thread;
//   int const rank;
//   int const group_leader;

//   __device_inline__ TwoshotFP8LineCodec(int thread, int rank)
//       : thread(thread), rank(rank), group_leader((threadIdx.x / 8) * 8) {
//     static_assert(kRankTileSize % 16 == 0,
//                   "kRankTileSize must be 16B aligned.");
//     set_fp16_ovfl(true);
//   }

//   __device_inline__ void send(int32x4_t* __restrict__ send_buffer,
//                               int32x4_t const* __restrict__ data) {
//     for (int k = 0; k < kRankAtoms; k++) {
//       int32x4_t const atom = data[k];

//       // abs(w)
//       int32x4_t w;
//       {
//         half const* x = reinterpret_cast<half const*>(&atom);
//         half* y = reinterpret_cast<half*>(&w);
//         for (int i = 0; i < 8; i++) {
//           y[i] = __habs(x[i]);
//         }
//       }

//       // max(w)
//       int wmax;
//       {
//         int a, b;
//         int* dw = reinterpret_cast<int*>(&w);
//         asm volatile("v_pk_max_f16 %0, %1, %2"
//                      : "=v"(a)
//                      : "v"(dw[0]), "v"(dw[1]));
//         asm volatile("v_pk_max_f16 %0, %1, %2"
//                      : "=v"(b)
//                      : "v"(dw[2]), "v"(dw[3]));
//         asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(wmax) : "v"(a),
//         "v"(b));

//         // Reduce the max among a group of 8 threads
//         // Note: This is basically 2 blocks of 32 values setup as the
//         // upper/lower halves of the fp16x2_t
//         for (int i = 1; i < 8; i <<= 1) {
//           int x = __shfl_down(wmax, i);
//           asm volatile("v_pk_max_f16 %0, %1, %2"
//                        : "=v"(wmax)
//                        : "v"(wmax), "v"(x));
//         }

//         // Share with the cohort
//         wmax = __shfl(wmax, group_leader);
//       }

//       // Derive scales
//       int decoding_scale;
//       int encoding_scale;
//       asm volatile("v_pk_mul_f16 %0, %1, %2"
//                    : "=v"(decoding_scale)
//                    : "v"(wmax), "v"(kScaleFactor));
//       asm volatile("v_pk_add_f16 %0, %1, %2"
//                    : "=v"(encoding_scale)
//                    : "v"(decoding_scale), "v"(kScaleEpsilon));
//       encoding_scale = __builtin_bit_cast(
//           int, h2rcp(__builtin_bit_cast(half2, encoding_scale)));

//       // Apply scales to get quantized values
//       for (int i = 0; i < 4; i++) {
//         asm volatile("v_pk_mul_f16 %0, %1, %2"
//                      : "=v"(w[i])
//                      : "v"(atom[i]), "v"(encoding_scale));
//       }

//       // Convert to packed FP8
//       fp32x8_t wf;
//       {
//         half2 const* x = reinterpret_cast<half2 const*>(&w);
//         float2* y = reinterpret_cast<float2*>(&wf);
//         for (int i = 0; i < 4; i++) {
//           y[i] = __half22float2(x[i]);
//         }
//       }

//       int32x2_t qw;
//       qw[0] = __builtin_amdgcn_cvt_pk_fp8_f32(wf[0], wf[1], qw[0], 0);
//       qw[0] = __builtin_amdgcn_cvt_pk_fp8_f32(wf[2], wf[3], qw[0], 1);
//       qw[1] = __builtin_amdgcn_cvt_pk_fp8_f32(wf[4], wf[5], qw[1], 0);
//       qw[1] = __builtin_amdgcn_cvt_pk_fp8_f32(wf[6], wf[7], qw[1], 1);

//       // Write quantized atom to send_buffer
//       // note: only the group leader stores the scale
//       uint8_t* atom_ptr =
//           reinterpret_cast<uint8_t*>(send_buffer + k *
//           kRankBufferTileStride);
//       int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
//       int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
//                     (thread / 8);

//       __builtin_nontemporal_store(qw, qw_ptr);
//       if (threadIdx.x == group_leader) {
//         __builtin_nontemporal_store(decoding_scale, qs_ptr);
//       }
//     }
//   }

//   __device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
//                               int32x4_t* __restrict__ data) {
//     for (int k = 0; k < kRankAtoms; k++) {
//       // Directly read quantized atom from recv_buffer
//       uint8_t* atom_ptr = reinterpret_cast<uint8_t*>(*recv_buffer);
//       int32x2_t* qw_ptr = reinterpret_cast<int32x2_t*>(atom_ptr) + thread;
//       int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
//                     (thread / 8);

//       int32x2_t qw = __builtin_nontemporal_load(qw_ptr);
//       int qs = __builtin_nontemporal_load(qs_ptr);

//       *recv_buffer += kRankBufferTileStride;

//       // Unpack FP8
//       int32x4_t w;
//       {
//         for (int i = 0; i < 2; i++) {
//           fp32x2_t wf0 = __builtin_amdgcn_cvt_pk_f32_fp8(qw[i], 0);
//           fp32x2_t wf1 = __builtin_amdgcn_cvt_pk_f32_fp8(qw[i], 1);

//           asm volatile("v_cvt_pkrtz_f16_f32 %0, %1, %2"
//                        : "=v"(w[i * 2 + 0])
//                        : "v"(wf0[0]), "v"(wf0[1]));
//           asm volatile("v_cvt_pkrtz_f16_f32 %0, %1, %2"
//                        : "=v"(w[i * 2 + 1])
//                        : "v"(wf1[0]), "v"(wf1[1]));
//         }
//       }

//       // Apply decoding scales
//       for (int i = 0; i < 4; i++) {
//         asm volatile("v_pk_mul_f16 %0, %1, %2"
//                      : "=v"(w[i])
//                      : "v"(w[i]), "v"(qs));
//       }

//       // That's pretty much it...
//       data[k] = w;
//     }
//   }
// };

// MARK: Q4 Line Codec
template <int world_size>
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

  // Q4 configuration
  static int constexpr kScaleFactor =
      0xB000B000;  // {-1/8.0h, -1/8.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xC800C800;      // {-8, -8}, fp16x2_t
  static int constexpr kRangeMax = 0x47004700;      // {+7, +7}, fp16x2_t
  static int constexpr kRangeBias = 0x00080008;     // {+8, +8}, int16x2_t

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
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(wmax) : "v"(a), "v"(b));

        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(wmin) : "v"(a), "v"(b));

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          asm volatile("v_pk_max_f16 %0, %1, %2"
                       : "=v"(wmax)
                       : "v"(wmax), "v"(x));

          int y = __shfl_down(wmin, i);
          asm volatile("v_pk_min_f16 %0, %1, %2"
                       : "=v"(wmin)
                       : "v"(wmin), "v"(y));
        }

        half2 wmaxh2 = __builtin_bit_cast(half2, wmax);
        half2 wminh2 = __builtin_bit_cast(half2, wmin);
        half2 wblockmaxh2;

        wblockmaxh2.x =
            __half2float(__habs(wmaxh2.x)) > __half2float(__habs(wminh2.x))
                ? wmaxh2.x
                : wminh2.x;
        wblockmaxh2.y =
            __half2float(__habs(wmaxh2.y)) > __half2float(__habs(wminh2.y))
                ? wmaxh2.y
                : wminh2.y;
        wblockmax = __builtin_bit_cast(int, wblockmaxh2);

        // Share with the cohort
        wblockmax = __shfl(wblockmax, group_leader);
      }

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      asm volatile("v_pk_mul_f16 %0, %1, %2"
                   : "=v"(decoding_scale)
                   : "v"(wblockmax), "v"(kScaleFactor));
      asm volatile("v_pk_add_f16 %0, %1, %2"
                   : "=v"(encoding_scale)
                   : "v"(decoding_scale), "v"(kScaleEpsilon));
      encoding_scale = __builtin_bit_cast(
          int, h2rcp(__builtin_bit_cast(half2, encoding_scale)));

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(atom[i]), "v"(encoding_scale));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMin));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMax));
      }

      // Convert from fp16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(__half2float(wh[i]));

        for (int i = 0; i < 4; i++) {
          asm volatile("v_pk_add_i16 %0, %1, %2"
                       : "=v"(q[i])
                       : "v"(q[i]), "v"(kRangeBias));
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
        static uint constexpr kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1032 =
            0xE408E408;  // {-1032.0, -1032.0}, fp16x2_t

        for (int i = 0; i < 4; i++) {
          int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kHalf2_1024;
          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(w[i])
                       : "v"(q4), "v"(kHalf2_1032));
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(qs));
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

// MARK: Q6 Line Codec
template <int world_size>
struct TwoshotQ6LineCodec {
  /*
      Int6-blocking Line codec for Twoshot collectives.
      We quantize the FP16 data to block-scaled Int64 in blocks of 32.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int6x8_t (4B + 2B) and a fp16 scale shared among 32 values.
  static int constexpr kRankAtoms = kAtoms / kWorldSize;
  static int constexpr kRankTileStride = 1664;
  static int constexpr kRankTileQ2Offset = 1024;
  static int constexpr kRankTileScaleOffset = 1536;
  static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

  static int constexpr kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  // Q4 configuration
  static int constexpr kScaleFactor =
      0xA800A800;  // {-1/32.0h, -1/32.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xD000D000;      // {-32, -32}, fp16x2_t
  static int constexpr kRangeMax = 0x4FC04FC0;      // {+31, +31}, fp16x2_t
  static int constexpr kRangeBias = 0x00200020;     // {+32, +32}, int16x2_t

  int const thread;
  int const rank;
  int const group_leader;

  __device_inline__ TwoshotQ6LineCodec(int thread, int rank)
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
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(wmax) : "v"(a), "v"(b));

        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(wmin) : "v"(a), "v"(b));

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          asm volatile("v_pk_max_f16 %0, %1, %2"
                       : "=v"(wmax)
                       : "v"(wmax), "v"(x));

          int y = __shfl_down(wmin, i);
          asm volatile("v_pk_min_f16 %0, %1, %2"
                       : "=v"(wmin)
                       : "v"(wmin), "v"(y));
        }

        half2 wmaxh2 = __builtin_bit_cast(half2, wmax);
        half2 wminh2 = __builtin_bit_cast(half2, wmin);
        half2 wblockmaxh2;

        wblockmaxh2.x =
            __half2float(__habs(wmaxh2.x)) > __half2float(__habs(wminh2.x))
                ? wmaxh2.x
                : wminh2.x;
        wblockmaxh2.y =
            __half2float(__habs(wmaxh2.y)) > __half2float(__habs(wminh2.y))
                ? wmaxh2.y
                : wminh2.y;
        wblockmax = __builtin_bit_cast(int, wblockmaxh2);

        // Share with the cohort
        wblockmax = __shfl(wblockmax, group_leader);
      }

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      asm volatile("v_pk_mul_f16 %0, %1, %2"
                   : "=v"(decoding_scale)
                   : "v"(wblockmax), "v"(kScaleFactor));
      asm volatile("v_pk_add_f16 %0, %1, %2"
                   : "=v"(encoding_scale)
                   : "v"(decoding_scale), "v"(kScaleEpsilon));
      encoding_scale = __builtin_bit_cast(
          int, h2rcp(__builtin_bit_cast(half2, encoding_scale)));

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(atom[i]), "v"(encoding_scale));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMin));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMax));
      }

      // Convert from fp16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(__half2float(wh[i]));

        for (int i = 0; i < 4; i++) {
          asm volatile("v_pk_add_i16 %0, %1, %2"
                       : "=v"(q[i])
                       : "v"(q[i]), "v"(kRangeBias));
        }
      }

      // Pack 8 x q6 into int32_t + int16_t
      uint32_t q4w;
      uint16_t q2w = 0;
      q4w = (q[0] & 0x000F000F) | ((q[1] & 0x000F000F) << 4) |
            ((q[2] & 0x000F000F) << 8) | ((q[3] & 0x000F000F) << 12);
      {
        int16_t* tw = reinterpret_cast<int16_t*>(&q);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          q2w |= (tw[i] >> 4) << (i * 2);
        }
      }

      // Write quantized atom to send_buffer
      // note: only the group leader stores the scale
      uint8_t* atom_ptr =
          reinterpret_cast<uint8_t*>(send_buffer + k * kRankBufferTileStride);
      uint32_t* q4w_ptr = reinterpret_cast<uint32_t*>(atom_ptr) + thread;
      uint16_t* q2w_ptr =
          reinterpret_cast<uint16_t*>(atom_ptr + kRankTileQ2Offset) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      __builtin_nontemporal_store(q4w, q4w_ptr);
      __builtin_nontemporal_store(q2w, q2w_ptr);
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
      uint32_t* q4w_ptr = reinterpret_cast<uint32_t*>(atom_ptr) + thread;
      uint16_t* q2w_ptr =
          reinterpret_cast<uint16_t*>(atom_ptr + kRankTileQ2Offset) + thread;
      int* qs_ptr = reinterpret_cast<int*>(atom_ptr + kRankTileScaleOffset) +
                    (thread / 8);

      uint32_t q4w = __builtin_nontemporal_load(q4w_ptr);
      uint16_t q2w = __builtin_nontemporal_load(q2w_ptr);
      int qs = __builtin_nontemporal_load(qs_ptr);

      *recv_buffer += kRankBufferTileStride;

      // Unpack q6 into fp16x8_t
      int32x4_t w;
      {
        static uint constexpr kMask000F = 0x000F000F;
        static uint constexpr kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1056 =
            0xE420E420;  // {-1056.0, -1056.0}, fp16x2_t

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int32_t q4 = q4w & kMask000F;
          int32_t q2 = (q2w & 0x3) | ((q2w & 0xC) << 14);
          q4w >>= 4;
          q2w >>= 4;

          int32_t q6 = q4 | (q2 << 4) | kHalf2_1024;
          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(w[i])
                       : "v"(q6), "v"(kHalf2_1056));
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(qs));
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

// MARK: Q8 Line Codec
template <int world_size>
struct TwoshotQ8LineCodec {
  /*
      Int8-blocking Line codec for Twoshot collectives.
      We quantize the FP16 data to block-scaled Int8 in blocks of 32.
  */

  static int constexpr kAtoms = 8;
  static int constexpr kAtomStride = 256;
  static int constexpr kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int8x8_t (8B) and a fp16 scale shared among 32 values.
  static int constexpr kRankAtoms = kAtoms / kWorldSize;
  static int constexpr kRankTileStride = 2176;
  static int constexpr kRankTileScaleOffset = 2048;
  static int constexpr kRankTileSize = kRankTileStride * kRankAtoms;

  static int constexpr kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static int constexpr kTileSize = kRankTileSize * kWorldSize;

  // Q4 configuration
  static int constexpr kScaleFactor =
      0xA000A000;  // {-1/128.0h, -1/128.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xD800D800;      // {-128, -128}, fp16x2_t
  static int constexpr kRangeMax = 0x57F057F0;      // {+127, +127}, fp16x2_t
  static int constexpr kRangeBias = 0x00800080;     // {+128, +128}, int16x2_t

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
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(wmax) : "v"(a), "v"(b));

        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(wmin) : "v"(a), "v"(b));

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          asm volatile("v_pk_max_f16 %0, %1, %2"
                       : "=v"(wmax)
                       : "v"(wmax), "v"(x));

          int y = __shfl_down(wmin, i);
          asm volatile("v_pk_min_f16 %0, %1, %2"
                       : "=v"(wmin)
                       : "v"(wmin), "v"(y));
        }

        half2 wmaxh2 = __builtin_bit_cast(half2, wmax);
        half2 wminh2 = __builtin_bit_cast(half2, wmin);
        half2 wblockmaxh2;

        wblockmaxh2.x =
            __half2float(__habs(wmaxh2.x)) > __half2float(__habs(wminh2.x))
                ? wmaxh2.x
                : wminh2.x;
        wblockmaxh2.y =
            __half2float(__habs(wmaxh2.y)) > __half2float(__habs(wminh2.y))
                ? wmaxh2.y
                : wminh2.y;
        wblockmax = __builtin_bit_cast(int, wblockmaxh2);

        // Share with the cohort
        wblockmax = __shfl(wblockmax, group_leader);
      }

      // Derive scales
      int decoding_scale;
      int encoding_scale;
      asm volatile("v_pk_mul_f16 %0, %1, %2"
                   : "=v"(decoding_scale)
                   : "v"(wblockmax), "v"(kScaleFactor));
      asm volatile("v_pk_add_f16 %0, %1, %2"
                   : "=v"(encoding_scale)
                   : "v"(decoding_scale), "v"(kScaleEpsilon));
      encoding_scale = __builtin_bit_cast(
          int, h2rcp(__builtin_bit_cast(half2, encoding_scale)));

      // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(atom[i]), "v"(encoding_scale));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMin));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMax));
      }

      // Convert from fp16x2_t to uint16x2_t
      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(__half2float(wh[i]));

        for (int i = 0; i < 4; i++) {
          asm volatile("v_pk_add_i16 %0, %1, %2"
                       : "=v"(q[i])
                       : "v"(q[i]), "v"(kRangeBias));
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
        static uint constexpr kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1152 =
            0xE480E480;  // {-1152.0, -1152.0}, fp16x2_t

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int32_t q8 = ((qw[i / 2] >> ((i % 2) * 8)) & kMask00FF) | kHalf2_1024;
          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(w[i])
                       : "v"(q8), "v"(kHalf2_1152));
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(qs));
      }

      // That's pretty much it...
      data[k] = w;
    }
  }
};

template <int world_size>
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

  // static int constexpr kScaleFactor = 0x1C001C00;  // {1/256.0h, 1/256.0h},
  // fp16x2_t
  static int constexpr kScaleFactor =
      0x1C041C04;  // {1/255.0h, 1/255.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0x00000000;      // {0, 0}, fp16x2_t
  static int constexpr kRangeMax = 0x5BF85BF8;      // {+255, +255}, fp16x2_t
  static int constexpr kRangeBias = 0x00800080;     // {+128, +128}, int16x2_t

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
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(wmax) : "v"(a), "v"(b));

        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(a)
                     : "v"(atom[0]), "v"(atom[1]));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(b)
                     : "v"(atom[2]), "v"(atom[3]));
        asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(wmin) : "v"(a), "v"(b));

        // Reduce the max among a group of 8 threads
        // Note: This is basically 2 blocks of 32 values setup as the
        // upper/lower halves of the fp16x2_t
        for (int i = 1; i < 8; i <<= 1) {
          int x = __shfl_down(wmax, i);
          asm volatile("v_pk_max_f16 %0, %1, %2"
                       : "=v"(wmax)
                       : "v"(wmax), "v"(x));

          int y = __shfl_down(wmin, i);
          asm volatile("v_pk_min_f16 %0, %1, %2"
                       : "=v"(wmin)
                       : "v"(wmin), "v"(y));
        }

        // Share with the cohort
        wblockmax = __shfl(wmax, group_leader);
        wblockmin = __shfl(wmin, group_leader);
      }

      // Derive zeros and scales
      int decoding_zero = wblockmin;
      int decoding_scale;
      int encoding_scale;
      static int constexpr kNegOne = 0xBC00BC00;  // {-1, -1}, fp16x2_t

      // MI300 lacks packed fp16 sub instruction. So we do -1 * min + max
      asm volatile("v_pk_fma_f16 %0, %1, %2 %3"
                   : "=v"(decoding_scale)
                   : "v"(kNegOne), "v"(decoding_zero), "v"(wblockmax));
      asm volatile("v_pk_mul_f16 %0, %1, %2"
                   : "=v"(decoding_scale)
                   : "v"(decoding_scale), "v"(kScaleFactor));
      asm volatile("v_pk_add_f16 %0, %1, %2"
                   : "=v"(encoding_scale)
                   : "v"(decoding_scale), "v"(kScaleEpsilon));
      encoding_scale = __builtin_bit_cast(
          int, h2rcp(__builtin_bit_cast(half2, encoding_scale)));

      // // Apply scales to get quantized values
      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_fma_f16 %0, %1, %2 %3"
                     : "=v"(w[i])
                     : "v"(kNegOne), "v"(decoding_zero), "v"(atom[i]));
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(encoding_scale));
        asm volatile("v_pk_max_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMin));
        asm volatile("v_pk_min_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(kRangeMax));
      }

      // Convert from fp16x8_t to uint8x8_t and pack into int32x2_t
      int32x2_t qw;
      {
        unsigned char* qi = reinterpret_cast<unsigned char*>(&qw);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (unsigned char)__half2float(wh[i]);
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
        half* wh = reinterpret_cast<half*>(&w);
        unsigned char* qi = reinterpret_cast<unsigned char*>(&qw);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          wh[i] = __float2half((float)qi[i]);
        }
      }

      // Apply decoding scales and zeros
      for (int i = 0; i < 4; i++) {
        asm volatile("v_pk_mul_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(qs));
        asm volatile("v_pk_add_f16 %0, %1, %2"
                     : "=v"(w[i])
                     : "v"(w[i]), "v"(qz));
      }

      data[k] = w;
    }
  }
};

// MARK: Twoshot All Reduce
template <class LineCodec>
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
      half const* __restrict__ A,  // input
      half* __restrict__ B,        // output
      int const N,                 // number of elements
      int const block,             // block index
      int const num_blocks,        // number of blocks
      int const world_size,  // unused - only kept around for API consistency
      int const rank,        // rank index
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

    BufferResource src_buffer(const_cast<half*>(A), N * sizeof(half));
    int src_offset = block * kTileSize + thread * sizeof(int32x4_t);
    int32x4_t* src = reinterpret_cast<int32x4_t*>(const_cast<half*>(A));

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
          int32x4_t& tA_fragment = tA[i];
          int32x4_t& tR_fragment = tR[i];

          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(tR_fragment[0])
                       : "v"(tR_fragment[0]), "v"(tA_fragment[0]));
          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(tR_fragment[1])
                       : "v"(tR_fragment[1]), "v"(tA_fragment[1]));
          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(tR_fragment[2])
                       : "v"(tR_fragment[2]), "v"(tA_fragment[2]));
          asm volatile("v_pk_add_f16 %0, %1, %2"
                       : "=v"(tR_fragment[3])
                       : "v"(tR_fragment[3]), "v"(tA_fragment[3]));
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
    BufferResource dst_buffer(B, N * sizeof(half));
    int dst_offset = block * kTileSize + thread * sizeof(int32x4_t);
    int32x4_t* dst = reinterpret_cast<int32x4_t*>(B);

    for (int i = 0; i < kAtoms; i++) {
      buffer_store_dwordx4(tA[i], dst_buffer.descriptor, dst_offset, 0, 0);
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};
