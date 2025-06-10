#pragma once

#include <hip/hip_runtime.h>
#include "base.h"

namespace quickreduce {

struct CodecBase {
  const int thread;
  const int rank;
  const int group_leader;
  __quickreduce_device_inline__ CodecBase(int thread, int rank)
      : thread(thread),
        rank(rank),
        group_leader((threadIdx.x / kThreadGroupSize) * kThreadGroupSize) {}
};

// Default full precision codec.
template <typename T, int world_size>
struct CodecFP16 : public CodecBase {
  static constexpr int kWorldSize = world_size;
  static constexpr int kRankAtoms = kAtoms / kWorldSize;

  // Codec tile size process by this workgroup.
  // Each thread processes atoms of f16x8_t (16B).
  static constexpr int kRankTransmittedTileSize =
      kBlockSize * kRankAtoms * sizeof(int32x4_t);
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTransmittedTileSize must be 16B aligned.");

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  __quickreduce_device_inline__ CodecFP16(int thread, int rank)
      : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                                          const int32x4_t* __restrict__ data,
                                          const int* validity) {
    for (int i = 0; i < kRankAtoms; i++) {
      __builtin_nontemporal_store(data[i], send_buffer + thread);
      send_buffer += kAtomStride;
    }
  }

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
                                          int32x4_t* __restrict__ data) {
    for (int i = 0; i < kRankAtoms; i++) {
      data[i] = __builtin_nontemporal_load(*recv_buffer + thread);
      *recv_buffer += kAtomStride;
    }
  }
};

// Int4 symmetric quantization codec.
// We quantize the FP16 data to block-scaled Int4 in blocks of 4 *
// kThreadGroupSize.
template <typename T, int world_size>
struct CodecQ4Symm : public CodecBase {
  static constexpr int kWorldSize = world_size;

  // Codec tile size process by this workgroup.
  // Each threads processes a fragment of fp16x8_t (16B),
  // into a int4x8_t (4B) and a fp16 scale shared among 32 values.
  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1152;
  static constexpr int kRankTileScaleOffset = 1024;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTransmittedTileSize must be 16B aligned.");

  static constexpr int kRankBufferTileStride =
      kRankTileStride / sizeof(int32x4_t);

  // Total tile size for the collective communication.
  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  // Constants configuration

  // {-1/8.0h, -1/8.0h}, f16x2_t
  static constexpr int kScaleFactor =
      std::is_same<T, half>::value ? 0xB000B000 : 0xBE00BE00;

  // {1e-7, 1e-7}, f16x2_t
  static constexpr int kScaleEpsilon =
      std::is_same<T, half>::value ? 0x00010001 : 0x33D733D7;

  // {-8, -8}, f16x2_t
  static constexpr int kRangeMin =
      std::is_same<T, half>::value ? 0xC800C800 : 0xC100C100;

  // {+7, +7}, f16x2_t
  static constexpr int kRangeMax =
      std::is_same<T, half>::value ? 0x47004700 : 0x40E040E0;

  // {+8, +8}, int16x2_t
  static constexpr int kRangeBias = 0x00080008;

  __quickreduce_device_inline__ CodecQ4Symm(int thread, int rank)
      : CodecBase(thread, rank) {}

  __quickreduce_device_inline__ void send(int32x4_t* __restrict__ send_buffer,
                                          const int32x4_t* __restrict__ data,
                                          const int* validity) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      // Compute the absolute maximum of the atom in the thread group
      // In 2 blocks of values, upper/lower halves of the f16x2_t
      int wblockmax = group_abs_max<T>(atom);

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

  __quickreduce_device_inline__ void recv(int32x4_t** __restrict__ recv_buffer,
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
        static constexpr uint kMask000F = 0x000F000F;
        static constexpr uint kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1032 =
            0xE408E408;  // {-1032.0, -1032.0}, fp16x2_t

        for (int i = 0; i < 4; i++) {
          if constexpr (std::is_same<T, half>::value) {
            int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kHalf2_1024;
            asm volatile("v_pk_add_f16 %0, %1, %2"
                         : "=v"(w[i])
                         : "v"(q4), "v"(kHalf2_1032));
          } else {
            int32_t int16_2 = (qw >> (i * 4)) & kMask000F;
            int16_t low = static_cast<int16_t>(int16_2 & 0xFFFF);
            int16_t high = static_cast<int16_t>((int16_2 >> 16) & 0xFFFF);
            nv_bfloat16 bf_low = __float2bfloat16(static_cast<float>(low));
            nv_bfloat16 bf_high = __float2bfloat16(static_cast<float>(high));
            nv_bfloat162 bf2 = __halves2bfloat162(bf_low, bf_high);
            int32_t packed_bf16 = *reinterpret_cast<int32_t*>(&bf2);
            w[i] = packed_add<nv_bfloat16>(packed_bf16, kRangeMin);
          }
        }
      }

      // Apply decoding scales
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<T>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

// Oneshot AllReduce
template <typename T, int world_size>
struct AllReduceOneshot {
  static_assert(sizeof(T) == 2);

  __device__ static void run(
      T const* __restrict__ A,             // input
      T* __restrict__ B,                   // output
      int const N,                         // number of elements
      int const rank,                      // rank index
      uint8_t** __restrict__ buffer_list,  // communication buffers
      long const data_offset,              // offset to start of the data buffer
      int flag_color) {
    BufferResource src_buffer(const_cast<T*>(A), N * sizeof(T));
    BufferResource dst_buffer(B, N * sizeof(T));

    uint8_t* rank_buffer = buffer_list[rank];

    const int block_size = blockDim.x;
    const int thread = threadIdx.x;
    const int block = blockIdx.x;
    const int problem_size = (N + 3) / 4;

    int32x4_t tA, tB;
    long grid = gridDim.x;
    long data_stride = grid * block_size * sizeof(int32x4_t);
    long comm_flags0_offset = block * (world_size * sizeof(int));
    long comm_flags1_offset =
        comm_flags0_offset + grid * (world_size * sizeof(int));

    for (int idx = block * block_size + thread; idx < problem_size;
         idx += grid * block_size) {
      // load values
      tA = buffer_load_dwordx4(src_buffer.descriptor, idx * sizeof(int32x4_t),
                               0, 0);

      // Write rank data into this rank segment of every rank's communication
      // buffer.
#pragma unroll
      for (int r = 0; r < world_size; r++) {
        int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
            buffer_list[r] + data_offset + rank * data_stride +
            idx * sizeof(int32x4_t));
        __builtin_nontemporal_store(tA, send_buffer);
      }
    }

    __syncthreads();
    if (thread < world_size) {
      int r = thread;
      int* peer_flag_ptr = reinterpret_cast<int*>(
          buffer_list[r] + comm_flags0_offset + rank * sizeof(int));
      __atomic_store_n(peer_flag_ptr, flag_color, __ATOMIC_RELEASE);
      int* self_flag_ptr = reinterpret_cast<int*>(
          rank_buffer + comm_flags0_offset + r * sizeof(int));

      // Wait for the flags to be set.
      while (__atomic_load_n(self_flag_ptr, __ATOMIC_ACQUIRE) != flag_color) {
      }
    }
    __syncthreads();

    for (int idx = block * block_size + thread; idx < problem_size;
         idx += grid * block_size) {
      {
        int r = 0;
        // Read posted data from the rank's communication buffer.
        int32x4_t* recv_buffer = reinterpret_cast<int32x4_t*>(
            rank_buffer + data_offset + r * data_stride +
            idx * sizeof(int32x4_t));
        tA = __builtin_nontemporal_load(recv_buffer);
      }
#pragma unroll
      for (int r = 1; r < world_size; r++) {
        // Read posted data from the rank's communication buffer.
        int32x4_t* recv_buffer = reinterpret_cast<int32x4_t*>(
            rank_buffer + data_offset + r * data_stride +
            idx * sizeof(int32x4_t));
        tB = __builtin_nontemporal_load(recv_buffer);

        // Reduce the local data with the read data
        packed_assign_add<T>(&tA, &tB);
      }

      buffer_store_dwordx4(tA, dst_buffer.descriptor, idx * sizeof(int32x4_t),
                           0, 0);
    }

    __syncthreads();
    if (thread < world_size) {
      int r = thread;
      int* peer_flag_ptr = reinterpret_cast<int*>(
          buffer_list[r] + comm_flags1_offset + rank * sizeof(int));
      __atomic_store_n(peer_flag_ptr, flag_color, __ATOMIC_RELAXED);
      int* self_flag_ptr = reinterpret_cast<int*>(
          rank_buffer + comm_flags1_offset + r * sizeof(int));

      // Wait for the flags to be set.
      while (__atomic_load_n(self_flag_ptr, __ATOMIC_RELAXED) != flag_color) {
      }
    }
  }
};

// Twoshot All Reduce
template <typename T, class Codec>
struct AllReduceTwoshot {
  static_assert(sizeof(T) == 2);

  static constexpr int kWorldSize = Codec::kWorldSize;

  __device__ static void run(
      T const* __restrict__ input, T* __restrict__ output,
      int const N,                         // number of elements
      int const block,                     // block index
      int const num_blocks,                // number of blocks
      int const rank,                      // rank index
      uint8_t** __restrict__ buffer_list,  // communication buffers
      long const data_offset,              // offset to start of the data buffer
      int flag_color) {
    // Topology
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    uint8_t* rank_buffer = buffer_list[rank];
    Codec codec(thread, rank);

    // --------------------------------------------------------
    // Read input into registers
    int32x4_t tA[kAtoms];
    int tA_validity[kAtoms];

    BufferResource src_buffer(const_cast<T*>(input), N * sizeof(T));
    int src_offset = block * kTileSize + thread * sizeof(int32x4_t);
    int32x4_t* src = reinterpret_cast<int32x4_t*>(const_cast<T*>(input));

    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      tA_validity[i] = src_offset < N * sizeof(T);
      src_offset += kAtomStride * sizeof(int32x4_t);
    }

    // --------------------------------------------------------
    // Phase-1A: Write segment data into the communication buffer of the target
    // rank responsible for this segment.
    long comm_data0_offset = data_offset + block * Codec::kTransmittedTileSize;
    long comm_data1_offset =
        num_blocks * Codec::kTransmittedTileSize + comm_data0_offset;

    long comm_flags0_offset = block * (kWorldSize * sizeof(int));
    long comm_flags1_offset =
        num_blocks * (kWorldSize * sizeof(int)) + comm_flags0_offset;

    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data0_offset +
                                       rank * Codec::kRankTransmittedTileSize);
      codec.send(send_buffer, &tA[r * Codec::kRankAtoms],
                 &tA_validity[r * Codec::kRankAtoms]);
    }

    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      int* flag_ptr = reinterpret_cast<int*>(
          buffer_list[r] + comm_flags0_offset + rank * sizeof(int));
      set_sync_flag(flag_ptr, flag_color);
    }
    // --------------------------------------------------------
    // Phase-1B: Reduce the segment data from the communication buffers.
    int32x4_t tR[Codec::kRankAtoms] = {};
    {
      // Read the data from the communication buffer.
      int32x4_t* recv_buffer =
          reinterpret_cast<int32x4_t*>(rank_buffer + comm_data0_offset);
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + comm_flags0_offset);

      for (int r = 0; r < kWorldSize; r++) {
        // Wait for the flags to be set.
        if (thread == 0) {
          wait_sync_flag(&flag_ptr[r], flag_color);
        }
        __syncthreads();

        // note: we reuse tA as temp buffer here
        codec.recv(&recv_buffer, tA);

        for (int i = 0; i < Codec::kRankAtoms; i++) {
          packed_assign_add<T>(&tR[i], &tA[i]);
        }
      }
    }

    // Phase-2: Write the reduced segment to every other rank
    for (int r = 0; r < kWorldSize; r++) {
      int32x4_t* send_buffer =
          reinterpret_cast<int32x4_t*>(buffer_list[r] + comm_data1_offset +
                                       rank * Codec::kRankTransmittedTileSize);
      codec.send(send_buffer, tR, &tA_validity[rank * Codec::kRankAtoms]);
    }

    __syncthreads();
    if (thread < kWorldSize) {
      int r = thread;
      int* flag_ptr = reinterpret_cast<int*>(
          buffer_list[r] + comm_flags1_offset + rank * sizeof(int));
      set_sync_flag(flag_ptr, flag_color);
    }

    // Phase-2: Read the gather segments from the rank's communication buffer.
    {
      // Read the data from the communication buffer.
      int32x4_t* recv_buffer =
          reinterpret_cast<int32x4_t*>(rank_buffer + comm_data1_offset);
      int* flag_ptr = reinterpret_cast<int*>(rank_buffer + comm_flags1_offset);

      for (int r = 0; r < kWorldSize; r++) {
        // Wait for the flags to be set.
        if (thread == 0) {
          wait_sync_flag(&flag_ptr[r], flag_color);
        }
        __syncthreads();

        // Gather all reduced and final rank segments into tA.
        codec.recv(&recv_buffer, &tA[r * Codec::kRankAtoms]);
      }
    }

    // --------------------------------------------------------
    // Write the result to output.
    BufferResource dst_buffer(output, N * sizeof(T));
    int dst_offset = block * kTileSize + thread * sizeof(int32x4_t);
    int32x4_t* dst = reinterpret_cast<int32x4_t*>(output);

    for (int i = 0; i < kAtoms; i++) {
      buffer_store_dwordx4(tA[i], dst_buffer.descriptor, dst_offset, 0, 0);
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

}  // namespace quickreduce