// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang's TP4 quick_all_reduce JIT transport.
#pragma once

namespace quickreduce::tp4_q4 {

static constexpr uint32_t kMaxBlocks = 256;

#define QR_TP4_GLOBAL(T, p) ((__attribute__((address_space(1))) T*)(p))

// Each wave must finish remote stores before a different wave publishes flags.
__quickreduce_device_inline__ void wait_own_stores() {
  __builtin_amdgcn_s_waitcnt(0x0F70);
}

__quickreduce_device_inline__ void set_sync_flag(uint32_t* flag_ptr,
                                                 uint32_t flag) {
  __atomic_store_n(QR_TP4_GLOBAL(uint32_t, flag_ptr), flag, __ATOMIC_RELEASE);
}

__quickreduce_device_inline__ void wait_sync_flag(uint32_t* flag_ptr,
                                                  uint32_t flag) {
  while (__atomic_load_n(QR_TP4_GLOBAL(uint32_t, flag_ptr), __ATOMIC_RELAXED) !=
         flag) {
  }
}

struct CodecQ4 : public CodecBase {
  static constexpr int kWorldSize = 4;
  static constexpr int kAtoms = 8;
  static constexpr int kTileSize = kBlockSize * kAtoms * sizeof(int32x4_t);

  static constexpr int kRankAtoms = kAtoms / kWorldSize;
  static constexpr int kRankTileStride = 1152;
  static constexpr int kRankTransmittedTileSize = kRankTileStride * kRankAtoms;
  static_assert(kRankTransmittedTileSize % 16 == 0,
                "kRankTransmittedTileSize must be 16B aligned.");

  static constexpr int kTransmittedTileSize =
      kRankTransmittedTileSize * kWorldSize;

  static constexpr int kScaleFactor = 0xB000B000;

  static constexpr int kScaleEpsilon = 0x00010001;

  static constexpr int kRangeMin = 0xC800C800;

  static constexpr int kRangeMax = 0x47004700;

  static constexpr int kRangeBias = 0x00080008;

  __quickreduce_device_inline__ CodecQ4(int thread, int rank)
      : CodecBase(thread, rank) {}

  struct Packed {
    int32_t qw;
    int scale;
  };

  __quickreduce_device_inline__ void encode(int32x4_t const* __restrict__ data,
                                            Packed* __restrict__ packed) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32x4_t const atom = data[k];

      int wblockmax = group_abs_max<half>(atom);

      int decoding_scale;
      int encoding_scale;
      decoding_scale = packed_mul<half>(wblockmax, kScaleFactor);
      encoding_scale = packed_add<half>(decoding_scale, kScaleEpsilon);
      encoding_scale = packed_rcp<half>(encoding_scale);

      int32x4_t w;
      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<half>(atom[i], encoding_scale);
        w[i] = packed_max<half>(w[i], kRangeMin);
        w[i] = packed_min<half>(w[i], kRangeMax);
      }

      int32x4_t q;
      {
        int16_t* qi = reinterpret_cast<int16_t*>(&q);
        half* wh = reinterpret_cast<half*>(&w);
        for (int i = 0; i < 8; i++) qi[i] = (int16_t)rintf(T2float_cast(wh[i]));

        for (int i = 0; i < 4; i++) {
          q[i] = packed_add<int16_t>(q[i], kRangeBias);
        }
      }

      int qw = q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12);

      packed[k].qw = qw;
      packed[k].scale = decoding_scale;
    }
  }

  static constexpr int kWaveSegment = kRankAtoms * (kWavefront * 4 + 8 * 4);
  static_assert(kWaveSegment * (kBlockSize / kWavefront) ==
                    kRankTransmittedTileSize,
                "layout covers the rank tile");

  __quickreduce_device_inline__ void store(uint8_t* __restrict__ send_buffer,
                                           Packed const* __restrict__ packed) {
    int wave = thread / kWavefront, lane = thread % kWavefront;
    uint8_t* wave_buffer = send_buffer + wave * kWaveSegment;
    static_assert(kRankAtoms == 2,
                  "a lane moves its two atoms in one 8B access");
    int32x2_t qw = {packed[0].qw, packed[1].qw};
    __builtin_nontemporal_store(qw,
                                QR_TP4_GLOBAL(int32x2_t, wave_buffer) + lane);
    int slot = lane % (8 * kRankAtoms);
    int src_lane = (slot % 8) * kThreadGroupSize;
    int scale = 0;
    for (int k = 0; k < kRankAtoms; k++) {
      int v = __shfl(packed[k].scale, src_lane);
      scale = (slot / 8 == k) ? v : scale;
    }
    __builtin_nontemporal_store(
        scale,
        QR_TP4_GLOBAL(int, wave_buffer + kWavefront * 4 * kRankAtoms) + slot);
  }

  __quickreduce_device_inline__ void load(uint8_t* __restrict__ recv_buffer,
                                          Packed* __restrict__ packed) {
    int wave = thread / kWavefront, lane = thread % kWavefront;
    uint8_t* wave_buffer = recv_buffer + wave * kWaveSegment;
    int32x2_t qw = __builtin_nontemporal_load(
        QR_TP4_GLOBAL(int32x2_t, wave_buffer) + lane);
    packed[0].qw = qw[0];
    packed[1].qw = qw[1];
    int group = lane / kThreadGroupSize;
    for (int k = 0; k < kRankAtoms; k++) {
      packed[k].scale = __builtin_nontemporal_load(
          QR_TP4_GLOBAL(int, wave_buffer + kWavefront * 4 * kRankAtoms) +
          k * 8 + group);
    }
  }

  __quickreduce_device_inline__ void decode(Packed const* __restrict__ packed,
                                            int32x4_t* __restrict__ data) {
    for (int k = 0; k < kRankAtoms; k++) {
      int32_t qw = packed[k].qw;
      int qs = packed[k].scale;

      int32x4_t w;
      {
        static constexpr uint kMask000F = 0x000F000F;
        static constexpr uint kHalf2_1024 =
            0x64006400;  // {1024.0, 1024.0}, fp16x2_t
        static uint constexpr kHalf2_1032 =
            0xE408E408;  // {-1032.0, -1032.0}, fp16x2_t

        for (int i = 0; i < 4; i++) {
          int32_t q4 = ((qw >> (i * 4)) & kMask000F) | kHalf2_1024;
          w[i] = packed_add<half>(q4, kHalf2_1032);
        }
      }

      for (int i = 0; i < 4; i++) {
        w[i] = packed_mul<half>(w[i], qs);
      }

      data[k] = w;
    }
  }
};

template <class Codec>
struct AllReduceTwoshot {
  static constexpr int kWorldSize = Codec::kWorldSize;
  static constexpr int kAtoms = Codec::kAtoms;
  static constexpr int kTileSize = Codec::kTileSize;

  __quickreduce_device_inline__ static uint8_t* lane_peer(uint8_t* const* peer,
                                                          int i) {
    uint8_t* p = peer[0];
#pragma unroll
    for (int r = 1; r < kWorldSize; r++) {
      p = (i == r) ? peer[r] : p;
    }
    return p;
  }

  __quickreduce_device_inline__ static void wait_and_load(
      Codec& codec, uint8_t* recv_buffer, uint32_t* flag_ptr,
      uint32_t flag_color,
      typename Codec::Packed (&packed)[kWorldSize][Codec::kRankAtoms]) {
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    if (thread < kWorldSize) {
      wait_sync_flag(&flag_ptr[thread], flag_color);
    }
    __syncthreads();
    for (int r = 0; r < kWorldSize; r++) {
      codec.load(recv_buffer + r * Codec::kRankTransmittedTileSize, packed[r]);
    }
  }

  __quickreduce_device_inline__ static void raise_flags(uint8_t* const* peer,
                                                        uint32_t flags_offset,
                                                        int rank,
                                                        uint32_t flag_color) {
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    wait_own_stores();
    __syncthreads();
    if (thread < kWorldSize) {
      set_sync_flag(
          reinterpret_cast<uint32_t*>(lane_peer(peer, thread) + flags_offset +
                                      rank * sizeof(uint32_t)),
          flag_color);
    }
  }

  __device__ static void run(half const* __restrict__ input,
                             half* __restrict__ output, uint32_t const N,
                             int const block, int const rank,
                             uint8_t* const* peer, uint32_t const data_offset,
                             uint32_t flag_color, int64_t data_size_per_phase) {
    int thread = threadIdx.x + threadIdx.y * kWavefront;
    uint8_t* rank_buffer = lane_peer(peer, rank);
    Codec codec(thread, rank);
    int block_id = blockIdx.x;
    int32x4_t tA[kAtoms];

    BufferResource src_buffer(const_cast<half*>(input), N * sizeof(half));
    uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);
    for (int i = 0; i < kAtoms; i++) {
      tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
      src_offset += kAtomStride * sizeof(int32x4_t);
    }
    __builtin_amdgcn_sched_barrier(0);
    for (int i = 0; i < kAtoms; i++) {
      const nv_bfloat162* bf_buf =
          reinterpret_cast<const nv_bfloat162*>(&tA[i]);
      half2 half_buf[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        half_buf[j] = __float22half2_rn(__bfloat1622float2(bf_buf[j]));
      }
      tA[i] = *reinterpret_cast<const int32x4_t*>(half_buf);
    }

    uint32_t comm_data0_offset =
        data_offset + block_id * Codec::kTransmittedTileSize;
    uint32_t comm_data1_offset = data_size_per_phase + comm_data0_offset;

    uint32_t comm_flags0_offset = block_id * (kWorldSize * sizeof(uint32_t));
    uint32_t comm_flags1_offset = (data_offset / 2) + comm_flags0_offset;

    for (int r = 0; r < kWorldSize; r++) {
      typename Codec::Packed packed[Codec::kRankAtoms];
      codec.encode(&tA[r * Codec::kRankAtoms], packed);
      codec.store(
          peer[r] + comm_data0_offset + rank * Codec::kRankTransmittedTileSize,
          packed);
    }
    raise_flags(peer, comm_flags0_offset, rank, flag_color);

    int32x4_t tR[Codec::kRankAtoms] = {};
    {
      typename Codec::Packed packed[kWorldSize][Codec::kRankAtoms];
      wait_and_load(
          codec, rank_buffer + comm_data0_offset,
          reinterpret_cast<uint32_t*>(rank_buffer + comm_flags0_offset),
          flag_color, packed);
      for (int r = 0; r < kWorldSize; r++) {
        codec.decode(packed[r], tA);
        for (int i = 0; i < Codec::kRankAtoms; i++) {
          packed_assign_add<half>(&tR[i], &tA[i]);
        }
      }
    }

    {
      typename Codec::Packed packed[Codec::kRankAtoms];
      codec.encode(tR, packed);
      for (int r = 0; r < kWorldSize; r++) {
        codec.store(peer[r] + comm_data1_offset +
                        rank * Codec::kRankTransmittedTileSize,
                    packed);
      }
    }
    raise_flags(peer, comm_flags1_offset, rank, flag_color);

    {
      typename Codec::Packed packed[kWorldSize][Codec::kRankAtoms];
      wait_and_load(
          codec, rank_buffer + comm_data1_offset,
          reinterpret_cast<uint32_t*>(rank_buffer + comm_flags1_offset),
          flag_color, packed);
      for (int r = 0; r < kWorldSize; r++) {
        codec.decode(packed[r], &tA[r * Codec::kRankAtoms]);
      }
    }

    BufferResource dst_buffer(output, N * sizeof(half));
    uint32_t dst_offset = block * kTileSize + thread * sizeof(int32x4_t);

    for (int i = 0; i < kAtoms; i++) {
      const half2* half_buf = reinterpret_cast<const half2*>(&tA[i]);
      nv_bfloat162 bf16_buf[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        bf16_buf[j] = __float22bfloat162_rn(__half22float2(half_buf[j]));
      }
      buffer_store_dwordx4(*reinterpret_cast<const int32x4_t*>(bf16_buf),
                           dst_buffer.descriptor, dst_offset, 0, 0);
      dst_offset += kAtomStride * sizeof(int32x4_t);
    }
  }
};

__global__ __quickreduce_launch_bounds_two_shot__ static void allreduce(
    const half* input, half* output, uint32_t count, uint32_t num_blocks,
    int rank, uint8_t** buffers, uint32_t data_offset, uint32_t* counters,
    int64_t bytes_per_phase) {
  int block = blockIdx.x;
  uint32_t color = counters[blockIdx.x];
#if defined(__gfx950__)
  using Kernel = AllReduceTwoshot<CodecQ4>;
  uint8_t* peers[4];
  #pragma unroll
  for (int r = 0; r < 4; r++) {
    peers[r] = buffers[r];
  }
  uint8_t** selected_buffers = peers;
#else
  using Kernel =
      ::quickreduce::AllReduceTwoshot<half, ::quickreduce::CodecQ4<half, 4>,
                                      true>;
  uint8_t** selected_buffers = buffers;
#endif
  while (block < num_blocks) {
    Kernel::run(input, output, count, block, rank, selected_buffers,
                data_offset, color, bytes_per_phase);
    block += gridDim.x;
    color++;
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    counters[blockIdx.x] = color;
  }
}

#undef QR_TP4_GLOBAL

}  // namespace quickreduce::tp4_q4
