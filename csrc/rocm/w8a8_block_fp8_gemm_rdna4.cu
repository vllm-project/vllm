// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8 block-scaled FP8 GEMM for AMD RDNA4 (gfx1201), built on the gfx12 fp8
// WMMA instruction v_wmma_f32_16x16x16_fp8_fp8_w32.
//
//   C[m,n] = sum_k A[m,k]*As[m, k/128] * B[n,k]*Bs[k/128, n/128]
//
// Two kernel families, routed on M by the host entry:
//   M <= 64   dec::  BN=64 tiles, BM in {16,32,64}
//   M >  64   v40::  BM x BN in {64x128, 128x128, 128x256}
//
// Only the 128x128 quantisation block is supported: both families use a
// 128-wide k-tile and fold one scale per tile, so any other group_k would
// misalign the two scale streams. The host entry rejects it.
//
// No M padding is needed. A and As are read through raw buffer descriptors
// sized from the real M, so rows past M read as zero in hardware, and every
// store is guarded by `gr < M && gc < N`. The grid is sized from the real M.
//
// B may be wider than K: VLLM_ROCM_FP8_PADDING leaves weights as an [N, K]
// view of an [N, K+256] buffer, so the row stride is passed separately.

#include <cstdint>
#include <string>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>

#if defined(__HIPCC__) && defined(__gfx1201__)
  #define __HIP__RDNA4__
#endif

namespace vllm {
namespace rdna4_w8a8 {

using fp8_frag_t = int __attribute__((ext_vector_type(2)));   // 8 B, WMMA arg
using f32_acc_t = float __attribute__((ext_vector_type(8)));  // WMMA acc
using fp8_q_t = int __attribute__((ext_vector_type(4)));      // 16 B load
using f32x4_t = float __attribute__((ext_vector_type(4)));

// The only quantisation block this file implements.
constexpr int QUANT_BLOCK_N = 128;
constexpr int QUANT_BLOCK_K = 128;

constexpr int WAVE_SIZE = 32;

// ============================================================
// V40 FAMILY (M > 64): compute bound, wide tiles.
// ============================================================
namespace v40 {

constexpr int VK = 128;  // k-tile
constexpr int VWM = 32;  // wave tile M
constexpr int VWN = 64;  // wave tile N

template <int BM, int BN>
struct G {
  static constexpr int wm = BM / VWM;
  static constexpr int wn = BN / VWN;
  static constexpr int nw = wm * wn;
  static constexpr int bs = nw * WAVE_SIZE;
  static constexpr int atpr = bs / BM;
  static constexpr int achunk = VK / atpr;
  static constexpr int aq = achunk / 16;
  static constexpr int btpr = bs / BN;
  static constexpr int bchunk = VK / btpr;
  static constexpr int bq = bchunk / 16;
};

#if defined(__HIP__RDNA4__)

constexpr int VSTRIDE = 128;  // LDS row stride (no padding; XOR swizzle)

// gfx1201 raw-buffer descriptor word 3. A wrong value makes every buffer load
// silently return ZERO.
constexpr unsigned VRSRC = 0x31027000u;

__device__ __forceinline__ int vlds_off(int r, int p) {
  const int c = p >> 3;
  return r * 128 + (((c ^ (r & 15)) << 3) | (p & 7));
}

__device__ __forceinline__ int vclamp(long long b) {
  return b > 2147483647LL ? 2147483647 : (int)b;
}

// Returns an INTEGER vector; assigning to float would numerically convert the
// bit pattern instead of reinterpreting it.
__device__ __forceinline__ f32x4_t vbl4(__amdgpu_buffer_rsrc_t r, int off) {
  auto v = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0);
  f32x4_t f;
  __builtin_memcpy(&f, &v, sizeof(f32x4_t));
  return f;
}

  #define VBAR_ARRIVE()                                               \
    do {                                                              \
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); \
      __builtin_amdgcn_s_barrier_signal(-1);                          \
    } while (0)

  #define VBAR_WAIT()                                                 \
    do {                                                              \
      __builtin_amdgcn_s_barrier_wait(-1);                            \
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); \
    } while (0)

#endif  // __HIP__RDNA4__

template <int BM, int BN, int OUTPUT_TYPE>
__global__
    __attribute__((amdgpu_flat_work_group_size(G<BM, BN>::bs, G<BM, BN>::bs)))
    __attribute__((amdgpu_waves_per_eu(8))) void gemm_kernel(
        const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
        void* __restrict__ C, const float* __restrict__ As,
        const float* __restrict__ Bs, int M, int N, int K, int K_stride,
        int group_n, int group_k, int stride_As_m, int stride_Bs_n,
        int GROUP_SIZE_M) {
#if defined(__HIP__RDNA4__)
  using GG = G<BM, BN>;
  const int tid = threadIdx.x;
  const int wave = tid / WAVE_SIZE;
  const int lane = tid % WAVE_SIZE;
  const int lane_wrapped = lane % 16;
  const int lane_group = lane / 16;

  const int wave_m = wave / GG::wn;
  const int wave_n = wave % GG::wn;

  const int pid = blockIdx.x;
  const int num_pid_m = (M + BM - 1) / BM;
  const int num_pid_n = (N + BN - 1) / BN;
  const int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
  const int group_id = pid / num_pid_in_group;
  const int first_pid_m = group_id * GROUP_SIZE_M;
  const int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
  const int pid_m = first_pid_m + (pid % group_size_m);
  const int pid_n = (pid % num_pid_in_group) / group_size_m;

  const int block_m = pid_m * BM;
  const int block_n = pid_n * BN;

  __shared__ __align__(16) uint8_t smem_A[BM * VSTRIDE];
  __shared__ __align__(16) uint8_t smem_B[BN * VSTRIDE];
  __shared__ __align__(16) float smem_as[GG::nw * VWM * 4];

  f32_acc_t acc[2][4];
  #pragma unroll
  for (int i = 0; i < 2; ++i)
  #pragma unroll
    for (int j = 0; j < 4; ++j)
      acc[i][j] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int a_row = tid / GG::atpr;
  const int a_off = (tid % GG::atpr) * GG::achunk;
  const int b_row = tid / GG::btpr;
  const int b_off = (tid % GG::btpr) * GG::bchunk;

  const long long a_global_row = block_m + a_row;
  const long long b_global_row = block_n + b_row;
  const int num_k_steps = K / VK;

  const int as_row = block_m + wave_m * VWM + lane;
  const int bs_col = (block_n + wave_n * VWN) / group_n;

  const __amdgpu_buffer_rsrc_t rA = __builtin_amdgcn_make_buffer_rsrc(
      (void*)A, 0, vclamp((long long)M * K), VRSRC);
  const __amdgpu_buffer_rsrc_t rB = __builtin_amdgcn_make_buffer_rsrc(
      (void*)B, 0, vclamp((long long)N * K_stride), VRSRC);
  const __amdgpu_buffer_rsrc_t rAs = __builtin_amdgcn_make_buffer_rsrc(
      (void*)As, 0, vclamp((long long)M * stride_As_m * 4), VRSRC);

  int offA = (int)(a_global_row * K + a_off);
  int offB = (int)(b_global_row * K_stride + b_off);
  int offAs = as_row * stride_As_m * 4;
  int offBs = bs_col * 4;

  float* const my_as = smem_as + wave * VWM * 4;

  fp8_q_t qa[GG::aq], qb[GG::bq];

  VBAR_ARRIVE();
  for (int ki = 0; ki < num_k_steps; ++ki) {
  #pragma unroll
    for (int c = 0; c < GG::aq; ++c)
      qa[c] = __builtin_amdgcn_raw_buffer_load_b128(rA, offA + c * 16, 0, 0);
  #pragma unroll
    for (int c = 0; c < GG::bq; ++c)
      qb[c] = __builtin_amdgcn_raw_buffer_load_b128(rB, offB + c * 16, 0, 0);
    VBAR_WAIT();

  #pragma unroll
    for (int t = 0; t < GG::aq * 2; ++t) {
      fp8_frag_t f;
      f[0] = qa[t >> 1][(t & 1) * 2];
      f[1] = qa[t >> 1][(t & 1) * 2 + 1];
      *reinterpret_cast<fp8_frag_t*>(smem_A + vlds_off(a_row, a_off + t * 8)) =
          f;
    }
  #pragma unroll
    for (int t = 0; t < GG::bq * 2; ++t) {
      fp8_frag_t f;
      f[0] = qb[t >> 1][(t & 1) * 2];
      f[1] = qb[t >> 1][(t & 1) * 2 + 1];
      *reinterpret_cast<fp8_frag_t*>(smem_B + vlds_off(b_row, b_off + t * 8)) =
          f;
    }
    VBAR_ARRIVE();

    // One b128 covers 4 k-steps. The tail may over-read up to 3 floats, but
    // those slots are never read back because ki stops first.
    if ((ki & 3) == 0) {
      const f32x4_t q = vbl4(rAs, offAs);
      offAs += 16;
  #pragma unroll
      for (int s = 0; s < 4; ++s) my_as[s * VWM + lane] = q[s];
    }
    // readfirstlane marks Bs wave-uniform so this becomes a scalar load.
    const int uoff = __builtin_amdgcn_readfirstlane(offBs);
    const float b_s = *reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(Bs) + uoff);
    offBs += stride_Bs_n * 4;

    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront", "local");
    float a_s[16];
    {
      const float* base = my_as + (ki & 3) * VWM;
  #pragma unroll
      for (int i = 0; i < 2; ++i)
  #pragma unroll
        for (int h = 0; h < 2; ++h)
          *reinterpret_cast<f32x4_t*>(a_s + i * 8 + h * 4) =
              *reinterpret_cast<const f32x4_t*>(base + i * 16 + lane_group * 8 +
                                                h * 4);
    }
    VBAR_WAIT();

    f32_acc_t temp[2][4];
  #pragma unroll
    for (int i = 0; i < 2; ++i)
  #pragma unroll
      for (int j = 0; j < 4; ++j)
        temp[i][j] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  #pragma unroll
    for (int mk = 0; mk < 8; ++mk) {
      const int k_local = mk * 16;
      fp8_frag_t b_frag[4];
  #pragma unroll
      for (int j = 0; j < 4; ++j)
        b_frag[j] = *reinterpret_cast<const fp8_frag_t*>(
            smem_B + vlds_off(wave_n * VWN + j * 16 + lane_wrapped,
                              k_local + lane_group * 8));
  #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const fp8_frag_t a_frag = *reinterpret_cast<const fp8_frag_t*>(
            smem_A + vlds_off(wave_m * VWM + i * 16 + lane_wrapped,
                              k_local + lane_group * 8));
  #pragma unroll
        for (int j = 0; j < 4; ++j)
          temp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
              a_frag, b_frag[j], temp[i][j]);
      }
    }

  #pragma unroll
    for (int i = 0; i < 2; ++i)
  #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s = a_s[i * 8 + e] * b_s;
  #pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[i][j][e] = __builtin_fmaf(temp[i][j][e], s, acc[i][j][e]);
      }
    VBAR_ARRIVE();

    offA += VK;
    offB += VK;
  }
  VBAR_WAIT();

  #pragma unroll
  for (int i = 0; i < 2; ++i)
  #pragma unroll
    for (int j = 0; j < 4; ++j)
  #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int gr = block_m + wave_m * VWM + i * 16 + lane_group * 8 + e;
        const int gc = block_n + wave_n * VWN + j * 16 + lane_wrapped;
        if (gr < M && gc < N) {
          const long long idx = (long long)gr * N + gc;
          const float val = acc[i][j][e];
          if constexpr (OUTPUT_TYPE == 0) {
            reinterpret_cast<float*>(C)[idx] = val;
          } else if constexpr (OUTPUT_TYPE == 1) {
            reinterpret_cast<_Float16*>(C)[idx] = (_Float16)val;
          } else {
            unsigned int bits = __float_as_uint(val);
            bits += 0x7FFFu + ((bits >> 16) & 1u);
            reinterpret_cast<unsigned short*>(C)[idx] =
                (unsigned short)(bits >> 16);
          }
        }
      }
#endif  // __HIP__RDNA4__
}

// Launch helper so the (tile, output_type) cross product stays readable.
template <int BM, int BN>
void launch(int output_type, int total_blocks, hipStream_t stream,
            const uint8_t* a, const uint8_t* b, void* c, const float* as,
            const float* bs, int M, int N, int K, int K_stride, int group_n,
            int group_k, int stride_As_m, int stride_Bs_n, int gsm) {
  dim3 grid(total_blocks);
  dim3 block(G<BM, BN>::bs);
  switch (output_type) {
    case 0:
      gemm_kernel<BM, BN, 0><<<grid, block, 0, stream>>>(
          a, b, c, as, bs, M, N, K, K_stride, group_n, group_k, stride_As_m,
          stride_Bs_n, gsm);
      break;
    case 1:
      gemm_kernel<BM, BN, 1><<<grid, block, 0, stream>>>(
          a, b, c, as, bs, M, N, K, K_stride, group_n, group_k, stride_As_m,
          stride_Bs_n, gsm);
      break;
    default:
      gemm_kernel<BM, BN, 2><<<grid, block, 0, stream>>>(
          a, b, c, as, bs, M, N, K, K_stride, group_n, group_k, stride_As_m,
          stride_Bs_n, gsm);
      break;
  }
}

}  // namespace v40

// ============================================================
// SMALL-M FAMILY (M <= 64): bandwidth bound, small BN for more workgroups.
// ============================================================
namespace dec {

constexpr int DBK = 128;

template <int BM, int BN, int WN>
struct DG {
  static constexpr int nw = BN / WN;
  static constexpr int bs = nw * WAVE_SIZE;
  static constexpr int mt = BM / 16;
  static constexpr int nt = WN / 16;
  static constexpr int btpr = (bs >= BN) ? (bs / BN) : 1;
  static constexpr int brow = (bs >= BN) ? 1 : (BN / bs);
  static constexpr int bchunk = DBK / btpr;
  static constexpr int bq = bchunk / 16;
  static constexpr int atpr = (bs >= BM * 8) ? 8 : ((bs >= BM) ? (bs / BM) : 1);
  static constexpr int arow = (bs >= BM) ? 1 : (BM / bs);
  static constexpr int achunk = DBK / atpr;
  static constexpr int aq = achunk / 16;
};

#if defined(__HIP__RDNA4__)
constexpr int DSTRIDE = 128;  // LDS row stride, XOR-swizzled as in v40
#endif

template <int BM, int BN, int WN, int OUTPUT_TYPE>
__global__ __attribute__((amdgpu_flat_work_group_size(DG<BM, BN, WN>::bs,
                                                      DG<BM, BN, WN>::bs))) void
gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
            void* __restrict__ C, const float* __restrict__ As,
            const float* __restrict__ Bs, int M, int N, int K, int K_stride,
            int group_n, int group_k, int stride_As_m, int stride_Bs_n) {
#if defined(__HIP__RDNA4__)
  using G = DG<BM, BN, WN>;
  const int tid = threadIdx.x;
  const int wave = tid / WAVE_SIZE;
  const int lane = tid % WAVE_SIZE;
  const int lw = lane % 16;
  const int lg = lane / 16;

  const int block_n = blockIdx.x * BN;
  const int block_m = blockIdx.y * BM;
  const int num_k_steps = K / DBK;

  __shared__ __align__(16) uint8_t sA[BM * DSTRIDE];
  __shared__ __align__(16) uint8_t sB[BN * DSTRIDE];
  __shared__ __align__(16) float sAs[DG<BM, BN, WN>::nw * BM];

  f32_acc_t acc[G::mt][G::nt];
  #pragma unroll
  for (int i = 0; i < G::mt; ++i)
  #pragma unroll
    for (int j = 0; j < G::nt; ++j)
      acc[i][j] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int a_row = tid / G::atpr;
  const int a_off = (tid % G::atpr) * G::achunk;
  const int b_row = tid / G::btpr;
  const int b_off = (tid % G::btpr) * G::bchunk;

  const __amdgpu_buffer_rsrc_t rA = __builtin_amdgcn_make_buffer_rsrc(
      (void*)A, 0, v40::vclamp((long long)M * K), v40::VRSRC);
  const __amdgpu_buffer_rsrc_t rB = __builtin_amdgcn_make_buffer_rsrc(
      (void*)B, 0, v40::vclamp((long long)N * K_stride), v40::VRSRC);
  const __amdgpu_buffer_rsrc_t rAs = __builtin_amdgcn_make_buffer_rsrc(
      (void*)As, 0, v40::vclamp((long long)M * stride_As_m * 4), v40::VRSRC);

  // Rows past M read as 0 through the buffer bound and their scales are forced
  // to 0, so they contribute nothing and are never written back.
  int offA = (int)((long long)(block_m + a_row) * K + a_off);
  int offB = (int)((long long)(block_n + b_row) * K_stride + b_off);
  int offAs = (block_m + lane) * stride_As_m * 4;
  int offBs = ((block_n + wave * WN) / group_n) * 4;

  fp8_q_t qb[G::bq * G::brow];
  fp8_q_t qa[G::aq * G::arow];

  VBAR_ARRIVE();
  for (int ki = 0; ki < num_k_steps; ++ki) {
  #pragma unroll
    for (int r = 0; r < G::brow; ++r)
  #pragma unroll
      for (int c = 0; c < G::bq; ++c)
        qb[r * G::bq + c] = __builtin_amdgcn_raw_buffer_load_b128(
            rB, offB + r * G::bs * K_stride + c * 16, 0, 0);
  #pragma unroll
    for (int r = 0; r < G::arow; ++r)
  #pragma unroll
      for (int c = 0; c < G::aq; ++c)
        qa[r * G::aq + c] = __builtin_amdgcn_raw_buffer_load_b128(
            rA, offA + r * G::bs * K + c * 16, 0, 0);
    VBAR_WAIT();

  #pragma unroll
    for (int r = 0; r < G::brow; ++r)
  #pragma unroll
      for (int t = 0; t < G::bq * 2; ++t) {
        fp8_frag_t f;
        f[0] = qb[r * G::bq + (t >> 1)][(t & 1) * 2];
        f[1] = qb[r * G::bq + (t >> 1)][(t & 1) * 2 + 1];
        *reinterpret_cast<fp8_frag_t*>(
            sB + v40::vlds_off(b_row + r * G::bs, b_off + t * 8)) = f;
      }
  #pragma unroll
    for (int r = 0; r < G::arow; ++r)
  #pragma unroll
      for (int t = 0; t < G::aq * 2; ++t) {
        fp8_frag_t f;
        f[0] = qa[r * G::aq + (t >> 1)][(t & 1) * 2];
        f[1] = qa[r * G::aq + (t >> 1)][(t & 1) * 2 + 1];
        *reinterpret_cast<fp8_frag_t*>(
            sA + v40::vlds_off(a_row + r * G::bs, a_off + t * 8)) = f;
      }
    VBAR_ARRIVE();

    // Wave-private, so no workgroup barrier is needed; a shared buffer would
    // race, because the natural place to write it is AFTER the phase-B arrive.
    float* const my_as = sAs + wave * BM;
  #pragma unroll
    for (int r = 0; r < (BM + WAVE_SIZE - 1) / WAVE_SIZE; ++r) {
      const int rr = r * WAVE_SIZE + lane;
      if (rr < BM)
        my_as[rr] =
            (block_m + rr < M)
                ? __builtin_bit_cast(
                      float,
                      __builtin_amdgcn_raw_buffer_load_b32(
                          rAs, offAs + r * WAVE_SIZE * stride_As_m * 4, 0, 0))
                : 0.f;
    }
    const int uoff = __builtin_amdgcn_readfirstlane(offBs);
    const float b_s = *reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(Bs) + uoff);
    offAs += 4;
    offBs += stride_Bs_n * 4;

    VBAR_WAIT();
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront", "local");

    f32_acc_t temp[G::mt][G::nt];
  #pragma unroll
    for (int i = 0; i < G::mt; ++i)
  #pragma unroll
      for (int j = 0; j < G::nt; ++j)
        temp[i][j] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  #pragma unroll
    for (int mk = 0; mk < 8; ++mk) {
      const int kl = mk * 16;
      fp8_frag_t bf[G::nt];
  #pragma unroll
      for (int j = 0; j < G::nt; ++j)
        bf[j] = *reinterpret_cast<const fp8_frag_t*>(
            sB + v40::vlds_off(wave * WN + j * 16 + lw, kl + lg * 8));
  #pragma unroll
      for (int i = 0; i < G::mt; ++i) {
        const fp8_frag_t af = *reinterpret_cast<const fp8_frag_t*>(
            sA + v40::vlds_off(i * 16 + lw, kl + lg * 8));
  #pragma unroll
        for (int j = 0; j < G::nt; ++j)
          temp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
              af, bf[j], temp[i][j]);
      }
    }

  #pragma unroll
    for (int i = 0; i < G::mt; ++i)
  #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s = my_as[i * 16 + lg * 8 + e] * b_s;
  #pragma unroll
        for (int j = 0; j < G::nt; ++j)
          acc[i][j][e] = __builtin_fmaf(temp[i][j][e], s, acc[i][j][e]);
      }
    VBAR_ARRIVE();
    offA += DBK;
    offB += DBK;
  }
  VBAR_WAIT();

  #pragma unroll
  for (int i = 0; i < G::mt; ++i)
  #pragma unroll
    for (int j = 0; j < G::nt; ++j)
  #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int gr = block_m + i * 16 + lg * 8 + e;
        const int gc = block_n + wave * WN + j * 16 + lw;
        if (gr < M && gc < N) {
          const long long idx = (long long)gr * N + gc;
          const float val = acc[i][j][e];
          if constexpr (OUTPUT_TYPE == 0) {
            reinterpret_cast<float*>(C)[idx] = val;
          } else if constexpr (OUTPUT_TYPE == 1) {
            reinterpret_cast<_Float16*>(C)[idx] = (_Float16)val;
          } else {
            unsigned bits = __float_as_uint(val);
            bits += 0x7FFFu + ((bits >> 16) & 1u);
            reinterpret_cast<unsigned short*>(C)[idx] =
                (unsigned short)(bits >> 16);
          }
        }
      }
#endif  // __HIP__RDNA4__
}

template <int BM, int BN, int WN>
void launch(int output_type, int M, int N, hipStream_t stream, const uint8_t* a,
            const uint8_t* b, void* c, const float* as, const float* bs, int K,
            int K_stride, int group_n, int group_k, int stride_As_m,
            int stride_Bs_n) {
  dim3 grid(N / BN, (M + BM - 1) / BM);
  dim3 block(DG<BM, BN, WN>::bs);
  switch (output_type) {
    case 0:
      gemm_kernel<BM, BN, WN, 0><<<grid, block, 0, stream>>>(
          a, b, c, as, bs, M, N, K, K_stride, group_n, group_k, stride_As_m,
          stride_Bs_n);
      break;
    case 1:
      gemm_kernel<BM, BN, WN, 1><<<grid, block, 0, stream>>>(
          a, b, c, as, bs, M, N, K, K_stride, group_n, group_k, stride_As_m,
          stride_Bs_n);
      break;
    default:
      gemm_kernel<BM, BN, WN, 2><<<grid, block, 0, stream>>>(
          a, b, c, as, bs, M, N, K, K_stride, group_n, group_k, stride_As_m,
          stride_Bs_n);
      break;
  }
}

// BN is always 64; the host entry's N % 128 == 0 check covers N % 64 == 0.
// The BM choice at 33..64 is shape dependent: BM=32 doubles the workgroup
// count but reads B twice, BM=64 reads it once, so threshold on N.
inline void dispatch(int M, int N, int output_type, hipStream_t stream,
                     const uint8_t* a, const uint8_t* b, void* c,
                     const float* as, const float* bs, int K, int K_stride,
                     int group_n, int group_k, int stride_As_m,
                     int stride_Bs_n) {
  if (M <= 16) {
    launch<16, 64, 16>(output_type, M, N, stream, a, b, c, as, bs, K, K_stride,
                       group_n, group_k, stride_As_m, stride_Bs_n);
  } else if (M <= 32) {
    launch<32, 64, 16>(output_type, M, N, stream, a, b, c, as, bs, K, K_stride,
                       group_n, group_k, stride_As_m, stride_Bs_n);
  } else if (N >= 8192) {
    launch<64, 64, 16>(output_type, M, N, stream, a, b, c, as, bs, K, K_stride,
                       group_n, group_k, stride_As_m, stride_Bs_n);
  } else {
    launch<32, 64, 16>(output_type, M, N, stream, a, b, c, as, bs, K, K_stride,
                       group_n, group_k, stride_As_m, stride_Bs_n);
  }
}

}  // namespace dec

namespace {

// _rocm_C is one binary for every arch in the build, so this translation unit
// also exists on non-gfx1201 targets with empty kernel bodies; launching those
// would silently return zeros rather than fail.
bool current_device_is_gfx1201() {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  return props != nullptr &&
         std::string(props->gcnArchName).find("gfx1201") != std::string::npos;
}

int output_type_of(const torch::Tensor& C) {
  if (C.scalar_type() == torch::kFloat32) return 0;
  if (C.scalar_type() == torch::kFloat16) return 1;
  return 2;  // bf16
}

}  // namespace

}  // namespace rdna4_w8a8
}  // namespace vllm

void w8a8_block_fp8_gemm_rdna4(const torch::Tensor& A, const torch::Tensor& B,
                               const torch::Tensor& As, const torch::Tensor& Bs,
                               torch::Tensor& C, int64_t group_n,
                               int64_t group_k) {
  using namespace vllm::rdna4_w8a8;

  TORCH_CHECK(current_device_is_gfx1201(),
              "w8a8_block_fp8_gemm_rdna4 is only implemented for gfx1201");

  // Any other block would misalign the activation and weight scale streams.
  TORCH_CHECK(group_n == QUANT_BLOCK_N && group_k == QUANT_BLOCK_K,
              "w8a8_block_fp8_gemm_rdna4 only supports a ", QUANT_BLOCK_N, "x",
              QUANT_BLOCK_K, " quantisation block, got ", group_n, "x",
              group_k);

  TORCH_CHECK(
      A.is_cuda() && B.is_cuda() && As.is_cuda() && Bs.is_cuda() && C.is_cuda(),
      "all tensors must be on the GPU");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && As.dim() == 2 && Bs.dim() == 2 &&
                  C.dim() == 2,
              "all tensors must be 2D");

  TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fn ||
                  A.scalar_type() == torch::kUInt8,
              "A must be float8_e4m3fn or uint8");
  TORCH_CHECK(B.scalar_type() == torch::kFloat8_e4m3fn ||
                  B.scalar_type() == torch::kUInt8,
              "B must be float8_e4m3fn or uint8");
  TORCH_CHECK(As.scalar_type() == torch::kFloat32, "As must be float32");
  TORCH_CHECK(Bs.scalar_type() == torch::kFloat32, "Bs must be float32");
  TORCH_CHECK(C.scalar_type() == torch::kFloat32 ||
                  C.scalar_type() == torch::kFloat16 ||
                  C.scalar_type() == torch::kBFloat16,
              "C must be float32, float16 or bfloat16");

  const int M = (int)A.size(0);
  const int K = (int)A.size(1);
  const int N = (int)B.size(0);
  const int K_stride = (int)B.size(1);

  TORCH_CHECK(N % QUANT_BLOCK_N == 0, "N=", N, " must be divisible by ",
              QUANT_BLOCK_N);
  TORCH_CHECK(K % QUANT_BLOCK_K == 0, "K=", K, " must be divisible by ",
              QUANT_BLOCK_K);
  TORCH_CHECK(K_stride >= K, "B row stride ", K_stride, " must be >= K=", K);

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  // B may be a column-narrowed view of a wider buffer, so require row-major
  // with unit column stride rather than full contiguity.
  TORCH_CHECK(B.stride(1) == 1 && B.stride(0) == K_stride,
              "B must be row-major with unit column stride");
  TORCH_CHECK(As.is_contiguous(), "As must be contiguous");
  TORCH_CHECK(Bs.is_contiguous(), "Bs must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

  const int num_k_groups = K / QUANT_BLOCK_K;
  const int num_n_groups = N / QUANT_BLOCK_N;

  TORCH_CHECK(As.size(0) == M && As.size(1) == num_k_groups, "As must be [", M,
              ", ", num_k_groups, "], got [", As.size(0), ", ", As.size(1),
              "]");
  // Asserted, never inferred: for a square scale grid (N/gn == K/gk) a
  // transposed Bs is shape-indistinguishable from a correct one, so guessing
  // the orientation would silently return a wrong result. The caller
  // transposes once at load.
  TORCH_CHECK(Bs.size(0) == num_k_groups && Bs.size(1) == num_n_groups,
              "Bs must be [", num_k_groups, ", ", num_n_groups,
              "] (k-major), got [", Bs.size(0), ", ", Bs.size(1), "]");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C must be [", M, ", ", N, "]");

  if (M == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  auto stream = at::cuda::getCurrentCUDAStream();

  const uint8_t* a_ptr = reinterpret_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(B.data_ptr());
  const float* as_ptr = As.data_ptr<float>();
  const float* bs_ptr = Bs.data_ptr<float>();
  void* c_ptr = C.data_ptr();

  const int stride_As_m = num_k_groups;
  const int stride_Bs_n = num_n_groups;
  const int output_type = output_type_of(C);
  constexpr int GROUP_SIZE_M = 8;

  // The tile is a pure function of M: at small M a wide tile launches too few
  // workgroups to fill the device, so grid coverage beats occupancy.
  if (M > 64) {
    int BMs, BNs;
    if (M < 384) {
      BMs = 64;
      BNs = 128;
    } else if (M < 768) {
      BMs = 128;
      BNs = 128;
    } else {
      BMs = 128;
      BNs = 256;
    }
    // 128x256 needs N % 256 == 0; fall back rather than compute garbage.
    if (BNs == 256 && (N % 256) != 0) BNs = 128;

    const int total_blocks = ((M + BMs - 1) / BMs) * (N / BNs);

    if (BMs == 64) {
      v40::launch<64, 128>(output_type, total_blocks, stream, a_ptr, b_ptr,
                           c_ptr, as_ptr, bs_ptr, M, N, K, K_stride,
                           (int)group_n, (int)group_k, stride_As_m, stride_Bs_n,
                           GROUP_SIZE_M);
    } else if (BNs == 128) {
      v40::launch<128, 128>(output_type, total_blocks, stream, a_ptr, b_ptr,
                            c_ptr, as_ptr, bs_ptr, M, N, K, K_stride,
                            (int)group_n, (int)group_k, stride_As_m,
                            stride_Bs_n, GROUP_SIZE_M);
    } else {
      v40::launch<128, 256>(output_type, total_blocks, stream, a_ptr, b_ptr,
                            c_ptr, as_ptr, bs_ptr, M, N, K, K_stride,
                            (int)group_n, (int)group_k, stride_As_m,
                            stride_Bs_n, GROUP_SIZE_M);
    }
    return;
  }

  dec::dispatch(M, N, output_type, stream, a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
                K, K_stride, (int)group_n, (int)group_k, stride_As_m,
                stride_Bs_n);
}
