// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>

namespace vllm::batch_invariant {
namespace {

struct MatrixLayout {
  int64_t row, group, k;
};

__device__ uint32_t load_fragment(const uint8_t* values, MatrixLayout stride,
                                  int64_t row, int group, int k, int rows) {
  if (row >= rows) return 0;
  const uint8_t* base =
      values + row * stride.row + group * stride.group + k * stride.k;
  if (stride.k == 1 && reinterpret_cast<uintptr_t>(base) % 4 == 0) {
    return *reinterpret_cast<const uint32_t*>(base);
  }
  uint32_t packed = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    packed |= uint32_t(values[row * stride.row + group * stride.group +
                              (k + byte) * stride.k])
              << (8 * byte);
  }
  return packed;
}

template <typename Scale>
__device__ float load_scale(const Scale* scales, MatrixLayout stride,
                            int64_t row, int group, int block) {
  int column = sizeof(Scale) == 4 ? block / 4 : block;
  uint32_t bits =
      scales[row * stride.row + group * stride.group + column * stride.k];
  if constexpr (sizeof(Scale) == 4) bits = (bits >> (8 * (block % 4))) & 255;
  bits = bits == 0 ? 0x00400000 : bits == 255 ? 0x7FC00000 : bits << 23;
  return __uint_as_float(bits);
}

__device__ float accumulate_block(float result, float product, float a_scale,
                                  float b_scale) {
  float scaled;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(product), "f"(a_scale));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(scaled), "f"(b_scale));
  asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(result), "f"(scaled));
  return result;
}

template <typename Scale, typename Output>
__global__ void block32_mm_kernel(const uint8_t* a, const Scale* a_scale,
                                  const uint8_t* weight,
                                  const Scale* weight_scale, Output* output,
                                  int m, int n, int k, int groups,
                                  int scale_rows, MatrixLayout a_stride,
                                  MatrixLayout as_stride, MatrixLayout w_stride,
                                  MatrixLayout ws_stride) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row = blockIdx.x * 16 + lane / 4;
  const int column = blockIdx.y * 32 + warp * 8;
  const int group = blockIdx.z;
  float result[4] = {};
  for (int block = 0; block < k / 32; ++block) {
    const int offset = block * 32 + (lane % 4) * 4;
    uint32_t a0 = load_fragment(a, a_stride, row, group, offset, m);
    uint32_t a1 = load_fragment(a, a_stride, row + 8, group, offset, m);
    uint32_t a2 = load_fragment(a, a_stride, row, group, offset + 16, m);
    uint32_t a3 = load_fragment(a, a_stride, row + 8, group, offset + 16, m);
    uint32_t b0 =
        load_fragment(weight, w_stride, column + lane / 4, group, offset, n);
    uint32_t b1 = load_fragment(weight, w_stride, column + lane / 4, group,
                                offset + 16, n);
    float product[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};"
        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
#else
    asm volatile("trap;");
#endif
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const int out_row = row + (element >= 2 ? 8 : 0);
      const int out_column = column + (lane % 4) * 2 + (element % 2);
      if (out_row < m && out_column < n) {
        result[element] = accumulate_block(
            result[element], product[element],
            load_scale(a_scale, as_stride, out_row, group, block),
            load_scale(weight_scale, ws_stride, out_column / scale_rows, group,
                       block));
      }
    }
  }
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    const int out_row = row + (element >= 2 ? 8 : 0);
    const int out_column = column + (lane % 4) * 2 + (element % 2);
    if (out_row < m && out_column < n) {
      output[(int64_t(out_row) * groups + group) * n + out_column] =
          Output(result[element]);
    }
  }
}

MatrixLayout layout(const torch::Tensor& tensor, bool grouped, bool weight) {
  if (!grouped) return {tensor.stride(0), 0, tensor.stride(1)};
  return weight ? MatrixLayout{tensor.stride(1), tensor.stride(0),
                               tensor.stride(2)}
                : MatrixLayout{tensor.stride(0), tensor.stride(1),
                               tensor.stride(2)};
}

// Tiled kernels. Every output element runs the same instruction sequence as
// block32_mm_kernel: one m16n8k32 E4M3 MMA per K32 block with a zero
// accumulator, two FP32 scale multiplies and one FP32 add in ascending block
// order. Tiling only changes which thread issues it, so results are bitwise
// equal to the reference kernel for every M, tile shape and grid.

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem,
                                           bool valid) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(gmem), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem,
                                          bool valid) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst),
               "l"(gmem), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}

__device__ __forceinline__ void mma_k32(float (&d)[4], const uint32_t (&a)[4],
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "f"(0.0f));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ float scale_block(float product, float a_scale,
                                             float b_scale) {
  float scaled;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(product), "f"(a_scale));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(scaled), "f"(b_scale));
  return scaled;
}

__device__ __forceinline__ float add_rn(float a, float b) {
  float c;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

// Two outputs of one row: acc[i] += (p[i] * a) * w[i]. The packed FP32 ops
// flush subnormals, so callers use this only when every operand and result is
// a normal number (see packed_scale_safe); then each lane matches
// scale_block + add_rn bitwise.
__device__ __forceinline__ uint64_t pack2(float lo, float hi) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(lo), "f"(hi));
  return r;
}

// a2 = {a, a}, w2 = {w0, w1}.
__device__ __forceinline__ void scale_add_x2(float& acc0, float& acc1, float p0,
                                             float p1, uint64_t a2,
                                             uint64_t w2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm("{\n"
      ".reg .b64 x, c;\n"
      "mov.b64 x, {%2, %3};\n"
      "mov.b64 c, {%0, %1};\n"
      "mul.rn.f32x2 x, x, %4;\n"
      "mul.rn.f32x2 x, x, %5;\n"
      "add.rn.f32x2 c, c, x;\n"
      "mov.b64 {%0, %1}, c;\n"
      "}"
      : "+f"(acc0), "+f"(acc1)
      : "f"(p0), "f"(p1), "l"(a2), "l"(w2));
#else
  asm("trap;");
#endif
}

// A 32-bit scale word holds the E8M0 bytes of four consecutive K32 blocks,
// both for byte scales (four adjacent columns) and for packed INT32 scales.
__device__ __forceinline__ float decode_scale(uint32_t word, int block) {
  uint32_t bits = (word >> (8 * block)) & 255;
  bits = bits == 0 ? 0x00400000 : bits == 255 ? 0x7FC00000 : bits << 23;
  return __uint_as_float(bits);
}

// A nonzero K32 product of E4M3 codes is a multiple of 2^-18 below 2^23, so
// with both scale exponents in [73 - 127, 179 - 127] every scaled product,
// partial sum and scale is a normal FP32 number and no rounding mode or
// subnormal flush can differ from the scalar path.
__device__ __forceinline__ bool packed_scale_safe(uint32_t word) {
  bool safe = true;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t bits = (word >> (8 * b)) & 255;
    safe &= bits >= 73 && bits <= 179;
  }
  return safe;
}

template <typename Scale>
__device__ __forceinline__ const void* scale_word(const Scale* scales,
                                                  MatrixLayout s, int64_t row,
                                                  int group, int word) {
  if constexpr (sizeof(Scale) == 4) {
    return scales + row * s.row + group * s.group + int64_t(word) * s.k;
  } else {
    return scales + row * s.row + group * s.group + 4 * int64_t(word);
  }
}

struct TiledArgs {
  const uint8_t* a;
  const void* a_scale;
  const uint8_t* w;
  const void* w_scale;
  void* out;
  int m, n, k, groups, scale_rows;
  MatrixLayout as, ass, ws, wss;
};

// Rows of 16-byte chunks; chunk c of row r lives at c ^ (r & 7).
__device__ __forceinline__ int swz(int row, int chunk) {
  return chunk ^ (row & 7);
}

// One 128-byte stage (four K32 blocks) of a warp's MI x NI sub-tile.
// fa[kb * BM + row] = {a, a}; fw[kb * BN / 2 + col / 2] = {w_col, w_col+1}.
template <bool PACKED, int MI, int NI, int BM, int BN>
__device__ __forceinline__ void tile_stage(float (&acc)[MI][NI][4],
                                           const uint8_t* ta, const uint8_t* tw,
                                           const float2* fa, const float2* fw,
                                           int arow, int bcol, int lane) {
  constexpr int kRowBytes = 128;
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    uint32_t af[MI][4], bf[NI][2];
#pragma unroll
    for (int i = 0; i < MI; ++i) {
      const int mat = lane >> 3;
      const int r = arow + i * 16 + (lane & 7) + ((mat & 1) << 3);
      const int c = 2 * kb + (mat >> 1);
      ldmatrix_x4(af[i], ta + r * kRowBytes + swz(r, c) * 16);
    }
#pragma unroll
    for (int j = 0; j < NI; j += 2) {
      const int mat = lane >> 3;
      const int r = bcol + j * 8 + (lane & 7) + ((mat >> 1) << 3);
      const int c = 2 * kb + (mat & 1);
      uint32_t t[4];
      ldmatrix_x4(t, tw + r * kRowBytes + swz(r, c) * 16);
      bf[j][0] = t[0];
      bf[j][1] = t[1];
      bf[j + 1][0] = t[2];
      bf[j + 1][1] = t[3];
    }
    const float2* fak = fa + kb * BM + arow + lane / 4;
    const float2* fwk = fw + kb * (BN / 2) + bcol / 2 + lane % 4;
    if constexpr (PACKED) {
      uint64_t a2[MI][2], w2[NI];
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        a2[i][0] = *reinterpret_cast<const uint64_t*>(fak + i * 16);
        a2[i][1] = *reinterpret_cast<const uint64_t*>(fak + i * 16 + 8);
      }
#pragma unroll
      for (int j = 0; j < NI; ++j)
        w2[j] = *reinterpret_cast<const uint64_t*>(fwk + j * 4);
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) {
          float prod[4];
          mma_k32(prod, af[i], bf[j][0], bf[j][1]);
          scale_add_x2(acc[i][j][0], acc[i][j][1], prod[0], prod[1], a2[i][0],
                       w2[j]);
          scale_add_x2(acc[i][j][2], acc[i][j][3], prod[2], prod[3], a2[i][1],
                       w2[j]);
        }
    } else {
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) {
          float prod[4];
          mma_k32(prod, af[i], bf[j][0], bf[j][1]);
          const float a0 = fak[i * 16].x, a1 = fak[i * 16 + 8].x;
          const float2 w = fwk[j * 4];
          acc[i][j][0] = add_rn(acc[i][j][0], scale_block(prod[0], a0, w.x));
          acc[i][j][1] = add_rn(acc[i][j][1], scale_block(prod[1], a0, w.y));
          acc[i][j][2] = add_rn(acc[i][j][2], scale_block(prod[2], a1, w.x));
          acc[i][j][3] = add_rn(acc[i][j][3], scale_block(prod[3], a1, w.y));
        }
    }
  }
}

// Large-M kernel: BM x BN tile, BK = 128 bytes (4 K32 blocks) per stage,
// each warp owns a (BM / WM) x (BN / WN) sub-tile and accumulates in place.
template <typename Scale, typename Output, int BM, int BN, int WM, int WN,
          int STAGES, int MIN_BLOCKS>
__global__ void __launch_bounds__(WM * WN * 32, MIN_BLOCKS)
    block32_mm_tiled_kernel(TiledArgs p) {
  constexpr int kThreads = WM * WN * 32;
  constexpr int kRowBytes = 128;
  constexpr int MI = BM / WM / 16;
  constexpr int NI = BN / WN / 8;
  static_assert(NI % 2 == 0);
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sa = smem;
  uint8_t* sw = sa + STAGES * BM * kRowBytes;
  uint32_t* ssa = reinterpret_cast<uint32_t*>(sw + STAGES * BN * kRowBytes);
  uint32_t* ssw = ssa + STAGES * BM;
  float2* fa = reinterpret_cast<float2*>(ssw + STAGES * BN);
  float2* fw = fa + 4 * BM;

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int wm = warp / WN, wn = warp % WN;
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM, group = blockIdx.z;
  const int stages_total = p.k / kRowBytes;
  const auto* a_scale = static_cast<const Scale*>(p.a_scale);
  const auto* w_scale = static_cast<const Scale*>(p.w_scale);

  auto load_stage = [&](int stage, int slot) {
    const int k0 = stage * kRowBytes;
    for (int i = tid; i < BM * 8; i += kThreads) {
      const int r = i / 8, c = i % 8, row = m0 + r;
      const bool valid = row < p.m;
      const uint8_t* src = valid ? p.a + int64_t(row) * p.as.row +
                                       int64_t(group) * p.as.group + k0 + c * 16
                                 : p.a;
      cp_async16(sa + (slot * BM + r) * kRowBytes + swz(r, c) * 16, src, valid);
    }
    for (int i = tid; i < BN * 8; i += kThreads) {
      const int r = i / 8, c = i % 8, row = n0 + r;
      const bool valid = row < p.n;
      const uint8_t* src = valid ? p.w + int64_t(row) * p.ws.row +
                                       int64_t(group) * p.ws.group + k0 + c * 16
                                 : p.w;
      cp_async16(sw + (slot * BN + r) * kRowBytes + swz(r, c) * 16, src, valid);
    }
    for (int i = tid; i < BM + BN; i += kThreads) {
      if (i < BM) {
        const int row = m0 + i;
        const bool valid = row < p.m;
        cp_async4(
            ssa + slot * BM + i,
            valid ? scale_word(a_scale, p.ass, row, group, stage) : p.a_scale,
            valid);
      } else {
        const int r = i - BM, col = n0 + r;
        const bool valid = col < p.n;
        cp_async4(
            ssw + slot * BN + r,
            valid ? scale_word(w_scale, p.wss, col / p.scale_rows, group, stage)
                  : p.w_scale,
            valid);
      }
    }
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < stages_total) load_stage(s, s);
    cp_async_commit();
  }

  const int arow = wm * (BM / WM), bcol = wn * (BN / WN);
  for (int stage = 0; stage < stages_total; ++stage) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int next = stage + STAGES - 1;
      if (next < stages_total) load_stage(next, next % STAGES);
      cp_async_commit();
    }
    const int slot = stage % STAGES;
    const uint8_t* ta = sa + slot * BM * kRowBytes;
    const uint8_t* tw = sw + slot * BN * kRowBytes;
    const uint32_t* tsa = ssa + slot * BM;
    const uint32_t* tsw = ssw + slot * BN;
    int unsafe = 0;
    for (int i = tid; i < BM + BN; i += kThreads) {
      const uint32_t word = i < BM ? tsa[i] : tsw[i - BM];
      const bool valid = i < BM ? m0 + i < p.m : n0 + i - BM < p.n;
      unsafe |= valid && !packed_scale_safe(word);
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const float v = decode_scale(word, kb);
        if (i < BM) {
          fa[kb * BM + i] = make_float2(v, v);
        } else {
          const int c = i - BM;
          reinterpret_cast<float*>(fw + kb * (BN / 2) + c / 2)[c & 1] = v;
        }
      }
    }
    const bool packed = !__syncthreads_or(unsafe);
    if (packed)
      tile_stage<true, MI, NI, BM, BN>(acc, ta, tw, fa, fw, arow, bcol, lane);
    else
      tile_stage<false, MI, NI, BM, BN>(acc, ta, tw, fa, fw, arow, bcol, lane);
  }
  cp_async_wait<0>();

  auto* out = static_cast<Output*>(p.out);
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = m0 + arow + i * 16 + lane / 4 + (e >= 2 ? 8 : 0);
        const int col = n0 + bcol + j * 8 + (lane % 4) * 2 + (e % 2);
        if (row < p.m && col < p.n)
          out[(int64_t(row) * p.groups + group) * p.n + col] =
              Output(acc[i][j][e]);
      }
}

// Skinny-M kernel. The MMA runs with weight rows as its M operand and tokens
// as its N operand; the dot product per output is unchanged. A CTA owns
// 16 * RT weight rows and 8 * TT tokens. Warps compute the scaled products of
// different (K32 block, row tile, token tile) units of a stage and stage them
// in shared memory (double-buffered); one thread per output then adds them in
// ascending block order. K is spread over warps without changing the order of
// the reduction. Token rows past M are never loaded: MMA columns are
// independent, so their garbage only reaches discarded outputs.
template <typename Scale, typename Output, int RT, int TT, int KB, int STAGES>
__global__ void __launch_bounds__(256) block32_mm_skinny_kernel(TiledArgs p) {
  constexpr int kThreads = 256, kWarps = 8;
  constexpr int BN = 16 * RT, BT = 8 * TT;
  constexpr int kRowBytes = KB * 32;
  constexpr int kChunks = kRowBytes / 16;
  constexpr int kWords = KB / 4;
  constexpr int kOut = BN * BT;
  static_assert(KB % 4 == 0 && kChunks % 8 == 0);
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sw = smem;
  uint8_t* sa = sw + STAGES * BN * kRowBytes;
  uint32_t* ssw = reinterpret_cast<uint32_t*>(sa + STAGES * BT * kRowBytes);
  uint32_t* ssa = ssw + STAGES * BN * kWords;
  float* q = reinterpret_cast<float*>(ssa + STAGES * BT * kWords);

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BT, group = blockIdx.z;
  const int valid_tokens = min(BT, p.m - m0);
  const int blocks_total = p.k / 32;
  const int stages_total = (blocks_total + KB - 1) / KB;
  const auto* a_scale = static_cast<const Scale*>(p.a_scale);
  const auto* w_scale = static_cast<const Scale*>(p.w_scale);

  auto load_stage = [&](int stage, int slot) {
    const int k0 = stage * kRowBytes;
    for (int i = tid; i < BN * kChunks; i += kThreads) {
      const int r = i / kChunks, c = i % kChunks, row = n0 + r;
      const bool valid = row < p.n && k0 + c * 16 < p.k;
      const uint8_t* src = valid ? p.w + int64_t(row) * p.ws.row +
                                       int64_t(group) * p.ws.group + k0 + c * 16
                                 : p.w;
      cp_async16(sw + (slot * BN + r) * kRowBytes + swz(r, c) * 16, src, valid);
    }
    for (int i = tid; i < valid_tokens * kChunks; i += kThreads) {
      const int r = i / kChunks, c = i % kChunks, row = m0 + r;
      const bool valid = k0 + c * 16 < p.k;
      const uint8_t* src = valid ? p.a + int64_t(row) * p.as.row +
                                       int64_t(group) * p.as.group + k0 + c * 16
                                 : p.a;
      cp_async16(sa + (slot * BT + r) * kRowBytes + swz(r, c) * 16, src, valid);
    }
    for (int i = tid; i < (BN + valid_tokens) * kWords; i += kThreads) {
      const int r = i / kWords, wi = i % kWords;
      const int word = stage * kWords + wi;
      const bool in_k = word * 4 < blocks_total;
      if (r < BN) {
        const int col = n0 + r;
        const bool valid = col < p.n && in_k;
        cp_async4(
            ssw + (slot * BN + r) * kWords + wi,
            valid ? scale_word(w_scale, p.wss, col / p.scale_rows, group, word)
                  : p.w_scale,
            valid);
      } else {
        const int t = r - BN;
        cp_async4(
            ssa + (slot * BT + t) * kWords + wi,
            in_k ? scale_word(a_scale, p.ass, m0 + t, group, word) : p.a_scale,
            in_k);
      }
    }
  };

  constexpr int kOutPerThread = (kOut + kThreads - 1) / kThreads;
  float acc[kOutPerThread];
#pragma unroll
  for (int i = 0; i < kOutPerThread; ++i) acc[i] = 0.f;

  auto accumulate = [&](int stage) {
    const float* qs = q + (stage & 1) * KB * kOut;
    const int kb_end = min(KB, blocks_total - stage * KB);
#pragma unroll
    for (int i = 0; i < kOutPerThread; ++i) {
      const int o = tid + i * kThreads;
      if (o < kOut && o / BN < valid_tokens) {
#pragma unroll
        for (int kb = 0; kb < KB; ++kb)
          if (kb < kb_end) acc[i] = add_rn(acc[i], qs[kb * kOut + o]);
      }
    }
  };

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < stages_total) load_stage(s, s);
    cp_async_commit();
  }

  const int token_tiles = (valid_tokens + 7) / 8;
  for (int stage = 0; stage < stages_total; ++stage) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int next = stage + STAGES - 1;
      if (next < stages_total) load_stage(next, next % STAGES);
      cp_async_commit();
    }
    const int slot = stage % STAGES;
    const uint8_t* tw = sw + slot * BN * kRowBytes;
    const uint8_t* ta = sa + slot * BT * kRowBytes;
    const uint32_t* tsw = ssw + slot * BN * kWords;
    const uint32_t* tsa = ssa + slot * BT * kWords;
    float* qs = q + (stage & 1) * KB * kOut;
    const int units = KB * RT * token_tiles;
    for (int unit = warp; unit < units; unit += kWarps) {
      const int kb = unit / (RT * token_tiles);
      const int rt = unit / token_tiles % RT, tt = unit % token_tiles;
      uint32_t wf[4], af[2];
      {
        const int mat = lane >> 3;
        const int r = rt * 16 + (lane & 7) + ((mat & 1) << 3);
        const int c = 2 * kb + (mat >> 1);
        ldmatrix_x4(wf, tw + r * kRowBytes + swz(r, c) * 16);
      }
      {
        const int l = lane & 15;
        const int r = tt * 8 + (l & 7);
        const int c = 2 * kb + (l >> 3);
        ldmatrix_x2(af, ta + r * kRowBytes + swz(r, c) * 16);
      }
      float prod[4];
      mma_k32(prod, wf, af[0], af[1]);
      const int col0 = rt * 16 + lane / 4, tok0 = tt * 8 + (lane % 4) * 2;
      const float w0 = decode_scale(tsw[col0 * kWords + kb / 4], kb % 4);
      const float w1 = decode_scale(tsw[(col0 + 8) * kWords + kb / 4], kb % 4);
      const float a0 = decode_scale(tsa[tok0 * kWords + kb / 4], kb % 4);
      const float a1 = decode_scale(tsa[(tok0 + 1) * kWords + kb / 4], kb % 4);
      float* qb = qs + kb * kOut;
      qb[tok0 * BN + col0] = scale_block(prod[0], a0, w0);
      qb[(tok0 + 1) * BN + col0] = scale_block(prod[1], a1, w0);
      qb[tok0 * BN + col0 + 8] = scale_block(prod[2], a0, w1);
      qb[(tok0 + 1) * BN + col0 + 8] = scale_block(prod[3], a1, w1);
    }
    // Products of the previous stage were published by this stage's barrier.
    if (stage > 0) accumulate(stage - 1);
  }
  cp_async_wait<0>();
  __syncthreads();
  if (stages_total > 0) accumulate(stages_total - 1);

  auto* out = static_cast<Output*>(p.out);
#pragma unroll
  for (int i = 0; i < kOutPerThread; ++i) {
    const int o = tid + i * kThreads;
    const int row = m0 + o / BN, col = n0 + o % BN;
    if (o < kOut && o / BN < valid_tokens && col < p.n)
      out[(int64_t(row) * p.groups + group) * p.n + col] = Output(acc[i]);
  }
}

template <typename Scale, typename Output, int BM, int BN, int WM, int WN,
          int STAGES, int MIN_BLOCKS>
void launch_tiled(const TiledArgs& args, cudaStream_t stream) {
  constexpr int smem = STAGES * (BM + BN) * (128 + 4) + 4 * (BM + BN / 2) * 8;
  auto kernel = block32_mm_tiled_kernel<Scale, Output, BM, BN, WM, WN, STAGES,
                                        MIN_BLOCKS>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const dim3 grid((args.n + BN - 1) / BN, (args.m + BM - 1) / BM, args.groups);
  kernel<<<grid, WM * WN * 32, smem, stream>>>(args);
}

template <typename Scale, typename Output, int RT, int TT, int KB, int STAGES>
void launch_skinny(const TiledArgs& args, cudaStream_t stream) {
  constexpr int BN = 16 * RT, BT = 8 * TT;
  constexpr int smem =
      STAGES * (BN + BT) * (KB * 32 + KB) + 2 * KB * BN * BT * 4;
  auto kernel = block32_mm_skinny_kernel<Scale, Output, RT, TT, KB, STAGES>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const dim3 grid((args.n + BN - 1) / BN, (args.m + BT - 1) / BT, args.groups);
  kernel<<<grid, 256, smem, stream>>>(args);
}

// Decode kernel for M <= 8 * TT tokens per token group. A CTA owns one
// 16-row weight tile and a slice of at most L K32 blocks; its warps compute
// the scaled products of the slice (weight rows as the MMA M operand) into
// shared memory. With one slice the CTA adds them in ascending block order.
// With S slices each CTA publishes its products to a global workspace and the
// last CTA of the tile adds all of them in ascending block order, so splitting
// K over CTAs never changes the reduction order (stream-K with ordered fixup).
template <typename Scale, typename Output, int TT, int L>
__global__ void __launch_bounds__(128)
    block32_mm_gemv_kernel(TiledArgs p, int slices, int per, float* workspace,
                           int* counters) {
  constexpr int kThreads = 128, kWarps = 4;
  constexpr int BN = 16, BT = 8 * TT;
  constexpr int kRowBytes = L * 32;
  constexpr int kChunks = kRowBytes / 16;
  constexpr int kWords = L / 4;
  constexpr int kOut = BN * BT;
  static_assert(L % 4 == 0 && kChunks % 8 == 0);
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sw = smem;
  uint8_t* sa = sw + BN * kRowBytes;
  uint32_t* ssw = reinterpret_cast<uint32_t*>(sa + BT * kRowBytes);
  uint32_t* ssa = ssw + BN * kWords;
  float* prod_s = reinterpret_cast<float*>(ssa + BT * kWords);  // [L][kOut]
  __shared__ int last;

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int tile = blockIdx.x, slice = blockIdx.y;
  const int token_groups = (p.m + BT - 1) / BT;
  const int group = blockIdx.z / token_groups;
  const int m0 = blockIdx.z % token_groups * BT;
  const int n0 = tile * BN;
  const int valid_tokens = min(BT, p.m - m0);
  const int blocks_total = p.k / 32;
  const int b0 = slice * per;
  const int nb = min(per, blocks_total - b0);
  const auto* a_scale = static_cast<const Scale*>(p.a_scale);
  const auto* w_scale = static_cast<const Scale*>(p.w_scale);
  const int64_t k0 = int64_t(b0) * 32;

  for (int i = tid; i < BN * kChunks; i += kThreads) {
    const int r = i / kChunks, c = i % kChunks, row = n0 + r;
    const bool valid = row < p.n && c < 2 * nb;
    const uint8_t* src = valid ? p.w + int64_t(row) * p.ws.row +
                                     int64_t(group) * p.ws.group + k0 + c * 16
                               : p.w;
    cp_async16(sw + r * kRowBytes + swz(r, c) * 16, src, valid);
  }
  for (int i = tid; i < valid_tokens * kChunks; i += kThreads) {
    const int r = i / kChunks, c = i % kChunks, row = m0 + r;
    const bool valid = c < 2 * nb;
    const uint8_t* src = valid ? p.a + int64_t(row) * p.as.row +
                                     int64_t(group) * p.as.group + k0 + c * 16
                               : p.a;
    cp_async16(sa + r * kRowBytes + swz(r, c) * 16, src, valid);
  }
  for (int i = tid; i < (BN + valid_tokens) * kWords; i += kThreads) {
    const int r = i / kWords, wi = i % kWords;
    const int word = b0 / 4 + wi;
    const bool in_k = wi * 4 < nb;
    if (r < BN) {
      const int col = n0 + r;
      const bool valid = col < p.n && in_k;
      cp_async4(
          ssw + r * kWords + wi,
          valid ? scale_word(w_scale, p.wss, col / p.scale_rows, group, word)
                : p.w_scale,
          valid);
    } else {
      const int t = r - BN;
      cp_async4(
          ssa + t * kWords + wi,
          in_k ? scale_word(a_scale, p.ass, m0 + t, group, word) : p.a_scale,
          in_k);
    }
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int token_tiles = (valid_tokens + 7) / 8;
  for (int unit = warp; unit < nb * token_tiles; unit += kWarps) {
    const int kb = unit / token_tiles, tt = unit % token_tiles;
    uint32_t wf[4], af[2];
    {
      const int mat = lane >> 3;
      const int r = (lane & 7) + ((mat & 1) << 3);
      const int c = 2 * kb + (mat >> 1);
      ldmatrix_x4(wf, sw + r * kRowBytes + swz(r, c) * 16);
    }
    {
      const int l = lane & 15;
      const int r = tt * 8 + (l & 7);
      const int c = 2 * kb + (l >> 3);
      ldmatrix_x2(af, sa + r * kRowBytes + swz(r, c) * 16);
    }
    float prod[4];
    mma_k32(prod, wf, af[0], af[1]);
    const int col0 = lane / 4, tok0 = tt * 8 + (lane % 4) * 2;
    const float w0 = decode_scale(ssw[col0 * kWords + kb / 4], kb % 4);
    const float w1 = decode_scale(ssw[(col0 + 8) * kWords + kb / 4], kb % 4);
    const float a0 = decode_scale(ssa[tok0 * kWords + kb / 4], kb % 4);
    const float a1 = decode_scale(ssa[(tok0 + 1) * kWords + kb / 4], kb % 4);
    float* pb = prod_s + kb * kOut;
    pb[tok0 * BN + col0] = scale_block(prod[0], a0, w0);
    pb[(tok0 + 1) * BN + col0] = scale_block(prod[1], a1, w0);
    pb[tok0 * BN + col0 + 8] = scale_block(prod[2], a0, w1);
    pb[(tok0 + 1) * BN + col0 + 8] = scale_block(prod[3], a1, w1);
  }
  __syncthreads();

  const int outs = BN * valid_tokens;
  auto* out = static_cast<Output*>(p.out);
  auto store = [&](int o, float v) {
    const int row = m0 + o / BN, col = n0 + o % BN;
    if (col < p.n)
      out[(int64_t(row) * p.groups + group) * p.n + col] = Output(v);
  };
  if (slices == 1) {
    for (int o = tid; o < outs; o += kThreads) {
      float acc = 0.f;
      for (int kb = 0; kb < nb; ++kb) acc = add_rn(acc, prod_s[kb * kOut + o]);
      store(o, acc);
    }
    return;
  }

  // Workspace rows: [tile instance][block][valid outputs].
  const int64_t inst = (int64_t(blockIdx.z) * gridDim.x + tile);
  float* ws = workspace + inst * int64_t(blocks_total) * kOut;
  for (int i = tid; i < nb * outs; i += kThreads) {
    const int kb = i / outs, o = i % outs;
    __stcg(ws + int64_t(b0 + kb) * outs + o, prod_s[kb * kOut + o]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(counters + inst, 1) == slices - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  // Ordered fixup by the last CTA: stream the products through shared memory
  // in chunks of L blocks and add them in ascending block order.
  float acc = 0.f;
  for (int c0 = 0; c0 < blocks_total; c0 += L) {
    const int cn = min(L, blocks_total - c0);
    for (int i = tid; i < cn * outs; i += kThreads)
      prod_s[i] = __ldcg(ws + int64_t(c0) * outs + i);
    __syncthreads();
    if (tid < outs)
      for (int kb = 0; kb < cn; ++kb)
        acc = add_rn(acc, prod_s[kb * outs + tid]);
    __syncthreads();
  }
  if (tid < outs) store(tid, acc);
  if (tid == 0) counters[inst] = 0;
}

template <typename Scale, typename Output, int TT, int L>
void launch_gemv(const TiledArgs& args, int per, cudaStream_t stream,
                 const at::TensorOptions& options) {
  const int slices = (args.k / 32 + per - 1) / per;
  constexpr int BT = 8 * TT;
  static_assert(16 * BT <= 128, "one thread per output in the fixup");
  constexpr int smem = (16 + BT) * (L * 32 + L) + L * 16 * BT * 4;
  auto kernel = block32_mm_gemv_kernel<Scale, Output, TT, L>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const int tiles = (args.n + 15) / 16;
  const int token_groups = (args.m + BT - 1) / BT;
  const dim3 grid(tiles, slices, args.groups * token_groups);
  float* workspace = nullptr;
  int* counters = nullptr;
  torch::Tensor ws_t, cnt_t;
  if (slices > 1) {
    const int64_t inst = int64_t(tiles) * args.groups * token_groups;
    ws_t = torch::empty({inst * (args.k / 32) * 16 * BT},
                        options.dtype(torch::kFloat32));
    cnt_t = torch::zeros({inst}, options.dtype(torch::kInt32));
    workspace = ws_t.data_ptr<float>();
    counters = cnt_t.data_ptr<int>();
  }
  kernel<<<grid, 128, smem, stream>>>(args, slices, per, workspace, counters);
}

// variant: 0 auto, 2 skinny, 3 large, 4 decode (debug/tests).
template <typename Scale, typename Output>
void dispatch_tiled(const TiledArgs& args, int variant, int num_sms,
                    cudaStream_t stream, const at::TensorOptions& options) {
  const int64_t row_tiles = int64_t(args.groups) * ((args.n + 15) / 16);
  if (variant == 4 ||
      (variant == 0 && args.m <= 8 && row_tiles < 8 * int64_t(num_sms))) {
    // Split K so that about four CTAs per SM stream the weights; slices are
    // whole scale words (4 blocks) and at most L blocks.
    constexpr int L = 32;
    const int64_t tiles =
        int64_t(args.groups) * ((args.n + 15) / 16) * ((args.m + 7) / 8);
    const int blocks = args.k / 32;
    const int64_t want = (4 * int64_t(num_sms) + tiles - 1) / tiles;
    int per = int((blocks + want - 1) / want);
    per = std::min(L, std::max(4, (per + 3) / 4 * 4));
    launch_gemv<Scale, Output, 1, L>(args, per, stream, options);
    return;
  }
  const bool small = variant == 2 || (variant == 0 && args.m <= 128);
  if (small) {
    const bool wide = row_tiles >= 8 * int64_t(num_sms);
    if (args.m <= 8) {
      if (wide)
        launch_skinny<Scale, Output, 4, 1, 8, 4>(args, stream);
      else
        launch_skinny<Scale, Output, 1, 1, 16, 8>(args, stream);
    } else if (args.m <= 16) {
      if (wide)
        launch_skinny<Scale, Output, 2, 2, 16, 4>(args, stream);
      else
        launch_skinny<Scale, Output, 1, 2, 16, 6>(args, stream);
    } else {
      launch_skinny<Scale, Output, 1, 4, 16, 4>(args, stream);
    }
    return;
  }
  const int64_t big_ctas =
      int64_t(args.groups) * ((args.n + 127) / 128) * ((args.m + 127) / 128);
  if (variant == 0 && big_ctas < num_sms)
    launch_tiled<Scale, Output, 64, 64, 2, 2, 6, 1>(args, stream);
  else
    launch_tiled<Scale, Output, 128, 128, 2, 4, 3, 2>(args, stream);
}

bool tiled_supported(const torch::Tensor& a, const torch::Tensor& a_scale,
                     const torch::Tensor& weight,
                     const torch::Tensor& weight_scale, bool grouped,
                     int64_t k) {
  if (k % 128 != 0) return false;
  auto aligned = [](const torch::Tensor& t, int bytes) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
  };
  const auto as = layout(a, grouped, false), ws = layout(weight, grouped, true);
  if (as.k != 1 || ws.k != 1 || !aligned(a, 16) || !aligned(weight, 16))
    return false;
  if (as.row % 16 || as.group % 16 || ws.row % 16 || ws.group % 16)
    return false;
  if (a_scale.scalar_type() == at::kByte) {
    const auto ass = layout(a_scale, grouped, false),
               wss = layout(weight_scale, grouped, true);
    if (ass.k != 1 || wss.k != 1 || !aligned(a_scale, 4) ||
        !aligned(weight_scale, 4))
      return false;
    if (ass.row % 4 || ass.group % 4 || wss.row % 4 || wss.group % 4)
      return false;
  }
  return true;
}

}  // namespace

void block32_scaled_mm_variant(const torch::Tensor& a,
                               const torch::Tensor& a_scale,
                               const torch::Tensor& weight,
                               const torch::Tensor& weight_scale,
                               torch::Tensor& output, bool grouped,
                               int64_t weight_scale_group_rows,
                               int64_t variant) {
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a_scale.device() == a.device() &&
                  weight.device() == a.device() &&
                  weight_scale.device() == a.device() &&
                  output.device() == a.device(),
              "K32 tensors must share a CUDA device");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  weight.scalar_type() == a.scalar_type(),
              "K32 operands must be E4M3");
  TORCH_CHECK(a.dim() == (grouped ? 3 : 2) && weight.dim() == a.dim() &&
                  a_scale.dim() == a.dim() && weight_scale.dim() == a.dim(),
              "Invalid K32 operand ranks");
  TORCH_CHECK(a_scale.scalar_type() == weight_scale.scalar_type() &&
                  (a_scale.scalar_type() == at::kByte ||
                   a_scale.scalar_type() == at::kInt),
              "K32 scales must be paired E8M0 bytes or packed INT32");
  const int64_t m = a.size(0), groups = grouped ? a.size(1) : 1;
  const int64_t n = weight.size(grouped ? 1 : 0), k = a.size(-1);
  TORCH_CHECK(k > 0 && k % 32 == 0 && weight.size(-1) == k &&
                  weight_scale_group_rows > 0 &&
                  weight_scale_group_rows <= INT_MAX,
              "Invalid K32 geometry");
  const int64_t scale_columns =
      a_scale.scalar_type() == at::kInt ? (k / 32 + 3) / 4 : k / 32;
  TORCH_CHECK(
      a_scale.size(0) >= m && a_scale.size(-1) >= scale_columns &&
          weight_scale.size(grouped ? 1 : 0) >=
              (n + weight_scale_group_rows - 1) / weight_scale_group_rows &&
          weight_scale.size(-1) >= scale_columns,
      "K32 scale capacity is too small");
  TORCH_CHECK(
      !grouped || (weight.size(0) == groups && a_scale.size(1) == groups &&
                   weight_scale.size(0) == groups),
      "K32 groups must match");
  TORCH_CHECK(output.is_contiguous() && output.dim() == a.dim() &&
                  output.size(0) == m && output.size(-1) == n &&
                  (!grouped || output.size(1) == groups),
              "Invalid K32 output shape");
  TORCH_CHECK(output.scalar_type() == at::kFloat ||
                  output.scalar_type() == at::kHalf ||
                  output.scalar_type() == at::kBFloat16,
              "Invalid K32 output dtype");
  TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX && groups <= 65535 &&
                  (n + 31) / 32 <= 65535,
              "K32 launch geometry is too large");
  if (output.numel() == 0) return;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major * 10 + properties->minor >= 89,
              "K32 FP8 MMA requires SM89 or newer");
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto a_stride = layout(a, grouped, false),
             as_stride = layout(a_scale, grouped, false);
  const auto w_stride = layout(weight, grouped, true),
             ws_stride = layout(weight_scale, grouped, true);
  TORCH_CHECK(variant >= 0 && variant <= 4, "Invalid K32 kernel variant");
  if (variant != 1 &&
      tiled_supported(a, a_scale, weight, weight_scale, grouped, k)) {
    const TiledArgs args{reinterpret_cast<const uint8_t*>(a.data_ptr()),
                         a_scale.data_ptr(),
                         reinterpret_cast<const uint8_t*>(weight.data_ptr()),
                         weight_scale.data_ptr(),
                         output.data_ptr(),
                         int(m),
                         int(n),
                         int(k),
                         int(groups),
                         int(weight_scale_group_rows),
                         a_stride,
                         as_stride,
                         w_stride,
                         ws_stride};
    const int num_sms = properties->multiProcessorCount;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, output.scalar_type(), "block32_scaled_mm",
        [&] {
          if (a_scale.scalar_type() == at::kInt)
            dispatch_tiled<int32_t, scalar_t>(args, int(variant), num_sms,
                                              stream, a.options());
          else
            dispatch_tiled<uint8_t, scalar_t>(args, int(variant), num_sms,
                                              stream, a.options());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const dim3 grid((m + 15) / 16, (n + 31) / 32, groups);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, output.scalar_type(), "block32_scaled_mm", [&] {
        if (a_scale.scalar_type() == at::kInt) {
          block32_mm_kernel<int32_t, scalar_t><<<grid, 128, 0, stream>>>(
              reinterpret_cast<const uint8_t*>(a.data_ptr()),
              a_scale.data_ptr<int32_t>(),
              reinterpret_cast<const uint8_t*>(weight.data_ptr()),
              weight_scale.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), m,
              n, k, groups, weight_scale_group_rows, a_stride, as_stride,
              w_stride, ws_stride);
        } else {
          block32_mm_kernel<uint8_t, scalar_t><<<grid, 128, 0, stream>>>(
              reinterpret_cast<const uint8_t*>(a.data_ptr()),
              a_scale.data_ptr<uint8_t>(),
              reinterpret_cast<const uint8_t*>(weight.data_ptr()),
              weight_scale.data_ptr<uint8_t>(), output.data_ptr<scalar_t>(), m,
              n, k, groups, weight_scale_group_rows, a_stride, as_stride,
              w_stride, ws_stride);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void block32_scaled_mm(const torch::Tensor& a, const torch::Tensor& a_scale,
                       const torch::Tensor& weight,
                       const torch::Tensor& weight_scale, torch::Tensor& output,
                       bool grouped, int64_t weight_scale_group_rows) {
  block32_scaled_mm_variant(a, a_scale, weight, weight_scale, output, grouped,
                            weight_scale_group_rows, 0);
}

}  // namespace vllm::batch_invariant
