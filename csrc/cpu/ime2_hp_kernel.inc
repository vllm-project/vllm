// A100 IME2 int4 GEMM for vLLM. Same vmadot.hp tile as the ONNX
// BlkLen=32 kernel: eight K32 subblocks, N=32, symmetric zero point 8.
// Run it on cores 8-15. The X100 cores do not implement xsmtvdotii.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace vllm_ime2 {

static unsigned long ime_vlenb(void) {
    unsigned long v;
    __asm__ volatile("csrr %0, vlenb" : "=r"(v));
    return v;
}

// A100 Q4_0 path. Activations are quantized the way llama.cpp's
// quantize_a_row_i8_hp / quantize_a_4row_i8_hp do it: eight K32 subblocks
// share one fp16 superblock scale, each stored subblock scale is the ratio
// scale/scale_avg, and four rows share one scale. B stays packed nibbles
// (block_q4_0x32). The dot is vmadot.hp. Symmetric zero point only (zp = 8,
// folded into the fp16 row sum). A model that passes zero-point bytes keeps
// the int8-tile kernel.

constexpr size_t kHpNb = 32;
constexpr size_t kHpSub = 32;
constexpr size_t kHpSuper = 256;
constexpr size_t kHpSubs = 8;
constexpr size_t kHpBlkBytes = 64 + 512;          // fp16 scale[32] + qs
constexpr size_t kHpSuperBytes = kHpSubs * kHpBlkBytes;
constexpr size_t kHpASub = 2 + kHpSub;            // fp16 scale + int8[32]
constexpr size_t kHpARowSuper = kHpSubs * kHpASub + kHpSubs * 2 + 2;  // 290
constexpr size_t kHpA4Sub = 8 + kHpSub * 4;       // 4 fp16 slots + 4*int8[32]
constexpr size_t kHpA4Super = kHpSubs * kHpA4Sub + 4 * kHpSubs * 2 + 8;  // 1160
static_assert(kHpBlkBytes == 576, "q4_0x32 block");
static_assert(kHpSuperBytes == 4608, "K256 x N32 tile");
static_assert(kHpARowSuper == 290, "one-row HP superblock");
static_assert(kHpA4Super == 1160, "four-row HP superblock");
static_assert(kHpA4Super == 4 * kHpARowSuper, "four rows occupy 4 * row stride");

static size_t hp_k256(size_t BlockCountK)
{
    return (BlockCountK + (kHpSubs - 1)) / kHpSubs;
}

__attribute__((target("arch=+zfh")))
static uint16_t hp_f32_to_f16(float f)
{
    _Float16 h = static_cast<_Float16>(f);
    uint16_t u;
    std::memcpy(&u, &h, sizeof(u));
    return u;
}

__attribute__((target("arch=+zfh")))
static float hp_f16_to_f32(uint16_t u)
{
    _Float16 h;
    std::memcpy(&h, &u, sizeof(h));
    return static_cast<float>(h);
}

static void hp_store_f16(void* p, float f)
{
    const uint16_t u = hp_f32_to_f16(f);
    std::memcpy(p, &u, sizeof(u));
}

static float hp_load_f16(const void* p)
{
    uint16_t u;
    std::memcpy(&u, p, sizeof(u));
    return hp_f16_to_f32(u);
}

// (-sum) converted to fp16, then multiplied by 8 in fp16. Matches llama.
static void hp_store_asum(void* p, int sum)
{
    const float neg = hp_f16_to_f32(hp_f32_to_f16(static_cast<float>(-sum)));
    hp_store_f16(p, neg * 8.0f);
}

static int hp_quant_i8(float x, float inv)
{
    int q = static_cast<int>(std::lrintf(x * inv));
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return q;
}

#if defined(__riscv_v_intrinsic)
static bool
hp_vec32()
{
    return ime_vlenb() == 128;
}

// vfcvt then clamp matches lrintf on a full K32 block. The sum is the sum of
// those clamped ints, which is what the scalar path adds.
static void
hp_vec_quant_row(const float* x, float inv, int8_t* q, int& sum)
{
    const size_t vl = __riscv_vsetvl_e32m1(kHpSub);
    vfloat32m1_t vf = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(x, vl), inv, vl);
    vint32m1_t vi = __riscv_vfcvt_x_f_v_i32m1(vf, vl);
    vi = __riscv_vmin_vx_i32m1(vi, 127, vl);
    vi = __riscv_vmax_vx_i32m1(vi, -128, vl);
    const vint32m1_t red = __riscv_vredsum_vs_i32m1_i32m1(vi, __riscv_vmv_v_x_i32m1(0, vl), vl);
    sum = static_cast<int>(__riscv_vmv_x_s_i32m1_i32(red));
    const vint16mf2_t q16 = __riscv_vncvt_x_x_w_i16mf2(vi, vl);
    __riscv_vse8_v_i8mf4(q, __riscv_vncvt_x_x_w_i8mf4(q16, vl), vl);
}

static float
hp_vec_amax1(const float* x)
{
    const size_t vl = __riscv_vsetvl_e32m1(kHpSub);
    const vfloat32m1_t a = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(x, vl), vl);
    const vfloat32m1_t mx = __riscv_vfredmax_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(0.f, vl), vl);
    return __riscv_vfmv_f_s_f32m1_f32(mx);
}

static float
hp_vec_amax4(const float* r0, const float* r1, const float* r2, const float* r3)
{
    const size_t vl = __riscv_vsetvl_e32m1(kHpSub);
    vfloat32m1_t a = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(r0, vl), vl);
    a = __riscv_vfmax_vv_f32m1(a, __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(r1, vl), vl), vl);
    a = __riscv_vfmax_vv_f32m1(a, __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(r2, vl), vl), vl);
    a = __riscv_vfmax_vv_f32m1(a, __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(r3, vl), vl), vl);
    const vfloat32m1_t mx = __riscv_vfredmax_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(0.f, vl), vl);
    return __riscv_vfmv_f_s_f32m1_f32(mx);
}
#endif

bool
Ime2HpActive(size_t BlkLen, bool has_zp)
{
    return BlkLen == 32 && !has_zp && ime_vlenb() == 128;
}

size_t
Ime2HpQuantARowBytes(size_t K)
{
    const size_t bck = (K + kHpSub - 1) / kHpSub;
    return hp_k256(bck) * kHpARowSuper;
}

size_t
Ime2HpColumnStride(size_t BlockCountK)
{
    return hp_k256(BlockCountK) * (kHpSuperBytes / kHpNb);
}

size_t
Ime2HpPackedBSize(size_t N, size_t BlockCountK)
{
    const size_t tiles = (N + kHpNb - 1) / kHpNb;
    return tiles * hp_k256(BlockCountK) * kHpSuperBytes;
}

static void hp_quantize_m1(const float* row, size_t K, uint8_t* dst)
{
    const size_t bck = (K + kHpSub - 1) / kHpSub;
    const size_t k256 = hp_k256(bck);
    std::memset(dst, 0, k256 * kHpARowSuper);
    for (size_t s = 0; s < k256; ++s) {
        uint8_t* blk = dst + s * kHpARowSuper;
        float scales[kHpSubs];
        float scale_avg = 0.0f;
        for (size_t kk = 0; kk < kHpSubs; ++kk) {
            const size_t k0 = s * kHpSuper + kk * kHpSub;
            float amax = 0.0f;
            const size_t n = k0 < K ? std::min(kHpSub, K - k0) : 0;
#if defined(__riscv_v_intrinsic)
            if (n == kHpSub && hp_vec32()) {
                amax = hp_vec_amax1(row + k0);
            } else
#endif
            {
                for (size_t i = 0; i < n; ++i) {
                    const float v = std::fabs(row[k0 + i]);
                    if (v > amax) amax = v;
                }
            }
            scales[kk] = amax / 127.0f;
            scale_avg += scales[kk];
        }
        scale_avg /= static_cast<float>(kHpSubs);
        const float inv_avg = scale_avg != 0.0f ? 1.0f / scale_avg : 0.0f;
        hp_store_f16(blk + kHpSubs * kHpASub + kHpSubs * 2, scale_avg);
        for (size_t kk = 0; kk < kHpSubs; ++kk) {
            uint8_t* sub = blk + kk * kHpASub;
            hp_store_f16(sub, scales[kk] * inv_avg);
            int8_t* q = reinterpret_cast<int8_t*>(sub + 2);
            const size_t k0 = s * kHpSuper + kk * kHpSub;
            const size_t n = k0 < K ? std::min(kHpSub, K - k0) : 0;
            const float inv = scales[kk] != 0.0f ? 1.0f / scales[kk] : 0.0f;
            int sum = 0;
#if defined(__riscv_v_intrinsic)
            if (n == kHpSub && hp_vec32()) {
                hp_vec_quant_row(row + k0, inv, q, sum);
            } else
#endif
            {
                for (size_t i = 0; i < n; ++i) {
                    const int v = hp_quant_i8(row[k0 + i], inv);
                    q[i] = static_cast<int8_t>(v);
                    sum += v;
                }
            }
            hp_store_asum(blk + kHpSubs * kHpASub + kk * 2, sum);
        }
    }
}

static void hp_quantize_m4(const float* a, size_t lda, size_t K, uint8_t* dst)
{
    const size_t bck = (K + kHpSub - 1) / kHpSub;
    const size_t k256 = hp_k256(bck);
    std::memset(dst, 0, k256 * kHpA4Super);
    for (size_t s = 0; s < k256; ++s) {
        uint8_t* blk = dst + s * kHpA4Super;
        float scales[kHpSubs];
        float scale_avg = 0.0f;
        for (size_t kk = 0; kk < kHpSubs; ++kk) {
            const size_t k0 = s * kHpSuper + kk * kHpSub;
            float amax = 0.0f;
            const size_t n = k0 < K ? std::min(kHpSub, K - k0) : 0;
            const float* rows[4] = {
                a + 0 * lda + k0,
                a + 1 * lda + k0,
                a + 2 * lda + k0,
                a + 3 * lda + k0,
            };
#if defined(__riscv_v_intrinsic)
            if (n == kHpSub && hp_vec32()) {
                amax = hp_vec_amax4(rows[0], rows[1], rows[2], rows[3]);
            } else
#endif
            {
                for (size_t r = 0; r < 4; ++r) {
                    for (size_t i = 0; i < n; ++i) {
                        const float v = std::fabs(rows[r][i]);
                        if (v > amax) amax = v;
                    }
                }
            }
            scales[kk] = amax / 127.0f;
            scale_avg += scales[kk];
        }
        scale_avg /= static_cast<float>(kHpSubs);
        const float inv_avg = scale_avg != 0.0f ? 1.0f / scale_avg : 0.0f;
        hp_store_f16(blk + kHpSubs * kHpA4Sub + 4 * kHpSubs * 2, scale_avg);
        for (size_t kk = 0; kk < kHpSubs; ++kk) {
            uint8_t* sub = blk + kk * kHpA4Sub;
            hp_store_f16(sub, scales[kk] * inv_avg);
            const size_t k0 = s * kHpSuper + kk * kHpSub;
            const size_t n = k0 < K ? std::min(kHpSub, K - k0) : 0;
            const float inv = scales[kk] != 0.0f ? 1.0f / scales[kk] : 0.0f;
            for (size_t r = 0; r < 4; ++r) {
                int8_t* q = reinterpret_cast<int8_t*>(sub + 8 + r * kHpSub);
                const float* row = a + r * lda + k0;
                int sum = 0;
#if defined(__riscv_v_intrinsic)
                if (n == kHpSub && hp_vec32()) {
                    hp_vec_quant_row(row, inv, q, sum);
                } else
#endif
                {
                    for (size_t i = 0; i < n; ++i) {
                        const int v = hp_quant_i8(row[i], inv);
                        q[i] = static_cast<int8_t>(v);
                        sum += v;
                    }
                }
                hp_store_asum(blk + kHpSubs * kHpA4Sub + (r * kHpSubs + kk) * 2, sum);
            }
        }
    }
}

void
Ime2HpQuantizeA(const float* A, size_t lda, size_t M, size_t K, std::byte* QuantA)
{
    const size_t row_bytes = Ime2HpQuantARowBytes(K);
    uint8_t* dst = reinterpret_cast<uint8_t*>(QuantA);
    size_t m = 0;
    for (; m + 4 <= M; m += 4) {
        hp_quantize_m4(A + m * lda, lda, K, dst + m * row_bytes);
    }
    for (; m < M; ++m) {
        hp_quantize_m1(A + m * lda, K, dst + m * row_bytes);
    }
}

void
Ime2HpPack(
    size_t N, size_t BlockCountK, const uint8_t* src, const float* scales, uint8_t* dst)
{
    const size_t k256 = hp_k256(BlockCountK);
    const size_t tiles = (N + kHpNb - 1) / kHpNb;
    const size_t tile_bytes = k256 * kHpSuperBytes;
    if (src != nullptr) {
        std::memset(dst, 0, tiles * tile_bytes);
    }
    for (size_t n0 = 0; n0 < N; n0 += kHpNb) {
        uint8_t* tile = dst + (n0 / kHpNb) * tile_bytes;
        const size_t ncols = std::min(kHpNb, N - n0);
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            uint8_t* blk = tile + (kb / kHpSubs) * kHpSuperBytes + (kb % kHpSubs) * kHpBlkBytes;
            for (size_t c = 0; c < ncols; ++c) {
                const size_t col = n0 + c;
                if (scales != nullptr) {
                    hp_store_f16(blk + c * 2, scales[col * BlockCountK + kb]);
                }
                if (src != nullptr) {
                    std::memcpy(blk + 64 + c * 16, src + (col * BlockCountK + kb) * 16, 16);
                }
            }
        }
    }
}

// One row, a multiple of 32 columns. B is the first tile of this N range.
__attribute__((target("arch=+zfh,+zvfh,+xsmtvdotii"), noinline))
static void hp_m1_asm(
    const uint8_t* quant_a, const uint8_t* quant_b, float* c, size_t count_n, size_t k256)
{
    constexpr size_t NB_COLS = 32;
    const size_t b_tile_stride = k256 * kHpSuperBytes;
    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        const uint8_t* b_data = quant_b + (ni / NB_COLS) * b_tile_stride;
        const uint8_t* a_data = quant_a;
        float* dst_c = c + ni;
        __asm__ volatile(
            ".option push\n\t"
            ".option arch, +zfh,+zvfh,+xsmtvdotii\n\t"
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             t4, %[BK]               \n\t"
            "li             t0, 0x4c00              \n\t"
            "fmv.h.x        fa0, t0                 \n\t"
            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            "li             t5, 8                   \n\t"
            "addi           t6, %[A], 288           \n\t"
            "flh            ft1, (t6)               \n\t"
            "addi           t6, %[A], 272           \n\t"
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v16, v18, v18           \n\t"
            "vxor.vv        v17, v18, v18           \n\t"
            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v19, v18, v18           \n\t"
            "INNER_BLK_LOOP%=:                      \n\t"
            "flh            fa1, (t6)               \n\t"
            "addi           t6, t6, 2               \n\t"
            "flh            ft0, (%[A])             \n\t"
            "addi           %[A], %[A], 2           \n\t"
            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vle8.v         v3, (%[A])              \n\t"
            "addi           %[A], %[A], 32          \n\t"
            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v8, (%[B])              \n\t"
            "addi           %[B], %[B], 64          \n\t"
            "vl4r.v         v4, (%[B])              \n\t"
            "addi           %[B], %[B], 512         \n\t"
            "vfmul.vf       v8, v8, ft0             \n\t"
            "vfmul.vf       v9, v8, fa0             \n\t"
            "vfmul.vf       v10, v8, fa1            \n\t"
            "vfwmacc.vf     v31, ft1, v10           \n\t"
            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v0, v8, v9, 3           \n\t"
            "vsrl.vi        v28, v3, 4              \n\t"
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vnpack4.vv     v2, v3, v3, 3           \n\t"
            "vnpack4.vv     v3, v28, v28, 3         \n\t"
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"
            "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
            "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
            "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
            "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"
            "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
            "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
            "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"
            "addi           t5, t5, -1              \n\t"
            "bgtz           t5, INNER_BLK_LOOP%=    \n\t"
            "vpack.vv       v8, v16, v17, 1         \n\t"
            "vpack.vv       v12, v18, v19, 1        \n\t"
            "vpack.vv       v20, v8, v12, 2         \n\t"
            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t4, t4, -1              \n\t"
            "vfwmacc.vf     v31, ft1, v20           \n\t"
            "addi           %[A], t6, 2             \n\t"
            "bgtz           t4, BLK_LOOP%=          \n\t"
            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            ".option pop\n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k256)
            : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
              "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1",
              "ft0", "ft1", "memory");
    }
}

// Four rows, a multiple of 32 columns.
__attribute__((target("arch=+zfh,+zvfh,+xsmtvdotii"), noinline))
static void hp_m4_asm(
    const uint8_t* quant_a, const uint8_t* quant_b, float* c, size_t count_n, size_t k256, size_t ldc)
{
    constexpr size_t NB_COLS = 32;
    const size_t b_tile_stride = k256 * kHpSuperBytes;
    const unsigned long ldc_bytes = ldc * 4;
    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        const uint8_t* b_tile = quant_b + (ni / NB_COLS) * b_tile_stride;
        const uint8_t* a_block = quant_a;
        float* dst_c = c + ni;
        __asm__ volatile(
            ".option push\n\t"
            ".option arch, +zfh,+zvfh,+xsmtvdotii\n\t"
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v28, v28, v28             \n\t"
            "vxor.vv        v29, v29, v29             \n\t"
            "vxor.vv        v30, v30, v30             \n\t"
            "vxor.vv        v31, v31, v31             \n\t"
            "li             t0, 0x4c00                \n\t"
            "fmv.h.x        ft0, t0                   \n\t"
            "li             t4, 8                     \n\t"
            "addi           t2, t6, 1088              \n\t"
            ".align 4                                 \n\t"
            "_BLK_LPST%=:                             \n\t"
            "flh            fa1, 64(t2)               \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v18, v30, v30             \n\t"
            "vxor.vv        v19, v31, v31             \n\t"
            "vxor.vv        v20, v30, v30             \n\t"
            "vxor.vv        v21, v31, v31             \n\t"
            "_KsubBLK_LPST%=:                         \n\t"
            "flh            fa0,   0(t6)              \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"
            "fmul.h         fa2, fa0, ft0             \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v16, v12, fa0             \n\t"
            "vfmul.vf       v17, v12, fa2             \n\t"
            "flh            ft1, 0(t2)                \n\t"
            "flh            ft2, 16(t2)               \n\t"
            "flh            ft3, 32(t2)               \n\t"
            "flh            ft4, 48(t2)               \n\t"
            "addi           t3, t6, 8                 \n\t"
            "vsetvli        t0, x0, e8, m1            \n\t"
            "vl1r.v         v0, (t3)                  \n\t"
            "addi           t3, s5, 64                \n\t"
            "vl4r.v         v4, (t3)                  \n\t"
            "vsetvli        t0, x0, e8, m1            \n\t"
            "vsrl.vi        v1, v0, 4                 \n\t"
            "vnpack4.vv     v12, v0, v1, 3            \n\t"
            "vpack.vv       v0, v17, v16, 3           \n\t"
            "vupack.vv      v2, v12, v12, 2           \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v12, v16, ft1             \n\t"
            "vfmul.vf       v13, v16, ft2             \n\t"
            "vfmul.vf       v24, v16, ft3             \n\t"
            "vfmul.vf       v25, v16, ft4             \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfwmacc.vf     v28, fa1, v12             \n\t"
            "vfwmacc.vf     v29, fa1, v13             \n\t"
            "vfwmacc.vf     v30, fa1, v24             \n\t"
            "vfwmacc.vf     v31, fa1, v25             \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
            "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
            "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
            "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
            "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
            "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
            "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
            "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"
            "addi           t4, t4, -1                \n\t"
            "addi           t6, t6, 8+128             \n\t"
            "addi           t2, t2, 2                 \n\t"
            "addi           s5, s5, 64+512            \n\t"
            "bgtz           t4, _KsubBLK_LPST%=       \n\t"
            "vsetvli        t0, x0, e16, m1           \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"
            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfwmacc.vf     v28, fa1, v26             \n\t"
            "vfwmacc.vf     v30, fa1, v27             \n\t"
            "li             t4, 8                     \n\t"
            "addi           t5, t5, -1                \n\t"
            "addi           t6, t6, 72                \n\t"
            "addi           t2, t6, 1088              \n\t"
            "bgtz           t5, _BLK_LPST%=           \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "add            t2, %[LDC], %[DST]        \n\t"
            "vse32.v        v28, (%[DST])             \n\t"
            "add            t3, %[LDC], t2            \n\t"
            "vse32.v        v29, (t2)                 \n\t"
            "add            t2, %[LDC], t3            \n\t"
            "vse32.v        v30, (t3)                 \n\t"
            "vse32.v        v31, (t2)                 \n\t"
            ".option pop\n\t"
            : [A] "+r"(a_block), [B] "+r"(b_tile)
            : [DST] "r"(dst_c), [LDC] "r"(ldc_bytes), [BK] "r"(k256)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft0", "ft1",
              "ft2", "ft3", "ft4", "memory");
    }
}

static int hp_nibble(const uint8_t* qs, size_t i)
{
    const uint8_t byte = qs[i >> 1];
    return (i & 1) ? (byte >> 4) : (byte & 0x0f);
}

static void hp_tail_m1(
    const uint8_t* a, const uint8_t* b, float* c, size_t col0, size_t ncols, size_t k256)
{
    const size_t tile_bytes = k256 * kHpSuperBytes;
    for (size_t ccol = 0; ccol < ncols; ++ccol) {
        const size_t col = col0 + ccol;
        float acc = 0.0f;
        for (size_t s = 0; s < k256; ++s) {
            const uint8_t* ab = a + s * kHpARowSuper;
            const float scale_avg = hp_load_f16(ab + 288);
            for (size_t kk = 0; kk < kHpSubs; ++kk) {
                const uint8_t* sub = ab + kk * kHpASub;
                const float as = hp_load_f16(sub);
                const int8_t* q = reinterpret_cast<const int8_t*>(sub + 2);
                const float asum = hp_load_f16(ab + 272 + kk * 2);
                const uint8_t* blk =
                    b + (col / kHpNb) * tile_bytes + s * kHpSuperBytes + kk * kHpBlkBytes;
                const float bs = hp_load_f16(blk + (col % kHpNb) * 2);
                const uint8_t* qs = blk + 64 + (col % kHpNb) * 16;
                int dot = 0;
                for (size_t i = 0; i < kHpSub; ++i) {
                    dot += q[i] * hp_nibble(qs, i);
                }
                acc += scale_avg * as * bs * (static_cast<float>(dot) + asum);
            }
        }
        c[col] = acc;
    }
}

static void hp_tail_m4(
    const uint8_t* a, const uint8_t* b, float* c, size_t col0, size_t ncols, size_t k256, size_t ldc)
{
    const size_t tile_bytes = k256 * kHpSuperBytes;
    for (size_t r = 0; r < 4; ++r) {
        for (size_t ccol = 0; ccol < ncols; ++ccol) {
            const size_t col = col0 + ccol;
            float acc = 0.0f;
            for (size_t s = 0; s < k256; ++s) {
                const uint8_t* ab = a + s * kHpA4Super;
                const float scale_avg = hp_load_f16(ab + kHpSubs * kHpA4Sub + 64);
                for (size_t kk = 0; kk < kHpSubs; ++kk) {
                    const uint8_t* sub = ab + kk * kHpA4Sub;
                    const float as = hp_load_f16(sub);
                    const int8_t* q = reinterpret_cast<const int8_t*>(sub + 8 + r * kHpSub);
                    const float asum = hp_load_f16(ab + kHpSubs * kHpA4Sub + (r * kHpSubs + kk) * 2);
                    const uint8_t* blk =
                        b + (col / kHpNb) * tile_bytes + s * kHpSuperBytes + kk * kHpBlkBytes;
                    const float bs = hp_load_f16(blk + (col % kHpNb) * 2);
                    const uint8_t* qs = blk + 64 + (col % kHpNb) * 16;
                    int dot = 0;
                    for (size_t i = 0; i < kHpSub; ++i) {
                        dot += q[i] * hp_nibble(qs, i);
                    }
                    acc += scale_avg * as * bs * (static_cast<float>(dot) + asum);
                }
            }
            c[r * ldc + col] = acc;
        }
    }
}

static void hp_add_bias(float* c, size_t rows, size_t cols, size_t ldc, const float* bias)
{
    if (bias == nullptr) {
        return;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t n = 0; n < cols; ++n) {
            c[r * ldc + n] += bias[n];
        }
    }
}

void
Ime2HpGemm(
    const std::byte* QuantA,
    const std::byte* QuantB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias)
{
    const size_t k256 = hp_k256(BlockCountK);
    const size_t row_bytes = k256 * kHpARowSuper;
    const size_t nfull = CountN & ~size_t{31};
    const uint8_t* b = reinterpret_cast<const uint8_t*>(QuantB);
    const uint8_t* a = reinterpret_cast<const uint8_t*>(QuantA);
    float* c = C;
    size_t m = 0;
    for (; m + 4 <= CountM; m += 4) {
        if (nfull != 0) {
            hp_m4_asm(a, b, c, nfull, k256, ldc);
        }
        if (nfull < CountN) {
            hp_tail_m4(a, b, c, nfull, CountN - nfull, k256, ldc);
        }
        hp_add_bias(c, 4, CountN, ldc, Bias);
        a += 4 * row_bytes;
        c += 4 * ldc;
    }
    for (; m < CountM; ++m) {
        if (nfull != 0) {
            hp_m1_asm(a, b, c, nfull, k256);
        }
        if (nfull < CountN) {
            hp_tail_m1(a, b, c, nfull, CountN - nfull, k256);
        }
        hp_add_bias(c, 1, CountN, ldc, Bias);
        a += row_bytes;
        c += ldc;
    }
}

}  // namespace vllm_ime2
