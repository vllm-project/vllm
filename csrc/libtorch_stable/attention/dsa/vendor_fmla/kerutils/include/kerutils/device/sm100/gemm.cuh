#pragma once

#include <cute/tensor.hpp>

#include <kerutils/device/common.h>

namespace cute {

// Extensions to CuTe
// CuTe don't support UTCMMA with .ws, so we add it here
// Besides, CuTe's UTCMMA has an `elect_one_sync()` inside which is really
// disgusting, so we have our own variant without `elect_one_sync()` here

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT {
  static_assert(M == 32 || M == 64 || M == 128,
                "SM100_MMA_F16BF16_WS_TS_NOELECT M-mode size should be 32, 64 "
                "or 128 for 1 CTA cluster MMA.");
  static_assert(
      N == 64 || N == 128 || N == 256,
      "SM100_MMA_F16BF16_WS_TS_NOELECT N-mode size should be 32, 64 or 128");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scaleC,
                                   uint64_t const& idescE) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_TS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_WS_TS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::tmem_frg_1sm<
      a_type, a_type,
      UMMA::TmemAllocMode::NonInterleaved>;  // Actually this should be
                                             // "duplicated", however, our great
                                             // CuTe doesn't allow us to set it
                                             // to "duplicated", so we just set
                                             // it to NonInterleaved for a
                                             // correct address calculation
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout =
      Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout =
      Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout =
      Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>();

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits, Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                    b_major, a_neg,
                                    b_neg>::fma(tmem_a, desc_b, tmem_c,
                                                uint32_t(traits.accumulate_),
                                                idesc);
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT {
  static_assert(M == 32 || M == 64 || M == 128,
                "SM100_MMA_F16BF16_WS_SS_NOELECT M-mode size should be 32, 64 "
                "or 128 for 1 CTA cluster MMA.");
  static_assert(
      N == 64 || N == 128 || N == 256,
      "SM100_MMA_F16BF16_WS_SS_NOELECT N-mode size should be 32, 64 or 128");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scaleC,
                                   uint64_t const& idescE) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC));
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_WS_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout =
      Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout =
      Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout =
      Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits, Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                    b_major, a_neg,
                                    b_neg>::fma(desc_a, desc_b, tmem_c,
                                                uint32_t(traits.accumulate_),
                                                idesc);
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_2x1SM_TS_NOELECT {
  static_assert(M == 128 || M == 256,
                "SM100_MMA_F16BF16_2x1SM_TS_NOELECT M-mode size should be 128 "
                "or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256),
                "SM100_MMA_F16BF16_2x1SM_TS_NOELECT N-mode size should be a "
                "multiple of 16 between 16 and 256.");
  static_assert(
      a_major == UMMA::Major::K,
      "SM100_MMA_F16BF16_2x1SM_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scaleC,
                                   uint64_t const& idescE) {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, "
        "%8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM100_MMA_F16BF16_2x1SM_TS_NOELECT without "
        "CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_TS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_2x1SM_TS_NOELECT supports 16bit types");

  using FrgTypeA =
      UMMA::tmem_frg_2sm<a_type, a_type, UMMA::TmemAllocMode::Duplicated>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions' K extent is always 256 bits; convert to units of
  // element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_2>;
  using ALayout = Layout<Shape<_2, Shape<Int<M / 2>, Int<K>>>,
                         Stride<Int<M / 2>, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_2, Shape<Int<N / 2>, Int<K>>>,
                         Stride<Int<N / 2>, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_2, Shape<Int<M / 2>, Int<N>>>,
                         Stride<Int<M / 2>, Stride<_1, Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg, c_sat>();

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits, Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                       b_major, a_neg, b_neg,
                                       c_sat>::fma(tmem_a, desc_b, tmem_c,
                                                   uint32_t(traits.accumulate_),
                                                   idesc);
  }
};

// SM100_MMA_F16BF16_2x1SM_SS without elect_one_sync()
template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS_NOELECT {
  static_assert(M == 128 || M == 256,
                "SM100_MMA_F16BF16_2x1SM_SS_NOELECT M-mode size should be 128 "
                "or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256),
                "SM100_MMA_F16BF16_2x1SM_SS_NOELECT N-mode size should be a "
                "multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scaleC,
                                   uint64_t const& idescE) {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, "
        "%9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM100_MMA_F16BF16_2x1SM_SS_NOELECT without "
        "CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_2x1SM_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instructions's K extent is always 256bits, convert to units of
  // element
  constexpr static int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_2>;
  using ALayout = Layout<Shape<_2, Shape<Int<M / 2>, Int<K>>>,
                         Stride<Int<M / 2>, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_2, Shape<Int<N / 2>, Int<K>>>,
                         Stride<Int<N / 2>, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_2, Shape<Int<M / 2>, Int<N>>>,
                         Stride<Int<M / 2>, Stride<_1, Int<M>>>>;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits, Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                       b_major, a_neg,
                                       b_neg>::fma(desc_a, desc_b, tmem_c,
                                                   uint32_t(traits.accumulate_),
                                                   idesc);
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_TS_NOELECT {
  static_assert(M == 64 || M == 128,
                "SM100_MMA_F16BF16_TS_NOELECT M-mode size should be 64 or 128 "
                "for 1 CTA cluster MMA.");
  static_assert(
      (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
          (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
      "SM100_MMA_F16BF16_TS_NOELECT N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
  static_assert(a_major == UMMA::Major::K,
                "SM100_MMA_F16BF16_TS_NOELECT A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scaleC,
                                   uint64_t const& idescE) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, "
        "%8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F16BF16_TS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_TS_NOELECT supports 16bit types");

  using FrgTypeA =
      UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC =
      UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // Logical shape-K is always 256 bits; transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout =
      Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout =
      Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout =
      Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg, c_sat>();

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits, Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_TS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major,
                                 a_neg, b_neg,
                                 c_sat>::fma(tmem_a, desc_b, tmem_c,
                                             uint32_t(traits.accumulate_),
                                             idesc);
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT {
  static_assert(M == 64 || M == 128,
                "SM100_MMA_F16BF16_SS_NOELECT M-mode size should be 64 or 128 "
                "for 1 CTA cluster MMA.");
  static_assert(
      (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
          (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
      "SM100_MMA_F16BF16_SS_NOELECT N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scaleC,
                                   uint64_t const& idescE) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, "
        "%8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_SS_NOELECT supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout =
      Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout =
      Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout =
      Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(
      MMA_Traits const& traits, Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A, Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major,
                                 a_neg, b_neg>::fma(desc_a, desc_b, tmem_c,
                                                    uint32_t(
                                                        traits.accumulate_),
                                                    idesc);
  }
};

}  // namespace cute
