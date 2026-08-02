#pragma once
// SM100_MMA_F8F6F4_2x1SM_SS without elect_one_sync(), mirroring the
// F16BF16 NOELECT atoms in kerutils/device/sm100/gemm.cuh. The caller
// (single MMA warp, one issuing thread) already guarantees one issuer.
#include <cute/tensor.hpp>

namespace cute {

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F8F6F4_2x1SM_SS_NOELECT {
  static_assert(M == 128 || M == 256,
                "SM100_MMA_F8F6F4_2x1SM_SS_NOELECT M-mode size should be 128 "
                "or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256),
                "SM100_MMA_F8F6F4_2x1SM_SS_NOELECT N-mode size should be a "
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
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, "
        "%8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
          "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM100_MMA_F8F6F4_2x1SM_SS_NOELECT without "
        "CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 &&
                    cute::sizeof_bits_v<b_type> <= 8,
                "SM100_MMA_F8F6F4_2x1SM_SS_NOELECT supports <= 8bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_2sm<c_type>;

  // Size of instruction's K extent is always 256bits -> 32 elements for 8bit
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

    SM100_MMA_F8F6F4_2x1SM_SS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                      b_major, a_neg,
                                      b_neg>::fma(desc_a, desc_b, tmem_c,
                                                  uint32_t(traits.accumulate_),
                                                  idesc);
  }
};

}  // namespace cute
