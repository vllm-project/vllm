#pragma once

#include <cute/tensor.hpp>

#include <kerutils/device/common.h>

namespace cute {

// Extensions to CuTe
// CuTe's built-in SM100_TMA_2SM_LOAD_1D series requires the number of
// participating threads to be 2 (using ThrID = Layout<_2>;) and also splits the
// data, which is really annoying to use, so we modified our own version.
// Additionally, to keep it consistent with other parts that use SM90 TMA, we
// made it accept TMA::CacheHintSm90 instead of TMA::CacheHintSm100.

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD : Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////
struct SM100_TMA_2SM_LOAD_1D_NOSPLIT {
  CUTE_HOST_DEVICE static void copy([[maybe_unused]] void const* desc_ptr,
                                    [[maybe_unused]] uint64_t* mbar_ptr,
                                    [[maybe_unused]] uint64_t cache_hint,
                                    [[maybe_unused]] void* smem_ptr,
                                    [[maybe_unused]] int32_t const& crd0) {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar =
        cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3}], [%2], %4;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
          "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_2D_NOSPLIT {
  CUTE_HOST_DEVICE static void copy([[maybe_unused]] void const* desc_ptr,
                                    [[maybe_unused]] uint64_t* mbar_ptr,
                                    [[maybe_unused]] uint64_t cache_hint,
                                    [[maybe_unused]] void* smem_ptr,
                                    [[maybe_unused]] int32_t const& crd0,
                                    int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar =
        cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
          "r"(crd1), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_3D_NOSPLIT {
  CUTE_HOST_DEVICE static void copy([[maybe_unused]] void const* desc_ptr,
                                    [[maybe_unused]] uint64_t* mbar_ptr,
                                    [[maybe_unused]] uint64_t cache_hint,
                                    [[maybe_unused]] void* smem_ptr,
                                    [[maybe_unused]] int32_t const& crd0,
                                    int32_t const& crd1, int32_t const& crd2) {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar =
        cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
          "r"(crd1), "r"(crd2), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_4D_NOSPLIT {
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2, int32_t const& crd3) {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar =
        cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
          "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_5D_NOSPLIT {
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2, int32_t const& crd3,
                                    int32_t const& crd4) {
#if defined(CUTE_ARCH_TMA_SM100_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar =
        cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
          "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use tma without CUTE_ARCH_TMA_SM100_ENABLED.");
#endif
  }
};
struct SM100_TMA_2SM_LOAD_NOSPLIT {
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0) {
    return SM100_TMA_2SM_LOAD_1D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint,
                                               smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1) {
    return SM100_TMA_2SM_LOAD_2D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint,
                                               smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2) {
    return SM100_TMA_2SM_LOAD_3D_NOSPLIT::copy(desc_ptr, mbar_ptr, cache_hint,
                                               smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2, int32_t const& crd3) {
    return SM100_TMA_2SM_LOAD_4D_NOSPLIT::copy(
        desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2, int32_t const& crd3,
                                    int32_t const& crd4) {
    return SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
        desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
  using PREFETCH = typename SM90_TMA_LOAD::PREFETCH;
};
struct SM100_TMA_2SM_LOAD_NOSPLIT_OP : SM100_TMA_2SM_LOAD_NOSPLIT {};
// The non-executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT, NumBitsPerTMA, AuxParams_> {
  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  // SM100_TMA_2SM_LOAD_NOSPLIT arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr TmaDescriptor const* get_tma_descriptor() const {
    return &tma_desc_;
  }
  // Construct an executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_mbar
  CUTE_HOST_DEVICE constexpr Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP,
                                         NumBitsPerTMA>
  with(uint64_t& tma_mbar, [[maybe_unused]] uint16_t const& multicast_mask = 0,
       TMA::CacheHintSm90 const& cache_hint =
           TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }
  // Construct an executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_mbar (temp.
  // overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP,
                                         NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc, uint64_t& tma_mbar,
       [[maybe_unused]] uint16_t const& multicast_mask = 0,
       TMA::CacheHintSm90 const& cache_hint =
           TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }
  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape),
                               decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
  // Don't try to execute a copy with SM100_TMA_2SM_LOAD_NOSPLIT before calling
  // .with()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const& traits, Tensor<TS, SLayout> const& src,
      Tensor<TD, DLayout>& dst) = delete;
};
// The executable SM100_TMA_2SM_LOAD_NOSPLIT with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA>
    : TMA_LOAD_Unpack<SM100_TMA_2SM_LOAD_NOSPLIT_OP, NumBitsPerTMA> {
  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  // SM100_TMA_2SM_LOAD_NOSPLIT arguments
  tuple<TmaDescriptor const*,
        uint64_t*,  // smem mbarrier
        uint64_t    // cache hint
        > const opargs_;
  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint64_t cache)
      : opargs_(desc, mbar, cache) {}
};

}  // namespace cute
