#pragma once

#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>

namespace deep_gemm::mma::sm100 {

/// Shared memory descriptor
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_smem_desc(cute::UMMA::LayoutType layout, void* smem_ptr,
                                          const uint32_t& stride_byte_offset, const uint32_t& leading_byte_offset) {
    cute::UMMA::SmemDescriptor desc;

    // Set the version for SM100
    desc.version_ = 1;

    // Legacy mode
    desc.lbo_mode_ = 0;

    // Layout
    desc.layout_type_ = static_cast<uint8_t>(layout);

    // Start address
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // Base offset
    desc.base_offset_ = 0;

    // SBO and LBO
    desc.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;

    return desc;
}

CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    // NOTES: the UTCCP layout is K-major by default
    // Atom size: 8 x 128 bits
    // {SBO, LBO} means the byte stride between atoms on {MN, K}
    // Since the UTCCP we used is 128b-wide (only 1 atom on K), so LBO can be zero
    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

CUTLASS_DEVICE
void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

CUTLASS_DEVICE
static uint32_t get_atom_base(const cute::UMMA::LayoutType& layout_type) {
    return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

/// UMMA descriptors
// ReSharper disable once CppNotAllPathsReturnValue
template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
constexpr static cute::UMMA::LayoutType to_umma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");
    // A special case
    if constexpr ((cute::is_same_v<dtype_t, float> and kMajorMode == cute::UMMA::Major::MN) or kUseBase32) {
        DG_STATIC_ASSERT(kUseBase32, "Invalid swizzling base");
        return cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B;
    }

    // Normal cases
    if constexpr (kSwizzleMode == 0)   return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 16)  return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)  return cute::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)  return cute::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128) return cute::UMMA::LayoutType::SWIZZLE_128B;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
constexpr uint32_t get_umma_desc_stride_k() {
    return kMajorMode == cute::UMMA::Major::K ? 1 : tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
uint32_t advance_umma_desc_lo(const uint32_t& base, const uint32_t& offset, const uint32_t& k_idx) {
    return base + (((offset + k_idx * get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>()) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, bool kUseBase32 = false, typename dtype_t>
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_umma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
    const auto num_non_contiguous = 128 / get_atom_base(layout_type);
    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        // NOTES: for K-major layout, the swizzle must be the same as `BLOCK_K * sizeof(dtype_t)`
        // also, atom index must be 0, so that each block has exactly one swizzle atom on the K axis
        DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
        // {SBO, LBO} means the byte stride between atoms on {MN, K}
        // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
        const uint32_t leading_byte_offset = 0;
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    } else {
        constexpr uint32_t BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();

        // Must have no in-atom MN-idx
        // NOTES: no worries for the runtime assert, the `mn_idx` are constants at compilation time
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for non-swizzling
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    }
}

CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

CUTLASS_DEVICE void update_instr_desc_with_umma_n(
    cute::UMMA::InstrDescriptorBlockScaled& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

CUTLASS_DEVICE void update_instr_desc_with_umma_n(
    cute::UMMA::InstrDescriptor& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

} // namespace deep_gemm::mma::sm100
