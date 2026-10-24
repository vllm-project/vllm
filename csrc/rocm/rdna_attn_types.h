// The head dims, element types and GPU families the RDNA3.5 attention kernels
// are built for.
#pragma once

namespace rdna {

enum class HeadDim : int {
  k64 = 64,
  k96 = 96,
  k128 = 128,
  k256 = 256,
  k512 = 512
};

enum class Dtype : int { kFp16 = 0, kBf16 = 1 };

// The GPU families with their own config tables, each built for its archs
// only (cmake/rocm_rdna_attn.cmake): gfx1151, with 1536 VGPRs a SIMD lane, and
// gfx1150, with 1024, whose tables gfx1152 and gfx1153 also use.
enum class Arch : int { kGfx1151 = 0, kGfx1150 = 1 };
inline constexpr int kNumArches = 2;

}  // namespace rdna
