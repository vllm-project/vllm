// The head dims and element types the RDNA3.5 attention kernels are built for.
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

}  // namespace rdna
