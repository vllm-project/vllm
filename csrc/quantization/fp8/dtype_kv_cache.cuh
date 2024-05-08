#pragma once

namespace vllm {

enum class Fp8KVCacheDataType {
    kAuto = 0,
    kFp8E4M3 = 1,
    kFp8E5M2 = 2,
};
}
