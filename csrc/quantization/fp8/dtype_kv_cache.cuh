#pragma once

namespace vllm {

enum class Fp8KVCacheDataType {
    kAuto = 0,
    kFp8E4m3 = 1,
    kFp8E5m2 = 2,
};
}
