#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

extern "C" {
    void memcpy(void* dst, void* src, const int64_t bytes);
}

class SecurityTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(SecurityTest, BufferReadsNeverExceedDeclaredLength) {
    // Invariant: Memory copy operations never read beyond source buffer boundaries
    auto [src_size, copy_size] = GetParam();
    
    // Allocate source buffer with guard pages
    uint8_t* src = new uint8_t[src_size + 32];
    uint8_t* dst = new uint8_t[src_size + 32];
    
    // Fill source with pattern and add sentinel values
    for (size_t i = 0; i < src_size + 32; ++i) {
        src[i] = 0xAA;
        dst[i] = 0xBB;
    }
    
    // Add sentinel pattern after source buffer
    for (size_t i = src_size; i < src_size + 32; ++i) {
        src[i] = 0xCC;
    }
    
    // Perform the potentially dangerous copy
    memcpy(dst, src, static_cast<int64_t>(copy_size));
    
    // Verify sentinel values after source buffer remain unchanged
    bool sentinel_unchanged = true;
    for (size_t i = src_size; i < src_size + 32; ++i) {
        if (src[i] != 0xCC) {
            sentinel_unchanged = false;
            break;
        }
    }
    
    // Cleanup
    delete[] src;
    delete[] dst;
    
    // Assert no out-of-bounds read occurred
    EXPECT_TRUE(sentinel_unchanged) << "Buffer overflow detected: source buffer overread";
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    SecurityTest,
    ::testing::Values(
        // Exact exploit case: copy size exceeds source buffer
        std::make_tuple(64, 128),
        // Boundary case: copy size equals source buffer
        std::make_tuple(64, 64),
        // Valid input: copy size smaller than source buffer
        std::make_tuple(128, 64),
        // Large overflow case
        std::make_tuple(32, 320),
        // Unaligned boundary case
        std::make_tuple(100, 150)
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}