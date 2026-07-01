#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <sstream>

// Simulated tensor metadata structure mimicking what would come from shared memory
struct TensorMetadata {
    int64_t numel;
    int64_t element_size;
    void* shm_ptr;
    size_t shm_actual_size;
};

// Safe copy function that validates buffer sizes before memcpy
// This represents the FIXED version of the vulnerable code
// Returns false if the copy would be unsafe
bool safe_tensor_copy(void* dst, size_t dst_size, const TensorMetadata& shm) {
    // Check for integer overflow in numel * element_size
    if (shm.numel <= 0 || shm.element_size <= 0) {
        return false;
    }
    
    // Overflow check: numel * element_size must not overflow
    if (shm.numel > (int64_t)(std::numeric_limits<size_t>::max() / (size_t)shm.element_size)) {
        return false;
    }
    
    size_t copy_size = (size_t)shm.numel * (size_t)shm.element_size;
    
    // Validate that copy_size does not exceed actual shared memory size
    if (copy_size > shm.shm_actual_size) {
        return false;
    }
    
    // Validate that destination buffer is large enough
    if (copy_size > dst_size) {
        return false;
    }
    
    // Safe to copy
    std::memcpy(dst, shm.shm_ptr, copy_size);
    return true;
}

// Vulnerable copy function (simulating the original code)
// Returns the number of bytes it would copy (for analysis)
size_t vulnerable_copy_size(const TensorMetadata& shm) {
    // This simulates: shm.numel() * shm.element_size() without overflow check
    // We use unsigned arithmetic to detect overflow
    if (shm.numel <= 0 || shm.element_size <= 0) {
        return 0;
    }
    uint64_t numel = (uint64_t)shm.numel;
    uint64_t elem_size = (uint64_t)shm.element_size;
    uint64_t result = numel * elem_size;
    return (size_t)result;
}

class BufferOverflowSecurityTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BufferOverflowSecurityTest, NeverExceedsDeclaredLength) {
    // Invariant: Buffer reads must never exceed the declared destination buffer length.
    // The copy size (numel * element_size) must be validated against both
    // the actual shared memory size and the destination buffer size before memcpy.
    
    std::string payload = GetParam();
    
    // Parse payload as "numel:element_size:dst_size:shm_size"
    int64_t numel = 0, element_size = 0;
    size_t dst_size = 0, shm_size = 0;
    
    std::istringstream ss(payload);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, ':')) {
        tokens.push_back(token);
    }
    
    if (tokens.size() >= 4) {
        try {
            numel = std::stoll(tokens[0]);
            element_size = std::stoll(tokens[1]);
            dst_size = (size_t)std::stoull(tokens[2]);
            shm_size = (size_t)std::stoull(tokens[3]);
        } catch (...) {
            // Invalid payload format - test passes trivially
            SUCCEED();
            return;
        }
    } else {
        SUCCEED();
        return;
    }
    
    // Allocate actual buffers for testing
    // Cap allocation sizes to prevent test OOM
    const size_t MAX_ALLOC = 1024 * 1024; // 1MB max for test
    size_t safe_shm_size = std::min(shm_size, MAX_ALLOC);
    size_t safe_dst_size = std::min(dst_size, MAX_ALLOC);
    
    std::vector<uint8_t> shm_buffer(safe_shm_size > 0 ? safe_shm_size : 1, 0xAB);
    std::vector<uint8_t> dst_buffer(safe_dst_size > 0 ? safe_dst_size : 1, 0x00);
    
    TensorMetadata meta;
    meta.numel = numel;
    meta.element_size = element_size;
    meta.shm_ptr = shm_buffer.data();
    meta.shm_actual_size = safe_shm_size;
    
    bool result = safe_tensor_copy(dst_buffer.data(), safe_dst_size, meta);
    
    if (result) {
        // If copy succeeded, verify the copy size was within bounds
        if (numel > 0 && element_size > 0) {
            // Check no overflow occurred
            ASSERT_LE((uint64_t)numel, std::numeric_limits<uint64_t>::max() / (uint64_t)element_size)
                << "Integer overflow in numel * element_size";
            
            size_t copy_size = (size_t)numel * (size_t)element_size;
            
            // INVARIANT: copy size must never exceed destination buffer size
            ASSERT_LE(copy_size, safe_dst_size)
                << "Copy size " << copy_size << " exceeds destination buffer size " << safe_dst_size;
            
            // INVARIANT: copy size must never exceed actual shared memory size
            ASSERT_LE(copy_size, safe_shm_size)
                << "Copy size " << copy_size << " exceeds shared memory size " << safe_shm_size;
        }
    } else {
        // Copy was rejected - this is the safe behavior for oversized inputs
        // Verify that rejection was correct (i.e., the copy would have been unsafe)
        bool would_overflow = false;
        bool would_exceed_dst = false;
        bool would_exceed_shm = false;
        
        if (numel > 0 && element_size > 0) {
            if ((uint64_t)numel > std::numeric_limits<uint64_t>::max() / (uint64_t)element_size) {
                would_overflow = true;
            } else {
                size_t copy_size = (size_t)numel * (size_t)element_size;
                would_exceed_dst = (copy_size > safe_dst_size);
                would_exceed_shm = (copy_size > safe_shm_size);
            }
        }
        
        bool rejection_justified = (numel <= 0) || (element_size <= 0) || 
                                    would_overflow || would_exceed_dst || would_exceed_shm;
        
        ASSERT_TRUE(rejection_justified)
            << "Copy was rejected but would have been safe: numel=" << numel 
            << " element_size=" << element_size 
            << " dst_size=" << safe_dst_size 
            << " shm_size=" << safe_shm_size;
    }
}

// Additional direct overflow detection test
TEST(BufferOverflowDirectTest, IntegerOverflowDetected) {
    // INVARIANT: numel * element_size must never overflow size_t
    
    // Overflow case: large numel * large element_size
    TensorMetadata meta;
    meta.numel = std::numeric_limits<int64_t>::max();
    meta.element_size = 8;
    meta.shm_actual_size = 64;
    
    std::vector<uint8_t> shm_buf(64, 0);
    std::vector<uint8_t> dst_buf(64, 0);
    meta.shm_ptr = shm_buf.data();
    
    bool result = safe_tensor_copy(dst_buf.data(), 64, meta);
    ASSERT_FALSE(result) << "Should reject: numel * element_size overflows";
}

TEST(BufferOverflowDirectTest, OversizedNumericsRejected) {
    // INVARIANT: copy size exceeding destination must be rejected
    
    const size_t SHM_SIZE = 1024;
    const size_t DST_SIZE = 64; // destination is smaller than what tensor claims
    
    std::vector<uint8_t> shm_buf(SHM_SIZE, 0xCC);
    std::vector<uint8_t> dst_buf(DST_SIZE, 0x00);
    
    TensorMetadata meta;
    meta.numel = 256;       // 256 * 4 = 1024 bytes
    meta.element_size = 4;
    meta.shm_ptr = shm_buf.data();
    meta.shm_actual_size = SHM_SIZE;
    
    bool result = safe_tensor_copy(dst_buf.data(), DST_SIZE, meta);
    ASSERT_FALSE(result) << "Should reject: copy size 1024 > dst_size 64";
    
    // Verify destination buffer was not corrupted
    for (size_t i = 0; i < DST_SIZE; i++) {
        ASSERT_EQ(dst_buf[i], 0x00) << "Destination buffer corrupted at index " << i;
    }
}

TEST(BufferOverflowDirectTest, ManipulatedMetadataRejected) {
    // INVARIANT: Compromised peer cannot trigger heap overflow via manipulated metadata
    // Simulates a compromised peer sending inflated numel/element_size
    
    const size_t ACTUAL_SHM_SIZE = 128;
    const size_t DST_SIZE = 128;
    
    std::vector<uint8_t> shm_buf(ACTUAL_SHM_SIZE, 0xDE);
    std::vector<uint8_t> dst_buf(DST_SIZE, 0x00);
    
    // Attacker claims tensor is 10x larger than actual shared memory
    TensorMetadata meta;
    meta.numel = 320;           // Claims 320 * 4 = 1280 bytes
    meta.element_size = 4;
    meta.shm_ptr = shm_buf.data();
    meta.shm_actual_size = ACTUAL_SHM_SIZE; // Actual is only 128 bytes
    
    bool result = safe_tensor_copy(dst_buf.data(), DST_SIZE, meta);
    ASSERT_FALSE(result) << "Should reject: claimed size 1280 > actual shm size 128";
}

TEST(BufferOverflowDirectTest, ValidCopySucceeds) {
    // INVARIANT: Valid copies within bounds must succeed
    
    const size_t SIZE = 256;
    std::vector<uint8_t> shm_buf(SIZE, 0xAA);
    std::vector<uint8_t> dst_buf(SIZE, 0x00);
    
    TensorMetadata meta;
    meta.numel = 64;
    meta.element_size = 4;  // 64 * 4 = 256 bytes
    meta.shm_ptr = shm_buf.data();
    meta.shm_actual_size = SIZE;
    
    bool result = safe_tensor_copy(dst_buf.data(), SIZE, meta);
    ASSERT_TRUE(result) << "Valid copy should succeed";
    
    // Verify data was copied correctly
    for (size_t i = 0; i < SIZE; i++) {
        ASSERT_EQ(dst_buf[i], 0xAA) << "Data mismatch at index " << i;
    }
}

TEST(BufferOverflowDirectTest, ZeroNegativeValuesRejected) {
    // INVARIANT: Zero or negative numel/element_size must be rejected
    
    std::vector<uint8_t> shm_buf(64, 0);
    std::vector<uint8_t> dst_buf(64, 0);
    
    TensorMetadata meta;
    meta.shm_ptr = shm_buf.data();
    meta.shm_actual_size = 64;
    
    // Zero numel
    meta.numel = 0;
    meta.element_size = 4;
    ASSERT_FALSE(safe_tensor_copy(dst_buf.data(), 64, meta)) << "Zero numel should be rejected";
    
    // Negative numel
    meta.numel = -1;
    meta.element_size = 4;
    ASSERT_FALSE(safe_tensor_copy(dst_buf.data(), 64, meta)) << "Negative numel should be rejected";
    
    // Zero element_size
    meta.numel = 16;
    meta.element_size = 0;
    ASSERT_FALSE(safe_tensor_copy(dst_buf.data(), 64, meta)) << "Zero element_size should be rejected";
    
    // Negative element_size
    meta.numel = 16;
    meta.element_size = -4;
    ASSERT_FALSE(safe_tensor_copy(dst_buf.data(), 64, meta)) << "Negative element_size should be rejected";
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    BufferOverflowSecurityTest,
    ::testing::Values(
        // Format: "numel:element_size:dst_size:shm_size"
        // Normal valid case
        "16:4:64:64",
        // dst_size exactly matches copy size
        "16:4:64:128",
        // copy size exceeds dst by 2x
        "32:4:64:256",
        // copy size exceeds dst by 10x
        "160:4:64:1024",
        // copy size exceeds dst by 100x
        "1600:4:64:10000",
        // copy size exceeds shm_size (manipulated metadata)
        "1000:4:4096:128",
        // Integer overflow attempt: max int64 numel
        "9223372036854775807:8:64:64",
        // Integer overflow attempt: large numel and element_size
        "4611686018427387904:4:64:64",
        // Integer overflow: numel * element_size wraps to small value
        "2305843009213693952:8:64:64",
        // Zero numel
        "0:4:64:64",
        // Negative numel
        "-1:4:64:64",
        // Negative element_size
        "16:-4:64:64",
        // Both negative
        "-16:-4:64:64",
        // Extremely large element_size
        "2:9223372036854775807:64:64",
        // copy size == dst_size (boundary)
        "8:8:64:64",
        // copy size == dst_size - 1 (just under boundary)
        "7:8:64:64",
        // copy size == dst_size + 1 (just over boundary)
        "9:8:64:64",
        // Very large shm claimed, small actual
        "1048576:4:4194304:64",
        // Attacker sends element_size=1 but huge numel
        "1048576:1:64:64",
        // Attacker sends numel=1 but huge element_size
        "1:1048576:64:64"