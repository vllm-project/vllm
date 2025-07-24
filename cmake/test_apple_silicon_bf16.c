#include <arm_neon.h>

int main() {
    // Test if bf16 instructions are available
    #if defined(__ARM_FEATURE_BF16)
        return 0;  // bf16 is supported
    #else
        return 1;  // bf16 is not supported
    #endif
} 