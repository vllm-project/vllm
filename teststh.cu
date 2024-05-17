#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {

    #if defined(__CUDA_ARCH__)
    printf("cuda arch: %ld\n", __CUDA_ARCH__);
    #else
    printf("no cuda arch found.\n");
    #endif

    return 0;
}