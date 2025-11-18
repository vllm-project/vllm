#include <stdio.h>
extern void testCoalescing();

int main() {
    printf("=== Memory Coalescing Test ===\n");
    testCoalescing();
    printf("Test complete\n");
    return 0;
}
