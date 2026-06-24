#pragma once
#ifdef __APPLE__
#include <libkern/OSByteOrder.h>

// 映射 Apple 的位操作指令到 vLLM 期望的 builtin
static inline int __builtin_clzg(unsigned int x, int fallback) {
    return x == 0 ? fallback : __builtin_clz(x);
}
static inline int __builtin_clzg(unsigned long x, int fallback) {
    return x == 0 ? fallback : __builtin_clzl(x);
}
static inline int __builtin_clzg(unsigned long long x, int fallback) {
    return x == 0 ? fallback : __builtin_clzll(x);
}

static inline int __builtin_ctzg(unsigned int x, int fallback) {
    return x == 0 ? fallback : __builtin_ctz(x);
}
static inline int __builtin_ctzg(unsigned long x, int fallback) {
    return x == 0 ? fallback : __builtin_ctzl(x);
}
static inline int __builtin_ctzg(unsigned long long x, int fallback) {
    return x == 0 ? fallback : __builtin_ctzll(x);
}
#endif
