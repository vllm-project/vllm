#pragma once
#ifdef __APPLE__
#include <libkern/OSByteOrder.h>

// 注入缺失的内置位操作函数，供 Apple Clang 编译使用
static inline int __builtin_clzg(unsigned int x, int n) { return x == 0 ? n : __builtin_clz(x); }
static inline int __builtin_clzg(unsigned long x, int n) { return x == 0 ? n : __builtin_clzl(x); }
static inline int __builtin_clzg(unsigned long long x, int n) { return x == 0 ? n : __builtin_clzll(x); }

static inline int __builtin_ctzg(unsigned int x, int n) { return x == 0 ? n : __builtin_ctz(x); }
static inline int __builtin_ctzg(unsigned long x, int n) { return x == 0 ? n : __builtin_ctzl(x); }
static inline int __builtin_ctzg(unsigned long long x, int n) { return x == 0 ? n : __builtin_ctzll(x); }
#endif
#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP

#if defined(__x86_64__)
  // x86 implementation
  #include "cpu_types_x86.hpp"
#elif defined(__POWER9_VECTOR__)
  // ppc implementation
  #include "cpu_types_vsx.hpp"
#elif defined(__s390x__)
  // s390 implementation
  #include "cpu_types_vxe.hpp"
#elif defined(__aarch64__)
  // arm implementation
  #include "cpu_types_arm.hpp"
#elif defined(__riscv_v)
  // riscv implementation
  #include "cpu_types_riscv.hpp"
#else
  #warning "unsupported vLLM cpu implementation, vLLM will compile with scalar"
  #include "cpu_types_scalar.hpp"
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#endif
