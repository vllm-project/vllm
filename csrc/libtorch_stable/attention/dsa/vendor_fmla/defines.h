#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

struct int32x8_t {
  int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
  float2 a01, a23, a45, a67;
};

struct bf16x8 {
  __nv_bfloat162 a01;
  __nv_bfloat162 a23;
  __nv_bfloat162 a45;
  __nv_bfloat162 a67;
};
