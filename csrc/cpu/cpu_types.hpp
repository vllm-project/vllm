#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP



#if defined(__x86_64__)
  // x86 implementation
  #include "cpu_types_x86.hpp"
#elif defined(__powerpc__) 
// ppc implementation
#pragma message("power pc triggered \n\n\n")
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

#include <c10/util/Exception.h>

namespace cpu_utils {
// Without OpenMP the omp pragmas compile to serial loops, so report 1: kernels
// that barrier on the thread count would otherwise deadlock.
inline int get_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  TORCH_WARN_ONCE(
      "vLLM CPU was built without OpenMP; running single-threaded.");
  return 1;
#endif
}
}  // namespace cpu_utils

#endif
