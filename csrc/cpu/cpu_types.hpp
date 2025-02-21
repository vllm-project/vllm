#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP

#if defined(__x86_64__)
  // x86 implementation
  #include "cpu_types_x86.hpp"
#elif defined(__POWER9_VECTOR__)
  // ppc implementation
  #include "cpu_types_vsx.hpp"
#elif defined(__aarch64__)
  // arm implementation
  #include "cpu_types_arm.hpp"
#else
  #warning "unsupported vLLM cpu implementation"
#endif

#endif