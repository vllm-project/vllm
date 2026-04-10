#ifndef CPU_TYPES_RISCV_HPP
#define CPU_TYPES_RISCV_HPP

// Routes to the right concrete RVV vector-types header for the VLEN this
// translation unit was compiled for. The compiler defines __riscv_v_min_vlen
// from the zvl<N>b extension in -march. To support a new VLEN, add a
// cpu_types_riscv_<N>.hpp file and an additional branch below.

#ifndef __riscv_vector
  #error "cpu_types_riscv.hpp included in a non-RVV translation unit"
#endif

#ifndef __riscv_v_min_vlen
  #error "compiler did not define __riscv_v_min_vlen; pass -march=...zvl<N>b"
#endif

#if __riscv_v_min_vlen == 128
  #include "cpu_types_riscv_128.hpp"
#elif __riscv_v_min_vlen == 256
  #include "cpu_types_riscv_256.hpp"
#else
  #error "unsupported __riscv_v_min_vlen; add a cpu_types_riscv_<N>.hpp"
#endif

#endif  // CPU_TYPES_RISCV_HPP
