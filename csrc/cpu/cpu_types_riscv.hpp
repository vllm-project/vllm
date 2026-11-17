#ifndef CPU_TYPES_RISCV_HPP
#define CPU_TYPES_RISCV_HPP

// RISC-V Vector (RVV) CPU type definitions for vLLM.
//
// Supports multiple VLENs via compile-time dispatch. The compiler defines
// __riscv_v_min_vlen from the zvl<N>b extension in -march. The defs header
// maps VLEN to the correct LMUL suffixes, and the impl header provides
// VLEN-independent class implementations.
//
// To add support for a new VLEN, add the LMUL mapping in
// cpu_types_riscv_defs.hpp (the impl header needs no changes).

#ifndef __riscv_vector
  #error "cpu_types_riscv.hpp included in a non-RVV translation unit"
#endif

#ifndef __riscv_v_min_vlen
  #error "compiler did not define __riscv_v_min_vlen; pass -march=...zvl<N>b"
#endif

#include "cpu_types_riscv_defs.hpp"
#include "cpu_types_riscv_impl.hpp"

#endif  // CPU_TYPES_RISCV_HPP
