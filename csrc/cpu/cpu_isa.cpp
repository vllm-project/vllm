// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// AMX-FP8 is reported at CPUID leaf 0x1E, subleaf 1, EAX[4].
#if defined(__x86_64__) || defined(_M_X64)
  #include <cpuid.h>

bool runtime_has_amx_fp8() {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid_max(0, nullptr) < 0x1e) return false;
  if (__get_cpuid_count(0x1e, 1, &eax, &ebx, &ecx, &edx)) {
    return (eax >> 4) & 1u;
  }
  return false;
}

bool cpu_has_amx_fp8() { return runtime_has_amx_fp8(); }
#else
bool cpu_has_amx_fp8() { return false; }
#endif
