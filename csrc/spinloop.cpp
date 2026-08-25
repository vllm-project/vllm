#include <Python.h>

extern "C" {

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#if defined(__aarch64__)
  #include <sys/auxv.h>
#endif

#if defined(__i386__) || defined(__x86_64__)
  #include <cpuid.h>
  #include <x86intrin.h>
#endif

#if defined(CLOCK_MONOTONIC_RAW)
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC_RAW
#else
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC
#endif

#define CPU_SUPPORT_NONE 0
#define CPU_SUPPORT_MONITORX 1
#define CPU_SUPPORT_WFET 2
#define CPU_SUPPORT_WAITPKG 3

#if defined(__i386__) || defined(__x86_64__)
  // Bounded wait budget per iteration, in microseconds. umwait is a bounded
  // monitor/timeout wait; the caller's callback re-checks the predicate
  // between waits, so a short budget bounds the worst-case re-check latency.
  #define UMWAIT_BUDGET_US 50
#endif

#if defined(__aarch64__)
  // Bit 31 per the arm64 ELF hwcaps ABI (FEAT_WFxT).
  #ifndef HWCAP2_WFXT
    #define HWCAP2_WFXT (1UL << 31)
  #endif
  // Bounded wait budget per iteration, in microseconds. WFET is a bounded
  // event/timeout wait; the caller's callback re-checks the predicate
  // between waits, so a short budget bounds the worst-case re-check
  // latency. Experimental: not validated for production use.
  #define WFET_BUDGET_US 50
#endif

#define MWAITX_DEFAULT_TIMEOUT_CYCLES 1000000

typedef struct {
  unsigned int cpu_support;
  unsigned int max_monitor_line_size;
} spinloop_state_t;

static void determine_cpu_support(spinloop_state_t* state) {
  state->cpu_support = CPU_SUPPORT_NONE;
  state->max_monitor_line_size = 0;

#if defined(__i386__) || defined(__x86_64__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 1) {
    // AMD CPU (possible monitorx/mwaitx support)
    if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163) {
      if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 1 &&
          eax >= 0x80000001 &&
          __get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 1) {
        if ((ecx & (1 << 29)) != 0) {
          state->cpu_support = CPU_SUPPORT_MONITORX;
        }
      }
    }
  }

  if (state->cpu_support == CPU_SUPPORT_MONITORX) {
    if (__get_cpuid(5, &eax, &ebx, &ecx, &edx) == 1) {
      state->max_monitor_line_size = ebx & 0xff;
    }
  } else if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 1 &&
             (edx & (1 << 5)) != 0) {
    // Intel WAITPKG (umonitor/umwait/tpause), CPUID.(EAX=7,ECX=0):EDX[5].
    state->cpu_support = CPU_SUPPORT_WAITPKG;
  }
#endif

#if defined(__aarch64__)
  // FEAT_WFxT (WFET/WFIT): ELF HWCAP2 bit 31. Runtime-gated so baseline
  // armv8 builds load everywhere and only wait via WFET where present.
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  if (hwcap2 & HWCAP2_WFXT) {
    state->cpu_support = CPU_SUPPORT_WFET;
  }
#endif
}

// Bounded-wait status shared by the arch helpers (wfet / umwait). Pure C:
// no Python API is touched (caller holds no GIL). WAIT_OK, WAIT_SPUN_OUT
// when the overall timeout expired, or WAIT_CLOCK_ERR when clock_gettime
// fails (caller defers the PyErr).
enum bounded_wait_status { WAIT_OK, WAIT_SPUN_OUT, WAIT_CLOCK_ERR };

#if defined(__aarch64__)
// Bounded WFET wait for the fallback path. Computes the absolute CNTVCT
// deadline from the remaining caller timeout and the per-wait budget, then
// waits once.
static enum bounded_wait_status wfet_bounded_wait(
    double timeout, const struct timespec* t_start) {
  double remaining_s = INFINITY;
  if (timeout > 1e-9) {
    struct timespec t_now;
    if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
      return WAIT_CLOCK_ERR;
    }
    const double elapsed = (double)(t_now.tv_sec - t_start->tv_sec) +
                           (t_now.tv_nsec - t_start->tv_nsec) * 1e-9;
    remaining_s = timeout - elapsed;
    if (remaining_s <= 0) {
      return WAIT_SPUN_OUT;
    }
  }
  const double budget_s = (double)WFET_BUDGET_US * 1e-6;
  const double wait_s = remaining_s < budget_s ? remaining_s : budget_s;
  uint64_t now_ticks, freq_hz, wait_ticks, deadline_ticks;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(now_ticks));
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq_hz));
  if (freq_hz == 0) {
    // Unreachable on compliant hardware; degrade to a plain yield.
    __asm__ volatile("yield" ::: "memory");
    return WAIT_OK;
  }
  wait_ticks = (uint64_t)(wait_s * (double)freq_hz);
  if (wait_ticks < 1) {
    wait_ticks = 1;  // rounding must not produce an immediate wake
  }
  deadline_ticks = now_ticks + wait_ticks;
  // wfet x16: register operand encoded in the .inst immediate
  // (encoding d5031000 | Rd); x16 is an interprocedural scratch register
  // not carrying live values here.
  __asm__ volatile(
      "mov x16, %[deadline]\n"
      ".inst 0xd5031010\n"
      :
      : [deadline] "r"(deadline_ticks)
      : "x16", "memory");
  return WAIT_OK;
}
#endif

#if defined(__i386__) || defined(__x86_64__)
// Bounded umwait for the fallback path on WAITPKG parts: umonitor arms the
// address range, then umwait sleeps until a store there (or a zero-extend of
// the counter deadline in TSC ticks, or an interrupt) wakes us. The deadline
// is capped by the remaining caller timeout and the per-wait budget, matching
// wfet_bounded_wait. Pure C: no Python API is touched (caller holds no GIL).
// Returns WAIT_OK, WAIT_SPUN_OUT when the overall timeout expired, or
// WAIT_CLOCK_ERR when clock_gettime fails (caller defers the PyErr).
static enum bounded_wait_status umwait_bounded_wait(
    const void* addr, double timeout, const struct timespec* t_start) {
  double remaining_s = INFINITY;
  if (timeout > 1e-9) {
    struct timespec t_now;
    if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
      return WAIT_CLOCK_ERR;
    }
    const double elapsed = (double)(t_now.tv_sec - t_start->tv_sec) +
                           (t_now.tv_nsec - t_start->tv_nsec) * 1e-9;
    remaining_s = timeout - elapsed;
    if (remaining_s <= 0) {
      return WAIT_SPUN_OUT;
    }
  }
  const double budget_s = (double)UMWAIT_BUDGET_US * 1e-6;
  const double wait_s = remaining_s < budget_s ? remaining_s : budget_s;
  // umwait's counter deadline is in TSC ticks. Determine the TSC frequency
  // once per call from the invariant-TSC leaf when present; fall back to
  // rdtsc/rdtscp only if the leaf is missing (never on WAITPKG parts, which
  // are all post-Skylake invariant-TSC).
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0x15, &eax, &ebx, &ecx, &edx) != 1 || eax == 0 || ebx == 0 ||
      ecx == 0) {
    return WAIT_OK;  // no invariant TSC ratio: degrade to the next iteration
  }
  const double tsc_hz = (double)ecx * (double)ebx / (double)eax;
  if (!(tsc_hz > 0.0)) {
    return WAIT_OK;
  }
  unsigned int lo, hi;
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  const uint64_t now_tsc = ((uint64_t)hi << 32) | lo;
  const uint64_t deadline_tsc = now_tsc + (uint64_t)(wait_s * tsc_hz);
  const uint64_t counter =
      deadline_tsc & 0xFFFFFFFFFFFFFFFFULL;  // umwait reads EDX:EAX directly
  // umonitor arms the wakeup range; the callback re-check in the outer loop
  // closes the arm/store race, mirroring the monitorx pattern. umwait
  // ecx=0 permits C0.1 (deeper wake latency, lower power). EFLAGS.ZF is set
  // when the wake came from the counter deadline rather than the monitor;
  // either way the loop re-checks the predicate, so ZF is ignored.
  const unsigned int d_lo = (unsigned int)(counter & 0xFFFFFFFFULL);
  const unsigned int d_hi = (unsigned int)(counter >> 32);
  // x86-64 SysV: rdi/rsi are scratch inputs; rax is umonitor's address; the
  // 64-bit deadline is staged in rsi:rdi and moved to rdx:rax before umwait.
  // Encoded via .byte (SDM: umonitor rax = f3 0f ae f6; umwait ecx =
  // f2 0f ae f1) because some otherwise-fine toolchains predate the WAITPKG
  // mnemonics; baseline x86-64 assemblers accept the raw bytes. The
  // instructions only execute on CPUID-gated WAITPKG parts.
  __asm__ volatile(
      "movq %%rdi, %%rax\n"             // rax = address for umonitor
      ".byte 0xf3, 0x0f, 0xae, 0xf0\n"  // umonitor rax
      "movl %1, %%eax\n"                // eax = counter low 32
      "movl %2, %%edx\n"                // edx = counter high 32
      "xorl %%ecx, %%ecx\n"             // control = 0 (C0.1 permitted)
      ".byte 0xf2, 0x0f, 0xae, 0xf1\n"  // umwait ecx
      :
      : "D"(addr), "r"(d_lo), "r"(d_hi)
      : "rax", "rcx", "rdx", "memory");
  return WAIT_OK;
}
#endif

static PyObject* method_spinloop(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  Py_buffer buffer;
  PyObject* callback;
  double timeout = 0.;

  spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(self);
  if (state == NULL) {
    PyErr_SetString(PyExc_TypeError, "Failed to retrieve module state!");
    return NULL;
  }

  static const char* keywords[] = {"buffer", "callback", "timeout", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*O|d", (char**)keywords,
                                   &buffer, &callback, &timeout)) {
    return NULL;
  }

  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "callback parameter must be callable!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  struct timespec t_start;
  if (clock_gettime(TIMEOUT_CLOCK, &t_start) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  bool result = false;
  bool error = false;
  bool clock_error = false;  // deferred: PyErr raised after GIL reacquire
  bool timed_out = false;    // WFET exhausted the overall timeout
#if defined(__aarch64__)
  enum bounded_wait_status wfet_status = WAIT_OK;
#endif
#if defined(__i386__) || defined(__x86_64__)
  enum bounded_wait_status umwait_status = WAIT_OK;
#endif
  bool have_timeout = (timeout > 1e-9);
  unsigned int iteration = 0;
#if defined(__i386__) || defined(__x86_64__)
  const bool buffer_qualifies = (buffer.len <= state->max_monitor_line_size);
#endif

  while (true) {
    PyObject* res = PyObject_CallNoArgs(callback);
    if (res == NULL) {
      error = true;
      break;
    }
    int ok = (res == Py_True);
    Py_DECREF(res);

    if (ok) {
      result = true;
      break;
    }

    // Check timeout at most every 16 iterations to avoid clock_gettime and
    // comparison cost
    if (have_timeout && (iteration & 15u) == 0) {
      struct timespec t_now;
      if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
        error = true;
        break;
      }

      const double elapsed = (double)(t_now.tv_sec - t_start.tv_sec) +
                             (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
      if (elapsed >= timeout) {
        result = false;
        break;
      }
    }
    ++iteration;

#if defined(__i386__) || defined(__x86_64__)
    // monitorx + mwaitx with qualified buffer
    if (buffer_qualifies && state->cpu_support == CPU_SUPPORT_MONITORX) {
      _mm_monitorx(buffer.buf, 0, 0);

      // Check once more in case the buffer has been modified while we were
      // arming the monitor hardware
      res = PyObject_CallNoArgs(callback);
      if (res == NULL) {
        error = true;
        break;
      }
      ok = (res == Py_True);
      Py_DECREF(res);

      if (ok) {
        result = true;
        break;
      }

      // Run mwaitx with enabled timeout (bit 1). The actual timeout value
      // is not very important, we just want to ensure we don't lock up
      // here for too long.
      Py_BEGIN_ALLOW_THREADS _mm_mwaitx((1 << 1), 0,
                                        MWAITX_DEFAULT_TIMEOUT_CYCLES);
      Py_END_ALLOW_THREADS
    }

    // Fallback: Busy poll
    else {
#endif
      // Give other threads a chance to be scheduled
      Py_BEGIN_ALLOW_THREADS
      // clang-format off: preprocessor-guarded arch branches confuse the
// formatter's brace/indent tracking; keep hand-aligned.
#if defined(__i386__) || defined(__x86_64__)
      if (state->cpu_support == CPU_SUPPORT_WAITPKG) {
        umwait_status = umwait_bounded_wait(buffer.buf, timeout, &t_start);
        if (umwait_status == WAIT_CLOCK_ERR) {
          clock_error = true;
          error = true;
        } else if (umwait_status == WAIT_SPUN_OUT) {
          timed_out = true;
        }
        // WAIT_OK: continue the outer loop; the callback re-check decides.
      } else {
        __builtin_ia32_pause();
      }
#elif defined(__aarch64__)
      if (state->cpu_support == CPU_SUPPORT_WFET) {
        wfet_status = wfet_bounded_wait(timeout, &t_start);
        if (wfet_status == WAIT_CLOCK_ERR) {
          clock_error = true;
          error = true;
        } else if (wfet_status == WAIT_SPUN_OUT) {
          timed_out = true;
        }
        // WAIT_OK: continue the outer loop; the callback re-check decides.
      } else {
        __asm__ volatile("yield" ::: "memory");
      }
#endif
      Py_END_ALLOW_THREADS
// clang-format on
#if defined(__i386__) || defined(__x86_64__)
    }
#endif

    if (error || timed_out) {
      break;
    }
  }

  PyBuffer_Release(&buffer);

  if (clock_error) {
    PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
    return NULL;
  }

  if (error) {
    return NULL;
  }

  if (result) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyMethodDef spinloop_methods[] = {
    {"spinloop", (PyCFunction)(void (*)(void))method_spinloop,
     METH_VARARGS | METH_KEYWORDS, "Wait for store with callback"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef spinloop_module = {
    PyModuleDef_HEAD_INIT,
    "spinloop",
    "Hardware-optimized spinloops for Python",
    sizeof(spinloop_state_t),
    spinloop_methods,
    NULL, /* m_slots */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL, /* m_free */
};

PyMODINIT_FUNC PyInit_spinloop(void) {
  PyObject* m = PyModule_Create(&spinloop_module);
  if (m != NULL) {
    spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(m);
    if (state != NULL) {
      determine_cpu_support(state);
    }
  }
  return m;
}

}  // extern "C"
