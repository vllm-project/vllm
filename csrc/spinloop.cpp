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

#if defined(__x86_64__) || defined(__aarch64__)
  // Longest single armed wait (UMONITOR/UMWAIT, or the exclusive monitor plus
  // WFET), in microseconds. A store to the watched bytes ends it early. The
  // loop then re-runs the callback, which also picks up conditions outside
  // the watched bytes, so this bounds how late those are noticed. Linux caps
  // UMWAIT further through IA32_UMWAIT_CONTROL.
  #define ARMED_WAIT_BUDGET_US 1000
  // Longest single timed wait (TPAUSE, or WFET without the monitor), used
  // when the watched bytes cannot be armed. Only the deadline ends it.
  #define TIMED_WAIT_BUDGET_US 50
#endif

#if defined(__aarch64__)
  // Bit 31 per the arm64 ELF hwcaps ABI (FEAT_WFxT).
  #ifndef HWCAP2_WFXT
    #define HWCAP2_WFXT (1UL << 31)
  #endif
  // An exclusive load is only guaranteed to watch the bytes it loads, so the
  // armed WFET wait watches one naturally aligned 64-bit word.
  #define EXCLUSIVE_WORD_BYTES 8
#endif

// About 330 us at 3 GHz: the longest armed MWAITX wait.
#define MWAITX_DEFAULT_TIMEOUT_CYCLES 1000000
// About 50 us at 3 GHz: the MWAITX counterpart of TIMED_WAIT_BUDGET_US.
#define MWAITX_TIMED_CYCLES 150000

typedef struct {
  unsigned int cpu_support;
  // Size of the range one MONITORX or UMONITOR arm watches; 0 when unknown
  // or when UMONITOR may be disabled, which keeps every wait unarmed (timed).
  unsigned int max_monitor_line_size;
#if defined(__x86_64__)
  // TSC frequency in Hz from the invariant-TSC leaf, cached so the wait path
  // never runs CPUID. WAITPKG is selected only when it is usable.
  double tsc_hz;
#endif
} spinloop_state_t;

static void determine_cpu_support(spinloop_state_t* state) {
  state->cpu_support = CPU_SUPPORT_NONE;
  state->max_monitor_line_size = 0;
#if defined(__x86_64__)
  state->tsc_hz = 0.0;
#endif

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
#endif

#if defined(__x86_64__)
  // WAITPKG (umonitor/umwait/tpause), CPUID.(EAX=7,ECX=0):ECX[5]. x86-64
  // only: the waits read a 64-bit TSC deadline from EDX:EAX. The deadline is
  // in TSC ticks, so WAITPKG also needs a usable invariant-TSC leaf (0x15).
  if (state->cpu_support == CPU_SUPPORT_NONE &&
      __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 1 &&
      (ecx & (1 << 5)) != 0) {
    if (__get_cpuid(0x15, &eax, &ebx, &ecx, &edx) == 1 && eax != 0 &&
        ebx != 0 && ecx != 0) {
      state->tsc_hz = (double)ecx * (double)ebx / (double)eax;
    }
    if (state->tsc_hz > 0.0) {
      state->cpu_support = CPU_SUPPORT_WAITPKG;
    }
  }
#endif

#if defined(__i386__) || defined(__x86_64__)
  // MONITORX and UMONITOR both watch one monitor line, whose size the
  // MONITOR leaf reports.
  bool monitor_works = state->cpu_support == CPU_SUPPORT_MONITORX;
  #if defined(__x86_64__)
  // Microcode may make UMONITOR set up no monitor on some Intel parts
  // (Sapphire Rapids and later Xeons), so a UMWAIT would only end at its
  // deadline. Arm it only where CPUID.(EAX=7,ECX=2):EDX[7] MONITOR_MITG_NO
  // says the processor is unaffected; elsewhere use the timed TPAUSE wait,
  // which is unaffected. Intel guidance 826393.
  if (state->cpu_support == CPU_SUPPORT_WAITPKG &&
      __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 1 && eax >= 2 &&
      __get_cpuid_count(7, 2, &eax, &ebx, &ecx, &edx) == 1 &&
      (edx & (1u << 7)) != 0) {
    monitor_works = true;
  }
  #endif
  if (monitor_works && __get_cpuid(5, &eax, &ebx, &ecx, &edx) == 1) {
    state->max_monitor_line_size = ebx & 0xff;
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

// Whether [buf, buf + len) lies inside one naturally aligned block of `block`
// bytes. One monitor arm, or one exclusive load, watches a single such block:
// a store to bytes outside it does not end the wait.
static bool within_block(const void* buf, Py_ssize_t len, uintptr_t block) {
  if (len <= 0 || block == 0) {
    return false;
  }
  const uintptr_t first = (uintptr_t)buf;
  const uintptr_t last = first + (uintptr_t)len - 1;
  return first / block == last / block;
}

#if defined(__x86_64__) || defined(__aarch64__)
enum wait_window { WINDOW_OPEN, WINDOW_EXPIRED, WINDOW_CLOCK_ERROR };

// Length of the next hardware wait: `budget_s`, cut to what is left of a
// positive `timeout` started at `t_start`.
static enum wait_window next_wait_s(double timeout,
                                    const struct timespec* t_start,
                                    double budget_s, double* wait_s) {
  *wait_s = budget_s;
  if (timeout <= 1e-9) {
    return WINDOW_OPEN;
  }
  struct timespec t_now;
  if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
    return WINDOW_CLOCK_ERROR;
  }
  const double elapsed = (double)(t_now.tv_sec - t_start->tv_sec) +
                         (t_now.tv_nsec - t_start->tv_nsec) * 1e-9;
  const double remaining = timeout - elapsed;
  if (remaining <= 0) {
    return WINDOW_EXPIRED;
  }
  if (remaining < budget_s) {
    *wait_s = remaining;
  }
  return WINDOW_OPEN;
}
#endif

#if defined(__x86_64__)
// The WAITPKG instructions are emitted as bytes because some otherwise-fine
// toolchains predate the mnemonics. They only run on CPUID-gated parts.
// Control 1 in ECX selects C0.1, the shallower and faster-waking state.

static inline void umonitor(const void* addr) {
  // umonitor rax
  __asm__ volatile(".byte 0xf3, 0x0f, 0xae, 0xf0" : : "a"(addr) : "memory");
}

static inline uint64_t tsc_deadline(double wait_s, double tsc_hz) {
  unsigned int lo, hi;
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return (((uint64_t)hi << 32) | lo) + (uint64_t)(wait_s * tsc_hz);
}

static inline void umwait(uint64_t deadline) {
  // umwait ecx
  __asm__ volatile(".byte 0xf2, 0x0f, 0xae, 0xf1"
                   :
                   : "c"(1u), "a"((uint32_t)deadline),
                     "d"((uint32_t)(deadline >> 32))
                   : "cc", "memory");
}

static inline void tpause(uint64_t deadline) {
  // tpause ecx
  __asm__ volatile(".byte 0x66, 0x0f, 0xae, 0xf1"
                   :
                   : "c"(1u), "a"((uint32_t)deadline),
                     "d"((uint32_t)(deadline >> 32))
                   : "cc", "memory");
}
#endif

#if defined(__aarch64__)
static inline uint64_t cntvct_deadline(double wait_s) {
  uint64_t now_ticks, freq_hz;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(now_ticks));
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq_hz));
  uint64_t wait_ticks = (uint64_t)(wait_s * (double)freq_hz);
  if (wait_ticks < 1) {
    wait_ticks = 1;  // rounding must not produce an immediate wake
  }
  return now_ticks + wait_ticks;
}

// wfet x16 (encoding d5031000 | Rd). x16 is an intra-procedure-call scratch
// register that carries no live value here.
static inline void wfet(uint64_t deadline) {
  __asm__ volatile(
      "mov x16, %[deadline]\n"
      ".inst 0xd5031010\n"
      :
      : [deadline] "r"(deadline)
      : "x16", "memory");
}

// Wait until another agent stores to `word`, or until `deadline`, unless
// `word` already differs from `snapshot`. The exclusive load arms this core's
// monitor; a store that clears it signals an event that ends the WFET. SEVL
// then WFE first clears any stale event. Nothing between the exclusive load
// and the WFET stores to memory, and CLREX drops the reservation afterwards.
static inline void armed_wfet(const uint64_t* word, uint64_t snapshot,
                              uint64_t deadline) {
  uint64_t seen;
  __asm__ volatile(
      "sevl\n"
      "wfe\n"
      "ldxr %[seen], [%[word]]\n"
      "cmp %[seen], %[snapshot]\n"
      "b.ne 1f\n"
      "mov x16, %[deadline]\n"
      ".inst 0xd5031010\n"
      "1:\n"
      "clrex\n"
      : [seen] "=&r"(seen)
      : [word] "r"(word), [snapshot] "r"(snapshot), [deadline] "r"(deadline)
      : "x16", "cc", "memory");
}
#endif

// Call `callback` until it returns True or `timeout` seconds pass, waiting in
// between. Where the CPU allows it, the wait is armed on `buffer`, so that a
// store to it ends the wait early; `buffer` must then hold every byte the
// callback reads.
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
  bool have_timeout = (timeout > 1e-9);
  unsigned int iteration = 0;
#if defined(__i386__) || defined(__x86_64__)
  // Whether one MONITORX or UMONITOR arm covers every watched byte.
  const bool monitor_covers =
      within_block(buffer.buf, buffer.len, state->max_monitor_line_size);
#endif
#if defined(__aarch64__)
  // The snapshot and the exclusive load read the whole aligned word around
  // the watched bytes, which may include bytes outside the exported buffer
  // (in shm_broadcast, other readers' flags in the same metadata row). The
  // load cannot fault: an 8-byte-aligned word never crosses a page, and it
  // shares a page with the first watched byte. A store to the extra bytes
  // only ends the wait early, which re-runs the callback.
  const bool wfet_armed =
      state->cpu_support == CPU_SUPPORT_WFET &&
      within_block(buffer.buf, buffer.len, EXCLUSIVE_WORD_BYTES);
  const uint64_t* watched_word =
      (const uint64_t*)((uintptr_t)buffer.buf &
                        ~(uintptr_t)(EXCLUSIVE_WORD_BYTES - 1));
#endif

  while (true) {
#if defined(__aarch64__)
    // Taken before the callback, so a store landing after the callback has
    // read the bytes either changes the word seen by the exclusive load or
    // clears the monitor it arms. The callback's own read flag sits in the
    // same word, so a written flag cycling 1 -> 0 -> 1 still shows.
    uint64_t snapshot = 0;
    if (wfet_armed) {
      snapshot = __atomic_load_n(watched_word, __ATOMIC_RELAXED);
    }
#endif
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

    // Check the timeout before every hardware wait, which may last up to its
    // budget; in a pure pause/yield spin, check every 16 iterations to save
    // the clock_gettime cost.
    const bool waits_in_hardware = state->cpu_support != CPU_SUPPORT_NONE;
    if (have_timeout && (waits_in_hardware || (iteration & 15u) == 0)) {
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

    bool waited = false;
#if defined(__i386__) || defined(__x86_64__)
    // monitorx + mwaitx. When the watched bytes span two monitor lines, a
    // store to the second line does not end the wait, so it becomes a short
    // timed wait that a store to the first line can still end early.
    if (state->cpu_support == CPU_SUPPORT_MONITORX) {
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

      // Run mwaitx with enabled timeout (bit 1). Its timer counts at the
      // P0 frequency, which is unknown here; the cycle counts assume about
      // 3 GHz. A positive timeout is re-checked before every wait, so it is
      // overrun by at most one wait.
      const unsigned int mwaitx_cycles =
          monitor_covers ? MWAITX_DEFAULT_TIMEOUT_CYCLES : MWAITX_TIMED_CYCLES;
      Py_BEGIN_ALLOW_THREADS _mm_mwaitx((1 << 1), 0, mwaitx_cycles);
      Py_END_ALLOW_THREADS waited = true;
    }
#endif
#if defined(__x86_64__)
    if (!waited && state->cpu_support == CPU_SUPPORT_WAITPKG) {
      if (monitor_covers) {
        umonitor(buffer.buf);
        // As with MONITORX: a store that landed before the arm is seen here,
        // one that lands after it ends the umwait.
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
      }
      double wait_s;
      const enum wait_window window = next_wait_s(
          timeout, &t_start,
          (monitor_covers ? ARMED_WAIT_BUDGET_US : TIMED_WAIT_BUDGET_US) * 1e-6,
          &wait_s);
      if (window == WINDOW_CLOCK_ERROR) {
        PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
        error = true;
        break;
      }
      if (window == WINDOW_EXPIRED) {
        break;
      }
      const uint64_t deadline = tsc_deadline(wait_s, state->tsc_hz);
      Py_BEGIN_ALLOW_THREADS if (monitor_covers) { umwait(deadline); }
      else {
        tpause(deadline);
      }
      Py_END_ALLOW_THREADS waited = true;
    }
#endif
#if defined(__aarch64__)
    if (!waited && state->cpu_support == CPU_SUPPORT_WFET) {
      double wait_s;
      const enum wait_window window = next_wait_s(
          timeout, &t_start,
          (wfet_armed ? ARMED_WAIT_BUDGET_US : TIMED_WAIT_BUDGET_US) * 1e-6,
          &wait_s);
      if (window == WINDOW_CLOCK_ERROR) {
        PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
        error = true;
        break;
      }
      if (window == WINDOW_EXPIRED) {
        break;
      }
      const uint64_t deadline = cntvct_deadline(wait_s);
      Py_BEGIN_ALLOW_THREADS if (wfet_armed) {
        armed_wfet(watched_word, snapshot, deadline);
      }
      else {
        wfet(deadline);
      }
      Py_END_ALLOW_THREADS waited = true;
    }
#endif
    if (!waited) {
      // Busy poll, giving other threads a chance to be scheduled.
      Py_BEGIN_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
          __asm__ volatile("yield" :: : "memory");
#endif
      Py_END_ALLOW_THREADS
    }
  }

  PyBuffer_Release(&buffer);

  if (error) {
    return NULL;
  }

  if (result) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

// Full store-to-load fence for the MessageQueue parked-flag handshake: mfence
// or a locked op on x86, dmb ish on Arm. Python's lock-based fences order only
// acquire and release accesses, which does not order a store before a later
// load of another location on weakly ordered CPUs.
static PyObject* method_full_fence(PyObject* self, PyObject* args) {
  (void)self;
  (void)args;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  // Not Py_RETURN_NONE: built against 3.12+ headers it skips the incref that
  // a 3.11 interpreter needs, because only 3.12+ makes None immortal.
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef spinloop_methods[] = {
    {"spinloop", (PyCFunction)(void (*)(void))method_spinloop,
     METH_VARARGS | METH_KEYWORDS, "Wait for store with callback"},
    {"full_fence", method_full_fence, METH_NOARGS,
     "Sequentially consistent memory fence"},
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
