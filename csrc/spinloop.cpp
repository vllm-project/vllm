#include <Python.h>

extern "C" {

#include <stdbool.h>
#include <time.h>

#if defined(__i386__) || defined(__x86_64__)
  #include <cpuid.h>
  #include <mwaitxintrin.h>
#endif

#if defined(CLOCK_MONOTONIC_RAW)
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC_RAW
#else
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC
#endif

#define CPU_SUPPORT_NONE 0
#define CPU_SUPPORT_MONITORX 1

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
  }
#endif
}

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
  const bool buffer_qualifies = (buffer.len <= state->max_monitor_line_size);

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
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield" :: : "memory");
#endif
      Py_END_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
    }
#endif
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

static PyMethodDef spinloop_methods[] = {
    {"spinloop", (PyCFunction)method_spinloop, METH_VARARGS | METH_KEYWORDS,
     "Wait for store with callback"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef spinloop_module = {
    PyModuleDef_HEAD_INIT, "spinloop",
    "Hardware-optimized spinloops for Python", sizeof(spinloop_state_t),
    spinloop_methods};

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
