// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// CUDA Checkpoint/Restore extension using cuCheckpointProcess* APIs.
// Requires NVIDIA driver >= 570.
// CUDA-only (no ROCm support - HIP checkpoint APIs not yet available).

#include <dlfcn.h>

#include <iostream>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// CUDA driver types
typedef int CUresult;
typedef void* CUcheckpointHandle;

// cuCheckpointProcess API function pointer types
typedef CUresult (*cuCheckpointProcessSuspend_t)(CUcheckpointHandle* handle);
typedef CUresult (*cuCheckpointProcessResume_t)(CUcheckpointHandle handle);
typedef CUresult (*cuCheckpointProcessGetState_t)(CUcheckpointHandle handle,
                                                  int* state);

// Function pointers (loaded dynamically)
static cuCheckpointProcessSuspend_t p_cuCheckpointProcessSuspend = nullptr;
static cuCheckpointProcessResume_t p_cuCheckpointProcessResume = nullptr;
static cuCheckpointProcessGetState_t p_cuCheckpointProcessGetState = nullptr;

static bool apis_loaded = false;

// Error handling
static char error_msg[4096];

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    CUresult error = condition;                                                \
    if (error != 0) {                                                          \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %d at %s:%d", error, \
               __FILE__, __LINE__);                                            \
      std::cerr << error_msg << std::endl;                                     \
      PyErr_SetString(PyExc_RuntimeError, error_msg);                          \
      return nullptr;                                                          \
    }                                                                          \
  } while (0)

static bool load_checkpoint_apis() {
  if (apis_loaded) return true;

  // Try to load from the already-loaded CUDA driver
  void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
  }
  if (!handle) {
    // Try loading fresh
    handle = dlopen("libcuda.so.1", RTLD_LAZY);
  }
  if (!handle) {
    return false;
  }

  p_cuCheckpointProcessSuspend =
      (cuCheckpointProcessSuspend_t)dlsym(handle, "cuCheckpointProcessSuspend");
  p_cuCheckpointProcessResume =
      (cuCheckpointProcessResume_t)dlsym(handle, "cuCheckpointProcessResume");
  p_cuCheckpointProcessGetState = (cuCheckpointProcessGetState_t)dlsym(
      handle, "cuCheckpointProcessGetState");

  // All three must be available
  if (p_cuCheckpointProcessSuspend && p_cuCheckpointProcessResume &&
      p_cuCheckpointProcessGetState) {
    apis_loaded = true;
    return true;
  }

  p_cuCheckpointProcessSuspend = nullptr;
  p_cuCheckpointProcessResume = nullptr;
  p_cuCheckpointProcessGetState = nullptr;
  return false;
}

// ---------------------------------------------------------------------------
// Python-exposed functions

static PyObject* python_checkpoint_suspend(PyObject* self, PyObject* args) {
  if (!apis_loaded) {
    PyErr_SetString(PyExc_RuntimeError,
                    "CUDA checkpoint APIs not available. "
                    "Requires NVIDIA driver >= 570.");
    return nullptr;
  }

  CUcheckpointHandle handle = nullptr;
  CUDA_CHECK(p_cuCheckpointProcessSuspend(&handle));

  return PyLong_FromUnsignedLongLong(
      reinterpret_cast<unsigned long long>(handle));
}

static PyObject* python_checkpoint_resume(PyObject* self, PyObject* args) {
  if (!apis_loaded) {
    PyErr_SetString(PyExc_RuntimeError,
                    "CUDA checkpoint APIs not available. "
                    "Requires NVIDIA driver >= 570.");
    return nullptr;
  }

  unsigned long long handle_val;
  if (!PyArg_ParseTuple(args, "K", &handle_val)) {
    return nullptr;
  }

  CUcheckpointHandle handle = reinterpret_cast<CUcheckpointHandle>(handle_val);
  CUDA_CHECK(p_cuCheckpointProcessResume(handle));

  Py_RETURN_NONE;
}

static PyObject* python_checkpoint_get_state(PyObject* self, PyObject* args) {
  if (!apis_loaded) {
    PyErr_SetString(PyExc_RuntimeError,
                    "CUDA checkpoint APIs not available. "
                    "Requires NVIDIA driver >= 570.");
    return nullptr;
  }

  unsigned long long handle_val;
  if (!PyArg_ParseTuple(args, "K", &handle_val)) {
    return nullptr;
  }

  CUcheckpointHandle handle = reinterpret_cast<CUcheckpointHandle>(handle_val);
  int state = 0;
  CUDA_CHECK(p_cuCheckpointProcessGetState(handle, &state));

  return PyLong_FromLong(state);
}

static PyObject* python_is_available(PyObject* self, PyObject* args) {
  bool available = load_checkpoint_apis();
  return PyBool_FromLong(available ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Python module definition

static PyMethodDef module_methods[] = {
    {"checkpoint_suspend", (PyCFunction)python_checkpoint_suspend, METH_NOARGS,
     "Suspend the CUDA process and return a checkpoint handle."},
    {"checkpoint_resume", (PyCFunction)python_checkpoint_resume, METH_VARARGS,
     "Resume the CUDA process from a checkpoint handle."},
    {"checkpoint_get_state", (PyCFunction)python_checkpoint_get_state,
     METH_VARARGS, "Get the state of a checkpoint handle."},
    {"is_available", (PyCFunction)python_is_available, METH_NOARGS,
     "Check if CUDA checkpoint APIs are available (driver >= 570)."},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef cuda_checkpoint_module = {
    PyModuleDef_HEAD_INIT, "cuda_checkpoint",
    "CUDA checkpoint/restore for process suspend/resume", -1, module_methods};

PyMODINIT_FUNC PyInit_cuda_checkpoint(void) {
  PyObject* module = PyModule_Create(&cuda_checkpoint_module);
  if (!module) {
    return NULL;
  }

  // Try to load APIs at import time
  load_checkpoint_apis();

  return module;
}

}  // extern "C"
