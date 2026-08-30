// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// CUDA Checkpoint/Restore extension using cuCheckpointProcess* APIs.
// Requires NVIDIA driver >= 570 (CUDA 12.8).
// CUDA-only (no ROCm support - HIP checkpoint APIs not yet available).
//
// The CUDA Checkpointing driver API is PID-based and has no "suspend"/
// "resume" entry points. Checkpointing a process is a two-step sequence
// (lock then checkpoint); restoring is the inverse (restore then unlock):
//
//   RUNNING --Lock--> LOCKED --Checkpoint--> CHECKPOINTED
//   CHECKPOINTED --Restore--> LOCKED --Unlock--> RUNNING
//
// See:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html

#include <dlfcn.h>

#include <iostream>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// CUDA driver types. CUresult and CUprocessState are plain ints in the ABI.
typedef int CUresult;
typedef int CUprocessState;

// cuCheckpointProcess API function pointer types. All operations are keyed
// by process id. The trailing "args" struct is optional on every call and
// may be passed as NULL, so we bind it as an opaque pointer and always pass
// nullptr (default behavior: no lock timeout, default checkpoint/restore).
typedef CUresult (*cuCheckpointProcessLock_t)(int pid, void* args);
typedef CUresult (*cuCheckpointProcessCheckpoint_t)(int pid, void* args);
typedef CUresult (*cuCheckpointProcessRestore_t)(int pid, void* args);
typedef CUresult (*cuCheckpointProcessUnlock_t)(int pid, void* args);
typedef CUresult (*cuCheckpointProcessGetState_t)(int pid,
                                                  CUprocessState* state);

// Function pointers (loaded dynamically).
static cuCheckpointProcessLock_t p_cuCheckpointProcessLock = nullptr;
static cuCheckpointProcessCheckpoint_t p_cuCheckpointProcessCheckpoint =
    nullptr;
static cuCheckpointProcessRestore_t p_cuCheckpointProcessRestore = nullptr;
static cuCheckpointProcessUnlock_t p_cuCheckpointProcessUnlock = nullptr;
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
    if (!handle) {
      handle = dlopen("libcuda.so", RTLD_LAZY);
    }
  }
  if (!handle) {
    return false;
  }

  p_cuCheckpointProcessLock =
      (cuCheckpointProcessLock_t)dlsym(handle, "cuCheckpointProcessLock");
  p_cuCheckpointProcessCheckpoint = (cuCheckpointProcessCheckpoint_t)dlsym(
      handle, "cuCheckpointProcessCheckpoint");
  p_cuCheckpointProcessRestore =
      (cuCheckpointProcessRestore_t)dlsym(handle, "cuCheckpointProcessRestore");
  p_cuCheckpointProcessUnlock =
      (cuCheckpointProcessUnlock_t)dlsym(handle, "cuCheckpointProcessUnlock");
  p_cuCheckpointProcessGetState = (cuCheckpointProcessGetState_t)dlsym(
      handle, "cuCheckpointProcessGetState");

  // All entry points must be available (added together in driver >= 570).
  if (p_cuCheckpointProcessLock && p_cuCheckpointProcessCheckpoint &&
      p_cuCheckpointProcessRestore && p_cuCheckpointProcessUnlock &&
      p_cuCheckpointProcessGetState) {
    apis_loaded = true;
    return true;
  }

  p_cuCheckpointProcessLock = nullptr;
  p_cuCheckpointProcessCheckpoint = nullptr;
  p_cuCheckpointProcessRestore = nullptr;
  p_cuCheckpointProcessUnlock = nullptr;
  p_cuCheckpointProcessGetState = nullptr;
  return false;
}

static bool ensure_available() {
  if (!apis_loaded) {
    PyErr_SetString(PyExc_RuntimeError,
                    "CUDA checkpoint APIs not available. "
                    "Requires NVIDIA driver >= 570.");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Python-exposed functions. Each maps 1:1 onto a driver entry point; the
// lock/checkpoint and restore/unlock sequencing is orchestrated in Python.

static PyObject* python_process_lock(PyObject* self, PyObject* args) {
  if (!ensure_available()) return nullptr;

  int pid;
  if (!PyArg_ParseTuple(args, "i", &pid)) {
    return nullptr;
  }

  CUDA_CHECK(p_cuCheckpointProcessLock(pid, nullptr));
  Py_RETURN_NONE;
}

static PyObject* python_process_checkpoint(PyObject* self, PyObject* args) {
  if (!ensure_available()) return nullptr;

  int pid;
  if (!PyArg_ParseTuple(args, "i", &pid)) {
    return nullptr;
  }

  CUDA_CHECK(p_cuCheckpointProcessCheckpoint(pid, nullptr));
  Py_RETURN_NONE;
}

static PyObject* python_process_restore(PyObject* self, PyObject* args) {
  if (!ensure_available()) return nullptr;

  int pid;
  if (!PyArg_ParseTuple(args, "i", &pid)) {
    return nullptr;
  }

  CUDA_CHECK(p_cuCheckpointProcessRestore(pid, nullptr));
  Py_RETURN_NONE;
}

static PyObject* python_process_unlock(PyObject* self, PyObject* args) {
  if (!ensure_available()) return nullptr;

  int pid;
  if (!PyArg_ParseTuple(args, "i", &pid)) {
    return nullptr;
  }

  CUDA_CHECK(p_cuCheckpointProcessUnlock(pid, nullptr));
  Py_RETURN_NONE;
}

static PyObject* python_get_state(PyObject* self, PyObject* args) {
  if (!ensure_available()) return nullptr;

  int pid;
  if (!PyArg_ParseTuple(args, "i", &pid)) {
    return nullptr;
  }

  CUprocessState state = 0;
  CUDA_CHECK(p_cuCheckpointProcessGetState(pid, &state));

  return PyLong_FromLong(state);
}

static PyObject* python_is_available(PyObject* self, PyObject* args) {
  bool available = load_checkpoint_apis();
  return PyBool_FromLong(available ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Python module definition

static PyMethodDef module_methods[] = {
    {"process_lock", (PyCFunction)python_process_lock, METH_VARARGS,
     "Lock the CUDA process (RUNNING -> LOCKED). Takes a pid."},
    {"process_checkpoint", (PyCFunction)python_process_checkpoint,
     METH_VARARGS,
     "Checkpoint a locked CUDA process (LOCKED -> CHECKPOINTED). Takes a "
     "pid."},
    {"process_restore", (PyCFunction)python_process_restore, METH_VARARGS,
     "Restore a checkpointed CUDA process (CHECKPOINTED -> LOCKED). Takes a "
     "pid."},
    {"process_unlock", (PyCFunction)python_process_unlock, METH_VARARGS,
     "Unlock the CUDA process (LOCKED -> RUNNING). Takes a pid."},
    {"get_state", (PyCFunction)python_get_state, METH_VARARGS,
     "Get the CUprocessState of a process. Takes a pid, returns an int."},
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
