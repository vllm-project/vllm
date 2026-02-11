// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// UCC Allgather extension for async CPU-side collectives over InfiniBand.
// This extension provides non-blocking allgather operations that can overlap
// with GPU computation for DP rank synchronization.

#include <cstring>
#include <iostream>
#include <vector>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <ucc/api/ucc.h>

// Error message buffer
static char error_msg[4096];

// Context structure to hold UCC library and context handles
struct UCCContext {
  ucc_lib_h lib;
  ucc_context_h context;
  ucc_team_h team;
  int rank;
  int world_size;
  bool team_created;
};

// Handle structure for non-blocking operations
struct UCCHandle {
  ucc_coll_req_h request;
  bool completed;
};

// ---------------------------------------------------------------------------
// Helper macros

#define UCC_CHECK(cmd)                                                         \
  do {                                                                         \
    ucc_status_t status = (cmd);                                               \
    if (status != UCC_OK) {                                                    \
      snprintf(error_msg, sizeof(error_msg), "UCC error: %s at %s:%d",         \
               ucc_status_string(status), __FILE__, __LINE__);                 \
      PyErr_SetString(PyExc_RuntimeError, error_msg);                          \
      return nullptr;                                                          \
    }                                                                          \
  } while (0)

#define UCC_CHECK_GOTO(cmd, label)                                             \
  do {                                                                         \
    ucc_status_t status = (cmd);                                               \
    if (status != UCC_OK) {                                                    \
      snprintf(error_msg, sizeof(error_msg), "UCC error: %s at %s:%d",         \
               ucc_status_string(status), __FILE__, __LINE__);                 \
      PyErr_SetString(PyExc_RuntimeError, error_msg);                          \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// OOB (Out-of-Band) allgather callback structure
// This is used during team creation to exchange addresses between ranks

struct OOBInfo {
  PyObject *oob_allgather_fn;
  int rank;
  int world_size;
};

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **request) {
  OOBInfo *info = (OOBInfo *)coll_info;

  // Acquire GIL for Python callback
  PyGILState_STATE gstate = PyGILState_Ensure();

  // Create bytes object from send buffer
  PyObject *send_bytes = PyBytes_FromStringAndSize((const char *)sbuf, msglen);
  if (!send_bytes) {
    PyGILState_Release(gstate);
    return UCC_ERR_NO_MESSAGE;
  }

  // Call Python OOB allgather function
  PyObject *result =
      PyObject_CallFunctionObjArgs(info->oob_allgather_fn, send_bytes, nullptr);
  Py_DECREF(send_bytes);

  if (!result) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return UCC_ERR_NO_MESSAGE;
  }

  // Result should be a list of bytes objects
  if (!PyList_Check(result)) {
    PyErr_SetString(PyExc_TypeError,
                    "OOB allgather must return a list of bytes");
    Py_DECREF(result);
    PyGILState_Release(gstate);
    return UCC_ERR_NO_MESSAGE;
  }

  Py_ssize_t list_size = PyList_Size(result);
  if (list_size != info->world_size) {
    PyErr_SetString(PyExc_ValueError, "OOB allgather returned wrong size list");
    Py_DECREF(result);
    PyGILState_Release(gstate);
    return UCC_ERR_NO_MESSAGE;
  }

  // Copy received data to receive buffer
  char *rbuf_ptr = (char *)rbuf;
  for (Py_ssize_t i = 0; i < list_size; i++) {
    PyObject *item = PyList_GetItem(result, i);
    if (!PyBytes_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "OOB allgather list items must be bytes");
      Py_DECREF(result);
      PyGILState_Release(gstate);
      return UCC_ERR_NO_MESSAGE;
    }
    Py_ssize_t item_size;
    char *item_data;
    PyBytes_AsStringAndSize(item, &item_data, &item_size);
    if ((size_t)item_size != msglen) {
      PyErr_SetString(PyExc_ValueError, "OOB allgather item size mismatch");
      Py_DECREF(result);
      PyGILState_Release(gstate);
      return UCC_ERR_NO_MESSAGE;
    }
    memcpy(rbuf_ptr + i * msglen, item_data, msglen);
  }

  Py_DECREF(result);
  PyGILState_Release(gstate);

  // OOB is synchronous, so we return a non-null "completed" request
  *request = (void *)1;
  return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *request) {
  // Our OOB is synchronous, always completed
  return UCC_OK;
}

static ucc_status_t oob_allgather_free(void *request) {
  // Nothing to free for synchronous OOB
  return UCC_OK;
}

// ---------------------------------------------------------------------------
// Python-exposed functions

// py_ucc_init(rank, world_size) -> ctx_ptr
static PyObject *py_ucc_init(PyObject *self, PyObject *args) {
  int rank, world_size;

  if (!PyArg_ParseTuple(args, "ii", &rank, &world_size)) {
    return nullptr;
  }

  UCCContext *ctx = new UCCContext();
  ctx->rank = rank;
  ctx->world_size = world_size;
  ctx->team_created = false;
  ctx->lib = nullptr;
  ctx->context = nullptr;
  ctx->team = nullptr;

  // Initialize UCC library
  ucc_lib_config_h lib_config;
  ucc_status_t status;

  status = ucc_lib_config_read(nullptr, nullptr, &lib_config);
  if (status != UCC_OK) {
    delete ctx;
    snprintf(error_msg, sizeof(error_msg),
             "Failed to read UCC lib config: %s", ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  ucc_lib_params_t lib_params = {};
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;

  status = ucc_init(&lib_params, lib_config, &ctx->lib);
  ucc_lib_config_release(lib_config);

  if (status != UCC_OK) {
    delete ctx;
    snprintf(error_msg, sizeof(error_msg), "Failed to init UCC: %s",
             ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  // Create UCC context
  ucc_context_config_h ctx_config;
  status = ucc_context_config_read(ctx->lib, nullptr, &ctx_config);
  if (status != UCC_OK) {
    ucc_finalize(ctx->lib);
    delete ctx;
    snprintf(error_msg, sizeof(error_msg),
             "Failed to read UCC context config: %s", ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  ucc_context_params_t ctx_params = {};
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE;
  ctx_params.type = UCC_CONTEXT_SHARED;

  status = ucc_context_create(ctx->lib, &ctx_params, ctx_config, &ctx->context);
  ucc_context_config_release(ctx_config);

  if (status != UCC_OK) {
    ucc_finalize(ctx->lib);
    delete ctx;
    snprintf(error_msg, sizeof(error_msg), "Failed to create UCC context: %s",
             ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  return PyLong_FromUnsignedLongLong((unsigned long long)ctx);
}

// py_ucc_create_team(ctx_ptr, oob_allgather_fn) -> None
static PyObject *py_ucc_create_team(PyObject *self, PyObject *args) {
  unsigned long long ctx_ptr;
  PyObject *oob_allgather_fn;

  if (!PyArg_ParseTuple(args, "KO", &ctx_ptr, &oob_allgather_fn)) {
    return nullptr;
  }

  if (!PyCallable_Check(oob_allgather_fn)) {
    PyErr_SetString(PyExc_TypeError, "oob_allgather_fn must be callable");
    return nullptr;
  }

  UCCContext *ctx = (UCCContext *)ctx_ptr;

  // Set up OOB info for team creation
  OOBInfo oob_info;
  oob_info.oob_allgather_fn = oob_allgather_fn;
  oob_info.rank = ctx->rank;
  oob_info.world_size = ctx->world_size;

  // Create team parameters
  ucc_team_params_t team_params = {};
  team_params.mask =
      UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = &oob_info;
  team_params.oob.n_oob_eps = ctx->world_size;
  team_params.ep = ctx->rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;

  ucc_status_t status;

  // Release GIL during team creation (except for OOB callbacks)
  // Note: OOB callbacks will re-acquire GIL as needed
  Py_BEGIN_ALLOW_THREADS;
  status = ucc_team_create_post(&ctx->context, 1, &team_params, &ctx->team);
  Py_END_ALLOW_THREADS;

  if (status != UCC_OK) {
    snprintf(error_msg, sizeof(error_msg), "Failed to post team creation: %s",
             ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  // Wait for team creation to complete
  while (true) {
    Py_BEGIN_ALLOW_THREADS;
    status = ucc_team_create_test(ctx->team);
    Py_END_ALLOW_THREADS;

    if (status == UCC_OK) {
      break;
    } else if (status == UCC_INPROGRESS) {
      // Progress the context
      Py_BEGIN_ALLOW_THREADS;
      ucc_context_progress(ctx->context);
      Py_END_ALLOW_THREADS;
    } else {
      snprintf(error_msg, sizeof(error_msg), "Team creation failed: %s",
               ucc_status_string(status));
      PyErr_SetString(PyExc_RuntimeError, error_msg);
      return nullptr;
    }
  }

  ctx->team_created = true;
  Py_RETURN_NONE;
}

// py_ucc_allgather_async(ctx_ptr, send_buf, recv_buf) -> handle_ptr
static PyObject *py_ucc_allgather_async(PyObject *self, PyObject *args) {
  unsigned long long ctx_ptr;
  Py_buffer send_buf, recv_buf;

  if (!PyArg_ParseTuple(args, "Ky*y*", &ctx_ptr, &send_buf, &recv_buf)) {
    return nullptr;
  }

  UCCContext *ctx = (UCCContext *)ctx_ptr;

  if (!ctx->team_created) {
    PyBuffer_Release(&send_buf);
    PyBuffer_Release(&recv_buf);
    PyErr_SetString(PyExc_RuntimeError, "UCC team not created");
    return nullptr;
  }

  UCCHandle *handle = new UCCHandle();
  handle->completed = false;
  handle->request = nullptr;

  // Set up allgather collective arguments
  ucc_coll_args_t coll_args = {};
  coll_args.mask = 0;
  coll_args.coll_type = UCC_COLL_TYPE_ALLGATHER;
  coll_args.src.info.buffer = send_buf.buf;
  coll_args.src.info.count = send_buf.len;
  coll_args.src.info.datatype = UCC_DT_UINT8;
  coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
  coll_args.dst.info.buffer = recv_buf.buf;
  coll_args.dst.info.count = recv_buf.len;
  coll_args.dst.info.datatype = UCC_DT_UINT8;
  coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

  ucc_status_t status;

  // Release GIL during UCC calls
  Py_BEGIN_ALLOW_THREADS;
  status = ucc_collective_init(&coll_args, &handle->request, ctx->team);
  Py_END_ALLOW_THREADS;

  PyBuffer_Release(&send_buf);
  PyBuffer_Release(&recv_buf);

  if (status != UCC_OK) {
    delete handle;
    snprintf(error_msg, sizeof(error_msg), "Failed to init collective: %s",
             ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_BEGIN_ALLOW_THREADS;
  status = ucc_collective_post(handle->request);
  Py_END_ALLOW_THREADS;

  if (status != UCC_OK) {
    ucc_collective_finalize(handle->request);
    delete handle;
    snprintf(error_msg, sizeof(error_msg), "Failed to post collective: %s",
             ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  return PyLong_FromUnsignedLongLong((unsigned long long)handle);
}

// py_ucc_test(handle_ptr) -> bool (True if completed)
static PyObject *py_ucc_test(PyObject *self, PyObject *args) {
  unsigned long long handle_ptr;

  if (!PyArg_ParseTuple(args, "K", &handle_ptr)) {
    return nullptr;
  }

  UCCHandle *handle = (UCCHandle *)handle_ptr;

  if (handle->completed) {
    Py_RETURN_TRUE;
  }

  ucc_status_t status;
  Py_BEGIN_ALLOW_THREADS;
  status = ucc_collective_test(handle->request);
  Py_END_ALLOW_THREADS;

  if (status == UCC_OK) {
    handle->completed = true;
    // Finalize the request
    Py_BEGIN_ALLOW_THREADS;
    ucc_collective_finalize(handle->request);
    Py_END_ALLOW_THREADS;
    handle->request = nullptr;
    Py_RETURN_TRUE;
  } else if (status == UCC_INPROGRESS) {
    Py_RETURN_FALSE;
  } else {
    snprintf(error_msg, sizeof(error_msg), "Collective test failed: %s",
             ucc_status_string(status));
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }
}

// py_ucc_wait(handle_ptr) -> None
static PyObject *py_ucc_wait(PyObject *self, PyObject *args) {
  unsigned long long handle_ptr;

  if (!PyArg_ParseTuple(args, "K", &handle_ptr)) {
    return nullptr;
  }

  UCCHandle *handle = (UCCHandle *)handle_ptr;

  if (handle->completed) {
    Py_RETURN_NONE;
  }

  ucc_status_t status;

  // Wait for completion
  while (true) {
    Py_BEGIN_ALLOW_THREADS;
    status = ucc_collective_test(handle->request);
    Py_END_ALLOW_THREADS;

    if (status == UCC_OK) {
      break;
    } else if (status == UCC_INPROGRESS) {
      // Keep waiting - could add a small sleep here if needed
      continue;
    } else {
      snprintf(error_msg, sizeof(error_msg), "Collective wait failed: %s",
               ucc_status_string(status));
      PyErr_SetString(PyExc_RuntimeError, error_msg);
      return nullptr;
    }
  }

  handle->completed = true;

  // Finalize the request
  Py_BEGIN_ALLOW_THREADS;
  ucc_collective_finalize(handle->request);
  Py_END_ALLOW_THREADS;
  handle->request = nullptr;

  Py_RETURN_NONE;
}

// py_ucc_progress(ctx_ptr) -> None
static PyObject *py_ucc_progress(PyObject *self, PyObject *args) {
  unsigned long long ctx_ptr;

  if (!PyArg_ParseTuple(args, "K", &ctx_ptr)) {
    return nullptr;
  }

  UCCContext *ctx = (UCCContext *)ctx_ptr;

  Py_BEGIN_ALLOW_THREADS;
  ucc_context_progress(ctx->context);
  Py_END_ALLOW_THREADS;

  Py_RETURN_NONE;
}

// py_ucc_free_handle(handle_ptr) -> None
static PyObject *py_ucc_free_handle(PyObject *self, PyObject *args) {
  unsigned long long handle_ptr;

  if (!PyArg_ParseTuple(args, "K", &handle_ptr)) {
    return nullptr;
  }

  UCCHandle *handle = (UCCHandle *)handle_ptr;

  if (handle->request && !handle->completed) {
    // Cancel and finalize if not completed
    ucc_collective_finalize(handle->request);
  }

  delete handle;
  Py_RETURN_NONE;
}

// py_ucc_destroy(ctx_ptr) -> None
static PyObject *py_ucc_destroy(PyObject *self, PyObject *args) {
  unsigned long long ctx_ptr;

  if (!PyArg_ParseTuple(args, "K", &ctx_ptr)) {
    return nullptr;
  }

  UCCContext *ctx = (UCCContext *)ctx_ptr;

  if (ctx->team_created && ctx->team) {
    ucc_team_destroy(ctx->team);
  }

  if (ctx->context) {
    ucc_context_destroy(ctx->context);
  }

  if (ctx->lib) {
    ucc_finalize(ctx->lib);
  }

  delete ctx;
  Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// Module definition

static PyMethodDef module_methods[] = {
    {"init", (PyCFunction)py_ucc_init, METH_VARARGS,
     "Initialize UCC context. Returns context pointer."},
    {"create_team", (PyCFunction)py_ucc_create_team, METH_VARARGS,
     "Create UCC team using OOB allgather for bootstrap."},
    {"allgather_async", (PyCFunction)py_ucc_allgather_async, METH_VARARGS,
     "Start async allgather. Returns handle pointer."},
    {"test", (PyCFunction)py_ucc_test, METH_VARARGS,
     "Test if collective is complete. Returns True/False."},
    {"wait", (PyCFunction)py_ucc_wait, METH_VARARGS,
     "Wait for collective to complete."},
    {"progress", (PyCFunction)py_ucc_progress, METH_VARARGS,
     "Progress the UCC context."},
    {"free_handle", (PyCFunction)py_ucc_free_handle, METH_VARARGS,
     "Free a collective handle."},
    {"destroy", (PyCFunction)py_ucc_destroy, METH_VARARGS,
     "Destroy UCC context and free resources."},
    {nullptr, nullptr, 0, nullptr}  // sentinel
};

static struct PyModuleDef ucc_allgather_module = {
    PyModuleDef_HEAD_INIT,
    "ucc_allgather",
    "UCC-based async allgather for CPU-side collectives over InfiniBand",
    -1,
    module_methods};

PyMODINIT_FUNC PyInit_ucc_allgather(void) {
  PyObject *module = PyModule_Create(&ucc_allgather_module);
  if (!module) {
    return nullptr;
  }
  return module;
}

}  // extern "C"
