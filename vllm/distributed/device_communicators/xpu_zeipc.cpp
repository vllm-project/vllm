// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Minimal Level Zero IPC interop for torch XPU tensors, JIT-compiled by
// xpu_p2p_communicator.py via torch.utils.cpp_extension. Host-only code:
// no device kernels, so a plain C++ compiler suffices (the SYCL headers
// are used only for get_native<> interop with torch's queue).
//
// The Level Zero context/device are taken from the sycl::queue behind
// torch.xpu.current_stream(); memory opened here is therefore valid in
// torch's own context, and copies enqueued on that (in-order) queue are
// ordered against torch ops with no extra synchronization.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstring>

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

namespace {

sycl::queue* g_q = nullptr;
ze_context_handle_t g_ctx = nullptr;
ze_device_handle_t g_dev = nullptr;

PyObject* zerr(const char* what, ze_result_t r) {
  PyErr_Format(PyExc_RuntimeError, "%s failed: 0x%x", what, (unsigned)r);
  return nullptr;
}

// init(queue_ptr) -> None
// queue_ptr is torch.xpu.Stream.sycl_queue (the sycl::queue* as an int).
PyObject* py_init(PyObject*, PyObject* args) {
  unsigned long long qp;
  if (!PyArg_ParseTuple(args, "K", &qp)) return nullptr;
  g_q = reinterpret_cast<sycl::queue*>(qp);
  g_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      g_q->get_context());
  g_dev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(g_q->get_device());
  Py_RETURN_NONE;
}

// export_buf(ptr) -> (handle_bytes, fd, offset)
// The IPC handle covers the whole allocation containing ptr; offset locates
// ptr within it. On Linux the handle wraps a dma-buf fd (its first 8 bytes),
// which is only valid in this process: the caller must pass it to the peer
// via SCM_RIGHTS, not as raw bytes.
PyObject* py_export(PyObject*, PyObject* args) {
  unsigned long long ptr;
  if (!PyArg_ParseTuple(args, "K", &ptr)) return nullptr;
  void* base = nullptr;
  size_t sz = 0;
  ze_result_t r = zeMemGetAddressRange(g_ctx, (void*)ptr, &base, &sz);
  if (r != ZE_RESULT_SUCCESS) return zerr("zeMemGetAddressRange", r);
  ze_ipc_mem_handle_t h;
  std::memset(&h, 0, sizeof(h));
  r = zeMemGetIpcHandle(g_ctx, base, &h);
  if (r != ZE_RESULT_SUCCESS) return zerr("zeMemGetIpcHandle", r);
  uint64_t fd;
  std::memcpy(&fd, h.data, 8);
  return Py_BuildValue("(y#Kk)", h.data, (Py_ssize_t)sizeof(h.data), fd,
                       (unsigned long)((char*)ptr - (char*)base));
}

// open_buf(handle_bytes, fd, offset) -> ptr
// fd is the dma-buf fd as received via SCM_RIGHTS; it replaces the sender's
// fd inside the handle before opening.
PyObject* py_open(PyObject*, PyObject* args) {
  const char* hb;
  Py_ssize_t hlen;
  unsigned long long fd;
  unsigned long off;
  if (!PyArg_ParseTuple(args, "y#Kk", &hb, &hlen, &fd, &off)) return nullptr;
  ze_ipc_mem_handle_t h;
  std::memset(&h, 0, sizeof(h));
  std::memcpy(h.data, hb,
              (size_t)hlen < sizeof(h.data) ? (size_t)hlen : sizeof(h.data));
  uint64_t fd64 = fd;
  std::memcpy(h.data, &fd64, 8);
  void* base = nullptr;
  ze_result_t r = zeMemOpenIpcHandle(g_ctx, g_dev, h, 0, &base);
  if (r != ZE_RESULT_SUCCESS) return zerr("zeMemOpenIpcHandle", r);
  return PyLong_FromUnsignedLongLong((unsigned long long)((char*)base + off));
}

// close_buf(base_ptr) -> None ; base_ptr must be the value open_buf returned
// minus the offset passed to it (i.e. offset 0 openings can pass it as is).
PyObject* py_close(PyObject*, PyObject* args) {
  unsigned long long ptr;
  if (!PyArg_ParseTuple(args, "K", &ptr)) return nullptr;
  ze_result_t r = zeMemCloseIpcHandle(g_ctx, (void*)ptr);
  if (r != ZE_RESULT_SUCCESS) return zerr("zeMemCloseIpcHandle", r);
  Py_RETURN_NONE;
}

// copy(dst_ptr, src_ptr, nbytes) -> None ; enqueued on torch's queue
PyObject* py_copy(PyObject*, PyObject* args) {
  unsigned long long dst, src, n;
  if (!PyArg_ParseTuple(args, "KKK", &dst, &src, &n)) return nullptr;
  g_q->memcpy((void*)dst, (void*)src, (size_t)n);
  Py_RETURN_NONE;
}

// wait() -> None ; host-synchronize torch's queue. Much cheaper than
// torch.xpu.synchronize(), which measurably added ~19us per call.
PyObject* py_wait(PyObject*, PyObject*) {
  g_q->wait();
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {{"init", py_init, METH_VARARGS, nullptr},
                         {"export_buf", py_export, METH_VARARGS, nullptr},
                         {"open_buf", py_open, METH_VARARGS, nullptr},
                         {"close_buf", py_close, METH_VARARGS, nullptr},
                         {"copy", py_copy, METH_VARARGS, nullptr},
                         {"wait", py_wait, METH_NOARGS, nullptr},
                         {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "vllm_xpu_zeipc", nullptr, -1,
                          methods};

}  // namespace

PyMODINIT_FUNC PyInit_vllm_xpu_zeipc(void) { return PyModule_Create(&mod); }
