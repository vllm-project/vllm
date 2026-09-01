// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Device-side-synchronized 2-rank all-reduce kernel for XPU, JIT-compiled by
// xpu_p2p_dev_communicator.py via torch.utils.cpp_extension.
//
// The reduce kernel itself is OpenCL C, compiled AT RUNTIME through the SYCL
// kernel_compiler extension (the driver's built-in compiler does the device
// compilation), so a plain C++ host compiler suffices - no icpx needed, same
// toolchain as the host-sync extension (xpu_zeipc.cpp).
//
// One kernel per collective does everything the CUDA custom all-reduce does
// with Signal/RankSignals: publish the local chunk into the peer-visible
// staging buffer, handshake through per-workgroup flags, reduce.  The host
// only enqueues; it never blocks on the peer.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <string>

#include <sycl/sycl.hpp>

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

sycl::queue* g_q = nullptr;
sycl::kernel* g_k[3] = {nullptr, nullptr, nullptr};

const char* kKernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float8 bf2f8(ushort8 h) { return as_float8(convert_uint8(h) << 16); }
inline float bf2f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort f2bf(float f) {
  uint u = as_uint(f);
  if (isnan(f)) return (ushort)((u >> 16) | 0x0040);
  u += 0x7fff + ((u >> 16) & 1);   /* round to nearest even */
  return (ushort)(u >> 16);
}
inline ushort8 f2bf8(float8 f) {
  uint8 u = as_uint8(f);
  ushort8 r = convert_ushort8((u + 0x7fff + ((u >> 16) & 1)) >> 16);
  ushort8 nanv = convert_ushort8((u >> 16) | (uint8)0x0040);
  return select(r, nanv, convert_ushort8(isnan(f) & 1) << 15);
}

/* One flag (its own 64-byte line) per workgroup, monotonically increasing
   sequence numbers so the flags are never reset.  Both sides of the
   exchange MUST be all_svm_devices-scope atomics: a plain volatile load
   spins on a stale cache line forever, because an inbound PCIe write from
   the peer GPU does not invalidate this device's cache - only the coherent
   access that a system-scope atomic compiles to observes it (measured on
   2x Arc B70: a volatile poll misses a mid-spin arrival even after 100 s,
   an atomic poll sees it immediately). */
inline void handshake(volatile __global atomic_uint* lflags,
                      volatile __global atomic_uint* pflags, uint seq) {
  size_t wg = get_group_id(0);
  if (get_local_id(0) == 0) {
    atomic_store_explicit(&pflags[wg * 16], seq, memory_order_release,
                          memory_scope_all_svm_devices);
    while (atomic_load_explicit(&lflags[wg * 16], memory_order_acquire,
                                memory_scope_all_svm_devices) < seq)
      ;
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Workgroup wg owns elements [wg*chunk, min(n, wg*chunk+chunk)).  Both
   ranks launch the identical grid, so peer workgroup wg publishes exactly
   the chunk this workgroup reads: per-workgroup flags are a sufficient
   handshake and no kernel-wide barrier is needed.  Phase 1 stages the own
   chunk for the peer, phase 2 reduces the local input against the peer's
   staged chunk.  8-wide vectors with a scalar tail; a two-operand add
   rounds once, which is bit-identical to fp32 accumulation. */
#define AR_BODY(REDV, REDS)                                              \
  size_t wg = get_group_id(0);                                           \
  size_t start = wg * chunk;                                             \
  size_t end = min(n, start + chunk);                                    \
  size_t lid = get_local_id(0);                                          \
  size_t ls = get_local_size(0);                                         \
  size_t vend = start + ((end - start) / 8) * 8;                         \
  for (size_t i = start + lid * 8; i < vend; i += ls * 8)                \
    vstore8(vload8(0, input + i), 0, my_stage + i);                      \
  if (lid == 0)                                                          \
    for (size_t i = vend; i < end; i++) my_stage[i] = input[i];          \
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);                              \
  handshake(lflags, pflags, seq);                                        \
  for (size_t i = start + lid * 8; i < vend; i += ls * 8) {              \
    REDV;                                                                \
  }                                                                      \
  if (lid == 0)                                                          \
    for (size_t i = vend; i < end; i++) { REDS; }

__kernel void ar_bf16(__global ushort* dst, __global const ushort* input,
                      __global ushort* my_stage,
                      __global const ushort* peer_stage,
                      volatile __global atomic_uint* lflags,
                      volatile __global atomic_uint* pflags, uint seq,
                      ulong n, ulong chunk) {
  AR_BODY(vstore8(f2bf8(bf2f8(vload8(0, input + i)) +
                        bf2f8(vload8(0, peer_stage + i))),
                  0, dst + i),
          dst[i] = f2bf(bf2f(input[i]) + bf2f(peer_stage[i])))
}

__kernel void ar_f16(__global half* dst, __global const half* input,
                     __global half* my_stage,
                     __global const half* peer_stage,
                     volatile __global atomic_uint* lflags,
                     volatile __global atomic_uint* pflags, uint seq,
                     ulong n, ulong chunk) {
  AR_BODY(vstore_half8_rte(vload_half8(0, input + i) +
                               vload_half8(0, peer_stage + i),
                           0, dst + i),
          dst[i] = (half)((float)input[i] + (float)peer_stage[i]))
}

__kernel void ar_f32(__global float* dst, __global const float* input,
                     __global float* my_stage,
                     __global const float* peer_stage,
                     volatile __global atomic_uint* lflags,
                     volatile __global atomic_uint* pflags, uint seq,
                     ulong n, ulong chunk) {
  AR_BODY(vstore8(vload8(0, input + i) + vload8(0, peer_stage + i), 0,
                  dst + i),
          dst[i] = input[i] + peer_stage[i])
}
)CLC";

// init(queue_ptr) -> None ; compiles the kernels for torch's queue.
PyObject* py_init(PyObject*, PyObject* args) {
  unsigned long long qp;
  if (!PyArg_ParseTuple(args, "K", &qp)) return nullptr;
  g_q = reinterpret_cast<sycl::queue*>(qp);
  try {
    auto src = syclex::create_kernel_bundle_from_source(
        g_q->get_context(), syclex::source_language::opencl,
        std::string(kKernelSource));
    // static: keeps the executable bundle (and its kernels) alive
    static auto kb =
        syclex::build(src, syclex::properties{syclex::build_options{
                               std::vector<std::string>{"-cl-std=CL3.0"}}});
    static sycl::kernel ks[3] = {kb.ext_oneapi_get_kernel("ar_bf16"),
                                 kb.ext_oneapi_get_kernel("ar_f16"),
                                 kb.ext_oneapi_get_kernel("ar_f32")};
    for (int i = 0; i < 3; i++) g_k[i] = &ks[i];
  } catch (const std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "kernel setup failed: %s", e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

// launch(dtype_code, dst, input, my_stage, peer_stage, lflags, pflags,
//        seq, n, chunk, nwg, wgsize) -> None ; enqueue only, never blocks.
PyObject* py_launch(PyObject*, PyObject* args) {
  int dt;
  unsigned long long dst, in, ms, ps, lf, pf, n, chunk;
  unsigned int seq;
  int nwg, wgsize;
  if (!PyArg_ParseTuple(args, "iKKKKKKIKKii", &dt, &dst, &in, &ms, &ps, &lf,
                        &pf, &seq, &n, &chunk, &nwg, &wgsize))
    return nullptr;
  try {
    g_q->submit([&](sycl::handler& h) {
      h.set_arg(0, reinterpret_cast<void*>(dst));
      h.set_arg(1, reinterpret_cast<void*>(in));
      h.set_arg(2, reinterpret_cast<void*>(ms));
      h.set_arg(3, reinterpret_cast<void*>(ps));
      h.set_arg(4, reinterpret_cast<void*>(lf));
      h.set_arg(5, reinterpret_cast<void*>(pf));
      h.set_arg(6, (uint32_t)seq);
      h.set_arg(7, (uint64_t)n);
      h.set_arg(8, (uint64_t)chunk);
      h.parallel_for(sycl::nd_range<1>((size_t)nwg * wgsize, (size_t)wgsize),
                     *g_k[dt]);
    });
  } catch (const std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "launch failed: %s", e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

// wait() -> None ; host-synchronize torch's queue (init/smoke-test only).
PyObject* py_wait(PyObject*, PyObject*) {
  g_q->wait();
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {{"init", py_init, METH_VARARGS, nullptr},
                         {"launch", py_launch, METH_VARARGS, nullptr},
                         {"wait", py_wait, METH_NOARGS, nullptr},
                         {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "vllm_xpu_zeipc_dev", nullptr,
                          -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit_vllm_xpu_zeipc_dev(void) { return PyModule_Create(&mod); }
