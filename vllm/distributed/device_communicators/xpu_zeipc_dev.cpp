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
// only enqueues; it never blocks on the peer.  The all-gather kernel is the
// same protocol without the reduction.
//
// The kernel is graph-capture safe: it is submitted to whatever queue the
// caller passes (the one currently recording, if any), and it carries no
// per-call state from the host.  The sequence number that drives the
// handshake and selects the staging slot is a per-workgroup counter in
// device memory that the kernel itself advances, so a replayed launch keeps
// making progress with arguments frozen at capture time.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <string>

#include <sycl/sycl.hpp>

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

// ar_bf16, ar_f16, ar_f32, ag
sycl::kernel* g_k[4] = {nullptr, nullptr, nullptr, nullptr};

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
   rounds once, which is bit-identical to fp32 accumulation.

   seq comes from a per-workgroup counter (own cache line, like the flags)
   that only this rank's kernels touch, so plain loads and stores suffice;
   launches on an in-order queue serialize, so each launch sees the value
   the previous one left.  Its parity picks one of two staging slots.
   That double buffer needs no release barrier: this rank reuses a slot at
   seq+2 only after its seq+1 handshake completed, which required the
   peer's seq+1 signal, which the peer sends only after its own kernel seq
   (every read of this rank's slot included) finished on its in-order
   queue. */
#define SEQ_PROLOGUE                                                     \
  size_t wg = get_group_id(0);                                           \
  size_t lid = get_local_id(0);                                          \
  __local uint lseq;                                                     \
  if (lid == 0) {                                                        \
    uint s = counters[wg * 16] + 1;                                      \
    counters[wg * 16] = s;                                               \
    lseq = s;                                                            \
  }                                                                      \
  work_group_barrier(CLK_LOCAL_MEM_FENCE);                               \
  uint seq = lseq;                                                       \
  my_stage += (seq & 1) * slot;                                          \
  peer_stage += (seq & 1) * slot;                                        \
  size_t start = wg * chunk;                                             \
  size_t end = min(n, start + chunk);                                    \
  size_t ls = get_local_size(0);

#define AR_BODY(REDV, REDS)                                              \
  SEQ_PROLOGUE                                                           \
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
                      volatile __global atomic_uint* pflags,
                      __global uint* counters, ulong n, ulong chunk,
                      ulong slot) {
  AR_BODY(vstore8(f2bf8(bf2f8(vload8(0, input + i)) +
                        bf2f8(vload8(0, peer_stage + i))),
                  0, dst + i),
          dst[i] = f2bf(bf2f(input[i]) + bf2f(peer_stage[i])))
}

__kernel void ar_f16(__global half* dst, __global const half* input,
                     __global half* my_stage,
                     __global const half* peer_stage,
                     volatile __global atomic_uint* lflags,
                     volatile __global atomic_uint* pflags,
                     __global uint* counters, ulong n, ulong chunk,
                     ulong slot) {
  AR_BODY(vstore_half8_rte(vload_half8(0, input + i) +
                               vload_half8(0, peer_stage + i),
                           0, dst + i),
          dst[i] = (half)((float)input[i] + (float)peer_stage[i]))
}

__kernel void ar_f32(__global float* dst, __global const float* input,
                     __global float* my_stage,
                     __global const float* peer_stage,
                     volatile __global atomic_uint* lflags,
                     volatile __global atomic_uint* pflags,
                     __global uint* counters, ulong n, ulong chunk,
                     ulong slot) {
  AR_BODY(vstore8(vload8(0, input + i) + vload8(0, peer_stage + i), 0,
                  dst + i),
          dst[i] = input[i] + peer_stage[i])
}

/* 2-rank all-gather along dim 0 of a contiguous input: pure placement, so
   it works on bytes and serves every dtype.  Phase 1 stages the local
   chunk for the peer and copies it into this rank's slice of dst; phase 2
   copies the peer's staged chunk into the peer's slice.  Same
   per-workgroup handshake and double buffer as the all-reduce, on its own
   slots, flags and counters. */
__kernel void ag(__global uchar* dst, __global const uchar* input,
                 __global uchar* my_stage, __global const uchar* peer_stage,
                 volatile __global atomic_uint* lflags,
                 volatile __global atomic_uint* pflags,
                 __global uint* counters, ulong n, ulong chunk, ulong slot,
                 ulong my_off, ulong peer_off) {
  SEQ_PROLOGUE
  size_t vend = start + ((end - start) / 16) * 16;
  __global uchar* mine = dst + my_off;
  __global uchar* theirs = dst + peer_off;
  for (size_t i = start + lid * 16; i < vend; i += ls * 16) {
    uchar16 v = vload16(0, input + i);
    vstore16(v, 0, my_stage + i);
    vstore16(v, 0, mine + i);
  }
  if (lid == 0)
    for (size_t i = vend; i < end; i++) {
      uchar v = input[i];
      my_stage[i] = v;
      mine[i] = v;
    }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  handshake(lflags, pflags, seq);
  for (size_t i = start + lid * 16; i < vend; i += ls * 16)
    vstore16(vload16(0, peer_stage + i), 0, theirs + i);
  if (lid == 0)
    for (size_t i = vend; i < end; i++) theirs[i] = peer_stage[i];
}
)CLC";

// init(queue_ptr) -> None ; compiles the kernels for the queue's context.
// queue_ptr is torch.xpu.Stream.sycl_queue (the sycl::queue* as an int).
PyObject* py_init(PyObject*, PyObject* args) {
  unsigned long long qp;
  if (!PyArg_ParseTuple(args, "K", &qp)) return nullptr;
  auto* q = reinterpret_cast<sycl::queue*>(qp);
  try {
    auto src = syclex::create_kernel_bundle_from_source(
        q->get_context(), syclex::source_language::opencl,
        std::string(kKernelSource));
    // static: keeps the executable bundle (and its kernels) alive
    static auto kb =
        syclex::build(src, syclex::properties{syclex::build_options{
                               std::vector<std::string>{"-cl-std=CL3.0"}}});
    static sycl::kernel ks[4] = {
        kb.ext_oneapi_get_kernel("ar_bf16"), kb.ext_oneapi_get_kernel("ar_f16"),
        kb.ext_oneapi_get_kernel("ar_f32"), kb.ext_oneapi_get_kernel("ag")};
    for (int i = 0; i < 4; i++) g_k[i] = &ks[i];
  } catch (const std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "kernel setup failed: %s", e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

// launch(queue_ptr, dtype_code, dst, input, my_stage, peer_stage, lflags,
//        pflags, counters, n, chunk, slot, nwg, wgsize) -> None
// Enqueue only, never blocks.  queue_ptr must be the caller's current
// stream on every call: during graph capture that is the recording queue,
// and a launch on any other queue would run eagerly and be missing from
// the replay.  my_stage/peer_stage are slot-0 base pointers; slot is the
// element stride to slot 1.
PyObject* py_launch(PyObject*, PyObject* args) {
  unsigned long long qp;
  int dt;
  unsigned long long dst, in, ms, ps, lf, pf, ctr, n, chunk, slot;
  int nwg, wgsize;
  if (!PyArg_ParseTuple(args, "KiKKKKKKKKKKii", &qp, &dt, &dst, &in, &ms, &ps,
                        &lf, &pf, &ctr, &n, &chunk, &slot, &nwg, &wgsize))
    return nullptr;
  try {
    reinterpret_cast<sycl::queue*>(qp)->submit([&](sycl::handler& h) {
      h.set_arg(0, reinterpret_cast<void*>(dst));
      h.set_arg(1, reinterpret_cast<void*>(in));
      h.set_arg(2, reinterpret_cast<void*>(ms));
      h.set_arg(3, reinterpret_cast<void*>(ps));
      h.set_arg(4, reinterpret_cast<void*>(lf));
      h.set_arg(5, reinterpret_cast<void*>(pf));
      h.set_arg(6, reinterpret_cast<void*>(ctr));
      h.set_arg(7, (uint64_t)n);
      h.set_arg(8, (uint64_t)chunk);
      h.set_arg(9, (uint64_t)slot);
      h.parallel_for(sycl::nd_range<1>((size_t)nwg * wgsize, (size_t)wgsize),
                     *g_k[dt]);
    });
  } catch (const std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "launch failed: %s", e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

// launch_ag(queue_ptr, dst, input, my_stage, peer_stage, lflags, pflags,
//           counters, n, chunk, slot, my_off, peer_off, nwg, wgsize) -> None
// Byte-granular all-gather launch; same queue and staging conventions as
// launch().  my_off/peer_off are the byte offsets of the two rank slices
// in dst.
PyObject* py_launch_ag(PyObject*, PyObject* args) {
  unsigned long long qp, dst, in, ms, ps, lf, pf, ctr, n, chunk, slot, moff,
      poff;
  int nwg, wgsize;
  if (!PyArg_ParseTuple(args, "KKKKKKKKKKKKKii", &qp, &dst, &in, &ms, &ps, &lf,
                        &pf, &ctr, &n, &chunk, &slot, &moff, &poff, &nwg,
                        &wgsize))
    return nullptr;
  try {
    reinterpret_cast<sycl::queue*>(qp)->submit([&](sycl::handler& h) {
      h.set_arg(0, reinterpret_cast<void*>(dst));
      h.set_arg(1, reinterpret_cast<void*>(in));
      h.set_arg(2, reinterpret_cast<void*>(ms));
      h.set_arg(3, reinterpret_cast<void*>(ps));
      h.set_arg(4, reinterpret_cast<void*>(lf));
      h.set_arg(5, reinterpret_cast<void*>(pf));
      h.set_arg(6, reinterpret_cast<void*>(ctr));
      h.set_arg(7, (uint64_t)n);
      h.set_arg(8, (uint64_t)chunk);
      h.set_arg(9, (uint64_t)slot);
      h.set_arg(10, (uint64_t)moff);
      h.set_arg(11, (uint64_t)poff);
      h.parallel_for(sycl::nd_range<1>((size_t)nwg * wgsize, (size_t)wgsize),
                     *g_k[3]);
    });
  } catch (const std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "launch_ag failed: %s", e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {{"init", py_init, METH_VARARGS, nullptr},
                         {"launch", py_launch, METH_VARARGS, nullptr},
                         {"launch_ag", py_launch_ag, METH_VARARGS, nullptr},
                         {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "vllm_xpu_zeipc_dev", nullptr,
                          -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit_vllm_xpu_zeipc_dev(void) { return PyModule_Create(&mod); }
