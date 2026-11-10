# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot INT4 all-reduce over the copy (SDMA) engines for two ranks (TP2), so
that dual batch overlap (DBO) can hide it.

Why: QuickReduce and RCCL move the data with CU kernels. Under DBO those kernels
run next to the other microbatch's GEMMs and both slow down: on MI355X, a
QuickReduce of an 8k-token microbatch next to a GEMM burst on another stream
hides only 1-9% of the shorter of the two. A peer copy on the SDMA engines needs
no CUs and hides 89-100% under the same GEMMs.

How, per call k (slot k % 2, everything on the caller's stream, no host
sync):

1. quantize x into send[slot]: INT4, one fp16 scale per 32 values (4.5 bits
   per value);
2. wait until the peer has consumed our call k - 2 (own counter
   ``consumed`` >= k - 1, hipStreamWaitValue64);
3. copy send[slot] into the peer's recv[slot] (hipMemcpyAsync with
   hipMemcpyDeviceToDeviceNoCU: the SDMA engines);
4. peer.arrived = k + 1 (hipStreamWriteValue64 into the peer's memory);
5. wait until own ``arrived`` >= k + 1: the peer's call k is in recv[slot];
6. out = dq(rank 0's message) + dq(rank 1's message) in fp32, then the input
   dtype. Both ranks add the same two exact fp32 values (a 4-bit integer times
   an fp16 scale is exact in fp32), so the outputs are bitwise identical on the
   two ranks;
7. peer.consumed = k + 1.

Link bytes per direction: 0.5625 B per value, the same as a two-shot all-reduce
at 4.5 bits, with one quantization instead of two. CU work: one quantize and
one dequantize-add pass, both memory-bound.

Shared buffers are uncached (hipDeviceMallocUncached, like the custom
all-reduce buffers on ROCm), so the peer's SDMA writes are never hidden behind
stale L2 lines. Each rank allocates its receive slots and counters on its own
GPU and maps the peer's through IPC; the push is a copy into IPC-mapped memory.

Invariants: every rank must call all_reduce in the same order with the same
sizes (true for TP collectives), or the surviving rank's stream waits forever
(hipStreamWaitValue64 has no timeout). Calls run on the current stream. Stream
write/wait operations cannot be captured into a graph, so callers must not use
this inside CUDA graph capture.

ROCm only. The constructor probes the HIP calls this module needs (the NoCU copy
kind and the stream write/wait operations) and raises at startup if the
runtime or driver lacks them.
"""

import ctypes
import time

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

GROUP = 32
_ALIGN = 256
_HIP_D2D_NOCU = 1024  # hipMemcpyDeviceToDeviceNoCU
_HIP_MALLOC_UNCACHED = 0x3  # hipDeviceMallocUncached
_HIP_IPC_LAZY_PEER = 1  # hipIpcMemLazyEnablePeerAccess
_WAIT_GTE = 0x0  # hipStreamWaitValueGte
_SLOTS = 2
_PROBE_TIMEOUT_S = 10.0
# Launch configs (groups per program, warps), measured on MI355X: within 1-4% of
# the best per size for 2k-16k tokens of hidden size 6144.
_Q_LAUNCH = (128, 1)
_DQ_LAUNCH = (128, 2)


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


@triton.jit
def _quant_int4_kernel(x_ptr, q_ptr, s_ptr, n_groups, BLOCK_G: tl.constexpr):
    g = tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = g < n_groups
    x = tl.load(
        x_ptr + g[:, None] * 32 + tl.arange(0, 32)[None, :], mask=gm[:, None], other=0.0
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    s = tl.minimum(amax / 7.0, 65504.0).to(tl.float16)
    sf = s.to(tl.float32)
    inv = tl.where(sf > 0.0, 1.0 / sf, 0.0)
    q = (
        tl.minimum(tl.maximum(tl.floor(x * inv[:, None] + 0.5), -7.0), 7.0).to(tl.int32)
        + 8
    )
    lo, hi = tl.split(tl.reshape(q, (BLOCK_G, 16, 2)))
    tl.store(
        q_ptr + g[:, None] * 16 + tl.arange(0, 16)[None, :],
        (lo | (hi << 4)).to(tl.uint8),
        mask=gm[:, None],
    )
    tl.store(s_ptr + g, s, mask=gm)


@triton.jit
def _dq_sum_kernel(
    qa_ptr, sa_ptr, qb_ptr, sb_ptr, out_ptr, n_groups, BLOCK_G: tl.constexpr
):
    g = tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = g < n_groups
    qoff = g[:, None] * 16 + tl.arange(0, 16)[None, :]
    pa = tl.load(qa_ptr + qoff, mask=gm[:, None], other=0).to(tl.int32)
    pb = tl.load(qb_ptr + qoff, mask=gm[:, None], other=0).to(tl.int32)
    sa = tl.load(sa_ptr + g, mask=gm, other=0.0).to(tl.float32)[:, None]
    sb = tl.load(sb_ptr + g, mask=gm, other=0.0).to(tl.float32)[:, None]
    lo = ((pa & 15) - 8).to(tl.float32) * sa + ((pb & 15) - 8).to(tl.float32) * sb
    hi = ((pa >> 4) - 8).to(tl.float32) * sa + ((pb >> 4) - 8).to(tl.float32) * sb
    y = tl.reshape(tl.join(lo, hi), (BLOCK_G, 32))
    tl.store(
        out_ptr + g[:, None] * 32 + tl.arange(0, 32)[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=gm[:, None],
    )


class _Ptr:
    """A raw device pointer for a Triton launch (reads .data_ptr() and .dtype)."""

    def __init__(self, ptr: int, dtype: torch.dtype):
        self.ptr, self.dtype = ptr, dtype

    def data_ptr(self) -> int:
        return self.ptr


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


class _Hip:
    def __init__(self):
        L = self.lib = ctypes.CDLL("libamdhip64.so")
        vp, sz, u, u64 = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_uint64,
        )
        sig = {
            "hipExtMallocWithFlags": [ctypes.POINTER(vp), sz, u],
            "hipMemset": [vp, ctypes.c_int, sz],
            "hipDeviceSynchronize": [],
            "hipIpcGetMemHandle": [ctypes.POINTER(_IpcHandle), vp],
            "hipIpcOpenMemHandle": [ctypes.POINTER(vp), _IpcHandle, u],
            "hipIpcCloseMemHandle": [vp],
            "hipFree": [vp],
            "hipMemcpyAsync": [vp, vp, sz, ctypes.c_int, vp],
            "hipStreamWriteValue64": [vp, vp, u64, u],
            "hipStreamWaitValue64": [vp, vp, u64, u, u64],
        }
        for name, args in sig.items():
            f = getattr(L, name)
            f.argtypes, f.restype = args, ctypes.c_int

    def __call__(self, name: str, *args, ok: tuple = ()) -> None:
        err = getattr(self.lib, name)(*args)
        if err != 0 and err not in ok:
            raise RuntimeError(f"{name} failed: hip error {err}")


class DmaAllReduce:
    """All-reduce (sum) of bf16/fp16 tensors between the two ranks of ``group``
    (a CPU/gloo group, for the handle exchange at init). Tensors up to
    ``max_bytes``; ``numel`` must be a multiple of 32. Collective: both ranks
    construct it in the same order."""

    def __init__(self, group, device: torch.device, max_bytes: int):
        assert dist.get_world_size(group) == 2, "DmaAllReduce is for two ranks"
        self.rank = dist.get_rank(group)
        self.device = device
        self.hip = _Hip()
        self.max_elems = max_bytes // 2
        # scales start here in a slot at max size
        self.q_off_s = _align(self.max_elems // 2)
        self.slot_bytes = _align(self.q_off_s + self.max_elems // GROUP * 2)
        self.calls = 0
        with torch.cuda.device(device):
            self.send = self._malloc(_SLOTS * self.slot_bytes)
            # shared region: [arrived u64][consumed u64] ... recv slots from _ALIGN on
            self.shared = self._malloc(_ALIGN + _SLOTS * self.slot_bytes)
            self.hip("hipDeviceSynchronize")
            h = _IpcHandle()
            self.hip(
                "hipIpcGetMemHandle", ctypes.byref(h), ctypes.c_void_p(self.shared)
            )
            handles = [None, None]
            dist.all_gather_object(handles, bytes(h), group=group)
            peer = ctypes.c_void_p()
            self.hip(
                "hipIpcOpenMemHandle",
                ctypes.byref(peer),
                _IpcHandle.from_buffer_copy(handles[1 - self.rank]),
                _HIP_IPC_LAZY_PEER,
            )
            self.peer_shared = peer.value
            error = self._probe()
        # Every rank raises together, so no rank is left waiting for a peer.
        errors = [None, None]
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            self.close()
            raise RuntimeError(
                "VLLM_DBO_TP_DMA_AR needs HIP stream memory operations and "
                "hipMemcpyDeviceToDeviceNoCU, which this ROCm runtime or driver "
                f"does not provide (per rank: {errors}). Unset VLLM_DBO_TP_DMA_AR "
                "or upgrade ROCm."
            )
        dist.barrier(group=group)
        self.arrived, self.consumed = self.shared, self.shared + 8
        self.peer_arrived, self.peer_consumed = self.peer_shared, self.peer_shared + 8
        self.recv, self.peer_recv = self.shared + _ALIGN, self.peer_shared + _ALIGN

    def _probe(self) -> str | None:
        """Exercise, on this rank only, the HIP calls the all-reduce needs and
        compile the Triton kernels for both dtypes, so a missing feature fails
        at startup and the first microbatch does not stall on JIT. Returns an
        error string, or None."""
        try:
            stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            flag = ctypes.c_void_p(self.shared)  # own counters are 0 until used
            src, dst = self.send, self.send + _ALIGN
            self.hip(
                "hipMemcpyAsync",
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                _ALIGN,
                _HIP_D2D_NOCU,
                stream,
            )
            self.hip("hipStreamWriteValue64", stream, flag, 1, 0)
            self.hip(
                "hipStreamWaitValue64", stream, flag, 1, _WAIT_GTE, 0xFFFFFFFFFFFFFFFF
            )
            u8, f16 = torch.uint8, torch.float16
            (qg, qw), (dg, dw) = _Q_LAUNCH, _DQ_LAUNCH
            for dtype in (torch.bfloat16, torch.float16):
                x = torch.zeros(GROUP, dtype=dtype, device=self.device)
                _quant_int4_kernel[(1,)](
                    x,
                    _Ptr(src, u8),
                    _Ptr(src + 16, f16),
                    1,
                    BLOCK_G=qg,
                    num_warps=qw,
                    enable_fp_fusion=False,
                )
                _dq_sum_kernel[(1,)](
                    _Ptr(src, u8),
                    _Ptr(src + 16, f16),
                    _Ptr(src, u8),
                    _Ptr(src + 16, f16),
                    x,
                    1,
                    BLOCK_G=dg,
                    num_warps=dw,
                )
            done = torch.cuda.Event()
            done.record()
            deadline = time.monotonic() + _PROBE_TIMEOUT_S
            while not done.query():
                if time.monotonic() > deadline:
                    return "probe timed out"
                time.sleep(0.001)
            self.hip("hipMemset", ctypes.c_void_p(self.shared), 0, 16)  # counters
            self.hip("hipDeviceSynchronize")
        except Exception as e:  # missing symbol, HIP error, Triton failure
            return f"{type(e).__name__}: {e}"
        return None

    def close(self) -> None:
        """Release the buffers and the IPC mapping. Collective in spirit: call
        only after no all-reduce is in flight."""
        hip = self.hip
        if getattr(self, "peer_shared", None):
            hip("hipIpcCloseMemHandle", ctypes.c_void_p(self.peer_shared), ok=(1,))
            self.peer_shared = None
        for name in ("shared", "send"):
            ptr = getattr(self, name, None)
            if ptr:
                hip("hipFree", ctypes.c_void_p(ptr), ok=(1,))
                setattr(self, name, None)

    def _malloc(self, nbytes: int) -> int:
        p = ctypes.c_void_p()
        self.hip("hipExtMallocWithFlags", ctypes.byref(p), nbytes, _HIP_MALLOC_UNCACHED)
        self.hip("hipMemset", p, 0, nbytes)
        return p.value

    def should_reduce(self, x: torch.Tensor) -> bool:
        return (
            x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous()
            and x.numel() % GROUP == 0
            and 0 < x.numel() <= self.max_elems
        )

    def all_reduce(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert self.should_reduce(x)
        out = torch.empty_like(x) if out is None else out
        n = x.numel()
        ng = n // GROUP
        s_off = _align(n // 2)
        nbytes = s_off + ng * 2
        k = self.calls
        slot = k % _SLOTS
        send = self.send + slot * self.slot_bytes
        recv = self.recv + slot * self.slot_bytes
        peer_recv = self.peer_recv + slot * self.slot_bytes
        stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        u8, f16 = torch.uint8, torch.float16
        (qg, qw), (dg, dw) = _Q_LAUNCH, _DQ_LAUNCH
        # No FMA contraction: floor(x * inv + 0.5) rounds twice, so a value on an
        # exact half-step tie quantizes the same as the reference.
        _quant_int4_kernel[(triton.cdiv(ng, qg),)](
            x,
            _Ptr(send, u8),
            _Ptr(send + s_off, f16),
            ng,
            BLOCK_G=qg,
            num_warps=qw,
            enable_fp_fusion=False,
        )
        if k >= _SLOTS:
            self.hip(
                "hipStreamWaitValue64",
                stream,
                ctypes.c_void_p(self.consumed),
                k - _SLOTS + 1,
                _WAIT_GTE,
                0xFFFFFFFFFFFFFFFF,
            )
        self.hip(
            "hipMemcpyAsync",
            ctypes.c_void_p(peer_recv),
            ctypes.c_void_p(send),
            nbytes,
            _HIP_D2D_NOCU,
            stream,
        )
        self.hip(
            "hipStreamWriteValue64",
            stream,
            ctypes.c_void_p(self.peer_arrived),
            k + 1,
            0,
        )
        self.hip(
            "hipStreamWaitValue64",
            stream,
            ctypes.c_void_p(self.arrived),
            k + 1,
            _WAIT_GTE,
            0xFFFFFFFFFFFFFFFF,
        )
        a, b = (send, recv) if self.rank == 0 else (recv, send)
        _dq_sum_kernel[(triton.cdiv(ng, dg),)](
            _Ptr(a, u8),
            _Ptr(a + s_off, f16),
            _Ptr(b, u8),
            _Ptr(b + s_off, f16),
            out,
            ng,
            BLOCK_G=dg,
            num_warps=dw,
        )
        self.hip(
            "hipStreamWriteValue64",
            stream,
            ctypes.c_void_p(self.peer_consumed),
            k + 1,
            0,
        )
        # Counted only once the whole call is enqueued, so the ranks' counters
        # stay equal if an enqueue raises.
        self.calls = k + 1
        return out
