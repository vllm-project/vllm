# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot INT4 all-reduce over the copy (SDMA) engines for two ranks (TP2), so
that dual batch overlap (DBO) can hide it.

Why: QuickReduce and RCCL move the data with CU kernels. Under DBO those kernels
run next to the other microbatch's GEMMs and both slow down: on MI355X, a
QuickReduce of an 8k-token microbatch next to a GEMM burst on another stream
hides only 1-9% of the shorter of the two. A peer copy on the SDMA engines needs
no CUs and hides 89-100% under the same GEMMs.

How, per call k (slot k % slots, everything on the caller's stream, no host
sync):

1. quantize x into send[slot]: INT4, one fp16 scale per 32 values (4.5 bits
   per value);
2. wait until the peer has consumed our call k - slots (own counter
   ``consumed`` >= k - slots + 1, hipStreamWaitValue64);
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
stale L2 lines. Every rank must call all_reduce in the same order with the same
sizes (true for TP collectives); calls run on the current stream. Stream
write/wait operations cannot be captured into a graph, so callers must not use
this inside CUDA graph capture.

Buffer layouts (``layout``):

- ``ipc``: each rank allocates its receive slots and counters on its own GPU
  and maps the peer's through IPC; the push is a copy into IPC-mapped memory.
- ``peer``: each rank allocates its outbox (slots and its counters) on the
  peer's GPU and the peer maps it through IPC, so the push is a true
  peer-to-peer copy and every wait polls memory in the waiting rank's own HBM.
  Same speed as ``ipc`` on MI355X.
"""

import ctypes

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

GROUP = 32
_ALIGN = 256
_HIP_D2D_NOCU = 1024  # hipMemcpyDeviceToDeviceNoCU
_HIP_D2D = 3  # hipMemcpyDeviceToDevice
_HIP_MALLOC_UNCACHED = 0x3  # hipDeviceMallocUncached
_HIP_IPC_LAZY_PEER = 1  # hipIpcMemLazyEnablePeerAccess
_WAIT_GTE = 0x0  # hipStreamWaitValueGte
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
            "hipMemcpyAsync": [vp, vp, sz, ctypes.c_int, vp],
            "hipStreamWriteValue64": [vp, vp, u64, u],
            "hipStreamWaitValue64": [vp, vp, u64, u, u64],
            "hipSetDevice": [ctypes.c_int],
            "hipDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
            "hipDeviceGetByPCIBusId": [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p],
            "hipDeviceEnablePeerAccess": [ctypes.c_int, u],
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
    ``max_bytes``; ``numel`` must be a multiple of 32."""

    def __init__(
        self,
        group,
        device: torch.device,
        max_bytes: int = 256 << 20,
        slots: int = 2,
        copy_kind: int = _HIP_D2D_NOCU,
        layout: str = "ipc",
    ):
        assert dist.get_world_size(group) == 2, "DmaAllReduce is for two ranks"
        assert layout in ("ipc", "peer"), layout
        self.rank = dist.get_rank(group)
        self.device = device
        self.hip = _Hip()
        self.slots = slots
        self.copy_kind = copy_kind
        self.layout = layout
        self.max_elems = max_bytes // 2
        self.q_off_s = _align(
            self.max_elems // 2
        )  # scales start here in a slot at max size
        self.slot_bytes = _align(self.q_off_s + self.max_elems // GROUP * 2)
        self.calls = 0
        if layout == "peer":
            self._init_peer(group)
            return
        with torch.cuda.device(device):
            self.send = self._malloc(self.slots * self.slot_bytes)
            # shared region: [arrived u64][consumed u64] ... recv slots from _ALIGN on
            self.shared_bytes = _ALIGN + self.slots * self.slot_bytes
            self.shared = self._malloc(self.shared_bytes)
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
        dist.barrier(group=group)
        self.arrived, self.consumed = self.shared, self.shared + 8
        self.peer_arrived, self.peer_consumed = self.peer_shared, self.peer_shared + 8
        self.recv, self.peer_recv = self.shared + _ALIGN, self.peer_shared + _ALIGN

    def _init_peer(self, group) -> None:
        """Layout "peer": outbox = [arrived u64][consumed u64], then the slots from
        _ALIGN, allocated on the peer's GPU."""
        hip = self.hip
        with torch.cuda.device(self.device):
            me = torch.cuda.current_device()
            bus = ctypes.create_string_buffer(64)
            hip("hipDeviceGetPCIBusId", bus, 64, me)
            buses = [None, None]
            dist.all_gather_object(buses, bus.value, group=group)
            peer = ctypes.c_int()
            hip("hipDeviceGetByPCIBusId", ctypes.byref(peer), buses[1 - self.rank])
            self.peer_device = peer.value
            hip(
                "hipDeviceEnablePeerAccess", self.peer_device, 0, ok=(704,)
            )  # 704: already enabled
            self.send = self._malloc(self.slots * self.slot_bytes)
            self.shared_bytes = _ALIGN + self.slots * self.slot_bytes
            h = _IpcHandle()
            hip("hipSetDevice", self.peer_device)
            try:
                self.box = self._malloc(self.shared_bytes)
                hip("hipDeviceSynchronize")
                hip("hipIpcGetMemHandle", ctypes.byref(h), ctypes.c_void_p(self.box))
            finally:
                hip("hipSetDevice", me)
            torch.cuda.set_device(me)
            hip("hipDeviceSynchronize")
            handles = [None, None]
            dist.all_gather_object(handles, bytes(h), group=group)
            inbox = ctypes.c_void_p()
            hip(
                "hipIpcOpenMemHandle",
                ctypes.byref(inbox),
                _IpcHandle.from_buffer_copy(handles[1 - self.rank]),
                _HIP_IPC_LAZY_PEER,
            )
            self.inbox = inbox.value
        dist.barrier(group=group)
        # Waits poll the peer's outbox, which lives in our HBM; signals and data
        # go into our outbox on the peer's GPU.
        self.arrived, self.consumed = self.inbox, self.inbox + 8
        self.peer_arrived, self.peer_consumed = self.box, self.box + 8
        self.recv, self.peer_recv = self.inbox + _ALIGN, self.box + _ALIGN

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
        k, self.calls = self.calls, self.calls + 1
        slot = k % self.slots
        send = self.send + slot * self.slot_bytes
        recv = self.recv + slot * self.slot_bytes
        peer_recv = self.peer_recv + slot * self.slot_bytes
        stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        u8, f16 = torch.uint8, torch.float16
        (qg, qw), (dg, dw) = _Q_LAUNCH, _DQ_LAUNCH
        _quant_int4_kernel[(triton.cdiv(ng, qg),)](
            x, _Ptr(send, u8), _Ptr(send + s_off, f16), ng, BLOCK_G=qg, num_warps=qw
        )
        if k >= self.slots:
            self.hip(
                "hipStreamWaitValue64",
                stream,
                ctypes.c_void_p(self.consumed),
                k - self.slots + 1,
                _WAIT_GTE,
                0xFFFFFFFFFFFFFFFF,
            )
        self.hip(
            "hipMemcpyAsync",
            ctypes.c_void_p(peer_recv),
            ctypes.c_void_p(send),
            nbytes,
            self.copy_kind,
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
        return out


def reference_int4_one_shot(xs: list[torch.Tensor]) -> torch.Tensor:
    """What all_reduce computes, in torch (for tests): the sum over ranks of
    dq(q(x_r)) in fp32, then x's dtype."""
    acc = None
    for x in xs:
        g = x.float().reshape(-1, GROUP)
        s = (
            torch.clamp(g.abs().amax(dim=1) / 7.0, max=65504.0)
            .to(torch.float16)
            .float()
        )
        inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
        q = torch.clamp(torch.floor(g * inv[:, None] + 0.5), -7, 7)
        d = q * s[:, None]
        acc = d if acc is None else acc + d
    return acc.reshape(xs[0].shape).to(xs[0].dtype)
