# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP stage hand-off on SDMA copy engines (ROCm, both stages on one node).

The receiving stage owns a ring of slots shared with the sender over CUDA IPC. A
sender thread waits for the step's event, copies the tensors into the step's slot
on SDMA engines (no CUs) and bumps a ``ready`` counter in host shared memory; the
receiver bumps ``freed`` once the step that read a slot has finished on the GPU.

The copies go through ROCr's ``hsa_amd_memory_async_copy_on_engine`` with
explicit engines: on HIP 7.16 a HIP peer copy runs as a blit kernel on the
sender's CUs (``hipMemcpyFlagExtPreferCE`` does not apply to device-to-device
batches), and ``hipMemcpyDeviceToDeviceNoCU`` picks engines whose peer copy
fails on MI355X.
"""

import ctypes
import mmap
import os
import queue
import threading
import time
import uuid
from collections import deque
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

NUM_SLOTS = 3
ALIGN = 256
# Metadata sends the receiver has not yet confirmed (via ``freed``) before
# ``send`` blocks on the oldest. In steady state at most ~NUM_SLOTS + the
# scheduler's batch queue are in flight, so this only bounds a stalled peer.
_MAX_PENDING_META = 64

_HSA_DEVICE_TYPE_GPU = 1
_HSA_AGENT_INFO_DEVICE = 17
_HSA_AMD_AGENT_INFO_BDFID = 0xA006
_HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1
_HSA_SIGNAL_CONDITION_LT = 2
_HSA_WAIT_STATE_ACTIVE = 1
# Two even engines fill one xGMI direction; odd engines hang on MI355X.
_ENGINES = (1 << 2, 1 << 4)


class _Agent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Signal(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Hsa:
    """The few ROCr calls of an SDMA peer copy, over ctypes."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("libhsa-runtime64.so.1")
        lib.hsa_init.restype = ctypes.c_int
        self._cb_type = ctypes.CFUNCTYPE(ctypes.c_int, _Agent, ctypes.c_void_p)
        lib.hsa_iterate_agents.argtypes = [self._cb_type, ctypes.c_void_p]
        lib.hsa_agent_get_info.argtypes = [_Agent, ctypes.c_int, ctypes.c_void_p]
        lib.hsa_amd_signal_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_Signal),
        ]
        lib.hsa_signal_wait_scacquire.restype = ctypes.c_int64
        lib.hsa_signal_wait_scacquire.argtypes = [
            _Signal,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.hsa_signal_store_relaxed.argtypes = [_Signal, ctypes.c_int64]
        lib.hsa_amd_memory_async_copy_on_engine.argtypes = [
            ctypes.c_void_p,
            _Agent,
            ctypes.c_void_p,
            _Agent,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
            _Signal,
            ctypes.c_uint32,
            ctypes.c_bool,
        ]
        if lib.hsa_init() != 0:
            raise RuntimeError("hsa_init failed")
        self.lib = lib

    def gpu_agent(self, device: torch.device) -> _Agent:
        p = torch.cuda.get_device_properties(device)
        want = (p.pci_bus_id << 8) | (p.pci_device_id << 3)
        found: list[_Agent] = []

        def visit(agent, _):
            kind = ctypes.c_uint32()
            self.lib.hsa_agent_get_info(
                agent, _HSA_AGENT_INFO_DEVICE, ctypes.byref(kind)
            )
            if kind.value == _HSA_DEVICE_TYPE_GPU:
                bdf = ctypes.c_uint32()
                self.lib.hsa_agent_get_info(
                    agent, _HSA_AMD_AGENT_INFO_BDFID, ctypes.byref(bdf)
                )
                if bdf.value & ~0x7 == want:
                    found.append(_Agent(agent.handle))
            return 0

        self.lib.hsa_iterate_agents(self._cb_type(visit), None)
        if not found:
            raise RuntimeError(f"no HSA agent for {device}")
        return found[0]

    def signal(self) -> _Signal:
        s = _Signal()
        if self.lib.hsa_amd_signal_create(
            1, 0, None, _HSA_AMD_SIGNAL_AMD_GPU_ONLY, ctypes.byref(s)
        ):
            raise RuntimeError("hsa_amd_signal_create failed")
        return s

    def copy(self, dst, dst_agent, src, src_agent, size, signal, engine) -> None:
        self.lib.hsa_signal_store_relaxed(signal, 1)
        if self.lib.hsa_amd_memory_async_copy_on_engine(
            dst, dst_agent, src, src_agent, size, 0, None, signal, engine, True
        ):
            raise RuntimeError("hsa_amd_memory_async_copy_on_engine failed")

    def wait(self, signal) -> None:
        while (
            self.lib.hsa_signal_wait_scacquire(
                signal,
                _HSA_SIGNAL_CONDITION_LT,
                1,
                1_000_000_000,
                _HSA_WAIT_STATE_ACTIVE,
            )
            > 0
        ):
            pass


class _Counters:
    """Two monotonic uint64 counters in host shared memory: ready (sender) and
    freed (receiver)."""

    def __init__(self, name: str, create: bool) -> None:
        path = f"/dev/shm/{name}"
        if create:
            with open(path, "wb") as f:
                f.write(b"\0" * 16)
        with open(path, "r+b") as f:
            self._mm = mmap.mmap(f.fileno(), 16)
        self.ready = ctypes.c_uint64.from_buffer(self._mm, 0)
        self.freed = ctypes.c_uint64.from_buffer(self._mm, 8)
        self.path = path

    @staticmethod
    def wait(counter: ctypes.c_uint64, at_least: int) -> None:
        spins = 0
        while counter.value < at_least:
            spins += 1
            time.sleep(0 if spins < 1000 else 20e-6)


def _layout(tensors: list[tuple[str, torch.Tensor]]) -> tuple[list[int], int]:
    offsets, end = [], 0
    for _, t in tensors:
        offsets.append(end)
        end += (t.numel() * t.element_size() + ALIGN - 1) // ALIGN * ALIGN
    return offsets, end


class PPSdmaSender:
    """The sending side of one stage boundary (this rank -> dst)."""

    def __init__(self, dst: int, max_num_tokens: int) -> None:
        self.dst = dst
        self.max_num_tokens = max_num_tokens
        self.seq = 0
        self.slots: list[torch.Tensor] | None = None
        # (seq, isend_object handle) of the metadata sends, oldest first.
        self._pending: deque[tuple[int, Any]] = deque()
        self._jobs: queue.Queue = queue.Queue()
        self._error: BaseException | None = None

    def _setup(self, tensors, num_tokens: int) -> dict:
        per_token = sum(t.numel() * t.element_size() for _, t in tensors) / max(
            1, num_tokens
        )
        slot_bytes = int(per_token * self.max_num_tokens) + ALIGN * (len(tensors) + 1)
        name = f"vllm_pp_sdma_{uuid.uuid4().hex}"
        self.counters = _Counters(name, create=True)
        self.hsa = _Hsa()
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        return dict(slot_bytes=slot_bytes, num_slots=NUM_SLOTS, counters=name)

    def _open_slots(self, shared) -> None:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        self.slots = [rebuild_cuda_tensor(*args) for args in shared]
        peer = self.slots[0].device
        self.src_agent = self.hsa.gpu_agent(self.device)
        self.dst_agent = self.hsa.gpu_agent(peer)
        self.signals = [self.hsa.signal() for _ in _ENGINES]
        threading.Thread(target=self._loop, daemon=True, name="pp-sdma-send").start()
        logger.info(
            "PP hand-off on SDMA: %d slots of %.0f MiB in the next stage, engines %s",
            len(self.slots),
            self.slots[0].numel() / 2**20,
            [e.bit_length() - 1 for e in _ENGINES],
        )

    def send(self, tensor_dict: dict[str, Any], pp_group) -> None:
        if self._error is not None:
            raise RuntimeError("PP SDMA hand-off failed") from self._error
        tensors = [
            (k, v.contiguous())
            for k, v in tensor_dict.items()
            if isinstance(v, torch.Tensor)
        ]
        others = {
            k: v for k, v in tensor_dict.items() if not isinstance(v, torch.Tensor)
        }
        num_tokens = max((t.shape[0] for _, t in tensors if t.dim()), default=0)
        setup = None
        if self.slots is None:
            setup = self._setup(tensors, num_tokens)
        self.seq += 1
        slot = self.seq % NUM_SLOTS
        offsets, total = _layout(tensors)
        meta = dict(
            seq=self.seq,
            slot=slot,
            tensors=[
                (k, tuple(t.shape), t.dtype, off)
                for (k, t), off in zip(tensors, offsets)
            ],
            others=others,
            setup=setup,
        )
        self._post_meta(meta, pp_group)
        if setup is not None:
            self._open_slots(pp_group.recv_object(src=self.dst))
        assert self.slots is not None
        if total > self.slots[slot].numel():
            raise RuntimeError(
                f"PP SDMA slot of {self.slots[slot].numel()} B < step of {total} B"
            )
        done = torch.Event()
        done.record()
        self._jobs.put((self.seq, slot, tensors, offsets, done))

    def _post_meta(self, meta: dict, pp_group) -> None:
        """Send this step's metadata (isend), after reaping the sends the receiver
        has consumed."""
        self._reap_meta_sends()
        self._pending.append((meta["seq"], pp_group.isend_object(meta, dst=self.dst)))

    def _reap_meta_sends(self) -> None:
        """Release the metadata sends the receiver has already received.

        A gloo isend ``Work`` reports ``is_completed()`` False until ``wait()``
        is called, so polling ``is_completed()`` would never drain this list.
        The receiver ``recv_object``s step ``seq``'s metadata before it reads
        that step's slot and bumps ``freed`` to ``seq`` only after that step has
        run on its GPU, so every send with ``seq <= freed`` has been received
        and its ``wait()`` returns at once. Past ``_MAX_PENDING_META``
        unconfirmed sends, block on the oldest (the receiver posts that recv
        without needing anything from this thread).
        """
        if not self._pending:
            return
        freed = self.counters.freed.value
        pending = self._pending
        while pending and (pending[0][0] <= freed or len(pending) >= _MAX_PENDING_META):
            pending.popleft()[1].wait()

    def _loop(self) -> None:
        try:
            torch.accelerator.set_device_index(self.device)
            while True:
                seq, slot, tensors, offsets, done = self._jobs.get()
                done.synchronize()
                _Counters.wait(self.counters.freed, seq - NUM_SLOTS)
                assert self.slots is not None
                base = self.slots[slot].data_ptr()
                for (_, t), off in zip(tensors, offsets):
                    size = t.numel() * t.element_size()
                    if size == 0:
                        continue
                    chunk = (size // len(_ENGINES) + ALIGN - 1) // ALIGN * ALIGN
                    used = []
                    for k, engine in enumerate(_ENGINES):
                        lo = k * chunk
                        if lo >= size:
                            break
                        n = min(chunk, size - lo)
                        self.hsa.copy(
                            base + off + lo,
                            self.dst_agent,
                            t.data_ptr() + lo,
                            self.src_agent,
                            n,
                            self.signals[k],
                            engine,
                        )
                        used.append(self.signals[k])
                    for s in used:
                        self.hsa.wait(s)
                del tensors
                self.counters.ready.value = seq
        except BaseException as e:  # surfaced on the next send()
            self._error = e
            logger.exception("PP SDMA sender thread failed")


class PPSdmaReceiver:
    """The receiving side of one stage boundary (src -> this rank)."""

    def __init__(self, src: int) -> None:
        self.src = src
        self.slots: list[torch.Tensor] | None = None
        self._last_seq = 0
        self._frees: queue.Queue = queue.Queue()

    def release_previous(self) -> None:
        """Called at the top of a step: the previous step's slot is free once
        everything enqueued so far (its forward, sampling, draft) has run."""
        if self._last_seq:
            done = torch.Event()
            done.record()
            self._frees.put((self._last_seq, done))
            self._last_seq = 0

    def _free_loop(self) -> None:
        torch.accelerator.set_device_index(self.device)
        while True:
            seq, done = self._frees.get()
            done.synchronize()
            self.counters.freed.value = seq

    def recv(self, pp_group) -> dict[str, Any]:
        from torch.multiprocessing.reductions import reduce_tensor

        meta = pp_group.recv_object(src=self.src)
        if meta["setup"] is not None:
            setup = meta["setup"]
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
            self.slots = [
                torch.empty(setup["slot_bytes"], dtype=torch.uint8, device=self.device)
                for _ in range(setup["num_slots"])
            ]
            pp_group.send_object(
                [reduce_tensor(s)[1] for s in self.slots], dst=self.src
            )
            self.counters = _Counters(setup["counters"], create=False)
            os.unlink(self.counters.path)  # both sides hold the mapping
            threading.Thread(
                target=self._free_loop, daemon=True, name="pp-sdma-free"
            ).start()
        assert self.slots is not None
        _Counters.wait(self.counters.ready, meta["seq"])
        slot = self.slots[meta["slot"]]
        out: dict[str, Any] = dict(meta["others"])
        for key, shape, dtype, off in meta["tensors"]:
            n = 1
            for d in shape:
                n *= d
            nbytes = n * dtype.itemsize
            out[key] = slot[off : off + nbytes].view(dtype).view(shape)
        self._last_seq = meta["seq"]
        return out
