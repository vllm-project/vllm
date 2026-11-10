# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport on SDMA copy engines (ROCm gfx950, one node).

The protocol, the buffers and the receive side are `CopyEngineTransport`'s.
Only the peer copy differs: on HIP 7.16 a HIP peer copy runs as a blit kernel
on the sender's CUs (`hipMemcpyFlagExtPreferCE` does not apply to
device-to-device copies, and `hipMemcpyDeviceToDeviceNoCU` picks engines whose
peer copy fails on MI355X while returning success). The copies therefore go
through ROCr's `hsa_amd_memory_async_copy_on_engine` on explicit engines, kept
in order with the side stream by HSA signals. Per step N:

- the host queues the data copies behind a dependency signal, which the side
  stream opens (writes 0) after its `empty >= N - 1` wait;
- two 8-byte copies of N, queued behind the data copies, publish `full = N` to
  the receiver and `sent = N` to this rank; the next step's pack waits for
  `sent`, as the CUDA transport's waits for its copy's event.

The shared buffers come from `allocate_shared_buffer_and_handle`, which on ROCm
allocates uncached memory, so a stream wait never reads a stale L2 line.
"""

import ctypes
import functools
from typing import Any, NamedTuple

import torch

from vllm.distributed.pp_transport.copy_engine import CopyEngineTransport

# This rank's header words after full (0) and empty (8): `sent`, then the ring
# of 8-byte sources of the published step numbers (below _DATA_OFFSET = 256).
_SENT = 16
_SEQ_SOURCES = 64
# Steps whose signals may be in flight; an entry is reused _RING steps later.
_RING = 16
# Two even engines fill one xGMI direction; odd engines hang on MI355X.
_ENGINES = (1 << 2, 1 << 4)
_ALIGN = 256

_HSA_DEVICE_TYPE_GPU = 1
_HSA_AGENT_INFO_DEVICE = 17
_HSA_AMD_AGENT_INFO_BDFID = 0xA006
# GPU_ONLY signals have a value pointer, which a stream can write.
_HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1
_HSA_SIGNAL_CONDITION_LT = 2
_HSA_WAIT_STATE_ACTIVE = 1
_HIP_STREAM_WAIT_VALUE_GTE = 0


class _Agent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Signal(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Hsa:
    """The ROCr calls of an SDMA copy, over ctypes."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("libhsa-runtime64.so.1")
        self._visit_type = ctypes.CFUNCTYPE(ctypes.c_int, _Agent, ctypes.c_void_p)
        lib.hsa_iterate_agents.argtypes = [self._visit_type, ctypes.c_void_p]
        lib.hsa_agent_get_info.argtypes = [_Agent, ctypes.c_int, ctypes.c_void_p]
        lib.hsa_amd_signal_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_Signal),
        ]
        lib.hsa_amd_signal_value_pointer.argtypes = [
            _Signal,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.hsa_signal_store_relaxed.argtypes = [_Signal, ctypes.c_int64]
        lib.hsa_signal_load_relaxed.argtypes = [_Signal]
        lib.hsa_signal_load_relaxed.restype = ctypes.c_int64
        lib.hsa_signal_wait_scacquire.argtypes = [
            _Signal,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.hsa_signal_wait_scacquire.restype = ctypes.c_int64
        lib.hsa_amd_memory_async_copy_on_engine.argtypes = [
            ctypes.c_void_p,
            _Agent,
            ctypes.c_void_p,
            _Agent,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(_Signal),
            _Signal,
            ctypes.c_uint32,
            ctypes.c_bool,
        ]
        if lib.hsa_init():
            raise RuntimeError("hsa_init failed")
        self.lib = lib
        self.agents: dict[int, _Agent] = {}

        def visit(agent, _) -> int:
            kind = ctypes.c_uint32()
            lib.hsa_agent_get_info(agent, _HSA_AGENT_INFO_DEVICE, ctypes.byref(kind))
            if kind.value == _HSA_DEVICE_TYPE_GPU:
                bdf = ctypes.c_uint32()
                lib.hsa_agent_get_info(
                    agent, _HSA_AMD_AGENT_INFO_BDFID, ctypes.byref(bdf)
                )
                self.agents[bdf.value & ~0x7] = _Agent(agent.handle)
            return 0

        lib.hsa_iterate_agents(self._visit_type(visit), None)

    def agent(self, bdf: int) -> _Agent:
        if bdf not in self.agents:
            raise RuntimeError(f"no HSA GPU agent at PCI {bdf:#x}")
        return self.agents[bdf]

    def signal(self) -> _Signal:
        s = _Signal()
        if self.lib.hsa_amd_signal_create(
            1, 0, None, _HSA_AMD_SIGNAL_AMD_GPU_ONLY, ctypes.byref(s)
        ):
            raise RuntimeError("hsa_amd_signal_create failed")
        return s

    def value_pointer(self, signal: _Signal) -> int:
        p = ctypes.c_void_p()
        if self.lib.hsa_amd_signal_value_pointer(signal, ctypes.byref(p)):
            raise RuntimeError("hsa_amd_signal_value_pointer failed")
        assert p.value is not None
        return p.value

    def store(self, signal: _Signal, value: int) -> None:
        self.lib.hsa_signal_store_relaxed(signal, value)

    def load(self, signal: _Signal) -> int:
        return self.lib.hsa_signal_load_relaxed(signal)

    def wait_zero(self, signal: _Signal) -> None:
        while (
            self.lib.hsa_signal_wait_scacquire(
                signal, _HSA_SIGNAL_CONDITION_LT, 1, 1 << 40, _HSA_WAIT_STATE_ACTIVE
            )
            > 0
        ):
            pass

    def copy(
        self,
        dst: int,
        dst_agent: _Agent,
        src: int,
        src_agent: _Agent,
        size: int,
        deps: list[_Signal],
        done: _Signal,
        engine: int,
    ) -> None:
        """Queue a copy on `engine` that starts once every signal in `deps` is
        0 and decrements `done` when it finishes."""
        dep_array = (_Signal * len(deps))(*deps)
        if self.lib.hsa_amd_memory_async_copy_on_engine(
            dst,
            dst_agent,
            src,
            src_agent,
            size,
            len(deps),
            dep_array,
            done,
            engine,
            True,
        ):
            raise RuntimeError("hsa_amd_memory_async_copy_on_engine failed")


class _Hip:
    """HIP's stream memory operations, over ctypes."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("libamdhip64.so")
        lib.hipStreamWaitValue64.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_uint64,
        ]
        lib.hipStreamWriteValue64.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint,
        ]
        self.lib = lib

    @staticmethod
    def _check(err: int, call: str) -> None:
        if err:
            raise RuntimeError(f"{call} failed: {err}")

    def wait_flag(self, stream: torch.cuda.Stream, addr: int, at_least: int) -> None:
        self._check(
            self.lib.hipStreamWaitValue64(
                stream.cuda_stream,
                addr,
                at_least,
                _HIP_STREAM_WAIT_VALUE_GTE,
                0xFFFF_FFFF_FFFF_FFFF,
            ),
            "hipStreamWaitValue64",
        )

    def write_flag(self, stream: torch.cuda.Stream, addr: int, value: int) -> None:
        self._check(
            self.lib.hipStreamWriteValue64(stream.cuda_stream, addr, value, 0),
            "hipStreamWriteValue64",
        )


@functools.cache
def _hsa() -> _Hsa:
    return _Hsa()


@functools.cache
def _hip() -> _Hip:
    return _Hip()


def _bdf(device: int) -> int:
    """The PCI address HSA reports for `device` (BDFID without the function)."""
    p = torch.cuda.get_device_properties(device)
    return (p.pci_bus_id << 8) | (p.pci_device_id << 3)


class _StepSignals(NamedTuple):
    dep: _Signal  # Opened by the side stream.
    data: tuple[_Signal, ...]  # One per engine, counts its data copies.
    full: _Signal  # The copy of the step number into the receiver's `full`.
    sent: _Signal  # The copy of the step number into this rank's `sent`.


class _Sent:
    """A step's peer copy, as `CopyEngineTransport` keeps it."""

    def __init__(self, transport: "SdmaTransport", seq: int, full: _Signal):
        self.transport, self.seq, self.full = transport, seq, full
        self.done = False

    def query(self) -> bool:
        # A ring entry's signals are waited for before they are reused.
        if not self.done:
            self.done = (
                self.transport.seq - self.seq >= _RING or _hsa().load(self.full) == 0
            )
        return self.done


class SdmaTransport(CopyEngineTransport):
    _LINK = "xGMI"

    @staticmethod
    def _wait_flag(stream: torch.cuda.Stream, addr: int, at_least: int) -> None:
        _hip().wait_flag(stream, addr, at_least)

    @staticmethod
    def _write_flag(stream: torch.cuda.Stream, addr: int, value: int) -> None:
        _hip().write_flag(stream, addr, value)

    @classmethod
    def _platform_unavailable(cls) -> str | None:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return "PP SDMA copy engines need ROCm"
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return "PP SDMA copy engines need gfx950 (MI355X)"
        return None

    def _device_info(self, device: int) -> int:
        bdf = _bdf(device)
        _hsa().agent(bdf)
        return bdf

    def _setup_sender(self, next_info: int) -> None:
        hsa = _hsa()
        self.agent = hsa.agent(_bdf(torch.accelerator.current_device_index()))
        self.peer_agent = hsa.agent(next_info)
        self.ring = [
            _StepSignals(
                hsa.signal(),
                tuple(hsa.signal() for _ in _ENGINES),
                hsa.signal(),
                hsa.signal(),
            )
            for _ in range(_RING)
        ]

    def _peer_copy(self, rows: dict[str, int]) -> _Sent:
        hsa = _hsa()
        header = self.empty - 8  # This rank's shared buffer.
        i = self.seq % _RING
        sig = self.ring[i]
        if self.seq > _RING:
            # Resetting a signal that a queued copy still uses deadlocks it.
            hsa.wait_zero(sig.full)
            hsa.wait_zero(sig.sent)

        # Each key's rows split across the engines, in 256-byte pieces.
        copies: list[list[tuple[int, int]]] = [[] for _ in _ENGINES]
        for key, n in rows.items():
            r = self.send_layout[key]
            size = n * r.row_bytes
            piece = -(-size // len(_ENGINES) // _ALIGN) * _ALIGN
            for e in range(len(_ENGINES)):
                lo = e * piece
                if lo < size:
                    copies[e].append((r.offset + lo, min(piece, size - lo)))

        hsa.store(sig.dep, 1)
        for done, engine_copies in zip(sig.data, copies):
            hsa.store(done, len(engine_copies))
        hsa.store(sig.full, 1)
        hsa.store(sig.sent, 1)
        src = self.send_buf.data_ptr()
        for engine, done, engine_copies in zip(_ENGINES, sig.data, copies):
            for offset, size in engine_copies:
                hsa.copy(
                    self.peer_recv_buf + offset,
                    self.peer_agent,
                    src + offset,
                    self.agent,
                    size,
                    [sig.dep],
                    done,
                    engine,
                )
        # Behind `dep` too: with no rows the data signals start at 0.
        seq_source = header + _SEQ_SOURCES + 8 * i
        after_data = [sig.dep, *sig.data]
        hsa.copy(
            self.peer_full,
            self.peer_agent,
            seq_source,
            self.agent,
            8,
            after_data,
            sig.full,
            _ENGINES[0],
        )
        hsa.copy(
            header + _SENT,
            self.agent,
            seq_source,
            self.agent,
            8,
            after_data,
            sig.sent,
            _ENGINES[0],
        )

        # Side stream, after the base class's `empty` wait: the step number's
        # source word, then open the copies.
        self._write_flag(self.stream, seq_source, self.seq)
        self._write_flag(self.stream, hsa.value_pointer(sig.dep), 0)
        return _Sent(self, self.seq, sig.full)

    def _wait_done(self, stream: torch.cuda.Stream, done: Any) -> None:
        assert isinstance(done, _Sent)
        self._wait_flag(stream, self.empty - 8 + _SENT, done.seq)
