# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The flag ops and peer copy of `copy_engine.CopyEngineTransport` on ROCm
(gfx950): HIP stream memory ops and SDMA engines.

On HIP 7.16 a HIP peer copy runs as a blit kernel on the sender's CUs
(`hipMemcpyFlagExtPreferCE` does not apply to device-to-device copies, and
`hipMemcpyDeviceToDeviceNoCU` picks engines whose peer copy fails on MI355X
while returning success). The copies therefore go through ROCr's
`hsa_amd_memory_async_copy_on_engine` on explicit engines, kept in order with
the side stream by HSA signals. Per step N:

- the host queues the data copies behind a dependency signal, which the side
  stream opens (writes 0) after the transport's `empty >= N - 1` wait;
- two 8-byte copies of N, queued behind the data copies, publish `full = N` to
  the receiver and `sent = N` to this rank;
- the side stream waits for `sent >= N`, so the transport's event recorded on
  it covers the copies, as on CUDA.

The shared buffers come from `allocate_shared_buffer_and_handle`, which on ROCm
allocates uncached memory, so a stream wait never reads a stale L2 line.
"""

import ctypes
import functools
from typing import NamedTuple

import torch

# The copier's words in the shared buffer: `sent`, then the ring of 8-byte
# sources of the published step numbers.
_SEQ_SOURCES = 8
# Steps whose signals may be in flight; an entry is reused _RING steps later.
_RING = 16
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


@functools.cache
def _hip() -> ctypes.CDLL:
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
    return lib


@functools.cache
def _hsa() -> _Hsa:
    return _Hsa()


def _agent(pci: tuple[int, int]) -> _Agent:
    """The HSA agent at PCI (bus, device)."""
    bus, device = pci
    return _hsa().agent((bus << 8) | (device << 3))


class _StepSignals(NamedTuple):
    dep: _Signal  # Opened by the side stream.
    data: tuple[_Signal, ...]  # One per engine, counts its data copies.
    full: _Signal  # The copy of the step number into the receiver's `full`.
    sent: _Signal  # The copy of the step number into this rank's `sent`.


class SdmaCopyEngines:
    """ROCm's flag ops and, for a sender, its peer copy to the next stage."""

    @staticmethod
    def check_device(device: int) -> None:
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            raise RuntimeError("PP SDMA copies need gfx950 (MI355X)")

    @staticmethod
    def wait_flag(stream: torch.cuda.Stream, addr: int, at_least: int) -> None:
        mask = 0xFFFF_FFFF_FFFF_FFFF
        gte = _HIP_STREAM_WAIT_VALUE_GTE
        if err := _hip().hipStreamWaitValue64(
            stream.cuda_stream, addr, at_least, gte, mask
        ):
            raise RuntimeError(f"hipStreamWaitValue64 failed: {err}")

    @staticmethod
    def write_flag(stream: torch.cuda.Stream, addr: int, value: int) -> None:
        if err := _hip().hipStreamWriteValue64(stream.cuda_stream, addr, value, 0):
            raise RuntimeError(f"hipStreamWriteValue64 failed: {err}")

    def __init__(
        self,
        send_buf: int,
        peer_recv_buf: int,
        peer_full: int,
        words: int,
        peer_pci: tuple[int, int],
    ) -> None:
        """`words`: this rank's uncached shared-buffer bytes for `sent` and
        the step-number sources; `peer_pci`: the next stage's PCI address."""
        props = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        )
        self.agent = _agent((props.pci_bus_id, props.pci_device_id))
        self.peer_agent = _agent(peer_pci)
        self.send_buf = send_buf
        self.peer_recv_buf = peer_recv_buf
        self.peer_full = peer_full
        self.sent, self.seq_sources = words, words + _SEQ_SOURCES
        hsa = _hsa()
        self.ring = [
            _StepSignals(
                hsa.signal(),
                tuple(hsa.signal() for _ in _ENGINES),
                hsa.signal(),
                hsa.signal(),
            )
            for _ in range(_RING)
        ]

    def copy(
        self, stream: torch.cuda.Stream, layout: dict, rows: dict, seq: int
    ) -> None:
        """Queue, behind `stream`'s work, the copy of each key's first `rows`
        rows into the peer's receive region and the publish of `full = seq`,
        and make `stream` wait for them."""
        hsa = _hsa()
        i = seq % _RING
        sig = self.ring[i]
        if seq > _RING:
            # Resetting a signal that a queued copy still uses deadlocks it.
            hsa.wait_zero(sig.full)
            hsa.wait_zero(sig.sent)

        # Each key's rows split across the engines, in 256-byte pieces.
        copies: list[list[tuple[int, int]]] = [[] for _ in _ENGINES]
        for key, n in rows.items():
            r = layout[key]
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
        for engine, done, engine_copies in zip(_ENGINES, sig.data, copies):
            for offset, size in engine_copies:
                hsa.copy(
                    self.peer_recv_buf + offset,
                    self.peer_agent,
                    self.send_buf + offset,
                    self.agent,
                    size,
                    [sig.dep],
                    done,
                    engine,
                )
        # Behind `dep` too: with no rows the data signals start at 0.
        seq_source = self.seq_sources + 8 * i
        after_data = [sig.dep, *sig.data]
        for dst, agent, done in (
            (self.peer_full, self.peer_agent, sig.full),
            (self.sent, self.agent, sig.sent),
        ):
            hsa.copy(
                dst, agent, seq_source, self.agent, 8, after_data, done, _ENGINES[0]
            )

        self.write_flag(stream, seq_source, seq)
        self.write_flag(stream, hsa.value_pointer(sig.dep), 0)
        self.wait_flag(stream, self.sent, seq)
