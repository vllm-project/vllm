# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport over MoRI-IO on ROCm (`VLLM_PP_TRANSPORT=mori_io`).

Each non-first rank registers `NUM_SLOTS` receive slots with a MoRI
`IOEngine`, each sized for its largest step: its persistent intermediate
tensors (`max_num_tokens` rows of every key, aux hidden states included). Each
non-last rank registers as many send slots of its next stage's slot size.
Per step N, in slot `N % num_slots`:

- sender: pack the outputs into the send slot on the current stream. A
  background thread waits for the pack and for a free receive slot (a credit),
  writes the packed bytes into the next stage's slot with one MoRI-IO write,
  then sends `(N, slot, rows)` over ZMQ: the XGMI backend has no
  receiver-side completion, so that message signals it.
- receiver: wait for the message, copy the slot into the persistent tensors on
  the current stream, and return the slot (credit) once that copy is done.

Each rank connects to its previous and next stage only. A chain on one node
uses MoRI's XGMI backend, one that spans nodes its RDMA backend. Unlike
`copy_engine` on ROCm, this needs neither gfx950 nor particular free SDMA
engines.
"""

import json
import math
import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.pp_transport import (
    PPTransport,
    RowSpec,
    check_intermediate_tensors,
    row_specs,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# Steps a stage may send ahead of the next one: receive slots per boundary
# (and as many send slots). Three let a stage run two steps ahead of its
# consumer, enough to keep a PP2 / PP4 prefill pipeline busy.
NUM_SLOTS = 3
_ALIGN = 256
# How long a stage waits for its neighbor before reporting it hung.
_TIMEOUT_S = 600


def row_bytes(specs: dict[str, RowSpec]) -> dict[str, int]:
    return {k: dtype.itemsize * math.prod(shape) for k, (dtype, shape) in specs.items()}


def pack_offsets(
    sizes: dict[str, int], rows: dict[str, int]
) -> tuple[dict[str, int], int]:
    """Each key's byte offset in a slot holding `rows[key]` rows of
    `sizes[key]` bytes, and the bytes the step uses."""
    offsets, end = {}, 0
    for key, n in rows.items():
        offsets[key] = end
        end += -(-n * sizes[key] // _ALIGN) * _ALIGN
    return offsets, end


def slot_bytes(persistent: dict[str, torch.Tensor]) -> int:
    """A receive slot's size: the step that fills every persistent
    intermediate tensor."""
    rows = {k: t.shape[0] for k, t in persistent.items()}
    return pack_offsets(row_bytes(row_specs(persistent)), rows)[1]


def check_available() -> None:
    """Raise unless MoRI-IO can carry PP tensors here; the same on every rank."""
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_mori

    if not current_platform.is_rocm():
        raise RuntimeError("VLLM_PP_TRANSPORT=mori_io needs ROCm")
    if not has_mori():
        raise RuntimeError(
            "VLLM_PP_TRANSPORT=mori_io needs MoRI (https://github.com/ROCm/mori)"
        )


class _Worker:
    """A daemon thread running jobs in submission order. Each ZMQ socket is
    used by one thread only."""

    def __init__(self, name: str, device: int) -> None:
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        self.device = device
        threading.Thread(target=self._run, name=name, daemon=True).start()

    def submit(self, fn: Callable[..., None], *args: Any) -> Future:
        future: Future = Future()
        self.jobs.put((future, fn, args))
        return future

    def _run(self) -> None:
        torch.accelerator.set_device_index(self.device)
        while True:
            future, fn, args = self.jobs.get()
            try:
                future.set_result(fn(*args))
            except BaseException as e:
                future.set_exception(e)


def _agree(pp_group: "GroupCoordinator", fn: Callable[[], Any]) -> list[Any]:
    """Run `fn` on every rank and gather the results; raise on every rank if it
    raised on any."""
    out = error = None
    try:
        out = fn()
    except Exception as e:
        error = repr(e)
    gathered: list[Any] = [None] * pp_group.world_size
    torch.distributed.all_gather_object(
        gathered, (out, error), group=pp_group.cpu_group
    )
    if errors := [e for _, e in gathered if e]:
        raise RuntimeError(f"MoRI-IO PP transport setup failed: {errors[0]}")
    return [o for o, _ in gathered]


class MoriIOTransport(PPTransport):
    name = "mori_io"

    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_group: "GroupCoordinator",
        persistent: dict[str, torch.Tensor] | None,
    ) -> None:
        """Collective over the PP group. Raises on every rank alike if some
        rank cannot set up."""
        check_available()
        import zmq
        from mori.io import (
            BackendType,
            EngineDesc,
            IOEngine,
            IOEngineConfig,
            MemoryDesc,
            RdmaBackendConfig,
            XgmiBackendConfig,
        )

        from vllm.distributed.parallel_state import in_the_same_node_as
        from vllm.utils.network_utils import (
            get_ip,
            get_open_port,
            make_zmq_path,
            make_zmq_socket,
        )

        rank = pp_group.rank_in_group
        first, last = pp_group.is_first_rank, pp_group.is_last_rank
        one_node = all(in_the_same_node_as(pp_group.cpu_group, source_rank=0))
        self.num_slots = NUM_SLOTS
        device = torch.accelerator.current_device_index()
        ctx = zmq.Context.instance()
        ip = get_ip()

        def listen() -> tuple[Any, str]:
            path = make_zmq_path("tcp", ip, get_open_port())
            return make_zmq_socket(ctx, path, zmq.PULL, bind=True), path

        def register() -> dict[str, Any]:
            key = f"vllm-pp{rank}-{uuid.uuid4().hex[:8]}"
            self.engine = IOEngine(key, IOEngineConfig(ip, get_open_port()))
            if one_node:
                self.engine.create_backend(BackendType.XGMI, XgmiBackendConfig())
            else:
                # Completion is signaled over ZMQ.
                config = RdmaBackendConfig(enable_notification=False)
                self.engine.create_backend(BackendType.RDMA, config)
            info: dict[str, Any] = {"engine": self.engine.get_engine_desc().pack()}
            if not first:
                assert persistent is not None
                self.persistent = persistent
                self.recv_slot_bytes = slot_bytes(persistent)
                specs = row_specs(persistent)
                self.recv_row_bytes = row_bytes(specs)
                self.recv_buf = torch.empty(
                    self.num_slots * self.recv_slot_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
                recv_md = self.engine.register_torch_tensor(self.recv_buf)
                self.meta_sock, info["meta"] = listen()
                info["recv_md"] = recv_md.pack()
                info["slot_bytes"] = self.recv_slot_bytes
                info["specs"] = specs
            if not last:
                self.ack_sock, info["ack"] = listen()
            return info

        infos = _agree(pp_group, register)

        def connect() -> None:
            for peer in {rank - 1, rank + 1} & set(range(len(infos))):
                desc = EngineDesc.unpack(infos[peer]["engine"])
                self.engine.register_remote_engine(desc)
            if not first:
                self.ack_push = make_zmq_socket(
                    ctx, infos[rank - 1]["ack"], zmq.PUSH, bind=False
                )
            if not last:
                nxt = infos[rank + 1]
                self.send_slot_bytes = nxt["slot_bytes"]
                self.send_specs = nxt["specs"]
                self.send_row_bytes = row_bytes(self.send_specs)
                self.send_buf = torch.empty(
                    self.num_slots * self.send_slot_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
                send_md = self.engine.register_torch_tensor(self.send_buf)
                remote_md = MemoryDesc.unpack(nxt["recv_md"])
                self.session = self.engine.create_session(send_md, remote_md)
                if self.session is None:
                    raise RuntimeError(f"no MoRI-IO session to PP rank {rank + 1}")
                self.meta_push = make_zmq_socket(ctx, nxt["meta"], zmq.PUSH, bind=False)

        _agree(pp_group, connect)

        self.sent = self.received = 0
        if not first:
            self.acker = _Worker("mori-pp-ack", device)
            self.acks: list[Future] = []
        if not last:
            self.sender = _Worker("mori-pp-send", device)
            self.writes: list[Future | None] = [None] * self.num_slots
            self.credits = self.num_slots
        # Every stage logs: slot sizes differ per boundary.
        logger.info_once(
            "MoRI-IO PP transport on PP rank %d: %s backend, %d slots, "
            "receive slot %.1f MiB, send slot %.1f MiB",
            rank,
            "XGMI" if one_node else "RDMA",
            self.num_slots,
            getattr(self, "recv_slot_bytes", 0) / 2**20,
            getattr(self, "send_slot_bytes", 0) / 2**20,
            scope="process",
        )

    def send(self, tensors: dict, pp_group: "GroupCoordinator") -> None:
        check_intermediate_tensors(tensors, self.send_specs)
        rows = {k: t.shape[0] for k, t in tensors.items()}
        offsets, size = pack_offsets(self.send_row_bytes, rows)
        if size > self.send_slot_bytes:
            raise RuntimeError(
                f"A {size / 2**20:.1f} MiB PP step exceeds the "
                f"{self.send_slot_bytes / 2**20:.1f} MiB MoRI-IO slot sized for "
                "the next stage's persistent intermediate tensors."
            )
        slot = self.sent % self.num_slots
        if (previous := self.writes[slot]) is not None:
            previous.result()
        base = slot * self.send_slot_bytes
        for key, t in tensors.items():
            if n := t.numel() * t.element_size():
                start = base + offsets[key]
                dst = self.send_buf[start : start + n].view(t.dtype).view(t.shape)
                dst.copy_(t)
        packed = torch.cuda.Event()
        packed.record()
        meta = json.dumps([self.sent, slot, rows]).encode()
        self.writes[slot] = self.sender.submit(self._write, packed, base, size, meta)
        self.sent += 1

    def _write(
        self, packed: torch.cuda.Event, base: int, size: int, meta: bytes
    ) -> None:
        packed.synchronize()
        if self.credits:
            self.credits -= 1
        elif self.ack_sock.poll(_TIMEOUT_S * 1000):
            self.ack_sock.recv()
        else:
            raise TimeoutError(f"The next PP stage freed no slot in {_TIMEOUT_S} s")
        if size:
            uid = self.session.allocate_transfer_uid()
            status = self.session.write(base, base, size, uid)
            deadline = time.monotonic() + _TIMEOUT_S
            while not (status.Succeeded() or status.Failed()):
                if time.monotonic() > deadline:
                    raise TimeoutError("MoRI-IO PP write did not complete")
                time.sleep(0)
            if status.Failed():
                raise RuntimeError(f"MoRI-IO PP write failed: {status.Message()}")
        self.meta_push.send(meta)

    def recv(self, pp_group: "GroupCoordinator") -> "IntermediateTensors":
        """This step's tensors, as views of the persistent tensors, copied in
        from the slot when `.tensors` is first read."""
        from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

        while self.acks and self.acks[0].done():
            self.acks.pop(0).result()
        if not self.meta_sock.poll(_TIMEOUT_S * 1000):
            raise TimeoutError(
                f"No PP tensors from the previous stage in {_TIMEOUT_S} s"
            )
        seq, slot, rows = json.loads(self.meta_sock.recv())
        if seq != self.received:
            raise RuntimeError(f"Got PP step {seq}, expected {self.received}")
        self.received += 1
        offsets, _ = pack_offsets(self.recv_row_bytes, rows)
        base = slot * self.recv_slot_bytes
        out = {k: self.persistent[k][:n] for k, n in rows.items()}

        def copy_in() -> None:
            for key, dst in out.items():
                start = base + offsets[key]
                src = self.recv_buf[start : start + dst.numel() * dst.element_size()]
                dst.copy_(src.view(dst.dtype).view(dst.shape))
            copied = torch.cuda.Event()
            copied.record()
            self.acks.append(self.acker.submit(self._ack, copied))

        return AsyncIntermediateTensors(out, comm_postprocess=[copy_in])

    def _ack(self, copied: torch.cuda.Event) -> None:
        copied.synchronize()
        self.ack_push.send(b"")
