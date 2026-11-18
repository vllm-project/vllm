# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`torch.distributed` transport for the NCCL weight-transfer backend.

`PyNcclCommunicator` loads `libnccl` itself, so the `nccl` backend can only move
weights on CUDA and ROCm. On any other accelerator the communicator disables
itself and every collective becomes a no-op -- a whole update round then reports
success while the workers keep serving stale weights.

The backend does not actually need NCCL, though: the engines only ever ask their
communicator to `broadcast`, and every accelerator torch supports already has a
collective library behind `torch.distributed` (`xccl` on XPU). So keep the
backend and swap the transport. That matters because the RL control plane lives
in the trainer -- TRL and friends pick `backend="nccl"`, name NCCL in their
config schemas, and reach `packed_nccl_broadcast_producer` directly -- so a
separate backend name would need every trainer to learn about it, while a
transport chosen by the platform works with the trainers that exist today.

The group is stateless
(`stateless_init_torch_distributed_process_group`), as in the NCCL engine: the
trainer is a separate process and not a member of the workers' tensor- or
pipeline-parallel groups. Broadcasts therefore address the sender by its rank
*within the transfer group* (`group_src`), not by a global rank.
"""

import atexit
import contextlib
import threading
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)


def _reap_process_group(pending: list["ProcessGroup"]) -> threading.Thread:
    """Leave and release the group in `pending` on a thread nobody waits for.

    Releasing a process group finalizes its transport, and a transport whose peer
    is still in the group can take arbitrarily long to finalize: measured on
    oneCCL (`xccl`), `~ProcessGroupXCCL` blocks in the KVS handshake for exactly
    as long as the peer keeps its end open. In weight transfer the peer routinely
    outlives us -- the trainer finishes, the inference server keeps serving -- so
    this would hold the trainer's process open for the rest of the server's life.

    Neither `abort()` nor `shutdown()` avoids it: both return promptly and leave
    the finalization to the destructor. So everything from here on happens on a
    daemon thread: the interpreter never waits for one, and a thread still parked
    in the transport's teardown dies with the process. Unregistering goes with it,
    because that can drop the registry's reference and finalize right there.

    The group arrives inside a one-element list, and the caller must hold no
    reference of its own: whoever drops the *last* reference is the one who runs
    the destructor, so the caller handing over a group it still holds would race
    the reaper for it -- and lose about half the time.
    """

    def reap() -> None:
        from torch.distributed.distributed_c10d import _unregister_process_group

        # `abort()` rather than `shutdown()`: abort drops the local state, while
        # shutdown asks the transport to finalize, which means waiting on a peer
        # that may never arrive (and on oneCCL, failing hard enough to take the
        # process down if it has already died). A transfer group has nothing in
        # flight to lose -- every round ends synchronized.
        with contextlib.suppress(Exception):
            pending[0].abort()
        with contextlib.suppress(Exception):
            _unregister_process_group(pending[0].group_name)
        while pending:
            pending.pop()  # the destructor lands here, on nobody's critical path

    thread = threading.Thread(
        target=reap, name="weight-transfer-pg-reaper", daemon=True
    )
    thread.start()
    return thread


class _TransferGroupHandle:
    """Stand-in for `StatelessProcessGroup` on the `.group` attribute.

    Trainers reach into `communicator.group` to drop the rendezvous store and
    the socket behind it when they are done (TRL's `close_communicator` does),
    so the attribute has to exist and take assignment. A torch process group
    holds no such objects once the group is up, so both fields are inert here.
    """

    def __init__(self, process_group: "ProcessGroup") -> None:
        self.process_group = process_group
        self.store = None
        self.socket = None


class TorchDistTransport:
    """`PyNcclCommunicator`-shaped broadcast over `torch.distributed`.

    Implements the surface the weight-transfer engines and
    `packed_nccl_broadcast_*` use -- `broadcast`, `rank`, `world_size`,
    `available`/`disabled`, `group` -- so both sides of the transfer keep
    running the NCCL engine's code with no branch of their own.

    Deliberately *not* a `DeviceCommunicatorBase`, despite `XpuCommunicator` and
    friends issuing much the same `dist.broadcast`. That family wraps the groups
    vLLM's own world owns (`cpu_group`/`device_group`), whereas a transfer group
    is an ad-hoc rendezvous with a trainer process that is not in that world; its
    `broadcast` takes no stream to order against; and it has no
    `available`/`disabled`, which the callers here read. Hence `Transport` rather
    than `Communicator`: this is weight transfer's wire, not a device's
    collective layer.
    """

    def __init__(
        self,
        process_group: "ProcessGroup",
        device: torch.device,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        self.process_group = process_group
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.group = _TransferGroupHandle(process_group)
        self._reaper: threading.Thread | None = None
        # This transport either transfers or raises; it never degrades to the
        # no-op state `PyNcclCommunicator` falls back to.
        self.available = True
        self.disabled = False
        # Leave the group before the interpreter tears torch down. Nothing in
        # the weight-transfer control plane is required to close a communicator
        # (TRL only drops its reference), and a process that exits still holding
        # a live oneCCL group can block in the transport's own teardown.
        atexit.register(self.destroy)

    @classmethod
    def create(
        cls,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
        device: torch.device | int,
    ) -> "TorchDistTransport":
        """Join the trainer<->worker transfer group.

        The caller must already have selected this process's accelerator: the
        backend binds its communicator to the current device on first use.
        """
        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        if isinstance(device, int):
            device = torch.device(current_platform.device_type, device)
        backend = current_platform.dist_backend
        logger.info(
            "Joining weight transfer group %s:%s as rank %d/%d over %s",
            master_address,
            master_port,
            rank,
            world_size,
            backend,
        )
        process_group = stateless_init_torch_distributed_process_group(
            host=master_address,
            port=master_port,
            rank=rank,
            world_size=world_size,
            backend=backend,
        )
        assert not isinstance(process_group, tuple)  # return_store=False
        return cls(process_group, device, rank, world_size, backend)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        stream: torch.Stream | None = None,
    ) -> None:
        """Broadcast one tensor from `src`'s rank *within the transfer group*.

        `stream` is the stream the caller wants the collective ordered against,
        matching `PyNcclCommunicator.broadcast`; `torch.distributed` uses the
        current stream, so honor it by entering it.
        """
        ctx = (
            current_platform.stream(stream)
            if stream is not None
            else contextlib.nullcontext()
        )
        with ctx:
            dist.broadcast(tensor, group_src=src, group=self.process_group)
        self._rendezvous()

    def _rendezvous(self) -> None:
        """Re-couple the two ends of the transfer after one broadcast.

        A collective with `async_op=False` only orders the calling stream
        against the transport's; the host returns before the data has landed.
        Nothing else throttles the sender, so it enqueues its whole stream of
        tensors while the receiver is still loading the first one -- and some
        backends (oneCCL, behind `xccl`) then deadlock: once the two ends have
        drifted apart by more than a handful of collectives, none of them ever
        completes and both ranks spin in the next device synchronization. One
        barrier per broadcast bounds the drift at a single collective. A full
        update round issues a few hundred of them, and their cost has not been
        isolated from the transfer itself; if it proves material the bound can be
        loosened to every Nth broadcast rather than dropped.
        """
        dist.barrier(group=self.process_group)

    def destroy(self) -> None:
        """Leave the transfer group without waiting for the other end.

        Whichever side finishes first faces a peer that is still in the group, so
        nothing here may depend on that peer -- and leaving the group is exactly
        what can block on one, so `_reap_process_group` takes it from here. The
        communicator itself is unusable the moment this returns.
        """
        if self.process_group is None:
            return
        # Drop the exit hook with the group: it holds a strong reference to this
        # communicator, and a long-lived process that builds one per update round
        # would otherwise accumulate a registration for each.
        atexit.unregister(self.destroy)
        # Hand the group over by list, holding no reference to it here: see
        # `_reap_process_group` on why the loser of that race blocks.
        pending = [self.process_group]
        self.process_group = None  # type: ignore[assignment]
        self.group.process_group = None  # type: ignore[assignment]
        self.disabled = True
        self._reaper = _reap_process_group(pending)
