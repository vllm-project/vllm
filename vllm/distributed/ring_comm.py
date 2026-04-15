# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Ring communication primitives for Context Parallelism.

Provides asynchronous P2P send/recv in a ring topology with explicit
CUDA stream management for reliable communication-computation overlap.

The design follows NVIDIA TransformerEngine's context_parallel.py:
- A dedicated ``cp_stream`` handles all P2P transfers
- Events synchronize between the compute stream and the comm stream
- K and V are packed into a single contiguous buffer per P2P round

Adapted from vllm-omni (diffusion/distributed/comm.py) and
TransformerEngine (attention/dot_product_attention/context_parallel.py).
"""

import torch
import torch.distributed as dist


class RingComm:
    """Asynchronous ring P2P communicator with explicit stream overlap.

    Each rank sends to ``(rank + 1) % world_size`` and receives from
    ``(rank - 1) % world_size``.  All P2P transfers run on a dedicated
    CUDA stream (``cp_stream``) so they can overlap with attention
    kernels running on the default compute stream.

    Args:
        process_group: The torch distributed process group for the CP
            ranks.
        cp_stream: Optional dedicated CUDA stream for P2P communication.
            If *None*, a new stream is created on the current device.

    Usage::

        comm = RingComm(cp_group, cp_stream)

        for step in range(comm.world_size):
            if step + 1 < comm.world_size:
                next_kv = comm.send_recv(current_kv)
                comm.commit()  # launch P2P on cp_stream

            # ... run attention on the default (compute) stream ...

            if step + 1 < comm.world_size:
                comm.wait()  # sync cp_stream → compute stream
                current_kv = next_kv
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        cp_stream: torch.cuda.Stream | None = None,
    ):
        self._process_group = process_group
        self._ops: list[dist.P2POp] = []
        self._reqs: list[dist.Work] | None = None

        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        local_send = (self.rank + 1) % self.world_size
        local_recv = (self.rank - 1) % self.world_size
        self.send_rank = dist.get_global_rank(self._process_group, local_send)
        self.recv_rank = dist.get_global_rank(self._process_group, local_recv)

        if cp_stream is None:
            cp_stream = torch.cuda.Stream()
        self._cp_stream = cp_stream

        self._comm_event = torch.cuda.Event()
        self._compute_event = torch.cuda.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_recv(
        self,
        to_send: torch.Tensor,
        recv_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Queue an async send/recv pair.

        Multiple ``send_recv`` calls can be queued (e.g. K and V)
        before a single ``commit``.

        Args:
            to_send: Tensor to send.  Made contiguous if needed.
            recv_tensor: Optional pre-allocated receive buffer.

        Returns:
            The receive buffer (contents valid only after ``wait``).
        """
        if not to_send.is_contiguous():
            to_send = to_send.contiguous()

        if recv_tensor is None:
            res = torch.empty_like(to_send, memory_format=torch.contiguous_format)
        else:
            res = recv_tensor
            if not res.is_contiguous():
                res = res.contiguous()

        # TE pattern: alternate send/recv order by rank parity
        # to avoid potential deadlocks on some NCCL/RCCL versions
        if self.rank % 2 == 0:
            self._ops.append(
                dist.P2POp(
                    dist.isend, to_send, self.send_rank, group=self._process_group
                )
            )
            self._ops.append(
                dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
            )
        else:
            self._ops.append(
                dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
            )
            self._ops.append(
                dist.P2POp(
                    dist.isend, to_send, self.send_rank, group=self._process_group
                )
            )
        return res

    def commit(self) -> None:
        """Launch all queued P2P operations on the dedicated cp_stream.

        Records an event on the compute stream so cp_stream can wait
        for any preceding compute to finish before sending data, then
        dispatches the batched P2P ops on cp_stream.
        """
        if self._reqs is not None:
            raise RuntimeError(
                "commit() called while previous operations are still pending. "
                "Call wait() before committing new operations."
            )

        # compute stream records: "my tensors are ready to send"
        self._compute_event.record(torch.cuda.current_stream())
        # cp_stream waits for compute to finish writing the send buffers
        self._cp_stream.wait_event(self._compute_event)

        with torch.cuda.stream(self._cp_stream):
            self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        """Block until all committed P2P operations complete.

        Waits for P2P ops on cp_stream, then synchronises cp_stream
        back into the compute stream so subsequent compute can safely
        read the received tensors.
        """
        if self._reqs is None:
            raise RuntimeError(
                "wait() called before commit(). "
                "Queue operations with send_recv(), then call commit()."
            )

        # Wait for all P2P ops to finish on cp_stream
        with torch.cuda.stream(self._cp_stream):
            for req in self._reqs:
                req.wait()

        # cp_stream records: "recv buffers are filled"
        self._comm_event.record(self._cp_stream)
        # compute stream waits until recv data is ready
        torch.cuda.current_stream().wait_event(self._comm_event)

        self._reqs = None
        self._ops = []
