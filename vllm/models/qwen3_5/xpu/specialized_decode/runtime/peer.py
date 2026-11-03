# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The P2P mailboxes of the in-kernel TP all-reduces: one device region per
rank, exported with Level Zero IPC and imported by the peer, holding one mailbox
per mono layer / MTP prologue and the vocabulary-shard candidate exchange; and
the TP agreements that must precede anything collective."""

import os
import socket
import tempfile

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group."""
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _exchange_fd(fd: int, rank: int, group) -> int:
    """Swaps the dma-buf fds of the two ranks over a unix socket (SCM_RIGHTS):
    importing the peer's IPC handle needs no ptrace permission this way."""
    paths: list = [None] * dist.get_world_size(group)
    path = srv = None
    if rank == 0:
        path = os.path.join(tempfile.gettempdir(), f"vllm_xpu_mono_{os.getpid()}.sock")
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
    dist.all_gather_object(paths, path, group=group)
    if srv is not None:
        conn, _ = srv.accept()
    else:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(paths[0])
    with conn:
        socket.send_fds(conn, [b"f"], [fd])
        _, fds, _, _ = socket.recv_fds(conn, 1, 1)
    if srv is not None:
        assert path is not None
        srv.close()
        os.unlink(path)
    return fds[0]


class PeerRegion:
    """This rank's region and the peer's, mapped: one mailbox per name, then the
    candidate exchange. The constructor is collective over the TP CPU group (the
    handle and fd exchange): every rank reaches it after the TP agreements."""

    def __init__(self, names: list[str], rank: int, group):
        ops = torch.ops._xpu_C
        self.rank = rank
        self.mbox_bytes = int(ops.mono_mbox_bytes())
        self.slots = {name: i for i, name in enumerate(names)}
        self.lm_off = len(names) * self.mbox_bytes
        self.handle = ops.mono_ipc_create(
            self.lm_off + int(ops.mono_lmhead_mbox_bytes())
        )
        info = ops.mono_ipc_export(self.handle)
        infos: list = [None] * dist.get_world_size(group)
        dist.all_gather_object(infos, info, group=group)
        peer_fd = _exchange_fd(int(info[1]), rank, group)
        mode = ops.mono_ipc_open(self.handle, infos[1 - rank], peer_fd)
        self.own = int(ops.mono_ipc_ptr(self.handle, False))
        self.peer = int(ops.mono_ipc_ptr(self.handle, True))
        logger.info(
            "Qwen3.5 XPU mono decode: P2P mailboxes for %d kernels (IPC import "
            "mode %d)",
            len(names),
            mode,
        )

    def mailbox(self, name: str) -> tuple[int, int]:
        """(own, peer) addresses of the mailbox of ``name``."""
        off = self.slots[name] * self.mbox_bytes
        return self.own + off, self.peer + off

    def candidates(self) -> tuple[int, int]:
        """(own, peer) addresses of the candidate exchange."""
        return self.own + self.lm_off, self.peer + self.lm_off
