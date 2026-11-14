# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA owner proxy: per GPU, a process whose NIC gathers rows of the shards
it serves into other GPUs' landing buffers.

It has its own GIL and one thread that moves rows (its command thread sleeps
in `recv` once set up), so it never waits on a CPU-busy model thread. The GPU
process sends commands over a socket (fds by SCM_RIGHTS). For a GPU fetching
rows (requester) from GPUs holding them (owners):

1. The requester's GPU writes, per owner and table, chained RDMA WRITEs whose
   SGEs gather that owner's rows, a packet's worth each, into the requester's
   landing buffer, with the owner proxy's pointers. Each QP's last chain links
   a READ of the requester's memory, which returns only once the rows before
   it have landed, then a fenced inline WRITE of `seq` into that QP's flag for
   the table. The GPU then posts a WRITE_WITH_IMM that copies them into this
   owner's mailbox (`rdma.py`).
2. The owner process, asleep on its receive CQ's completion channel, posts the
   WRs as they are: its NIC gathers the rows, then writes the flags. Rows in
   host memory of the requester's node are gathered the same way by the
   requester's own owner process over loopback QPs, so no GPU reads host
   memory.

It wakes once per request and never spins.

Run as a script: only the standard library and `verbs.py` are imported.
"""

import ctypes
import errno
import faulthandler
import json
import mmap
import os
import signal
import socket
import struct
import sys
import threading
import traceback

if __package__:
    from vllm.distributed.engram_pool import verbs
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import verbs  # type: ignore[no-redef]

SGE_BYTES = 16
# WRs per posted chain; the requester's GPU signals each chain's last. Few WRs
# take one chain: each more QP costs a post and a READ round trip.
MIN_CHAIN, MAX_CHAIN = 16, 64
# Receive WRs kept posted for each requester.
RECV_DEPTH = 32
_PR_SET_PDEATHSIG = 1


def chain_length(num_wrs: int, qps: int) -> int:
    """WRs per chain when an owner posts `num_wrs` WRs over `qps` QPs."""
    return min(MAX_CHAIN, max(MIN_CHAIN, -(-num_wrs // qps)))


def mailbox_bytes(max_wrs: int, max_items: int) -> int:
    """A requester's WRs, then their SGEs."""
    return max_wrs * verbs.SEND_WR_BYTES + max_items * SGE_BYTES


def _failed(wc: ctypes.Array, count: int, what: str) -> bool:
    """Report failed completions in `wc`; True if any."""
    failed = False
    for i in range(count):
        status, _, vendor = struct.unpack_from("<III", wc, i * verbs.WC_BYTES + 8)
        if status:
            print(
                f"Engram pool RDMA {what} failed: ibv_wc status {status}, "
                f"vendor error {vendor:#x}",
                file=sys.stderr,
                flush=True,
            )
            failed = True
    return failed


class Owner:
    """Posts requesters' WRs, so its NIC gathers rows of this GPU's shard and
    of its node's host shards; rows and flags of this GPU land through it."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.keep: list = []

    def serve(self) -> None:
        while True:
            message, fds, _, _ = socket.recv_fds(self.sock, 1 << 20, 4)
            if not message:
                return
            command, *args = json.loads(message)
            try:
                if command == "register":
                    args.append(fds)
                result = ("ok", getattr(self, command)(*args))
            except Exception:
                result = ("error", traceback.format_exc())
            self.sock.sendall(json.dumps(result).encode())

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True).start()

    def init(self, device, world, rank, qps_per_peer, sq_depth):
        self.device = dev = verbs.Device(device)
        # CPUs of another NUMA node post twice as slowly.
        if cpus := verbs.local_cpus(device) & os.sched_getaffinity(0):
            os.sched_setaffinity(0, cpus)
        self.world, self.rank = world, rank
        self.gatherers = [p for p in range(world) if p != rank] + [rank]
        sge_cap = min(dev.max_sge, 16)
        depth = min(sq_depth, dev.max_qp_wr)
        # Gathers complete here unseen until a send queue fills, so it holds
        # every send queue's WRs.
        self.data_cq, self.data_channel = dev.create_cq(
            min(dev.max_cqe, len(self.gatherers) * qps_per_peer * depth),
            wakeable=True,
        )
        # Requests complete receives, which wake this process.
        self.recv_cq, self.recv_channel = dev.create_cq(
            len(self.gatherers) * RECV_DEPTH, wakeable=True
        )
        # Per gatherer (this GPU's too, over loopback) QPs for gathers, and per
        # requester one that its requests arrive on.
        self.qps: dict[int, list[int]] = {}
        self.qpns: list[list[int]] = [[] for _ in range(world)]
        notices: list[int] = [0] * world
        self.receivers: dict[int, tuple[int, int]] = {}
        for peer in self.gatherers:
            made = [
                dev.create_qp(self.data_cq, self.recv_cq, depth, sge_cap, 64)
                for _ in range(qps_per_peer)
            ]
            self.qps[peer] = [qp for qp, _ in made]
            self.qpns[peer] = [qpn for _, qpn in made]
            qp, notices[peer] = dev.create_qp(
                self.data_cq, self.recv_cq, 1, 1, 0, RECV_DEPTH
            )
            self.receivers[notices[peer]] = (peer, qp)
        self.loopback = [
            dev.create_qp(self.data_cq, self.recv_cq, 1) for _ in range(qps_per_peer)
        ]
        self.data_wc = ctypes.create_string_buffer(verbs.WC_BYTES * 64)
        gid, lid = dev.address()
        return {
            "address": [gid.hex(), lid],
            "qpns": self.qpns,
            "notices": notices,
            "max_sge": sge_cap,
            "mtu": verbs.MTU_BYTES.get(dev.mtu, 1024),
            "device": dev.describe(),
        }

    def connect(self, peers, requesters, max_wrs, max_items, max_tables):
        """Connect each peer's QPs (`peers` lists the peer, its owner's address
        and QPs for this GPU) and each requester's (its address and QP for
        this GPU); returns this GPU's mailboxes (per requester and table) and
        their rkey and lkey."""
        dev = self.device
        for peer, (gid, lid), theirs in peers:
            for qp, remote in zip(self.qps[peer], theirs):
                dev.connect(qp, remote, (bytes.fromhex(gid), lid))
        for qp, qpn, (partner, partner_qpn) in zip(
            self.qps[self.rank], self.qpns[self.rank], self.loopback
        ):
            dev.connect(qp, partner_qpn, dev.address())
            dev.connect(partner, qpn, dev.address())
        self.recv_wr = (ctypes.c_uint64 * 4)()
        by_peer = {peer: qp for peer, qp in self.receivers.values()}
        for peer, (gid, lid), remote in requesters:
            dev.connect(by_peer[peer], remote, (bytes.fromhex(gid), lid))
            for _ in range(RECV_DEPTH):
                dev.post_recv(by_peer[peer], ctypes.addressof(self.recv_wr))
        self.box_bytes = mailbox_bytes(max_wrs, max_items)
        self.max_tables = max_tables
        boxes = mmap.mmap(-1, self.world * max_tables * self.box_bytes)
        self.keep.append(boxes)
        self.boxes = ctypes.addressof(ctypes.c_char.from_buffer(boxes))
        _, lkey, rkey = dev.reg_host(self.boxes, len(boxes), verbs.FLAGS)
        return [self.boxes, rkey, lkey]

    def map_peer(self, pid, fd, size):
        """Register another process's host shard (a memfd) for this GPU's
        loopback gathers; returns its address here and lkey."""
        opened = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR)
        try:
            mapping = mmap.mmap(opened, size, flags=mmap.MAP_SHARED)
        finally:
            os.close(opened)
        self.keep.append(mapping)
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        return address, self.device.reg_host(address, size, verbs.LOCAL)[1]

    def register(self, kind, size, iova, access, fds):
        """Register host memory or HBM; returns its address, lkey and rkey."""
        (fd,) = fds
        try:
            if kind == "host":
                mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                self.keep.append(mapping)
                return address, *self.device.reg_host(address, size, access)[1:]
            return iova, *self.device.reg_dmabuf(fd, iova, size, access)[1:]
        finally:
            os.close(fd)

    def run(self) -> None:
        """Post each request's WRs from its mailbox, a chain per QP in turn."""
        device, recv_wr = self.device, ctypes.addressof(self.recv_wr)
        wc = ctypes.create_string_buffer(verbs.WC_BYTES * 64)
        device.notify_cq(self.recv_cq)
        while True:
            count = device.poll_cq(self.recv_cq, wc, 64)
            if not count:
                device.cq_event(self.recv_channel)
                device.notify_cq(self.recv_cq)
                continue
            self.retire(block=False)
            for i in range(count):
                status, _, _, _, imm, qpn = struct.unpack_from(
                    "<IIIIII", wc, i * verbs.WC_BYTES + 8
                )
                num_wrs, table = imm & 0xFFFFFF, imm >> 24
                peer, qp = self.receivers[qpn]
                try:
                    device.post_recv(qp, recv_wr)
                    if status:
                        print(
                            f"Engram pool RDMA request from GPU {peer} failed: "
                            f"ibv_wc status {status}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    box = self.boxes
                    box += (peer * self.max_tables + table) * self.box_bytes
                    qps = self.qps[peer]
                    chain = chain_length(num_wrs, len(qps))
                    for j, head in enumerate(range(0, num_wrs, chain)):
                        self.post(qps[j % len(qps)], box + head * verbs.SEND_WR_BYTES)
                except Exception:
                    traceback.print_exc()

    def post(self, qp: int, wr: int) -> None:
        device = self.device
        while error := device._post(qp, wr, device._bad_ref):
            if error != errno.ENOMEM:
                raise RuntimeError(f"ibv_post_send: {os.strerror(error)}")
            # The send queue is full: resume at the first WR that did not fit.
            wr = device._bad.value or 0
            self.retire(block=True)

    def retire(self, block: bool) -> None:
        """Reap completed gathers, freeing their send queue slots; with
        `block`, sleep until one completes."""
        device, wc = self.device, self.data_wc
        reaped = 0
        while count := device.poll_cq(self.data_cq, wc, 64):
            _failed(wc, count, "gather")
            reaped += count
        if reaped or not block:
            return
        device.notify_cq(self.data_cq)
        while not (count := device.poll_cq(self.data_cq, wc, 64)):
            device.cq_event(self.data_channel)
            device.notify_cq(self.data_cq)
        _failed(wc, count, "gather")


def main() -> None:
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != parent:
        return
    # Native crashes (ctypes into verbs) print where it was.
    faulthandler.enable()
    Owner(socket.socket(fileno=int(sys.argv[1]))).serve()
    # Its worker sleeps in a C call; leave it there.
    os._exit(0)


if __name__ == "__main__":
    main()
