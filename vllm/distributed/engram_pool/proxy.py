# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA proxies: per GPU, a requester process and an owner process that move
table rows between GPUs.

Each has its own GIL and one thread that moves rows (its command thread sleeps
in `recv` once set up), so neither waits on the other or on a CPU-busy model
thread, and captured fetches need no host thread at all. The GPU
process sends commands over a socket (fds by SCM_RIGHTS); fetches run through
memory shared with the GPU. For a GPU fetching rows (requester) from GPUs
holding them (owners):

1. The requester's GPU writes, per owner and table, chained RDMA WRITEs whose
   SGEs gather that owner's rows, a packet's worth each, into the requester's
   landing buffer, with the owner proxy's pointers. Each QP's last chain links
   a READ of the requester's memory, which returns only once the rows before
   it have landed, then a fenced inline WRITE of `seq` into that QP's flag for
   the table. Then the GPU raises `go = seq` (if any row is remote) and
   `issued`. A CUDA host function launched as the fetch starts posts a
   semaphore.
2. The requester process, asleep on that semaphore, waits for `issued` if
   woken before it, then posts the WRITE_WITH_IMM the GPU wrote for each owner
   and table, which copies its WRs and SGEs into that owner's mailbox.
3. The owner process, asleep on its receive CQ's completion channel, posts the
   WRs as they are: its NIC gathers the rows, then writes the flags. Rows in
   host memory of the requester's node are gathered the same way by the
   requester's own owner process over loopback QPs, so no GPU reads host
   memory.

Each process wakes once per fetch and never spins.

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
import time
import traceback

if __package__:
    from vllm.distributed.engram_pool import verbs
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import verbs  # type: ignore[no-redef]

# Shared control block, in int64 words: go, issued (fetches started), done
# (`-seq` if the requester failed), then per (table, gatherer) its first slot
# and first WR (one more each), its WRs that gather rows, and the request the
# GPU wrote for it (REQUEST_WORDS).
GO, ISSUED, DONE, STARTS, WR_STARTS = 0, 8, 16, 32, 544
NUM_WRS, REQUESTS = 1056, 1568
REQUEST_WORDS = 24
# The GPU's process-shared semaphore follows the words.
SEM_OFFSET = (REQUESTS + 512 * REQUEST_WORDS) * 8
CTRL_BYTES = SEM_OFFSET + 64
SGE_BYTES = 16
# WRs per posted chain; the requester's GPU signals each chain's last. Few WRs
# take one chain: each more QP costs a post and a READ round trip.
MIN_CHAIN, MAX_CHAIN = 16, 64
# Receive WRs kept posted for each requester, and requests in flight per QP.
RECV_DEPTH, REQUEST_DEPTH = 32, 64
_PR_SET_PDEATHSIG = 1
_NOTICE = -1
_U64 = 0xFFFFFFFFFFFFFFFF


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


class _Proxy:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.keep: list = []

    def _open(self, device: str, world: int, rank: int) -> verbs.Device:
        self.device = verbs.Device(device)
        # CPUs of another NUMA node post twice as slowly.
        if cpus := verbs.local_cpus(device) & os.sched_getaffinity(0):
            os.sched_setaffinity(0, cpus)
        self.world, self.rank = world, rank
        self.gatherers = [p for p in range(world) if p != rank] + [rank]
        return self.device

    def serve(self) -> None:
        while True:
            message, fds, _, _ = socket.recv_fds(self.sock, 1 << 20, 4)
            if not message:
                return
            command, *args = json.loads(message)
            try:
                if command in ("init", "register"):
                    args.append(fds)
                result = ("ok", getattr(self, command)(*args))
            except Exception:
                result = ("error", traceback.format_exc())
            self.sock.sendall(json.dumps(result).encode())

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        raise NotImplementedError


class Requester(_Proxy):
    """Hands each gatherer its WRs of every fetch."""

    def init(self, device, world, rank, max_tables, fds):
        dev = self._open(device, world, rank)
        self.ctrl = mmap.mmap(fds[0], CTRL_BYTES, flags=mmap.MAP_SHARED)
        os.close(fds[0])
        self.words = memoryview(self.ctrl).cast("B").cast("q")
        self.sem = ctypes.addressof(ctypes.c_char.from_buffer(self.ctrl)) + SEM_OFFSET
        self.cq, _ = dev.create_cq(len(self.gatherers) * REQUEST_DEPTH)
        made = {
            p: dev.create_qp(self.cq, self.cq, REQUEST_DEPTH, 2) for p in self.gatherers
        }
        self.qps = {p: qp for p, (qp, _) in made.items()}
        # The first table first, and remote owners first: their requests have
        # further to go.
        requests = ctypes.addressof(ctypes.c_char.from_buffer(self.ctrl))
        requests += REQUESTS * 8
        order = [t * world + p for t in range(max_tables) for p in self.gatherers]
        self.posts = [
            (vg, self.qps[vg % world], requests + vg * REQUEST_WORDS * 8)
            for vg in order
        ]
        return {"qpns": [made[p][1] for p in range(world)], "requests": requests}

    def connect(self, owners):
        """Connect to each gatherer's owner process: (gatherer, address, QP)."""
        for peer, (gid, lid), remote in owners:
            self.device.connect(self.qps[peer], remote, (bytes.fromhex(gid), lid))

    def register(self, iova, size, fds):
        """Register this GPU's descriptors (dmabuf); returns their lkey."""
        (fd,) = fds
        try:
            return self.device.reg_dmabuf(fd, iova, size, verbs.LOCAL)[1]
        finally:
            os.close(fd)

    def run(self) -> None:
        """Once the GPU raises go, post the request it wrote for each gatherer
        and table with rows of the fetch."""
        libc = ctypes.CDLL(None, use_errno=True)
        sched_yield, monotonic = libc.sched_yield, time.monotonic
        sem_wait, sem = libc.sem_wait, ctypes.c_void_p(self.sem)
        lock, unlock = libc.pthread_mutex_lock, libc.pthread_mutex_unlock
        mutex = ctypes.create_string_buffer(64)
        libc.pthread_mutex_init(mutex, None)
        lock(mutex)
        words, device = self.words, self.device
        bad = ctypes.byref(ctypes.c_void_p())
        wc = ctypes.create_string_buffer(verbs.WC_BYTES * 64)
        seq = issued = 0
        while True:
            if sem_wait(sem):
                continue
            # The wakeup was launched as the fetch started: wait out its few
            # kernels if they are still running (napping if that takes long).
            deadline = monotonic() + 2e-4
            while words[ISSUED] == issued:
                sched_yield() if monotonic() < deadline else time.sleep(5e-4)
            issued = words[ISSUED]
            # Fetches with no remote rows leave go as it was.
            if words[GO] == seq:
                continue
            seq = words[GO]
            # Release then acquire: read what go publishes only after go.
            unlock(mutex)
            lock(mutex)
            try:
                while count := device.poll_cq(self.cq, wc, 64):
                    if _failed(wc, count, "request"):
                        raise RuntimeError("Engram pool RDMA request failed")
                for peer, qp, at in self.posts:
                    if words[NUM_WRS + peer] and (error := device._post(qp, at, bad)):
                        raise RuntimeError(f"ibv_post_send: {os.strerror(error)}")
            except Exception:
                traceback.print_exc()
                words[DONE] = -seq


class Owner(_Proxy):
    """Posts requesters' WRs, so its NIC gathers rows of this GPU's shard and
    of its node's host shards; rows and flags of this GPU land through it."""

    def init(self, device, world, rank, qps_per_peer, sq_depth, fds):
        dev = self._open(device, world, rank)
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
        self.bad = ctypes.c_void_p()
        self.bad_ref = ctypes.byref(self.bad)
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
        self.recv_wr = (ctypes.c_uint64 * 4)(_NOTICE & _U64, 0, 0, 0)
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
        while error := self.device._post(qp, wr, self.bad_ref):
            if error != errno.ENOMEM:
                raise RuntimeError(f"ibv_post_send: {os.strerror(error)}")
            # The send queue is full: resume at the first WR that did not fit.
            wr = self.bad.value or 0
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
    role = {"requester": Requester, "owner": Owner}[sys.argv[2]]
    role(socket.socket(fileno=int(sys.argv[1]))).serve()
    # Its worker sleeps in a C call; leave it there.
    os._exit(0)


if __name__ == "__main__":
    main()
