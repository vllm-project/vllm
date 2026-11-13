# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA proxy: one process per GPU that posts the READs its GPU writes.

It has its own GIL, so a CPU-busy model thread never delays a transfer, and
captured fetches need no host thread at all. The GPU process sends commands
over a socket (fds by SCM_RIGHTS); a fetch is driven through shared memory:

1. The GPU writes chained `ibv_send_wr`s (with this process's addresses), each
   peer's start slot, then `go = seq`.
2. This process posts every peer's chains round-robin over its QPs, polls the
   completions and writes `done = seq` (`-seq` if any READ failed).

Run as a script: only the standard library and `verbs.py` are imported.
"""

import ctypes
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

# Shared control block, in int64 words. go and done sit on their own lines.
GO, DONE, STARTS = 0, 16, 32
WR_OFFSET = 4096
# WRs per posted chain; the GPU signals only each chain's last. Short chains
# spread a small fetch over more QPs, long ones cut posting calls.
SHORT_CHAIN, LONG_CHAIN, LONG_FROM = 64, 256, 2048


def chain_length(rows: int) -> int:
    """WRs per chain for a peer that serves `rows` rows of a fetch."""
    return LONG_CHAIN if rows >= LONG_FROM else SHORT_CHAIN


# Spin this long after the last fetch, then nap between checks.
SPIN_S, NAP_S = 200e-6, 30e-6
_PR_SET_PDEATHSIG, _PR_SET_TIMERSLACK = 1, 29


class Proxy:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.keep: list = []
        self.thread: threading.Thread | None = None
        self.running = True

    def init(self, device, world, rank, qps_per_peer, sq_depth, sizes, fds):
        self.device = dev = verbs.Device(device)
        self.world, self.rank = world, rank
        self.ctrl = mmap.mmap(fds[0], sizes[0], flags=mmap.MAP_SHARED)
        self.staging = mmap.mmap(fds[1], sizes[1], flags=mmap.MAP_SHARED)
        for fd in fds:
            os.close(fd)
        self.words = memoryview(self.ctrl).cast("B").cast("q")
        ctrl_va = ctypes.addressof(ctypes.c_char.from_buffer(self.ctrl))
        staging_va = ctypes.addressof(ctypes.c_char.from_buffer(self.staging))
        _, lkey, _ = dev.reg_host(staging_va, sizes[1])
        self.wr_va = ctrl_va + WR_OFFSET
        self.sq_depth = min(sq_depth, dev.max_qp_wr)
        # Chains a QP may have in flight before its SQ fills.
        self.capacity = self.sq_depth // LONG_CHAIN
        self.peers = [p for p in range(world) if p != rank]
        depth = self.capacity * qps_per_peer * max(len(self.peers), 1)
        self.cq = dev.create_cq(min(depth, dev.max_cqe))
        self.qps: dict[int, list[int]] = {}
        qpns: list[list[int]] = [[] for _ in range(world)]
        self.qp_of: dict[int, int] = {}
        for peer in self.peers:
            self.qps[peer] = []
            for _ in range(qps_per_peer):
                qp, qpn = dev.create_qp(self.cq, self.sq_depth)
                self.qps[peer].append(qp)
                qpns[peer].append(qpn)
                self.qp_of[qpn] = qp
        self.wc = ctypes.create_string_buffer(verbs.WC_BYTES * 64)
        gid, lid = dev.address()
        return {
            "address": [gid.hex(), lid],
            "qpns": qpns,
            "ctrl_va": ctrl_va,
            "staging_va": staging_va,
            "lkey": lkey,
            "device": dev.describe(),
        }

    def connect(self, peers):
        """`peers` lists each peer, its address and its QPs for this GPU."""
        for peer, (gid, lid), theirs in peers:
            for qp, remote in zip(self.qps[peer], theirs):
                self.device.connect(qp, remote, (bytes.fromhex(gid), lid))

    def register(self, kind, size, iova, fds):
        """Register a shard for peers' READs; returns its address and rkey."""
        (fd,) = fds
        try:
            if kind == "host":
                mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                self.keep.append(mapping)
                _, _, rkey = self.device.reg_host(address, size)
                return address, rkey
            _, _, rkey = self.device.reg_dmabuf(fd, iova, size)
            return iova, rkey
        finally:
            os.close(fd)

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self) -> None:
        words, last, idle = self.words, 0, time.monotonic()
        while self.running:
            seq = words[GO]
            if seq == last:
                if time.monotonic() - idle > SPIN_S:
                    time.sleep(NAP_S)
                continue
            last = seq
            try:
                self.transfer()
                words[DONE] = seq
            except Exception:
                traceback.print_exc()
                words[DONE] = -seq
            idle = time.monotonic()

    def transfer(self) -> None:
        words, post = self.words, self.device.post_send
        starts = words[STARTS : STARTS + self.world + 1].tolist()
        # Each peer's chains, interleaved so every peer starts at once.
        queues = [
            (
                self.qps[p],
                range(
                    starts[p], starts[p + 1], chain_length(starts[p + 1] - starts[p])
                ),
            )
            for p in self.peers
            if starts[p + 1] > starts[p]
        ]
        inflight = dict.fromkeys(self.qp_of.values(), 0)
        pending = index = 0
        while queues:
            remaining = []
            for qps, chunks in queues:
                if index >= len(chunks):
                    continue
                remaining.append((qps, chunks))
                qp = qps[index % len(qps)]
                while inflight[qp] >= self.capacity:
                    pending -= self.poll(inflight)
                post(qp, self.wr_va + chunks[index] * verbs.SEND_WR_BYTES)
                inflight[qp] += 1
                pending += 1
            queues = remaining
            index += 1
        while pending:
            pending -= self.poll(inflight)

    def poll(self, inflight) -> int:
        count = self.device.poll_cq(self.cq, self.wc, 64)
        raw = self.wc.raw
        for i in range(count):
            status, _, vendor, _, _, qpn = struct.unpack_from(
                "<IIIIII", raw, i * verbs.WC_BYTES + 8
            )
            if status:
                qp = self.qp_of[qpn]
                peer = next(p for p, qps in self.qps.items() if qp in qps)
                raise RuntimeError(
                    f"RDMA READ from peer {peer} failed: ibv_wc status {status}, "
                    f"vendor error {vendor:#x}"
                )
            inflight[self.qp_of[qpn]] -= 1
        return count

    def serve(self) -> None:
        while True:
            message, fds, _, _ = socket.recv_fds(self.sock, 1 << 20, 4)
            if not message:
                return
            command, *args = json.loads(message)
            if command == "stop":
                return
            try:
                if command in ("init", "register"):
                    args.append(fds)
                result = ("ok", getattr(self, command)(*args))
            except Exception:
                result = ("error", traceback.format_exc())
            self.sock.sendall(json.dumps(result).encode())


def main() -> None:
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    libc.prctl(_PR_SET_TIMERSLACK, 1000)  # ns: naps stay close to NAP_S
    if os.getppid() != parent:
        return
    # Native crashes (ctypes into verbs) print where every thread was.
    faulthandler.enable()
    proxy = Proxy(socket.socket(fileno=int(sys.argv[1])))
    try:
        proxy.serve()
    finally:
        proxy.running = False
        if proxy.thread is not None:
            proxy.thread.join()


if __name__ == "__main__":
    main()
