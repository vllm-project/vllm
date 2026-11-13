# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA proxy: one process per GPU that moves table rows between GPUs.

It has its own GIL, so a CPU-busy model thread never delays a transfer, and
captured fetches need no host thread at all. The GPU process sends commands
over a socket (fds by SCM_RIGHTS); fetches run through memory it shares with
its GPU. For a GPU fetching rows (requester) from GPUs holding them (owners):

1. The requester's GPU writes, per owner, chained RDMA WRITEs whose SGEs gather
   that owner's rows, a packet's worth each, into the requester's landing
   buffer, with the owner proxy's pointers. Then it raises `go = seq`.
2. The requester's proxy copies each owner's WRs and SGEs into that owner's
   mailbox with RDMA WRITEs, then rings it by writing the mailbox header.
3. The owner's proxy posts the WRs as they are, so its NIC gathers the rows.
   Rows in host memory of the requester's node are gathered the same way by
   the requester's own proxy over loopback QPs, so no GPU reads host memory.
   Once they complete, it writes `seq` (`-seq` if any failed) into that
   owner's flag in the requester's HBM, which the requester's GPU waits on:
   the flag lands after the rows, and the GPU sees its own memory in order.

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

# Shared control block, in int64 words: go and done (`-seq` if this proxy
# failed a request) on their own lines, then each owner's first slot and its
# first WR in the GPU's WRs (world + 1 each).
GO, DONE, STARTS, WR_STARTS = 0, 16, 32, 544
CTRL_BYTES = 8192
# A mailbox: a header word, then the requester's WRs and their SGEs.
HEADER_BYTES, SGE_BYTES = 64, 16
# WRs per posted chain at most; the requester's GPU signals each chain's last.
MAX_CHAIN = 64
# Spin this long after the last activity, then nap between checks.
SPIN_S, NAP_S = 200e-6, 30e-6
_PR_SET_PDEATHSIG, _PR_SET_TIMERSLACK = 1, 29
# Completion ids of this proxy's own posts; owners' WRs carry the requester.
_REQUEST, _ARRIVAL = -1, -2
_U64 = 0xFFFFFFFFFFFFFFFF


def chain_length(num_wrs: int, qps: int) -> int:
    """WRs per chain when an owner posts `num_wrs` WRs over `qps` QPs."""
    return min(MAX_CHAIN, max(1, -(-num_wrs // qps)))


def mailbox_bytes(max_wrs: int, max_items: int) -> int:
    return HEADER_BYTES + max_wrs * verbs.SEND_WR_BYTES + max_items * SGE_BYTES


class Proxy:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.keep: list = []
        self.thread: threading.Thread | None = None
        self.running = True

    def init(self, device, world, rank, qps_per_peer, sq_depth, fds):
        self.device = dev = verbs.Device(device)
        # CPUs of another NUMA node post twice as slowly.
        if cpus := verbs.local_cpus(device) & os.sched_getaffinity(0):
            os.sched_setaffinity(0, cpus)
        self.world, self.rank = world, rank
        self.ctrl = mmap.mmap(fds[0], CTRL_BYTES, flags=mmap.MAP_SHARED)
        os.close(fds[0])
        self.words = memoryview(self.ctrl).cast("B").cast("q")
        sge_cap = min(dev.max_sge, 16)
        self.peers = [p for p in range(world) if p != rank]
        self.cq = dev.create_cq(min(dev.max_cqe, 1 << 16))
        # QPs by index: their handles, each peer's, and posts in flight.
        self.qp_handles: list[int] = []
        self.qps: dict[int, list[int]] = {}
        qpns: list[list[int]] = [[] for _ in range(world)]
        self.qp_of: dict[int, int] = {}
        depth = min(sq_depth, dev.max_qp_wr)
        # Peers' QPs, then loopback pairs: this GPU gathers its node's rows.
        for peer in self.peers + [rank, -1]:
            self.qps[peer] = []
            for _ in range(qps_per_peer):
                qp, qpn = dev.create_qp(self.cq, depth, sge_cap, 64)
                self.qps[peer].append(len(self.qp_handles))
                self.qp_of[qpn] = len(self.qp_handles)
                self.qp_handles.append(qp)
                if peer != -1:
                    qpns[peer].append(qpn)
        self.loopback = self.qps.pop(-1)
        self.inflight = [0] * len(self.qp_handles)
        # Posts a QP may have in flight before its send queue fills.
        self.capacity = max(1, depth // (MAX_CHAIN + 3))
        self.wc = ctypes.create_string_buffer(verbs.WC_BYTES * 64)
        gid, lid = dev.address()
        return {
            "address": [gid.hex(), lid],
            "qpns": qpns,
            "max_sge": sge_cap,
            "mtu": verbs.MTU_BYTES.get(dev.mtu, 1024),
            "device": dev.describe(),
        }

    def connect(self, peers, max_wrs, max_items):
        """Connect each peer's QPs (`peers` lists the peer, its address and its
        QPs for this GPU); returns this GPU's mailboxes and arrival words."""
        for peer, (gid, lid), theirs in peers:
            for qp, remote in zip(self.qps[peer], theirs):
                self.device.connect(
                    self.qp_handles[qp], remote, (bytes.fromhex(gid), lid)
                )
        handles, numbers = self.qp_handles, {i: n for n, i in self.qp_of.items()}
        for qp, partner in zip(self.qps[self.rank], self.loopback):
            self.device.connect(handles[qp], numbers[partner], self.device.address())
            self.device.connect(handles[partner], numbers[qp], self.device.address())
        self.max_wrs = max_wrs
        self.box_bytes = mailbox_bytes(max_wrs, max_items)
        # Every requester's mailbox, then one arrival word per owner.
        boxes = mmap.mmap(-1, self.world * self.box_bytes)
        self.keep.append(boxes)
        self.boxes = ctypes.addressof(ctypes.c_char.from_buffer(boxes))
        # Headers must land after the WRs they ring.
        _, _, rkey = self.device.reg_host(self.boxes, len(boxes), verbs.FLAGS)
        self.box_words = memoryview(boxes).cast("B").cast("q")
        self.served = [0] * self.world
        self.answering: dict[int, list[int]] = {}
        return [self.boxes, rkey]

    def link(self, desc, desc_lkey, peers):
        """Requests read this GPU's WRs and SGEs from `desc`; `peers` maps each
        peer to its mailboxes, their rkey, its flags and their rkey."""
        self.desc, self.desc_lkey = desc, desc_lkey
        self.posts = {}
        for peer, (boxes, rkey, flags, flags_rkey) in peers.items():
            mine = boxes + self.rank * self.box_bytes
            remotes = (
                mine + HEADER_BYTES,
                mine + HEADER_BYTES + self.max_wrs * verbs.SEND_WR_BYTES,
                mine,
            )
            # Copy WRs, copy SGEs, then ring the header: in order on one QP.
            wrs = (ctypes.c_uint64 * 48)()
            sges = (ctypes.c_uint64 * 6)()
            header = (ctypes.c_uint64 * 1)()
            sges[4], sges[5] = ctypes.addressof(header), 8
            for i, remote in enumerate(remotes):
                wr = 16 * i
                wrs[wr] = _REQUEST & _U64
                wrs[wr + 1] = ctypes.addressof(wrs) + 128 * (i + 1) if i < 2 else 0
                wrs[wr + 2] = ctypes.addressof(sges) + 16 * i
                wrs[wr + 3] = 1  # one SGE, IBV_WR_RDMA_WRITE
                wrs[wr + 4] = verbs.SEND_SIGNALED | verbs.SEND_INLINE if i == 2 else 0
                wrs[wr + 5], wrs[wr + 6] = remote, rkey
            # This GPU's flag in the peer's HBM.
            flag = (ctypes.c_int64 * 1)()
            flag_sge = (ctypes.c_uint64 * 2)(ctypes.addressof(flag), 8)
            arrival = (ctypes.c_uint64 * 16)()
            arrival[0] = _ARRIVAL & _U64
            arrival[2] = ctypes.addressof(flag_sge)
            arrival[3] = 1
            arrival[4] = verbs.SEND_SIGNALED | verbs.SEND_INLINE
            arrival[5] = flags + self.rank * 8
            arrival[6] = flags_rkey
            self.posts[int(peer)] = (wrs, sges, header, arrival, flag, flag_sge)

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

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self) -> None:
        words, seq, idle = self.words, 0, time.monotonic()
        while self.running:
            busy = False
            try:
                if words[GO] != seq:
                    seq = words[GO]
                    busy = True
                    self.request(seq)
                busy |= self.answer()
                busy |= self.poll() > 0
            except Exception:
                traceback.print_exc()
                words[DONE] = -seq
            if busy:
                idle = time.monotonic()
            elif time.monotonic() - idle > SPIN_S:
                time.sleep(NAP_S)

    def post(self, qp: int, wr: int) -> None:
        while self.inflight[qp] >= self.capacity:
            self.poll()
        device = self.device
        if error := device._post(self.qp_handles[qp], wr, device._bad_ref):
            raise RuntimeError(f"ibv_post_send failed: {os.strerror(error)}")
        self.inflight[qp] += 1

    def request(self, seq: int) -> None:
        """Hand each owner its WRs of this fetch."""
        words, world = self.words, self.world
        starts = words[STARTS : STARTS + world + 1].tolist()
        wr_starts = words[WR_STARTS : WR_STARTS + world + 1].tolist()
        sges_at = self.desc + self.max_wrs * verbs.SEND_WR_BYTES
        lkey = self.desc_lkey << 32
        for peer in self.qps:
            num_wrs = wr_starts[peer + 1] - wr_starts[peer]
            if not num_wrs:
                continue
            wrs, sges, header, _, _, _ = self.posts[peer]
            chain = chain_length(num_wrs, len(self.qps[peer]))
            rows = starts[peer + 1] - starts[peer]
            sges[0] = self.desc + wr_starts[peer] * verbs.SEND_WR_BYTES
            sges[1] = num_wrs * verbs.SEND_WR_BYTES | lkey
            sges[2] = sges_at + starts[peer] * SGE_BYTES
            sges[3] = rows * SGE_BYTES | lkey
            # One word, so an owner never sees half a request.
            header[0] = (seq & 0xFFFFFFFF) << 32 | chain << 24 | num_wrs
            self.post(self.qps[peer][0], ctypes.addressof(wrs))

    def answer(self) -> bool:
        """Post the WRs of requests waiting in this GPU's mailboxes."""
        busy = False
        stride = self.box_bytes // 8
        for peer in self.qps:
            word = self.box_words[peer * stride]
            if word == self.served[peer]:
                continue
            self.served[peer] = word
            num_wrs, chain = word & 0xFFFFFF, word >> 24 & 0xFF
            first = self.boxes + peer * self.box_bytes + HEADER_BYTES
            heads = range(0, num_wrs, chain)
            self.answering[peer] = [word >> 32 & 0xFFFFFFFF, len(heads), 0]
            qps = self.qps[peer]
            for i, head in enumerate(heads):
                self.post(qps[i % len(qps)], first + head * verbs.SEND_WR_BYTES)
            busy = True
        return busy

    def poll(self) -> int:
        count = self.device.poll_cq(self.cq, self.wc, 64)
        raw = self.wc.raw
        for i in range(count):
            (wr_id,) = struct.unpack_from("<q", raw, i * verbs.WC_BYTES)
            status, _, vendor, _, _, qpn = struct.unpack_from(
                "<IIIIII", raw, i * verbs.WC_BYTES + 8
            )
            self.inflight[self.qp_of[qpn]] -= 1
            if status:
                print(
                    f"Engram pool RDMA WRITE {wr_id} failed: ibv_wc status "
                    f"{status}, vendor error {vendor:#x}",
                    file=sys.stderr,
                    flush=True,
                )
                if wr_id == _REQUEST:
                    raise RuntimeError("Engram pool RDMA request failed")
            if wr_id < 0:
                continue
            state = self.answering[wr_id]
            state[1] -= 1
            state[2] |= status
            if not state[1]:
                _, _, _, arrival, flag, _ = self.posts[wr_id]
                flag[0] = -state[0] if state[2] else state[0]
                self.post(self.qps[wr_id][0], ctypes.addressof(arrival))
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
