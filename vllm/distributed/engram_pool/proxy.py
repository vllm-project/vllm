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
   mailbox with RDMA WRITEs, writes the mailbox header, then rings the owner
   with a WRITE_WITH_IMM.
3. The owner's proxy posts the WRs as they are, so its NIC gathers the rows.
   Rows in host memory of the requester's node are gathered the same way by
   the requester's own proxy over loopback QPs, so no GPU reads host memory.
   Once they complete, it writes `seq` (`-seq` if any failed) into that
   owner's flag in the requester's HBM, which the requester's GPU waits on:
   the flag lands after the rows, and the GPU sees its own memory in order.

The proxy never spins: it sleeps in epoll until something needs it. Its GPU
posts a semaphore in the control block from a CUDA host function (libc's
`sem_post`) once go is up; a peer's WRITE_WITH_IMM completes a receive, and
the CQ's completion channel turns that, or the completion of one of its own
WRs, into an event.

Run as a script: only the standard library and `verbs.py` are imported.
"""

import ctypes
import faulthandler
import json
import mmap
import os
import select
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

# Shared control block, in int64 words: go and the doorbell peers ring on one
# line, done (`-seq` if this proxy failed a request) on the next, then each
# owner's first slot and its first WR in the GPU's WRs (world + 1 each).
GO, DOORBELL, DONE, STARTS, WR_STARTS = 0, 4, 16, 32, 544
# The GPU's process-shared semaphore follows the words.
SEM_OFFSET = 8192
CTRL_BYTES = SEM_OFFSET + 64
# A mailbox: a header word, then the requester's WRs and their SGEs.
HEADER_BYTES, SGE_BYTES = 64, 16
# WRs per posted chain at most; the requester's GPU signals each chain's last.
MAX_CHAIN = 64
# Receive WRs kept posted on each QP that peers ring.
RECV_DEPTH = 32
_PR_SET_PDEATHSIG = 1


# Completion ids of this proxy's own posts and receives; owners' WRs carry the
# requester.
_REQUEST, _ARRIVAL, _NOTICE = -1, -2, -3
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
        self.cq, self.channel, channel_fd = dev.create_cq(min(dev.max_cqe, 1 << 16))
        # The GPU's semaphore wakes a helper thread, which wakes the loop.
        self.sem = ctypes.addressof(ctypes.c_char.from_buffer(self.ctrl)) + SEM_OFFSET
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.wake = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.epoll = select.epoll()
        for fd in (channel_fd, self.wake):
            self.epoll.register(fd, select.EPOLLIN)
        self.channel_fd = channel_fd
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
                qp, qpn = dev.create_qp(self.cq, depth, sge_cap, 64, RECV_DEPTH)
                self.qps[peer].append(len(self.qp_handles))
                self.qp_of[qpn] = len(self.qp_handles)
                self.qp_handles.append(qp)
                if peer != -1:
                    qpns[peer].append(qpn)
        self.loopback = self.qps.pop(-1)
        self.inflight = [0] * len(self.qp_handles)
        self.pending = 0
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
        # Requesters ring this GPU on their first QP to it.
        self.recv_wr = (ctypes.c_uint64 * 4)(_NOTICE & _U64, 0, 0, 0)
        for qp in [self.qps[p][0] for p in self.peers] + [self.loopback[0]]:
            for _ in range(RECV_DEPTH):
                self.device.post_recv(handles[qp], ctypes.addressof(self.recv_wr))
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
        # Peers ring this GPU's doorbell in its control block.
        ctrl = ctypes.addressof(ctypes.c_char.from_buffer(self.ctrl))
        _, _, ctrl_rkey = self.device.reg_host(ctrl, CTRL_BYTES, verbs.FLAGS)
        return [self.boxes, rkey, ctrl, ctrl_rkey]

    def link(self, desc, desc_lkey, peers):
        """Requests read this GPU's WRs and SGEs from `desc`; `peers` maps each
        peer to its mailboxes, flags and control block, each with its rkey."""
        self.desc, self.desc_lkey = desc, desc_lkey
        self.posts = {}
        for peer, (boxes, rkey, ctrl, ctrl_rkey, flags, flags_rkey) in peers.items():
            mine = boxes + self.rank * self.box_bytes
            remotes = (
                (mine + HEADER_BYTES, rkey),
                (mine + HEADER_BYTES + self.max_wrs * verbs.SEND_WR_BYTES, rkey),
                (mine, rkey),
                (ctrl + DOORBELL * 8, ctrl_rkey),
            )
            # Copy WRs, copy SGEs, write the header, ring the doorbell: in
            # order on one QP.
            wrs = (ctypes.c_uint64 * 64)()
            sges = (ctypes.c_uint64 * 8)()
            header = (ctypes.c_uint64 * 1)()
            bell = (ctypes.c_int64 * 1)()
            sges[4], sges[5] = ctypes.addressof(header), 8
            sges[6], sges[7] = ctypes.addressof(bell), 8
            for i, (remote, key) in enumerate(remotes):
                wr = 16 * i
                wrs[wr] = _REQUEST & _U64
                wrs[wr + 1] = ctypes.addressof(wrs) + 128 * (i + 1) if i < 3 else 0
                wrs[wr + 2] = ctypes.addressof(sges) + 16 * i
                wrs[wr + 3] = 1  # one SGE, IBV_WR_RDMA_WRITE
                wrs[wr + 4] = verbs.SEND_INLINE if i >= 2 else 0
                wrs[wr + 5], wrs[wr + 6] = remote, key
            # The doorbell completes a receive at the owner, which wakes it.
            wrs[16 * 3 + 3] = 1 | verbs.WR_WRITE_WITH_IMM << 32
            wrs[16 * 3 + 4] |= verbs.SEND_SIGNALED
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
            self.posts[int(peer)] = (wrs, sges, header, bell, arrival, flag, flag_sge)

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
        threading.Thread(target=self.wait_gpu, daemon=True).start()

    def wait_gpu(self) -> None:
        """Turn each post of the GPU's semaphore into a wakeup of the loop."""
        sem_wait = self.libc.sem_wait
        while self.running:
            if sem_wait(ctypes.c_void_p(self.sem)) == 0:
                os.eventfd_write(self.wake, 1)

    def loop(self) -> None:
        words, seq = self.words, 0
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
                continue
            # Sleep until the GPU posts its semaphore or the CQ completes
            # anything; a completion before the CQ was armed raises no event.
            self.device.notify_cq(self.cq)
            if self.poll():
                continue
            for fd, _ in self.epoll.poll(1.0):
                if fd == self.wake:
                    os.eventfd_read(self.wake)
                else:
                    self.device.cq_event(self.channel)

    def post(self, qp: int, wr: int) -> None:
        while self.inflight[qp] >= self.capacity:
            self.poll()
        device = self.device
        if error := device._post(self.qp_handles[qp], wr, device._bad_ref):
            raise RuntimeError(f"ibv_post_send failed: {os.strerror(error)}")
        self.inflight[qp] += 1
        self.pending += 1

    def request(self, seq: int) -> None:
        """Hand each owner its WRs of this fetch."""
        words = self.words
        sges_at = self.desc + self.max_wrs * verbs.SEND_WR_BYTES
        lkey = self.desc_lkey << 32
        for peer in self.qps:
            first_wr = words[WR_STARTS + peer]
            num_wrs = words[WR_STARTS + peer + 1] - first_wr
            if not num_wrs:
                continue
            wrs, sges, header, bell, _, _, _ = self.posts[peer]
            chain = chain_length(num_wrs, len(self.qps[peer]))
            first = words[STARTS + peer]
            sges[0] = self.desc + first_wr * verbs.SEND_WR_BYTES
            sges[1] = num_wrs * verbs.SEND_WR_BYTES | lkey
            sges[2] = sges_at + first * SGE_BYTES
            sges[3] = (words[STARTS + peer + 1] - first) * SGE_BYTES | lkey
            # One word, so an owner never sees half a request.
            header[0] = (seq & 0xFFFFFFFF) << 32 | chain << 24 | num_wrs
            bell[0] = seq
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
            status, opcode, vendor, _, _, qpn = struct.unpack_from(
                "<IIIIII", raw, i * verbs.WC_BYTES + 8
            )
            if opcode >= verbs.WC_RECV:
                # A requester rang; answer() reads its mailbox.
                handle = self.qp_handles[self.qp_of[qpn]]
                self.device.post_recv(handle, ctypes.addressof(self.recv_wr))
                continue
            self.inflight[self.qp_of[qpn]] -= 1
            self.pending -= 1
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
                _, _, _, _, arrival, flag, _ = self.posts[wr_id]
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
            os.eventfd_write(proxy.wake, 1)
            proxy.thread.join()


if __name__ == "__main__":
    main()
