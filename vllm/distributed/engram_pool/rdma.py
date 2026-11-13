# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table rows from every peer's shard over RDMA, gathered by the owner's NIC.

A READ per row would cap a NIC at its message rate (~45M/s here, a fraction of
line rate for small rows). Instead each owner's NIC gathers a packet's worth of
rows per RDMA WRITE (one SGE per row) and writes them into the requester's
landing buffer. The requester's GPU builds those WRs; the proxies (`proxy.py`)
carry them to the owners and post them.

A fetch is five stream-ordered kernels and a host function, so it captures into
CUDA graphs:
1. `_plan_kernel` copies rows this GPU or its node's GPUs own straight into
   the output, zeroes dead ones, and numbers the rest within their owner's group.
2. `_wr_kernel` writes each owner's gathering WRs and SGEs, in the owner proxy's
   addresses, into this GPU's descriptor buffer.
3. `_go_kernel` (one program) publishes each gatherer's slots and WRs, bumps
   the device-side sequence number and raises the go flag. A CUDA host
   function (libc's `sem_post`) then wakes the sleeping proxy, and
   `_wait_kernel` (one program) holds the stream until every gatherer's flag
   says its rows have landed.
4. `_scatter_kernel` moves the landed rows from owner order into the output.

Shards live in host memory, or in dmabuf-registered HBM.
"""

import ctypes
import json
import os
import socket
import subprocess
import sys

import torch

from vllm.distributed.engram_pool import proxy, verbs
from vllm.distributed.engram_pool.memory import (
    CuMem,
    SharedHost,
    device_view,
    gpu_numa_node,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

SQ_DEPTH = 1024
# A failed or dead proxy must not hang the GPU forever.
TIMEOUT_NS = 60 * 10**9


def pick_device() -> str:
    """An active RDMA device on this GPU's NUMA node; that node's GPUs take
    its devices in turn."""
    devices = verbs.active_devices()
    if not devices:
        raise RuntimeError("Engram pool RDMA transport found no active RDMA device")
    gpu = torch.accelerator.current_device_index()
    node = gpu_numa_node(gpu)
    near = [d for d in devices if verbs.device_numa_node(d) == node] or devices
    peers = [
        i for i in range(torch.accelerator.device_count()) if gpu_numa_node(i) == node
    ]
    return near[peers.index(gpu) % len(near) if gpu in peers else 0]


@triton.jit(do_not_specialize=["num_items"])
def _plan_kernel(
    ids,
    meta,
    out,
    counts,
    owner_pos,
    remote_addr,
    key_out,
    num_items,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy rows of shards mapped here, zero dead ones, and place remote rows
    in their owner's group: `owner_pos` packs (owner, position in group), -1 if
    not remote.

    Items are `[..., TABLES, K]` row ids; `meta` rows hold each table's rows per
    shard, rows, and device arrays of shard bases and lkeys for their NICs,
    mapped shards and gatherers (words 1 to 6). Remote rows are grouped by the
    GPU whose NIC gathers them.
    """
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = item < num_items
    table = meta + (item // K % TABLES) * 8
    row = tl.load(ids + item, mask=valid, other=-1).to(tl.int64)
    rows_per_shard = tl.load(table + 1, mask=valid, other=1)
    live = valid & (row >= 0) & (row < tl.load(table + 2, mask=valid, other=0))
    owner = tl.where(live, row // rows_per_shard, 0)
    local = row - owner * rows_per_shard
    mapped = tl.load(table + 5, mask=live, other=0).to(tl.pointer_type(tl.int64))
    base = tl.load(mapped + owner, mask=live, other=0)
    direct = live & (base != 0)
    remote = live & ~direct
    shard = base.to(tl.pointer_type(tl.uint64))
    words = tl.arange(0, BLOCK_W)
    in_row = words[None, :] < WORDS
    data = tl.load(
        shard[:, None] + local[:, None] * WORDS + words[None, :],
        mask=direct[:, None] & in_row,
        other=0,
    )
    dst = out.to(tl.pointer_type(tl.uint64)) + item[:, None] * WORDS + words[None, :]
    # Remote rows come from the NIC.
    tl.store(dst, data, mask=(valid & ~remote)[:, None] & in_row)
    gatherers = tl.load(table + 6, mask=remote, other=0).to(tl.pointer_type(tl.int32))
    gatherer = tl.load(gatherers + owner, mask=remote, other=0).to(tl.int64)
    position = tl.atomic_add(counts + gatherer, 1, mask=remote)
    packed = (gatherer << 32) | position.to(tl.int64)
    tl.store(owner_pos + item, tl.where(remote, packed, -1), mask=valid)
    bases = tl.load(table + 3, mask=remote, other=0).to(tl.pointer_type(tl.int64))
    base = tl.load(bases + owner, mask=remote, other=0)
    tl.store(remote_addr + item, base + local * (WORDS * 8), mask=remote)
    keys = tl.load(table + 4, mask=remote, other=0).to(tl.pointer_type(tl.int32))
    tl.store(key_out + item, tl.load(keys + owner, mask=remote, other=0), mask=remote)


@triton.jit(do_not_specialize=["num_items", "rank", "landing", "landing_rkey"])
def _wr_kernel(
    owner_pos,
    counts,
    remote_addr,
    keys,
    boxes,
    desc,
    landing,
    landing_rkey,
    num_items,
    world,
    rank,
    MAX_WRS: tl.constexpr,
    ROWS_PER_WR: tl.constexpr,
    QPS: tl.constexpr,
    MAX_CHAIN: tl.constexpr,
    HEADER_BYTES: tl.constexpr,
    WORLD_P2: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write each remote row's SGE at its slot, and for every ROWS_PER_WR rows
    of an owner one RDMA WRITE to `landing`, chained as `proxy.chain_length`
    says with each chain's last signaled. WR pointers are into the owner's
    mailbox for this GPU (`boxes`)."""
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(owner_pos + item, mask=item < num_items, other=-1)
    remote = packed >= 0
    owner = packed >> 32
    position = packed & 0xFFFFFFFF
    peers = tl.arange(0, WORLD_P2)
    sizes = tl.load(counts + peers, mask=peers < world, other=0).to(tl.int64)
    before = peers[None, :] < owner[:, None]
    start = tl.sum(tl.where(before, sizes[None, :], 0), 1)
    wr_start = tl.sum(tl.where(before, tl.cdiv(sizes, ROWS_PER_WR)[None, :], 0), 1)
    size = tl.load(counts + owner, mask=remote, other=0).to(tl.int64)
    slot = start + position
    # ibv_sge: addr, length | lkey << 32.
    pair = tl.arange(0, 2)[None, :]
    key = tl.load(keys + item, mask=remote, other=0).to(tl.int64) & 0xFFFFFFFF
    sge = tl.where(
        pair == 0,
        tl.load(remote_addr + item, mask=remote, other=0)[:, None],
        (WORDS * 8 | key << 32)[:, None],
    )
    tl.store(desc + MAX_WRS * 16 + slot[:, None] * 2 + pair, sge, mask=remote[:, None])
    # ibv_send_wr words: wr_id, next, sg_list, num_sge | opcode << 32 (WRITE is
    # 0), send_flags, rdma.remote_addr, rdma.rkey.
    first = remote & (position % ROWS_PER_WR == 0)
    index = position // ROWS_PER_WR
    num_wrs = tl.cdiv(size, ROWS_PER_WR)
    chain = tl.minimum(tl.maximum(tl.cdiv(num_wrs, QPS), 1), MAX_CHAIN)
    last = (index % chain == chain - 1) | (index == num_wrs - 1)
    box = tl.load(boxes + owner, mask=first, other=0) + HEADER_BYTES
    wr_at = box + index * 128
    col = tl.arange(0, 16)[None, :]
    wr = tl.where(col == 0, rank.to(tl.int64), 0)
    wr = tl.where(col == 1, tl.where(last, 0, wr_at + 128)[:, None], wr)
    wr = tl.where(col == 2, (box + MAX_WRS * 128 + position * 16)[:, None], wr)
    rows = tl.minimum(size - position, ROWS_PER_WR)
    wr = tl.where(col == 3, rows[:, None], wr)
    wr = tl.where(col == 4, tl.where(last, 2, 0).to(tl.int64)[:, None], wr)
    wr = tl.where(col == 5, (landing + slot * (WORDS * 8))[:, None], wr)
    wr = tl.where(col == 6, landing_rkey.to(tl.int64), wr)
    record = desc + (wr_start + index)[:, None] * 16 + col
    tl.store(record, wr, mask=first[:, None])
    # The proxy raises the NIC's reads of these only after a later kernel's go.
    tl.inline_asm_elementwise(
        "membar.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _go_kernel(
    counts,
    seq_ptr,
    ctrl,
    world,
    GO: tl.constexpr,
    STARTS: tl.constexpr,
    WR_STARTS: tl.constexpr,
    ROWS_PER_WR: tl.constexpr,
    WORLD_P2: tl.constexpr,
):
    """Publish each gatherer's first slot and WR, then raise go with the next
    sequence number. Every fetch raises go, as every fetch wakes the proxy.

    Launch with one warp: every thread holds copies of these values, and the
    counts are read before they are reset."""
    peers = tl.arange(0, WORLD_P2)
    in_world = peers < world
    sizes = tl.load(counts + peers, mask=in_world, other=0).to(tl.int64)
    tl.store(ctrl + STARTS + peers, tl.cumsum(sizes, 0) - sizes, mask=in_world)
    tl.store(ctrl + STARTS + world, tl.sum(sizes))
    wrs = tl.cdiv(sizes, ROWS_PER_WR)
    tl.store(ctrl + WR_STARTS + peers, tl.cumsum(wrs, 0) - wrs, mask=in_world)
    tl.store(ctrl + WR_STARTS + world, tl.sum(wrs))
    tl.debug_barrier()
    tl.store(counts + peers, tl.zeros_like(peers), mask=in_world)
    # One read-modify-write, so every thread agrees on the sequence number.
    seq = tl.atomic_add(seq_ptr, 1) + 1
    tl.debug_barrier()
    tl.atomic_xchg(ctrl + GO, seq, sem="release", scope="sys")


@triton.jit(debug=True)
def _wait_kernel(
    seq_ptr,
    ctrl,
    flags,
    world,
    timeout_ns,
    DONE: tl.constexpr,
    STARTS: tl.constexpr,
    WORLD_P2: tl.constexpr,
):
    """Hold the stream until every gatherer of this fetch has written `seq`
    (low 32 bits, negated on failure) into its flag, after its rows."""
    peers = tl.arange(0, WORLD_P2)
    starts = tl.load(ctrl + STARTS + peers, mask=peers <= world, other=0)
    ends = tl.load(ctrl + STARTS + peers + 1, mask=peers < world, other=0)
    owners = (peers < world) & (ends > starts)
    seq = tl.load(seq_ptr)
    mine = seq & 0xFFFFFFFF
    deadline = tl.extra.cuda.globaltimer() + timeout_ns
    landed = tl.load(flags + peers, mask=owners, other=mine, volatile=True)
    waiting = tl.sum((landed != mine).to(tl.int32))
    failed = tl.load(ctrl + DONE, volatile=True) == -seq
    while (waiting > 0) & ~failed & (tl.extra.cuda.globaltimer() < deadline):
        landed = tl.load(flags + peers, mask=owners, other=mine, volatile=True)
        waiting = tl.sum((landed != mine).to(tl.int32))
        failed = tl.load(ctrl + DONE, volatile=True) == -seq
        failed |= tl.sum((landed == -mine).to(tl.int32)) > 0
    tl.device_assert(~failed, "Engram pool RDMA fetch failed")
    tl.device_assert(waiting == 0, "Engram pool RDMA fetch timed out")


@triton.jit(do_not_specialize=["num_items"])
def _scatter_kernel(
    owner_pos,
    ctrl,
    landing,
    out,
    num_items,
    STARTS: tl.constexpr,
    HOST: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy landed remote rows, in owner order, to their items in the output."""
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(owner_pos + item, mask=item < num_items, other=-1)
    remote = packed >= 0
    start = tl.load(ctrl + STARTS + (packed >> 32), mask=remote, other=0)
    slot = start + (packed & 0xFFFFFFFF)
    words = tl.arange(0, BLOCK_W)
    mask = remote[:, None] & (words[None, :] < WORDS)
    src = landing.to(tl.pointer_type(tl.uint64)) + slot[:, None] * WORDS + words
    # NICs rewrite host landing between fetches: bypass caches there.
    if HOST:
        data = tl.load(src, mask=mask, other=0, cache_modifier=".cv")
    else:
        data = tl.load(src, mask=mask, other=0)
    dst = out.to(tl.pointer_type(tl.uint64)) + item[:, None] * WORDS + words[None, :]
    tl.store(dst, data, mask=mask)


class RdmaTransport:
    """RC QPs from this GPU's proxy process to every peer's, and the buffers
    fetches go through. `group` provides rank_in_group, world_size and
    all_gather(bytes)."""

    def __init__(
        self, group, max_items: int, max_row_bytes: int, qps_per_peer: int = 0
    ) -> None:
        self.group = group
        self.rank, self.world = group.rank_in_group, group.world_size
        self.max_items = max_items
        with open("/proc/sys/kernel/random/boot_id") as f:
            self.node = f.read().strip()
        # Bound the WRITEs in flight per NIC: many QPs for few peers, few for many.
        self.qps_per_peer = qps_per_peer or min(
            16, max(2, 128 // max(self.world - 1, 1))
        )
        self.ctrl = SharedHost(proxy.CTRL_BYTES, "ctrl")
        # The proxy sleeps on this process-shared semaphore; a CUDA host
        # function posts it from the stream, with no Python on that path.
        libc = ctypes.CDLL(None, use_errno=True)
        self.sem = self.ctrl.cpu.data_ptr() + proxy.SEM_OFFSET
        if libc.sem_init(ctypes.c_void_p(self.sem), 1, 0):
            raise OSError(ctypes.get_errno(), "sem_init failed")
        self.sem_post = ctypes.cast(libc.sem_post, ctypes.c_void_p)
        self.launch_host_func = ctypes.CDLL("libcuda.so.1").cuLaunchHostFunc
        self.launch_host_func.argtypes = [ctypes.c_void_p] * 3
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock = ours
        self.process = subprocess.Popen(
            [sys.executable, proxy.__file__, str(theirs.fileno())],
            pass_fds=[theirs.fileno()],
        )
        theirs.close()
        info = self._call(
            "init",
            pick_device(),
            self.world,
            self.rank,
            self.qps_per_peer,
            SQ_DEPTH,
            fds=[os.dup(self.ctrl.fd)],
        )
        logger.info(
            "Engram pool RDMA proxy (pid %d) on %s, %d QPs per peer",
            self.process.pid,
            info["device"],
            self.qps_per_peer,
        )
        everyone = self._all_gather(
            [info["address"], info["qpns"], info["max_sge"], info["mtu"]]
        )
        # Every owner posts our WRs, so they fit every NIC and path.
        self.max_sge = min(e[2] for e in everyone)
        self.mtu = min(e[3] for e in everyone)
        self.max_wrs = -(-max_items // self.rows_per_wr(max_row_bytes)) + self.world
        mailboxes = self._call(
            "connect",
            [
                [peer, address, qpns[self.rank]]
                for peer, (address, qpns, _, _) in enumerate(everyone)
                if peer != self.rank
            ],
            self.max_wrs,
            max_items,
        )
        gpu = torch.device("cuda", torch.accelerator.current_device_index())
        from cuda.bindings import driver as cu

        _, ordering = cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
            cu.cuDeviceGet(gpu.index)[1],
        )
        # Rows, and the flags behind them, land in HBM if this GPU sees the
        # NIC's writes there in order without a flush; else in host memory.
        landing_bytes = max_items * max_row_bytes
        landing: SharedHost | tuple[CuMem, int]
        flags: SharedHost | tuple[CuMem, int]
        if ordering >= 100:  # CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
            hbm = [CuMem(n, rdma=True) for n in (landing_bytes, 8 * self.world)]
            landing, flags = [(memory, memory.map()) for memory in hbm]
            self.landing = device_view(landing[1], landing_bytes, hbm[0])
            self.flags = device_view(flags[1], 8 * self.world, hbm[1])
        else:
            landing = SharedHost(landing_bytes, "landing")
            flags = SharedHost(8 * self.world, "flags")
            self.landing, self.flags = landing.gpu, flags.gpu
        self.flags = self.flags.view(torch.int64)
        self.flags.zero_()
        self.landing_at, _, self.landing_rkey = self._register(landing, verbs.BULK)
        flags_at, _, flags_rkey = self._register(flags, verbs.FLAGS)
        self.host_landing = isinstance(landing, SharedHost)
        self._keep = [landing, flags]
        boxes = self._all_gather(mailboxes + [flags_at, flags_rkey])
        # This GPU's mailbox at each owner, in that owner proxy's addresses.
        box_bytes = proxy.mailbox_bytes(self.max_wrs, max_items)
        self.boxes = torch.tensor(
            [box + self.rank * box_bytes for box, *_ in boxes],
            dtype=torch.int64,
            device=gpu,
        )
        desc_bytes = self.max_wrs * verbs.SEND_WR_BYTES + max_items * proxy.SGE_BYTES
        memory = CuMem(desc_bytes, rdma=True)
        address = memory.map()
        self.desc = device_view(address, desc_bytes, memory).view(torch.int64)
        desc_at, desc_lkey, _ = self._register((memory, address), verbs.LOCAL)
        self._call(
            "link",
            desc_at,
            desc_lkey,
            dict(enumerate(boxes)),
        )
        # Peers' QPs and mailboxes must be ready before anyone requests.
        group.barrier()
        self._call("start")
        self.ctrl_words = self.ctrl.gpu.view(torch.int64)
        self.counts = torch.zeros(self.world, dtype=torch.int32, device=gpu)
        self.seq = torch.zeros(1, dtype=torch.int64, device=gpu)
        self.owner_pos = torch.empty(max_items, dtype=torch.int64, device=gpu)
        self.remote_addr = torch.empty(max_items, dtype=torch.int64, device=gpu)
        self.keys = torch.empty(max_items, dtype=torch.int32, device=gpu)

    def rows_per_wr(self, row_bytes: int) -> int:
        """Rows one WRITE gathers: a packet's worth, within every NIC's SGEs."""
        return max(1, min(self.max_sge, self.mtu // row_bytes))

    def _all_gather(self, value) -> list:
        return [
            json.loads(v) for v in self.group.all_gather(json.dumps(value).encode())
        ]

    def _call(self, command: str, *args, fds: list[int] | None = None):
        socket.send_fds(self.sock, [json.dumps([command, *args]).encode()], fds or [])
        for fd in fds or ():
            os.close(fd)
        status, result = json.loads(self.sock.recv(1 << 20))
        if status != "ok":
            raise RuntimeError(f"Engram pool RDMA proxy {command} failed:\n{result}")
        return result

    def _register(
        self, memory: SharedHost | tuple[CuMem, int], access: int
    ) -> list[int]:
        """Register host memory, or HBM as (allocation, address), with this
        GPU's NIC (`access` from `verbs`); returns its address for the NIC,
        lkey and rkey."""
        if isinstance(memory, SharedHost):
            fd = os.dup(memory.fd)
            return self._call("register", "host", memory.nbytes, 0, access, fds=[fd])
        cumem, address = memory
        fd = cumem.dmabuf_fd(address)
        return self._call("register", "dmabuf", cumem.size, address, access, fds=[fd])

    def register(self, shard: SharedHost | tuple[CuMem, int]) -> tuple[list, ...]:
        """Register this GPU's shard (host memory, or HBM as (allocation,
        address)) for gathers by its NIC. Returns, for every GPU's shard:
        its address and lkey for the NIC that gathers it, that NIC's rank,
        and its address on this GPU if this GPU reads it directly (else 0).

        A GPU reads only its own HBM shard directly. Its proxy gathers the host
        shards of its node over loopback; owners gather everything else.
        Collective: every GPU registers its shards in one order.
        """
        address, lkey, _ = self._register(shard, verbs.LOCAL)
        host = isinstance(shard, SharedHost)
        # Peers on this node open the memfd through /proc.
        mine = [address, lkey, self.node, os.getpid(), shard.fd if host else -1]  # type: ignore[union-attr]
        peers = self._all_gather(mine)
        bases, keys, gatherers, mapped = [], [], [], []
        for rank, (address, lkey, node, pid, fd) in enumerate(peers):
            gatherer = rank
            if host and node == self.node and rank != self.rank:
                try:
                    size = shard.nbytes  # type: ignore[union-attr]
                    address, lkey = self._call("map_peer", pid, fd, size)
                    gatherer = self.rank
                except RuntimeError as e:
                    logger.warning_once(
                        "Engram pool GPUs fetch their node's host shards from "
                        "their owners: %s",
                        e,
                    )
            bases.append(address)
            keys.append(lkey)
            gatherers.append(gatherer)
            own_hbm = rank == self.rank and not host
            mapped.append(shard[1] if own_hbm else 0)  # type: ignore[index]
        return bases, keys, gatherers, mapped

    def fetch(
        self, meta: torch.Tensor, ids: torch.Tensor, out: torch.Tensor, words: int
    ) -> None:
        """Gather rows `ids` (`[..., TABLES, K]`) of the tables in `meta` into
        `out`, on the current stream."""
        n = ids.numel()
        if n > self.max_items:
            raise ValueError(f"{n} rows exceed the pool's capacity {self.max_items}")
        if self.process.poll() is not None:
            raise RuntimeError("Engram pool RDMA proxy exited")
        if not n:
            return
        block_w = triton.next_power_of_2(words)
        world_p2 = triton.next_power_of_2(self.world)
        rows_per_wr = self.rows_per_wr(words * 8)
        _plan_kernel[(triton.cdiv(n, 64),)](
            ids,
            meta,
            out,
            self.counts,
            self.owner_pos,
            self.remote_addr,
            self.keys,
            n,
            TABLES=meta.shape[0],
            K=ids.shape[-1],
            WORDS=words,
            BLOCK_W=block_w,
            BLOCK=64,
        )
        _wr_kernel[(triton.cdiv(n, 128),)](
            self.owner_pos,
            self.counts,
            self.remote_addr,
            self.keys,
            self.boxes,
            self.desc,
            self.landing_at,
            self.landing_rkey,
            n,
            self.world,
            self.rank,
            MAX_WRS=self.max_wrs,
            ROWS_PER_WR=rows_per_wr,
            QPS=self.qps_per_peer,
            MAX_CHAIN=proxy.MAX_CHAIN,
            HEADER_BYTES=proxy.HEADER_BYTES,
            WORLD_P2=world_p2,
            WORDS=words,
            BLOCK=128,
        )
        _go_kernel[(1,)](
            self.counts,
            self.seq,
            self.ctrl_words,
            self.world,
            GO=proxy.GO,
            STARTS=proxy.STARTS,
            WR_STARTS=proxy.WR_STARTS,
            ROWS_PER_WR=rows_per_wr,
            WORLD_P2=world_p2,
            num_warps=1,
        )
        # Wake the sleeping proxy once go is up; no Python runs on this path.
        stream = torch.cuda.current_stream().cuda_stream
        if error := self.launch_host_func(stream, self.sem_post, self.sem):
            raise RuntimeError(f"cuLaunchHostFunc failed: {error}")
        _wait_kernel[(1,)](
            self.seq,
            self.ctrl_words,
            self.flags,
            self.world,
            TIMEOUT_NS,
            DONE=proxy.DONE,
            STARTS=proxy.STARTS,
            WORLD_P2=triton.next_power_of_2(self.world + 1),
            num_warps=1,
        )
        _scatter_kernel[(triton.cdiv(n, 64),)](
            self.owner_pos,
            self.ctrl_words,
            self.landing,
            out,
            n,
            STARTS=proxy.STARTS,
            HOST=self.host_landing,
            WORDS=words,
            BLOCK_W=block_w,
            BLOCK=64,
        )

    def close(self) -> None:
        if self.process.poll() is None:
            socket.send_fds(self.sock, [json.dumps(["stop"]).encode()], [])
            self.process.wait()
        self.sock.close()
