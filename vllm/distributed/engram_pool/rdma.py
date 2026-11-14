# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table rows from every peer's shard over RDMA, gathered by the owner's NIC.

A READ per row would cap a NIC at its message rate (~45M/s here, a fraction of
line rate for small rows). Instead each owner's NIC gathers a packet's worth of
rows per RDMA WRITE (one SGE per row) and writes them into the requester's
landing buffer. The requester's GPU builds those WRs; the proxies (`proxy.py`)
carry them to the owners and post them.

A fetch is stream-ordered kernels and a host function, so it captures into
CUDA graphs, and no SM touches a row until it has landed. `start`:
1. `_plan_kernel` numbers live rows within their (table, gatherer) group.
2. `_wr_kernel` writes each group's gathering WRs and SGEs, in the gatherer
   proxy's addresses, into this GPU's descriptor buffer. Its last program
   publishes each group's slots and WRs and the WRITE_WITH_IMM that hands them
   over, bumps the device-side sequence number and raises the go flag. A CUDA
   host function (libc's `sem_post`), launched beside the stream ahead of
   plan (`wake`), wakes the sleeping requester process.
Then `wait(table)`, before the layer that needs the table:
3. `_place_kernel` holds the stream until the flag of every QP that gathered
   the table's rows says they have landed, then moves them from group order
   into the output.

Shards live in host memory, or in dmabuf-registered HBM.
"""

import ctypes
import json
import os
import socket
import subprocess
import sys
from typing import Any

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
MIN_CHAIN: tl.constexpr = tl.constexpr(proxy.MIN_CHAIN)
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
    counts,
    owner_pos,
    remote_addr,
    key_out,
    num_items,
    world,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    VG_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Place each live row in the group of its table and gatherer (`table *
    world + gatherer`): `owner_pos` packs (group, position in group), -1 for
    dead rows. Reads no row data.

    Items are `[..., TABLES, K]` row ids; `meta` rows hold each table's rows per
    shard, rows, and device arrays of shard bases and lkeys for the NICs that
    gather them, and of those NICs' ranks (words 1 to 5).
    """
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = item < num_items
    index = item // K % TABLES
    table = meta + index * 8
    row = tl.load(ids + item, mask=valid, other=-1).to(tl.int64)
    rows_per_shard = tl.load(table + 1, mask=valid, other=1)
    live = valid & (row >= 0) & (row < tl.load(table + 2, mask=valid, other=0))
    owner = tl.where(live, row // rows_per_shard, 0)
    local = row - owner * rows_per_shard
    gatherers = tl.load(table + 5, mask=live, other=0).to(tl.pointer_type(tl.int32))
    gatherer = tl.load(gatherers + owner, mask=live, other=0).to(tl.int64)
    gatherer += index * world
    # Positions: ranks within the block, after one atomic per group.
    peers = tl.arange(0, VG_P2)
    mine = ((gatherer[:, None] == peers[None, :]) & live[:, None]).to(tl.int32)
    sizes = tl.sum(mine, 0)
    base = tl.atomic_add(counts + peers, sizes, mask=sizes > 0)
    before = tl.cumsum(mine, 0) - mine + base[None, :]
    position = tl.sum(tl.where(mine > 0, before, 0), 1).to(tl.int64)
    tl.store(
        owner_pos + item, tl.where(live, gatherer << 32 | position, -1), mask=valid
    )
    bases = tl.load(table + 3, mask=live, other=0).to(tl.pointer_type(tl.int64))
    base_at = tl.load(bases + owner, mask=live, other=0)
    tl.store(remote_addr + item, base_at + local * (WORDS * 8), mask=live)
    keys = tl.load(table + 4, mask=live, other=0).to(tl.pointer_type(tl.int32))
    tl.store(key_out + item, tl.load(keys + owner, mask=live, other=0), mask=live)


@triton.jit
def _wr_layout(rows, ROWS_PER_WR, QPS, MAX_CHAIN):
    """Gathering WRs, WRs per chain and QPs used for `rows` rows of one
    gatherer, chained as `proxy.chain_length` says."""
    num_wrs = tl.cdiv(rows, ROWS_PER_WR)
    chain = tl.minimum(tl.maximum(tl.cdiv(num_wrs, QPS), MIN_CHAIN), MAX_CHAIN)
    return num_wrs, chain, tl.minimum(tl.cdiv(num_wrs, chain), QPS)


@triton.jit
def _go(
    counts,
    seq_ptr,
    ctrl,
    starts,
    used,
    boxes,
    box_rkeys,
    desc,
    desc_lkey,
    requests,
    world,
    num_vg,
    GO: tl.constexpr,
    ISSUED: tl.constexpr,
    STARTS: tl.constexpr,
    WR_STARTS: tl.constexpr,
    NUM_WRS: tl.constexpr,
    REQUESTS: tl.constexpr,
    REQUEST_WORDS: tl.constexpr,
    BOX_BYTES: tl.constexpr,
    MAX_WRS: tl.constexpr,
    ROWS_PER_WR: tl.constexpr,
    QPS: tl.constexpr,
    MAX_CHAIN: tl.constexpr,
    VG_P2: tl.constexpr,
):
    """Publish each (table, gatherer) group's first slot and WR, its gathering
    WRs and the WRITE_WITH_IMM (imm: those WRs | table << 24) that copies its
    WRs and SGEs from `desc` to its mailbox (`requests`: these in the
    requester's addresses). Then raise go with the next sequence number if any
    row is remote, and always count the fetch in `issued`. Counts are read
    before they are reset."""
    peers = tl.arange(0, VG_P2)
    in_vg = peers < num_vg
    sizes = tl.load(counts + peers, mask=in_vg, other=0).to(tl.int64)
    first = tl.cumsum(sizes, 0) - sizes
    tl.store(ctrl + STARTS + peers, first, mask=in_vg)
    tl.store(ctrl + STARTS + num_vg, tl.sum(sizes))
    tl.store(starts + peers, first, mask=in_vg)
    num_wrs, _, num_qps = _wr_layout(sizes, ROWS_PER_WR, QPS, MAX_CHAIN)
    wrs = num_wrs + 2 * num_qps
    first_wr = tl.cumsum(wrs, 0) - wrs
    tl.store(ctrl + WR_STARTS + peers, first_wr, mask=in_vg)
    tl.store(ctrl + WR_STARTS + num_vg, tl.sum(wrs))
    tl.store(ctrl + NUM_WRS + peers, num_wrs, mask=in_vg)
    tl.store(used + peers, num_qps, mask=in_vg)
    # ibv_send_wr (sg_list, num_sge | WRITE_WITH_IMM << 32, IBV_SEND_SIGNALED
    # | imm << 32, remote_addr, rkey), then its two ibv_sge.
    col = tl.arange(0, 32)[None, :]
    lkey = desc_lkey.to(tl.int64) << 32
    sges = requests + (peers * REQUEST_WORDS + 16)[:, None] * 8
    imm = num_wrs | (peers // world) << 24
    wr = tl.where(col == 2, sges, 0)
    wr = tl.where(col == 3, 2 | 1 << 32, wr)
    wr = tl.where(col == 4, (2 | imm << 32)[:, None], wr)
    box = tl.load(boxes + peers % world, mask=in_vg, other=0)
    wr = tl.where(col == 5, (box + peers // world * BOX_BYTES)[:, None], wr)
    rkey = tl.load(box_rkeys + peers % world, mask=in_vg, other=0).to(tl.int64)
    wr = tl.where(col == 6, rkey[:, None], wr)
    wr = tl.where(col == 16, desc + first_wr[:, None] * 128, wr)
    wr = tl.where(col == 17, (wrs * 128)[:, None] | lkey, wr)
    wr = tl.where(col == 18, desc + MAX_WRS * 128 + first[:, None] * 16, wr)
    wr = tl.where(col == 19, (sizes * 16)[:, None] | lkey, wr)
    at = ctrl + REQUESTS + peers[:, None] * REQUEST_WORDS + col
    tl.store(at, wr, mask=in_vg[:, None] & (col < REQUEST_WORDS))
    tl.debug_barrier()
    tl.store(counts + peers, tl.zeros_like(peers), mask=in_vg)
    if tl.sum(sizes) > 0:
        # One read-modify-write, so every thread agrees on the sequence number.
        seq = tl.atomic_add(seq_ptr, 1) + 1
        tl.debug_barrier()
        tl.atomic_xchg(ctrl + GO, seq, sem="release", scope="sys")
    tl.debug_barrier()
    tl.atomic_add(ctrl + ISSUED, 1, sem="release", scope="sys")


@triton.jit(
    do_not_specialize=[
        "num_items",
        "num_vg",
        "rank",
        "landing",
        "landing_rkey",
        "flags",
        "flags_rkey",
        "desc_at",
        "desc_lkey",
        "requests",
    ]
)
def _wr_kernel(
    owner_pos,
    counts,
    remote_addr,
    keys,
    boxes,
    box_keys,
    desc,
    seq_ptr,
    landing,
    landing_rkey,
    flags,
    flags_rkey,
    num_items,
    world,
    num_vg,
    rank,
    ticket,
    ctrl,
    starts,
    used,
    box_rkeys,
    desc_at,
    desc_lkey,
    requests,
    GO: tl.constexpr,
    ISSUED: tl.constexpr,
    STARTS: tl.constexpr,
    WR_STARTS: tl.constexpr,
    NUM_WRS: tl.constexpr,
    REQUESTS: tl.constexpr,
    REQUEST_WORDS: tl.constexpr,
    BOX_BYTES: tl.constexpr,
    MAX_WRS: tl.constexpr,
    ROWS_PER_WR: tl.constexpr,
    QPS: tl.constexpr,
    MAX_CHAIN: tl.constexpr,
    VG_P2: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write each remote row's SGE at its slot, and for every ROWS_PER_WR rows
    of an owner one RDMA WRITE to `landing`, chained as `proxy.chain_length`
    says with each chain's last signaled. Each QP's last chain then links a
    READ of its flag, which returns once the QP's rows have landed, and a
    fenced inline WRITE of the next `seq` to it. Groups are (table, gatherer);
    WR pointers are into the gatherer's mailbox for this GPU and the table
    (`boxes`, then BOX_BYTES per table): WRs, flag WRs, then SGEs. The last
    program to finish then runs `_go`."""
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(owner_pos + item, mask=item < num_items, other=-1)
    remote = packed >= 0
    owner = packed >> 32
    position = packed & 0xFFFFFFFF
    peers = tl.arange(0, VG_P2)
    sizes = tl.load(counts + peers, mask=peers < num_vg, other=0).to(tl.int64)
    all_wrs, _, all_qps = _wr_layout(sizes, ROWS_PER_WR, QPS, MAX_CHAIN)
    before = peers[None, :] < owner[:, None]
    start = tl.sum(tl.where(before, sizes[None, :], 0), 1)
    wr_start = tl.sum(tl.where(before, (all_wrs + 2 * all_qps)[None, :], 0), 1)
    size = tl.load(counts + owner, mask=remote, other=0).to(tl.int64)
    num_wrs, chain, num_qps = _wr_layout(size, ROWS_PER_WR, QPS, MAX_CHAIN)
    box = tl.load(boxes + owner % world, mask=remote, other=0)
    box += owner // world * BOX_BYTES
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
    last = (index % chain == chain - 1) | (index == num_wrs - 1)
    head = index // chain
    read_at = box + (num_wrs + 2 * (head % QPS)) * 128
    tail = tl.where(head + QPS >= tl.cdiv(num_wrs, chain), read_at, 0)
    wr_at = box + index * 128
    sges = box + (num_wrs + 2 * num_qps) * 128
    col = tl.arange(0, 16)[None, :]
    wr = tl.where(col == 0, rank.to(tl.int64), 0)
    wr = tl.where(col == 1, tl.where(last, tail, wr_at + 128)[:, None], wr)
    wr = tl.where(col == 2, (sges + position * 16)[:, None], wr)
    rows = tl.minimum(size - position, ROWS_PER_WR)
    wr = tl.where(col == 3, rows[:, None], wr)
    wr = tl.where(col == 4, tl.where(last, 2, 0).to(tl.int64)[:, None], wr)
    wr = tl.where(col == 5, (landing + slot * (WORDS * 8))[:, None], wr)
    wr = tl.where(col == 6, landing_rkey.to(tl.int64), wr)
    record = desc + (wr_start + index)[:, None] * 16 + col
    tl.store(record, wr, mask=first[:, None])
    # Row q writes QP q's READ of flag (owner, q) into its own word 15, then
    # the WRITE (IBV_SEND_INLINE | IBV_SEND_FENCE) of seq from its word 14.
    # Each keeps its SGE in words 12-13.
    is_flag = (remote & (position < num_qps))[:, None]
    read = box + (num_wrs + 2 * position) * 128
    write = read + 128
    flag_at = (flags + (owner * QPS + position) * 8)[:, None]
    lkey = tl.load(box_keys + owner % world, mask=remote, other=0).to(tl.int64)
    lkey <<= 32
    wr = tl.where(col == 0, rank.to(tl.int64), 0)
    wr = tl.where(col == 1, write[:, None], wr)
    wr = tl.where(col == 2, (read + 96)[:, None], wr)
    wr = tl.where(col == 3, 1 | 4 << 32, wr)
    wr = tl.where(col == 5, flag_at, wr)
    wr = tl.where(col == 6, flags_rkey.to(tl.int64), wr)
    wr = tl.where(col == 12, (read + 120)[:, None], wr)
    wr = tl.where(col == 13, (8 | lkey)[:, None], wr)
    record = desc + (wr_start + num_wrs + 2 * position)[:, None] * 16 + col
    tl.store(record, wr, mask=is_flag)
    wr = tl.where(col == 0, rank.to(tl.int64), 0)
    wr = tl.where(col == 2, (write + 96)[:, None], wr)
    wr = tl.where(col == 3, 1, wr)
    wr = tl.where(col == 4, 8 | 1, wr)
    wr = tl.where(col == 5, flag_at, wr)
    wr = tl.where(col == 6, flags_rkey.to(tl.int64), wr)
    wr = tl.where(col == 12, (write + 112)[:, None], wr)
    wr = tl.where(col == 13, 8, wr)
    wr = tl.where(col == 14, tl.load(seq_ptr) + 1, wr)
    tl.store(record + 16, wr, mask=is_flag)
    # The NIC reads these only after go, which the last program raises.
    tl.inline_asm_elementwise(
        "membar.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    if tl.atomic_add(ticket, 1, sem="acq_rel") == tl.num_programs(0) - 1:
        tl.store(ticket, 0)
        _go(
            counts,
            seq_ptr,
            ctrl,
            starts,
            used,
            boxes,
            box_rkeys,
            desc_at,
            desc_lkey,
            requests,
            world,
            num_vg,
            GO,
            ISSUED,
            STARTS,
            WR_STARTS,
            NUM_WRS,
            REQUESTS,
            REQUEST_WORDS,
            BOX_BYTES,
            MAX_WRS,
            ROWS_PER_WR,
            QPS,
            MAX_CHAIN,
            VG_P2,
        )


@triton.jit(do_not_specialize=["num_words", "table"])
def _place_kernel(
    owner_pos,
    starts,
    landing,
    out,
    num_words,
    table,
    seq_ptr,
    ctrl,
    used,
    flags,
    world,
    timeout_ns,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    HOST: tl.constexpr,
    WORDS: tl.constexpr,
    DONE: tl.constexpr,
    QPS: tl.constexpr,
    QPS_P2: tl.constexpr,
    WORLD_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Once each QP that gathered rows of `table` in this fetch has written
    `seq` into its flag (after its rows), copy them, in group order, to their
    places in `out` (that table's `[..., K]` rows) and zero dead rows; one word
    per lane."""
    peers = tl.arange(0, WORLD_P2)
    in_world = peers < world
    vg = table * world + peers
    qps = tl.arange(0, QPS_P2)[None, :]
    mask = qps < tl.load(used + vg, mask=in_world, other=0)[:, None]
    at = flags + vg[:, None] * QPS + qps
    seq = tl.load(seq_ptr)
    deadline = tl.extra.cuda.globaltimer() + timeout_ns
    waiting = tl.sum(
        (tl.load(at, mask=mask, other=seq, volatile=True) != seq).to(tl.int32)
    )
    failed = tl.load(ctrl + DONE, volatile=True) == -seq
    while (waiting > 0) & ~failed & (tl.extra.cuda.globaltimer() < deadline):
        landed = tl.load(at, mask=mask, other=seq, volatile=True)
        waiting = tl.sum((landed != seq).to(tl.int32))
        failed = tl.load(ctrl + DONE, volatile=True) == -seq
    if failed | (waiting > 0):
        tl.device_print("Engram pool RDMA fetch failed (1) or timed out (0):", failed)
        tl.inline_asm_elementwise(
            "trap;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
    word = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = word < num_words
    row = word // WORDS
    item = row // K * (TABLES * K) + table * K + row % K
    packed = tl.load(owner_pos + item, mask=valid, other=-1)
    live = packed >= 0
    start = tl.load(starts + (packed >> 32), mask=live, other=0)
    src = (
        landing.to(tl.pointer_type(tl.uint64)) + (start + (packed & 0xFFFFFFFF)) * WORDS
    )
    src += word - row * WORDS
    # NICs rewrite host landing between fetches: bypass caches there.
    if HOST:
        data = tl.load(src, mask=live, other=0, cache_modifier=".cv")
    else:
        data = tl.load(src, mask=live, other=0)
    tl.store(out.to(tl.pointer_type(tl.uint64)) + word, data, mask=valid)


class RdmaTransport:
    """RC QPs from this GPU's proxy processes to every peer's, and the buffers
    fetches go through. `group` provides rank_in_group, world_size and
    all_gather(bytes)."""

    def __init__(
        self,
        group,
        max_items: int,
        max_row_bytes: int,
        max_tables: int = 1,
        qps_per_peer: int = 0,
    ) -> None:
        self.group = group
        self.rank, self.world = group.rank_in_group, group.world_size
        # Rows per table in a fetch, and in all.
        self.max_items, self.max_tables = max_items, max_tables
        total = max_items * max_tables
        with open("/proc/sys/kernel/random/boot_id") as f:
            self.node = f.read().strip()
        # Bound the WRITEs in flight per NIC: many QPs for few peers, few for many.
        self.qps_per_peer = qps_per_peer or min(
            16, max(2, 128 // max(self.world - 1, 1))
        )
        self.ctrl = SharedHost(proxy.CTRL_BYTES, "ctrl")
        # The requester sleeps on this process-shared semaphore; a CUDA host
        # function posts it from the stream, with no Python on that path.
        libc = ctypes.CDLL(None, use_errno=True)
        self.sem = self.ctrl.cpu.data_ptr() + proxy.SEM_OFFSET
        if libc.sem_init(ctypes.c_void_p(self.sem), 1, 0):
            raise OSError(ctypes.get_errno(), "sem_init failed")
        self.sem_post = ctypes.cast(libc.sem_post, ctypes.c_void_p)
        self.launch_host_func = ctypes.CDLL("libcuda.so.1").cuLaunchHostFunc
        self.launch_host_func.argtypes = [ctypes.c_void_p] * 3
        # A host function holds its stream until it returns: run it beside.
        self.wake_stream = torch.cuda.Stream()
        self._ready, self._joined = torch.cuda.Event(), torch.cuda.Event()
        self._woken = False
        self._compiled: dict[tuple, Any] = {}
        # Each its own process, so neither ever waits for the other's GIL.
        self.procs: dict[str, tuple[subprocess.Popen, socket.socket]] = {}
        for role in ("owner", "requester"):
            ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            process = subprocess.Popen(
                [sys.executable, proxy.__file__, str(theirs.fileno()), role],
                pass_fds=[theirs.fileno()],
            )
            theirs.close()
            self.procs[role] = (process, ours)
        device = pick_device()
        info = self._call(
            "owner", "init", device, self.world, self.rank, self.qps_per_peer, SQ_DEPTH
        )
        requester = self._call(
            "requester",
            "init",
            device,
            self.world,
            self.rank,
            max_tables,
            fds=[os.dup(self.ctrl.fd)],
        )
        requests, self.requests_at = requester["qpns"], requester["requests"]
        logger.info(
            "Engram pool RDMA proxies (pids %s) on %s, %d QPs per peer",
            [process.pid for process, _ in self.procs.values()],
            info["device"],
            self.qps_per_peer,
        )
        everyone = self._all_gather(
            [
                info["address"],
                info["qpns"],
                info["max_sge"],
                info["mtu"],
                info["notices"],
                requests,
            ]
        )
        # Every owner posts our WRs, so they fit every NIC and path.
        self.max_sge = min(e[2] for e in everyone)
        self.mtu = min(e[3] for e in everyone)
        # Per (table, gatherer): a partial WR, and a READ and a flag WR per QP.
        rows_per_wr = self.rows_per_wr(max_row_bytes)
        box_wrs = -(-max_items // rows_per_wr) + 1 + 2 * self.qps_per_peer
        groups = self.world * max_tables
        self.max_wrs = -(-total // rows_per_wr) + groups * (1 + 2 * self.qps_per_peer)
        mailboxes = self._call(
            "owner",
            "connect",
            [
                [peer, e[0], e[1][self.rank]]
                for peer, e in enumerate(everyone)
                if peer != self.rank
            ],
            [[peer, e[0], e[5][self.rank]] for peer, e in enumerate(everyone)],
            box_wrs,
            max_items,
            max_tables,
        )
        self._call(
            "requester",
            "connect",
            [[peer, e[0], e[4][self.rank]] for peer, e in enumerate(everyone)],
        )
        gpu = torch.device("cuda", torch.accelerator.current_device_index())
        from cuda.bindings import driver as cu

        _, ordering = cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
            cu.cuDeviceGet(gpu.index)[1],
        )
        # Rows, and the flags behind them, land in HBM if this GPU sees the
        # NIC's writes there in order without a flush; else in host memory.
        landing_bytes = total * max_row_bytes
        flags_bytes = 8 * groups * self.qps_per_peer
        landing: SharedHost | tuple[CuMem, int]
        flags: SharedHost | tuple[CuMem, int]
        if ordering >= 100:  # CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
            hbm = [CuMem(n, rdma=True) for n in (landing_bytes, flags_bytes)]
            landing, flags = [(memory, memory.map()) for memory in hbm]
            self.landing = device_view(landing[1], landing_bytes, hbm[0])
            self.flags = device_view(flags[1], flags_bytes, hbm[1])
        else:
            landing = SharedHost(landing_bytes, "landing")
            flags = SharedHost(flags_bytes, "flags")
            self.landing, self.flags = landing.gpu, flags.gpu
        self.flags = self.flags.view(torch.int64)
        self.flags.zero_()
        self.landing_at, _, self.landing_rkey = self._register(landing, verbs.BULK)
        self.flags_at, _, self.flags_rkey = self._register(flags, verbs.FLAGS)
        self.host_landing = isinstance(landing, SharedHost)
        self._keep = [landing, flags]
        boxes = self._all_gather(mailboxes)
        # This GPU's first mailbox at each owner, in that owner proxy's
        # addresses; one follows per table.
        self.box_bytes = proxy.mailbox_bytes(box_wrs, max_items)
        self.boxes = torch.tensor(
            [box + self.rank * max_tables * self.box_bytes for box, _, _ in boxes],
            dtype=torch.int64,
            device=gpu,
        )
        self.box_keys = torch.tensor(
            [lkey for _, _, lkey in boxes], dtype=torch.int32, device=gpu
        )
        self.box_rkeys = torch.tensor(
            [rkey for _, rkey, _ in boxes], dtype=torch.int32, device=gpu
        )
        desc_bytes = self.max_wrs * verbs.SEND_WR_BYTES + total * proxy.SGE_BYTES
        memory = CuMem(desc_bytes, rdma=True)
        address = memory.map()
        self.desc = device_view(address, desc_bytes, memory).view(torch.int64)
        self.desc_lkey = self._call(
            "requester",
            "register",
            address,
            desc_bytes,
            fds=[memory.dmabuf_fd(address)],
        )
        # Peers' QPs and mailboxes must be ready before anyone requests.
        group.barrier()
        for role in self.procs:
            self._call(role, "start")
        self.ctrl_words = self.ctrl.gpu.view(torch.int64)
        self.counts = torch.zeros(groups, dtype=torch.int32, device=gpu)
        self.starts = torch.zeros(groups, dtype=torch.int64, device=gpu)
        self.used = torch.zeros(groups, dtype=torch.int32, device=gpu)
        self.ticket = torch.zeros(1, dtype=torch.int32, device=gpu)
        self.desc_at = self.desc.data_ptr()
        self.world_p2 = triton.next_power_of_2(self.world)
        self.seq = torch.zeros(1, dtype=torch.int64, device=gpu)
        self.owner_pos = torch.empty(total, dtype=torch.int64, device=gpu)
        self.remote_addr = torch.empty(total, dtype=torch.int64, device=gpu)
        self.keys = torch.empty(total, dtype=torch.int32, device=gpu)

    def rows_per_wr(self, row_bytes: int) -> int:
        """Rows one WRITE gathers: a packet's worth, within every NIC's SGEs."""
        return max(1, min(self.max_sge, self.mtu // row_bytes))

    def _all_gather(self, value) -> list:
        return [
            json.loads(v) for v in self.group.all_gather(json.dumps(value).encode())
        ]

    def _call(self, role: str, command: str, *args, fds: list[int] | None = None):
        sock = self.procs[role][1]
        socket.send_fds(sock, [json.dumps([command, *args]).encode()], fds or [])
        for fd in fds or ():
            os.close(fd)
        status, result = json.loads(sock.recv(1 << 20))
        if status != "ok":
            raise RuntimeError(f"Engram pool RDMA proxy {command} failed:\n{result}")
        return result

    def _register(
        self, memory: SharedHost | tuple[CuMem, int], access: int
    ) -> list[int]:
        """Register host memory, or HBM as (allocation, address), with the
        owner process (`access` from `verbs`); returns its address for the NIC,
        lkey and rkey."""
        if isinstance(memory, SharedHost):
            fd = os.dup(memory.fd)
            args = ["host", memory.nbytes, 0, access]
        else:
            cumem, address = memory
            fd = cumem.dmabuf_fd(address)
            args = ["dmabuf", cumem.size, address, access]
        return self._call("owner", "register", *args, fds=[fd])

    def register(self, shard: SharedHost | tuple[CuMem, int]) -> tuple[list, ...]:
        """Register this GPU's shard (host memory, or HBM as (allocation,
        address)) for gathers by its NIC. Returns, for every GPU's shard:
        its address and lkey for the NIC that gathers it, and that NIC's rank.

        No SM reads a shard: this GPU's NIC gathers its own shard and its
        node's host shards over loopback; owners gather everything else.
        Collective: every GPU registers its shards in one order.
        """
        address, lkey, _ = self._register(shard, verbs.LOCAL)
        host = isinstance(shard, SharedHost)
        # Peers on this node open the memfd through /proc.
        mine = [address, lkey, self.node, os.getpid(), shard.fd if host else -1]  # type: ignore[union-attr]
        peers = self._all_gather(mine)
        bases, keys, gatherers = [], [], []
        for rank, (address, lkey, node, pid, fd) in enumerate(peers):
            gatherer = rank
            if host and node == self.node and rank != self.rank:
                try:
                    size = shard.nbytes  # type: ignore[union-attr]
                    address, lkey = self._call("owner", "map_peer", pid, fd, size)
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
        return bases, keys, gatherers

    def start(
        self, meta: torch.Tensor, ids: torch.Tensor, out: torch.Tensor, words: int
    ) -> None:
        """Start gathering rows `ids` (`[..., TABLES, K]`) of the tables in `meta`
        into `out` (`[TABLES, ..., K]` rows), on the current stream: the NICs
        move them while the stream goes on, and `wait` takes each table's."""
        n, tables, k = ids.numel(), meta.shape[0], ids.shape[-1]
        if n > self.max_items * self.max_tables or tables > self.max_tables:
            raise ValueError(f"{tables} tables of {n} rows exceed the pool's capacity")
        if any(process.poll() is not None for process, _ in self.procs.values()):
            raise RuntimeError("Engram pool RDMA proxy exited")
        self._fetch = (n, tables, k, words, out)
        if not n:
            return
        rows_per_wr = self.rows_per_wr(words * 8)
        num_vg, vg_p2 = self.world * tables, triton.next_power_of_2(self.world * tables)
        qps = self.qps_per_peer
        plan = self._kernel(
            _plan_kernel,
            (triton.cdiv(n, 256),),
            (ids, meta, self.counts, self.owner_pos, self.remote_addr, self.keys)
            + (n, self.world),
            4,
            dict(TABLES=tables, K=k, WORDS=words, VG_P2=vg_p2, BLOCK=256),
        )
        wr = self._kernel(
            _wr_kernel,
            (triton.cdiv(n, 128),),
            (self.owner_pos, self.counts, self.remote_addr, self.keys, self.boxes)
            + (self.box_keys, self.desc, self.seq, self.landing_at, self.landing_rkey)
            + (self.flags_at, self.flags_rkey, n, self.world, num_vg, self.rank)
            + (self.ticket, self.ctrl_words, self.starts, self.used, self.box_rkeys)
            + (self.desc_at, self.desc_lkey, self.requests_at),
            4,
            dict(
                GO=proxy.GO,
                ISSUED=proxy.ISSUED,
                STARTS=proxy.STARTS,
                WR_STARTS=proxy.WR_STARTS,
                NUM_WRS=proxy.NUM_WRS,
                REQUESTS=proxy.REQUESTS,
                REQUEST_WORDS=proxy.REQUEST_WORDS,
                BOX_BYTES=self.box_bytes,
                MAX_WRS=self.max_wrs,
                ROWS_PER_WR=rows_per_wr,
                QPS=qps,
                MAX_CHAIN=proxy.MAX_CHAIN,
                VG_P2=vg_p2,
                WORDS=words,
                BLOCK=128,
            ),
        )
        # Once nothing is left to compile, unless `wake` came earlier.
        if not self._woken:
            self.wake()
        self._woken = False
        stream = torch.cuda.current_stream()
        for launch in (plan, wr):
            launch(stream.cuda_stream)
        self._joined.record(self.wake_stream)
        stream.wait_event(self._joined)

    def wake(self) -> None:
        """Wake the requester for the next `start`, from a CUDA host function
        beside the current stream. It holds its stream until it returns
        (~45 us), so the stream's work until `start`'s kernels raise go hides
        that: call this as early before `start` as the stream allows."""
        stream = torch.cuda.current_stream()
        self._ready.record(stream)
        self.wake_stream.wait_event(self._ready)
        wake = self.wake_stream.cuda_stream
        if error := self.launch_host_func(wake, self.sem_post, self.sem):
            raise RuntimeError(f"cuLaunchHostFunc failed: {error}")
        self._woken = True

    def wait(self, table: int) -> None:
        """Hold the current stream until table `table`'s rows of the last
        `start` have landed, and place them."""
        n, tables, k, words, out = self._fetch
        if not n:
            return
        qps = self.qps_per_peer
        num_words = n // tables * words
        self._launch(
            _place_kernel,
            (triton.cdiv(num_words, 1024),),
            torch.cuda.current_stream().cuda_stream,
            (self.owner_pos, self.starts, self.landing, out[table], num_words, table)
            + (self.seq, self.ctrl_words, self.used, self.flags, self.world)
            + (TIMEOUT_NS,),
            TABLES=tables,
            K=k,
            HOST=self.host_landing,
            WORDS=words,
            DONE=proxy.DONE,
            QPS=qps,
            QPS_P2=triton.next_power_of_2(qps),
            WORLD_P2=self.world_p2,
            BLOCK=1024,
        )

    def _launch(self, kernel, grid, stream: int, args: tuple, num_warps=4, **consts):
        """Launch `kernel` with `args` then `consts` (its constexprs, in order)."""
        self._kernel(kernel, grid, args, num_warps, consts)(stream)

    def _kernel(self, kernel, grid, args: tuple, num_warps: int, consts: dict):
        """A launcher (taking the stream) of `kernel` on `grid`, compiled once
        per constexprs and tensor alignment; Triton's per-call dispatch would
        cost ~20 us per launch here."""
        aligned = tuple(a.data_ptr() % 16 == 0 for a in args if type(a) is torch.Tensor)
        key = (kernel, num_warps, *consts.values(), *aligned)
        if (compiled := self._compiled.get(key)) is None:
            compiled = kernel.warmup(*args, grid=grid, num_warps=num_warps, **consts)
            compiled._init_handles()
            self._compiled[key] = compiled
        values = (*args, *consts.values())
        x, y, z = (*grid, 1, 1)[:3]

        def launch(stream: int) -> None:
            compiled.run(
                x,
                y,
                z,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *values,
            )

        return launch

    def close(self) -> None:
        for process, sock in self.procs.values():
            sock.close()
            process.wait()
