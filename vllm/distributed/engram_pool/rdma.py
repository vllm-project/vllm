# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-sided RDMA READs of table rows from every peer's shard.

A fetch is four stream-ordered kernels and no host work, so it captures into
CUDA graphs:
1. `_plan_kernel` copies rows this GPU owns straight into the output, zeroes
   dead ones, and numbers the rest within their owner's group.
2. `_wr_kernel` writes each remote row's READ into memory shared with this
   GPU's proxy process (`proxy.py`), chained per owner.
3. `_go_wait_kernel` (one program) bumps the device-side sequence number,
   raises the go flag, and holds the stream until the proxy reports done.
4. `_receive_kernel` copies the landed rows from host staging into the output.

Shards live in host memory, or in dmabuf-registered HBM.
"""

import json
import os
import socket
import subprocess
import sys

import torch

from vllm.distributed.engram_pool import proxy, verbs
from vllm.distributed.engram_pool.memory import CuMem, SharedHost, gpu_numa_node
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

SQ_DEPTH = 4096
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


@triton.jit(do_not_specialize=["num_items", "rank"])
def _plan_kernel(
    ids,
    meta,
    out,
    counts,
    owner_pos,
    remote_addr,
    rkey_out,
    num_items,
    rank,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy owned rows, zero dead ones, and place remote rows in their owner's
    group: `owner_pos` packs (owner, position in group), -1 if not remote.

    Items are `[..., TABLES, K]` row ids; `meta` rows are each table's local
    shard, rows per shard, rows, and device arrays of peer bases and rkeys.
    """
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = item < num_items
    table = meta + (item // K % TABLES) * 8
    row = tl.load(ids + item, mask=valid, other=-1).to(tl.int64)
    rows_per_shard = tl.load(table + 1, mask=valid, other=1)
    live = valid & (row >= 0) & (row < tl.load(table + 2, mask=valid, other=0))
    owner = tl.where(live, row // rows_per_shard, 0)
    local = row - owner * rows_per_shard
    mine = live & (owner == rank)
    remote = live & (owner != rank)
    words = tl.arange(0, BLOCK_W)
    in_row = words[None, :] < WORDS
    shard = tl.load(table, mask=mine, other=0).to(tl.pointer_type(tl.uint64))
    data = tl.load(
        shard[:, None] + local[:, None] * WORDS + words[None, :],
        mask=mine[:, None] & in_row,
        other=0,
    )
    dst = out.to(tl.pointer_type(tl.uint64)) + item[:, None] * WORDS + words[None, :]
    # The receive kernel fills remote rows.
    tl.store(dst, data, mask=(valid & ~remote)[:, None] & in_row)
    position = tl.atomic_add(counts + owner, 1, mask=remote)
    packed = (owner << 32) | position.to(tl.int64)
    tl.store(owner_pos + item, tl.where(remote, packed, -1), mask=valid)
    bases = tl.load(table + 3, mask=remote, other=0).to(tl.pointer_type(tl.int64))
    base = tl.load(bases + owner, mask=remote, other=0)
    tl.store(remote_addr + item, base + local * (WORDS * 8), mask=remote)
    rkeys = tl.load(table + 4, mask=remote, other=0).to(tl.pointer_type(tl.int32))
    tl.store(rkey_out + item, tl.load(rkeys + owner, mask=remote, other=0), mask=remote)


@triton.jit(do_not_specialize=["num_items"])
def _wr_kernel(
    owner_pos,
    counts,
    remote_addr,
    rkeys,
    wrs,
    sges,
    wrs_proxy,
    sges_proxy,
    staging_proxy,
    lkey,
    num_items,
    world,
    SHORT_CHAIN: tl.constexpr,
    LONG_CHAIN: tl.constexpr,
    LONG_FROM: tl.constexpr,
    WORLD_P2: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write each remote row's READ at its slot in owner order, chained as
    `proxy.chain_length` says with each chain's last WR signaled. Pointers
    inside WRs are the proxy's addresses of the shared memory."""
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(owner_pos + item, mask=item < num_items, other=-1)
    remote = packed >= 0
    owner = packed >> 32
    position = packed & 0xFFFFFFFF
    peers = tl.arange(0, WORLD_P2)
    sizes = tl.load(counts + peers, mask=peers < world, other=0).to(tl.int64)
    start = tl.sum(tl.where(peers[None, :] < owner[:, None], sizes[None, :], 0), 1)
    size = tl.load(counts + owner, mask=remote, other=0)
    slot = start + position
    chain = tl.where(size >= LONG_FROM, LONG_CHAIN, SHORT_CHAIN)
    last = (position % chain == chain - 1) | (position == size - 1)
    zero = tl.zeros_like(slot)
    # ibv_send_wr as 64-bit words: wr_id, next, sg_list, num_sge | opcode << 32,
    # send_flags, rdma.remote_addr, rdma.rkey.
    wr = wrs + slot * 16
    tl.store(wr, zero, mask=remote)
    tl.store(wr + 1, tl.where(last, 0, wrs_proxy + (slot + 1) * 128), mask=remote)
    tl.store(wr + 2, sges_proxy + slot * 16, mask=remote)
    tl.store(wr + 3, zero + (1 | (4 << 32)), mask=remote)
    tl.store(wr + 4, tl.where(last, 2, 0).to(tl.int64), mask=remote)
    tl.store(wr + 5, tl.load(remote_addr + item, mask=remote, other=0), mask=remote)
    rkey = tl.load(rkeys + item, mask=remote, other=0).to(tl.int64)
    tl.store(wr + 6, rkey & 0xFFFFFFFF, mask=remote)
    # ibv_sge: addr, length | lkey << 32.
    sge = sges + slot * 2
    tl.store(sge, staging_proxy + item * (WORDS * 8), mask=remote)
    tl.store(sge + 1, zero + (WORDS * 8 | (lkey.to(tl.int64) << 32)), mask=remote)
    # A later kernel raises go; its release orders only its own writes, so make
    # these WRs visible to the CPU before this thread exits.
    tl.inline_asm_elementwise(
        "membar.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit(debug=True)
def _go_wait_kernel(
    counts,
    seq_ptr,
    ctrl,
    world,
    timeout_ns,
    GO: tl.constexpr,
    DONE: tl.constexpr,
    STARTS: tl.constexpr,
    WORLD_P2: tl.constexpr,
):
    """Hand this fetch's READs to the proxy and wait for them to land.

    Launch with one warp: every thread holds copies of these values, and the
    counts are read before they are reset."""
    peers = tl.arange(0, WORLD_P2)
    in_world = peers < world
    sizes = tl.load(counts + peers, mask=in_world, other=0).to(tl.int64)
    tl.store(ctrl + STARTS + peers, tl.cumsum(sizes, 0) - sizes, mask=in_world)
    total = tl.sum(sizes)
    tl.store(ctrl + STARTS + world, total)
    tl.debug_barrier()
    tl.store(counts + peers, tl.zeros_like(peers), mask=in_world)
    if total > 0:
        # One read-modify-write, so every thread agrees on the sequence number.
        seq = tl.atomic_add(seq_ptr, 1) + 1
        tl.debug_barrier()
        tl.atomic_xchg(ctrl + GO, seq, sem="release", scope="sys")
        deadline = tl.extra.cuda.globaltimer() + timeout_ns
        done = tl.load(ctrl + DONE, volatile=True)
        while (done != seq) & (done != -seq) & (tl.extra.cuda.globaltimer() < deadline):
            done = tl.load(ctrl + DONE, volatile=True)
        tl.device_assert(done != -seq, "Engram pool RDMA fetch failed")
        tl.device_assert(done == seq, "Engram pool RDMA fetch timed out")


@triton.jit(do_not_specialize=["num_items"])
def _receive_kernel(
    owner_pos,
    staging,
    out,
    num_items,
    WORDS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy landed remote rows from host staging into the output."""
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    remote = tl.load(owner_pos + item, mask=item < num_items, other=-1) >= 0
    words = tl.arange(0, BLOCK_W)
    mask = remote[:, None] & (words[None, :] < WORDS)
    index = item[:, None] * WORDS + words[None, :]
    # The NIC rewrites staging between fetches: bypass caches.
    data = tl.load(
        staging.to(tl.pointer_type(tl.uint64)) + index,
        mask=mask,
        other=0,
        cache_modifier=".cv",
    )
    tl.store(out.to(tl.pointer_type(tl.uint64)) + index, data, mask=mask)


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
        # Bound the READs in flight per NIC: many QPs for few peers, few for many.
        qps_per_peer = qps_per_peer or min(16, max(2, 128 // max(self.world - 1, 1)))
        wr_bytes = max_items * verbs.SEND_WR_BYTES
        self.ctrl = SharedHost(
            proxy.WR_OFFSET + wr_bytes + max_items * verbs.SGE_BYTES, "ctrl"
        )
        self.staging = SharedHost(max_items * max_row_bytes, "staging")
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
            qps_per_peer,
            SQ_DEPTH,
            (self.ctrl.nbytes, self.staging.nbytes),
            fds=[os.dup(self.ctrl.fd), os.dup(self.staging.fd)],
        )
        logger.info(
            "Engram pool RDMA proxy (pid %d) on %s, %d QPs per peer",
            self.process.pid,
            info["device"],
            qps_per_peer,
        )
        everyone = [
            json.loads(value)
            for value in group.all_gather(
                json.dumps([info["address"], info["qpns"]]).encode()
            )
        ]
        self._call(
            "connect",
            [
                [peer, address, qpns[self.rank]]
                for peer, (address, qpns) in enumerate(everyone)
                if peer != self.rank
            ],
        )
        # Peers' QPs must be ready to answer before anyone reads.
        group.barrier()
        self._call("start")

        gpu = torch.device("cuda", torch.accelerator.current_device_index())
        self.lkey = info["lkey"]
        self.wrs_proxy = info["ctrl_va"] + proxy.WR_OFFSET
        self.sges_proxy = self.wrs_proxy + wr_bytes
        self.staging_proxy = info["staging_va"]
        ctrl = self.ctrl.gpu
        self.ctrl_words = ctrl[: proxy.WR_OFFSET].view(torch.int64)
        self.wrs = ctrl[proxy.WR_OFFSET : proxy.WR_OFFSET + wr_bytes].view(torch.int64)
        self.sges = ctrl[proxy.WR_OFFSET + wr_bytes :].view(torch.int64)
        self.counts = torch.zeros(self.world, dtype=torch.int32, device=gpu)
        self.seq = torch.zeros(1, dtype=torch.int64, device=gpu)
        self.owner_pos = torch.empty(max_items, dtype=torch.int64, device=gpu)
        self.remote_addr = torch.empty(max_items, dtype=torch.int64, device=gpu)
        self.rkeys = torch.empty(max_items, dtype=torch.int32, device=gpu)

    def _call(self, command: str, *args, fds: list[int] | None = None):
        socket.send_fds(self.sock, [json.dumps([command, *args]).encode()], fds or [])
        for fd in fds or ():
            os.close(fd)
        status, result = json.loads(self.sock.recv(1 << 20))
        if status != "ok":
            raise RuntimeError(f"Engram pool RDMA proxy {command} failed:\n{result}")
        return result

    def register(self, shard: SharedHost | tuple[CuMem, int]) -> tuple[list, list]:
        """Register this GPU's shard (host memory, or HBM as (allocation,
        address)) and return every GPU's (address, rkey) in rank order.

        Collective: every GPU registers its shards in one order.
        """
        if isinstance(shard, SharedHost):
            address, rkey = self._call(
                "register", "host", shard.nbytes, 0, fds=[os.dup(shard.fd)]
            )
        else:
            memory, address = shard
            address, rkey = self._call(
                "register",
                "dmabuf",
                memory.size,
                address,
                fds=[memory.dmabuf_fd(address)],
            )
        peers = [
            json.loads(value)
            for value in self.group.all_gather(json.dumps([address, rkey]).encode())
        ]
        return [a for a, _ in peers], [k for _, k in peers]

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
        _plan_kernel[(triton.cdiv(n, 64),)](
            ids,
            meta,
            out,
            self.counts,
            self.owner_pos,
            self.remote_addr,
            self.rkeys,
            n,
            self.rank,
            TABLES=meta.shape[0],
            K=ids.shape[-1],
            WORDS=words,
            BLOCK_W=block_w,
            BLOCK=64,
        )
        _wr_kernel[(triton.cdiv(n, 256),)](
            self.owner_pos,
            self.counts,
            self.remote_addr,
            self.rkeys,
            self.wrs,
            self.sges,
            self.wrs_proxy,
            self.sges_proxy,
            self.staging_proxy,
            self.lkey,
            n,
            self.world,
            SHORT_CHAIN=proxy.SHORT_CHAIN,
            LONG_CHAIN=proxy.LONG_CHAIN,
            LONG_FROM=proxy.LONG_FROM,
            WORLD_P2=world_p2,
            WORDS=words,
            BLOCK=256,
        )
        _go_wait_kernel[(1,)](
            self.counts,
            self.seq,
            self.ctrl_words,
            self.world,
            TIMEOUT_NS,
            GO=proxy.GO,
            DONE=proxy.DONE,
            STARTS=proxy.STARTS,
            WORLD_P2=world_p2,
            num_warps=1,
        )
        _receive_kernel[(triton.cdiv(n, 64),)](
            self.owner_pos,
            self.staging.gpu,
            out,
            n,
            WORDS=words,
            BLOCK_W=block_w,
            BLOCK=64,
        )

    def close(self) -> None:
        if self.process.poll() is None:
            socket.send_fds(self.sock, [json.dumps(["stop"]).encode()], [])
            self.process.wait()
        self.sock.close()
