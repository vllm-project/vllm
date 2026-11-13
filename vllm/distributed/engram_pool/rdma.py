# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-sided RDMA READs of table rows from every peer's shard.

A fetch is four stream-ordered kernels and no host work, so it captures into
CUDA graphs:
1. `_plan_kernel` copies rows this GPU or its node's GPUs own straight into
   the output, zeroes dead ones, and numbers the rest within their owner's group.
2. `_wr_kernel` writes each remote row's READ into memory shared with this
   GPU's proxy process (`proxy.py`), chained per owner.
3. `_go_wait_kernel` (one program) bumps the device-side sequence number,
   raises the go flag, and holds the stream until the proxy reports done.
4. `_receive_kernel` copies the landed rows from host staging into the output,
   on platforms where the NIC cannot write them straight into the output.

Shards live in host memory, or in dmabuf-registered HBM.
"""

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


@triton.jit(do_not_specialize=["num_items"])
def _plan_kernel(
    ids,
    meta,
    out,
    counts,
    owner_pos,
    remote_addr,
    rkey_out,
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
    shard, rows, and device arrays of peer bases, rkeys and mapped shards
    (words 1 to 5).
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
    wrs_proxy,
    landing,
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
    wr_proxy = wrs_proxy + slot * 128
    raddr = tl.load(remote_addr + item, mask=remote, other=0)
    rkey = tl.load(rkeys + item, mask=remote, other=0).to(tl.int64) & 0xFFFFFFFF
    # ibv_send_wr words: wr_id, next, sg_list, num_sge | opcode << 32, send_flags,
    # rdma.remote_addr, rdma.rkey; RC READs ignore its tail, which holds the ibv_sge.
    col = tl.arange(0, 16)[None, :]
    wr = tl.where(col == 1, tl.where(last, 0, wr_proxy + 128)[:, None], 0)
    wr = tl.where(col == 2, (wr_proxy + 112)[:, None], wr)
    wr = tl.where(col == 3, 1 | (4 << 32), wr)
    wr = tl.where(col == 4, tl.where(last, 2, 0).to(tl.int64)[:, None], wr)
    wr = tl.where(col == 5, raddr[:, None], wr)
    wr = tl.where(col == 6, rkey[:, None], wr)
    wr = tl.where(col == 14, (landing + item * (WORDS * 8))[:, None], wr)
    wr = tl.where(col == 15, WORDS * 8 | (lkey.to(tl.int64) << 32), wr)
    # Whole 128 B records, so each one reaches host memory as one line.
    tl.store(wrs + slot[:, None] * 16 + col, wr, mask=remote[:, None])
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
        with open("/proc/sys/kernel/random/boot_id") as f:
            self.node = f.read().strip()
        self._peer_shards: list = []
        # Bound the READs in flight per NIC: many QPs for few peers, few for many.
        qps_per_peer = qps_per_peer or min(16, max(2, 128 // max(self.world - 1, 1)))
        self.ctrl = SharedHost(
            proxy.WR_OFFSET + max_items * verbs.SEND_WR_BYTES, "ctrl"
        )
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
            self.ctrl.nbytes,
            fds=[os.dup(self.ctrl.fd)],
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
        from cuda.bindings import driver as cu

        _, ordering = cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
            cu.cuDeviceGet(gpu.index)[1],
        )
        # Rows land in the HBM output if this GPU sees the NIC's writes unflushed.
        out_bytes = max_items * max_row_bytes
        self.staging: SharedHost | None = None
        landing: SharedHost | tuple[CuMem, int]
        if ordering >= 100:  # CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
            memory = CuMem(out_bytes, rdma=True)
            address = memory.map()
            self.output = device_view(address, out_bytes, memory)
            landing = (memory, address)
        else:
            self.staging = landing = SharedHost(out_bytes, "staging")
            self.output = torch.empty(out_bytes, dtype=torch.uint8, device=gpu)
        self.landing, self.lkey, _ = self._register(landing)
        self.wrs_proxy = info["ctrl_va"] + proxy.WR_OFFSET
        ctrl = self.ctrl.gpu
        self.ctrl_words = ctrl[: proxy.WR_OFFSET].view(torch.int64)
        self.wrs = ctrl[proxy.WR_OFFSET :].view(torch.int64)
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

    def _register(self, memory: SharedHost | tuple[CuMem, int]) -> list[int]:
        """Register host memory, or HBM as (allocation, address), with this
        GPU's NIC; returns its address for the NIC, lkey and rkey."""
        if isinstance(memory, SharedHost):
            fd = os.dup(memory.fd)
            return self._call("register", "host", memory.nbytes, 0, fds=[fd])
        cumem, address = memory
        fd = cumem.dmabuf_fd(address)
        return self._call("register", "dmabuf", cumem.size, address, fds=[fd])

    def register(self, shard: SharedHost | tuple[CuMem, int]) -> tuple[list, ...]:
        """Register this GPU's shard (host memory, or HBM as (allocation,
        address)) and map the host shards of its node. Returns every GPU's
        address and rkey for the NIC, and its shard's address here (0 if not
        mapped: read over RDMA).

        Collective: every GPU registers its shards in one order.
        """
        address, _, rkey = self._register(shard)
        host = isinstance(shard, SharedHost)
        # Peers on this node open the memfd through /proc.
        mine = [address, rkey, self.node, os.getpid(), shard.fd if host else -1]  # type: ignore[union-attr]
        peers = [
            json.loads(value)
            for value in self.group.all_gather(json.dumps(mine).encode())
        ]
        mapped = []
        for rank, (_, _, node, pid, fd) in enumerate(peers):
            pointer = 0
            if rank == self.rank:
                pointer = shard.gpu.data_ptr() if host else shard[1]  # type: ignore[union-attr,index]
            elif node == self.node and fd >= 0:
                try:
                    opened = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR)
                    peer = SharedHost(shard.nbytes, "", opened)  # type: ignore[union-attr]
                    self._peer_shards.append(peer)
                    pointer = peer.gpu.data_ptr()
                except (OSError, RuntimeError) as e:
                    logger.warning_once(
                        "Engram pool GPUs read host shards of their node over RDMA: %s",
                        e,
                    )
            mapped.append(pointer)
        return [p[0] for p in peers], [p[1] for p in peers], mapped

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
            self.wrs_proxy,
            self.landing,
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
        if self.staging is not None:
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
