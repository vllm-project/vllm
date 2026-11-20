# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table rows from every peer's shard over RDMA, gathered by the owner's NIC.

A READ per row would cap a NIC at its message rate (~45M/s here, a fraction of
line rate for small rows). Instead each owner's NIC gathers a packet's worth of
rows per RDMA WRITE (one SGE per row) and writes them into the requester's
landing buffer. The requester's GPU builds those WRs and hands each owner its
share itself; owner proxy processes (`proxy.py`) post them.

A fetch is two stream-ordered kernels, so it captures into CUDA graphs, needs
no host thread, and no SM touches a row until it has landed. `start`:
1. `_plan_kernel` numbers live rows within their (table, gatherer) group.
2. `_wr_kernel` writes each group's gathering WRs and SGEs, in the gatherer
   proxy's addresses, into this GPU's descriptor buffer. Its last program
   bumps the device-side sequence number and posts, per group, an RDMA
   WRITE_WITH_IMM that copies them into the gatherer's mailbox: it writes the
   mlx5 WQE into the send queue of this GPU's QP to that gatherer and rings
   the doorbell (GPU-initiated, as IBGDA).
Then, before the layer that needs a table, `wait_landed` (in the consumer's
kernel, or `_place_kernel`) spins until the flag of every QP that gathered the
table's rows says they have landed.

Shards live in host memory, or in dmabuf-registered HBM.
"""

import ctypes
import json
import mmap
import os
import socket
import subprocess
import sys
import threading
from typing import Any

import torch

from vllm.distributed.engram_pool import proxy, verbs
from vllm.distributed.engram_pool.memory import (
    CuMem,
    SharedHost,
    device_view,
)
from vllm.distributed.engram_pool.rows import (
    GATHERERS,
    READY_FAILED,
    READY_FLAGS,
    READY_LANDED,
    READY_QPS,
    READY_SEQ,
    READY_TIMEOUT,
    READY_USED,
    READY_WORLD,
    SHARD_BASES,
    SHARD_KEYS,
    locate,
    wait_landed,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

SQ_DEPTH = 1024
# Requests in flight per QP to a gatherer: a fetch posts one per table.
REQUEST_DEPTH = 64
MIN_CHAIN: tl.constexpr = tl.constexpr(proxy.MIN_CHAIN)
MAX_CHAIN: tl.constexpr = tl.constexpr(proxy.MAX_CHAIN)
# A failed transfer or dead proxy must not hang the GPU forever.
TIMEOUT_NS = 60 * 10**9


class _RequestWQE(ctypes.Structure):
    """The mlx5 WQE of a request: an RDMA WRITE_WITH_IMM of its WRs and of
    their SGEs (a data segment each) into a mailbox."""

    _fields_ = [
        ("ctrl", verbs.mlx5_wqe_ctrl_seg),
        ("raddr", verbs.mlx5_wqe_raddr_seg),
        ("wrs", verbs.mlx5_wqe_data_seg),
        ("sges", verbs.mlx5_wqe_data_seg),
    ]


def _word(struct, field: str, word_bytes: int) -> tl.constexpr:
    """The `word_bytes`-byte word of `struct` where `field` (dotted) starts."""
    offset = 0
    for name in field.split("."):
        offset += getattr(struct, name).offset
        struct = dict(struct._fields_)[name]
    return tl.constexpr(offset // word_bytes)


# ibv_send_wr and ibv_sge, in bytes and 8-byte words.
WR_BYTES: tl.constexpr = tl.constexpr(proxy.WR_BYTES)
SGE_BYTES: tl.constexpr = tl.constexpr(proxy.SGE_BYTES)
WR_WORDS: tl.constexpr = tl.constexpr(proxy.WR_BYTES // 8)
SGE_WORDS: tl.constexpr = tl.constexpr(proxy.SGE_BYTES // 8)
_WR_NEXT = _word(verbs.ibv_send_wr, "next", 8)
_WR_SG_LIST = _word(verbs.ibv_send_wr, "sg_list", 8)
_WR_NUM_SGE = _word(verbs.ibv_send_wr, "num_sge", 8)
_WR_SEND_FLAGS = _word(verbs.ibv_send_wr, "send_flags", 8)
_WR_REMOTE_ADDR = _word(verbs.ibv_send_wr, "remote_addr", 8)
_WR_RKEY = _word(verbs.ibv_send_wr, "rkey", 8)
# A flag WR keeps its SGE, then its 8-byte payload, in its `spare` words.
_WR_SGE = _word(verbs.ibv_send_wr, "spare", 8)
_WR_DATA: tl.constexpr = tl.constexpr(_WR_SGE + SGE_WORDS)
_WRITE: tl.constexpr = tl.constexpr(verbs.IBV_WR_RDMA_WRITE)
_READ: tl.constexpr = tl.constexpr(verbs.IBV_WR_RDMA_READ)
_SIGNALED: tl.constexpr = tl.constexpr(verbs.IBV_SEND_SIGNALED)
_INLINE_FENCE: tl.constexpr = tl.constexpr(verbs.IBV_SEND_INLINE | verbs.IBV_SEND_FENCE)
# The request WQE in bytes, 32-bit words and 16-byte segments.
WQE_BYTES: tl.constexpr = tl.constexpr(verbs.MLX5_SEND_WQE_BB)
WQE_WORDS: tl.constexpr = tl.constexpr(ctypes.sizeof(_RequestWQE) // 4)
_WQE_SEGMENTS: tl.constexpr = tl.constexpr(ctypes.sizeof(_RequestWQE) // 16)
_WQE_OPCODE = _word(_RequestWQE, "ctrl.opmod_idx_opcode", 4)
_WQE_QPN_DS = _word(_RequestWQE, "ctrl.qpn_ds", 4)
_WQE_IMM = _word(_RequestWQE, "ctrl.imm", 4)
_WQE_RADDR = _word(_RequestWQE, "raddr.raddr", 4)
_WQE_RKEY = _word(_RequestWQE, "raddr.rkey", 4)
_WQE_WRS = _word(_RequestWQE, "wrs", 4)
_WQE_SGES = _word(_RequestWQE, "sges", 4)
_WRITE_IMM: tl.constexpr = tl.constexpr(verbs.MLX5_OPCODE_RDMA_WRITE_IMM)


def node_id() -> str:
    """This machine's identity: GPUs with the same one share host memory."""
    with open("/proc/sys/kernel/random/boot_id") as f:
        return f.read().strip()


def pick_device() -> str:
    """This GPU's RDMA device: the active one closest to it (see `_nearest`).
    GPUs with the same closest devices take them round-robin."""
    devices = verbs.active_devices()
    if not devices:
        raise RuntimeError("Engram pool RDMA transport found no active RDMA device")
    nearest = [
        _nearest(devices, gpu) for gpu in range(torch.accelerator.device_count())
    ]
    gpu = torch.accelerator.current_device_index()
    peers = [i for i, near in enumerate(nearest) if near == nearest[gpu]]
    return nearest[gpu][peers.index(gpu) % len(nearest[gpu])]


def _nearest(devices: list[str], gpu: int) -> list[str]:
    """The `devices` closest to `gpu`: those sharing the longest sysfs PCIe
    path with it (the NIC behind its PCIe switch), then those on its NUMA node
    (for GPUs not on the NICs' PCIe tree, as with C2C-attached GPUs)."""
    props = torch.cuda.get_device_properties(gpu)
    bdf = (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )
    gpu_path = os.path.realpath(f"/sys/bus/pci/devices/{bdf}").split("/")
    node = current_platform.get_device_numa_node(gpu)

    def closeness(device: str) -> tuple[int, bool]:
        path = os.path.realpath(f"/sys/class/infiniband/{device}/device").split("/")
        return len(os.path.commonprefix([gpu_path, path])), (
            verbs.device_numa_node(device) == node
        )

    best = max(map(closeness, devices))
    return [d for d in devices if closeness(d) == best]


@triton.jit
def _rdma_wr(word, wr_id, next, sg_list, num_sge, opcode, send_flags, addr, rkey):
    """An RDMA WRITE or READ ibv_send_wr, as 8-byte words (`word`)."""
    wr = tl.where(word == 0, wr_id, 0)
    wr = tl.where(word == _WR_NEXT, next, wr)
    wr = tl.where(word == _WR_SG_LIST, sg_list, wr)
    # num_sge, then opcode.
    wr = tl.where(word == _WR_NUM_SGE, num_sge | opcode << 32, wr)
    wr = tl.where(word == _WR_SEND_FLAGS, send_flags, wr)
    wr = tl.where(word == _WR_REMOTE_ADDR, addr, wr)
    return tl.where(word == _WR_RKEY, rkey, wr)


@triton.jit
def _with_sge(value, word, at, addr, length, lkey):
    """`value` (8-byte words) with an ibv_sge at word `at`."""
    value = tl.where(word == at, addr, value)
    return tl.where(word == at + 1, length | lkey << 32, value)


@triton.jit
def _wr_layout(rows, ROWS_PER_WR, QPS):
    """Gathering WRs, WRs per chain and QPs used for `rows` rows of one
    gatherer, chained as `proxy.chain_length` says."""
    num_wrs = tl.cdiv(rows, ROWS_PER_WR)
    chain = tl.minimum(tl.maximum(tl.cdiv(num_wrs, QPS), MIN_CHAIN), MAX_CHAIN)
    return num_wrs, chain, tl.minimum(tl.cdiv(num_wrs, chain), QPS)


@triton.jit(do_not_specialize=["num_items"])
def _plan_kernel(
    ids,
    meta,
    counts,
    owner_pos,
    remote_addr,
    keys,
    num_items,
    world,
    ticket,
    starts,
    wr_starts,
    wr_counts,
    used,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    ROWS_PER_WR: tl.constexpr,
    QPS: tl.constexpr,
    VG_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Number live rows within their (table, gatherer) group into `owner_pos`
    (-1 if dead); the last program lays out each group's slots, WRs and QPs."""
    peers = tl.arange(0, VG_P2)
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = item < num_items
    index, table, owner, local, live = locate(ids, meta, item, valid, TABLES, K)
    gatherers = tl.load(table + GATHERERS, mask=live, other=0)
    gatherers = gatherers.to(tl.pointer_type(tl.int32))
    gatherer = tl.load(gatherers + owner, mask=live, other=0).to(tl.int64)
    gatherer += index * world
    # Positions: ranks within the block, after one atomic per group.
    mine = ((gatherer[:, None] == peers[None, :]) & live[:, None]).to(tl.int32)
    sizes = tl.sum(mine, 0)
    base = tl.atomic_add(counts + peers, sizes, mask=sizes > 0)
    before = tl.cumsum(mine, 0) - mine + base[None, :]
    position = tl.sum(tl.where(mine > 0, before, 0), 1).to(tl.int64)
    tl.store(
        owner_pos + item, tl.where(live, gatherer << 32 | position, -1), mask=valid
    )
    bases = tl.load(table + SHARD_BASES, mask=live, other=0)
    bases = bases.to(tl.pointer_type(tl.int64))
    base_at = tl.load(bases + owner, mask=live, other=0)
    tl.store(remote_addr + item, base_at + local * (WORDS * 8), mask=live)
    lkeys = tl.load(table + SHARD_KEYS, mask=live, other=0)
    lkey = tl.load(lkeys.to(tl.pointer_type(tl.int32)) + owner, mask=live)
    tl.store(keys + item, lkey, mask=live)
    tl.debug_barrier()
    if _last_program(ticket):
        in_vg = peers < world * TABLES
        group_rows = tl.load(counts + peers, mask=in_vg, other=0).to(tl.int64)
        group_wrs, _, group_qps = _wr_layout(group_rows, ROWS_PER_WR, QPS)
        all_wrs = group_wrs + 2 * group_qps
        tl.store(starts + peers, tl.cumsum(group_rows, 0) - group_rows, mask=in_vg)
        tl.store(wr_starts + peers, tl.cumsum(all_wrs, 0) - all_wrs, mask=in_vg)
        tl.store(wr_counts + peers, group_wrs, mask=in_vg)
        tl.store(used + peers, group_qps, mask=in_vg)


@triton.jit(
    do_not_specialize=[
        "num_items",
        "rank",
        "landing",
        "landing_rkey",
        "flags",
        "flags_rkey",
        "desc_lkey",
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
    rank,
    ticket,
    starts,
    wr_starts,
    wr_counts,
    used,
    box_rkeys,
    desc_lkey,
    queues,
    slots,
    BOX_BYTES: tl.constexpr,
    MAX_WRS: tl.constexpr,
    ROWS_PER_WR: tl.constexpr,
    QPS: tl.constexpr,
    VG_P2: tl.constexpr,
    WORDS: tl.constexpr,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write each group's gathering WRs and SGEs (in its gatherer's mailbox
    layout) and each row's slot; the last program posts every group's request."""
    peers = tl.arange(0, VG_P2)
    item = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(owner_pos + item, mask=item < num_items, other=-1)
    remote = packed >= 0
    owner = packed >> 32
    position = packed & 0xFFFFFFFF
    start = tl.load(starts + owner, mask=remote, other=0)
    wr_start = tl.load(wr_starts + owner, mask=remote, other=0)
    size = tl.load(counts + owner, mask=remote, other=0).to(tl.int64)
    num_wrs, chain, num_qps = _wr_layout(size, ROWS_PER_WR, QPS)
    box = tl.load(boxes + owner % world, mask=remote, other=0)
    box += owner // world * BOX_BYTES
    slot = start + position
    row = item // (TABLES * K) * K + item % K
    at = slots + item // K % TABLES * (num_items // TABLES) + row
    tl.store(at, tl.where(remote, slot, -1).to(tl.int32), mask=item < num_items)
    key = tl.load(keys + item, mask=remote, other=0).to(tl.int64) & 0xFFFFFFFF
    addr = tl.load(remote_addr + item, mask=remote, other=0)[:, None]
    word = tl.arange(0, SGE_WORDS)[None, :]
    sge = _with_sge(0, word, 0, addr, WORDS * 8, key[:, None])
    sge_at = desc + MAX_WRS * WR_WORDS + slot[:, None] * SGE_WORDS + word
    tl.store(sge_at, sge, mask=remote[:, None])
    first = remote & (position % ROWS_PER_WR == 0)
    index = position // ROWS_PER_WR
    last = (index % chain == chain - 1) | (index == num_wrs - 1)
    head = index // chain
    read_at = box + (num_wrs + 2 * (head % QPS)) * WR_BYTES
    tail = tl.where(head + QPS >= tl.cdiv(num_wrs, chain), read_at, 0)
    wr_at = box + index * WR_BYTES
    sges = box + (num_wrs + 2 * num_qps) * WR_BYTES
    wr_id = rank.to(tl.int64)
    word = tl.arange(0, WR_WORDS)[None, :]
    wr = _rdma_wr(
        word,
        wr_id,
        tl.where(last, tail, wr_at + WR_BYTES)[:, None],
        (sges + position * SGE_BYTES)[:, None],
        tl.minimum(size - position, ROWS_PER_WR)[:, None],
        _WRITE,
        tl.where(last, _SIGNALED, 0).to(tl.int64)[:, None],
        (landing + slot * (WORDS * 8))[:, None],
        landing_rkey.to(tl.int64),
    )
    record = desc + (wr_start + index)[:, None] * WR_WORDS + word
    tl.store(record, wr, mask=first[:, None])
    # Row q writes QP q's READ of flag (owner, q) into the READ's data word,
    # then the fenced inline WRITE of the next seq from the WRITE's.
    is_flag = (remote & (position < num_qps))[:, None]
    read = (box + (num_wrs + 2 * position) * WR_BYTES)[:, None]
    write = read + WR_BYTES
    lkey = tl.load(box_keys + owner % world, mask=remote, other=0).to(tl.int64)
    flag = (flags + (owner * QPS + position) * 8)[:, None]
    rkey = flags_rkey.to(tl.int64)
    wr = _rdma_wr(word, wr_id, write, read + _WR_SGE * 8, 1, _READ, 0, flag, rkey)
    wr = _with_sge(wr, word, _WR_SGE, read + _WR_DATA * 8, 8, lkey[:, None])
    record = desc + (wr_start + num_wrs + 2 * position)[:, None] * WR_WORDS + word
    tl.store(record, wr, mask=is_flag)
    wr = _rdma_wr(
        word, wr_id, 0, write + _WR_SGE * 8, 1, _WRITE, _INLINE_FENCE, flag, rkey
    )
    wr = _with_sge(wr, word, _WR_SGE, write + _WR_DATA * 8, 8, 0)
    wr = tl.where(word == _WR_DATA, tl.load(seq_ptr) + 1, wr)
    tl.store(record + WR_WORDS, wr, mask=is_flag)
    # The NIC reads these only after the doorbell the last program rings.
    _fence_sys()
    tl.debug_barrier()
    if _last_program(ticket):
        in_vg = peers < world * TABLES
        sizes = tl.load(counts + peers, mask=in_vg, other=0).to(tl.int64)
        all_wrs = tl.load(wr_counts + peers, mask=in_vg, other=0)
        tl.debug_barrier()
        tl.store(counts + peers, tl.zeros_like(peers), mask=in_vg)
        if tl.sum(sizes) > 0:
            # One read-modify-write, so every thread agrees on the sequence number.
            tl.atomic_add(seq_ptr, 1)
            _post_requests(
                peers,
                in_vg & (all_wrs > 0),
                all_wrs,
                sizes,
                tl.load(starts + peers, mask=in_vg, other=0),
                tl.load(wr_starts + peers, mask=in_vg, other=0),
                all_wrs + 2 * tl.load(used + peers, mask=in_vg, other=0),
                world,
                boxes,
                box_rkeys,
                desc.to(tl.int64),
                desc_lkey,
                queues,
                BOX_BYTES,
                MAX_WRS,
            )


@triton.jit
def _last_program(ticket):
    """Whether this program is the last of its launch to get here."""
    last = tl.atomic_add(ticket, 1, sem="acq_rel") == tl.num_programs(0) - 1
    if last:
        tl.store(ticket, 0)
    return last


@triton.jit
def _fence_sys():
    tl.inline_asm_elementwise(
        "membar.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _be32(x):
    """The low 32 bits of `x`, big-endian, as stored by a little-endian GPU."""
    x = x.to(tl.int64)
    swapped = (x & 0xFF) << 24 | (x & 0xFF00) << 8 | (x >> 8) & 0xFF00
    return (swapped | (x >> 24) & 0xFF).to(tl.uint32)


@triton.jit
def _with_be64(value, word, at, x):
    """`value` (32-bit words) with big-endian `x` at word `at`."""
    value = tl.where(word == at, _be32(x >> 32)[:, None], value)
    return tl.where(word == at + 1, _be32(x)[:, None], value)


@triton.jit
def _with_data_seg(value, word, at, length, lkey, addr):
    """`value` (32-bit words) with an mlx5_wqe_data_seg at word `at`."""
    value = tl.where(word == at, _be32(length)[:, None], value)
    value = tl.where(word == at + 1, lkey, value)
    return _with_be64(value, word, at + 2, addr)


@triton.jit
def _post_requests(
    vg,
    active,
    imm_wrs,
    sizes,
    group_first,
    group_first_wr,
    group_wrs,
    world,
    boxes,
    box_rkeys,
    desc_at,
    desc_lkey,
    queues,
    BOX_BYTES: tl.constexpr,
    MAX_WRS: tl.constexpr,
):
    """Post each active group's `_RequestWQE` on its gatherer's QP and ring
    the doorbell; `queues` rows are those `_map_queues` returns."""
    gatherer = vg % world
    queue = queues + gatherer * 8
    # Each group's WQE index on its QP: after the active groups before it there.
    other = vg[None, :]
    same = (gatherer[:, None] == other % world) & active[None, :]
    before = tl.sum((same & (other < vg[:, None])).to(tl.int64), 1)
    last = active & (tl.sum((same & (other > vg[:, None])).to(tl.int32), 1) == 0)
    index = tl.load(queue + 5, mask=active, other=0) + before
    wqe_count = tl.load(queue + 1, mask=active, other=1)
    qpn = tl.load(queue + 4, mask=active, other=0)
    wqe = tl.load(queue, mask=active, other=0) + index % wqe_count * WQE_BYTES
    box = tl.load(boxes + gatherer, mask=active, other=0) + vg // world * BOX_BYTES
    rkey = tl.load(box_rkeys + gatherer, mask=active, other=0)
    wrs_at = desc_at + group_first_wr * WR_BYTES
    sges_at = desc_at + MAX_WRS * WR_BYTES + group_first * SGE_BYTES
    # Big-endian 32-bit words; unsignaled, and imm raw, as the owner reads it.
    word = tl.arange(0, WQE_WORDS)[None, :]
    ctrl0 = _be32(index << 8 & 0xFFFF00 | _WRITE_IMM)[:, None]
    ctrl1 = _be32(qpn << 8 | _WQE_SEGMENTS)[:, None]
    imm = (imm_wrs | vg // world << 24).to(tl.uint32)[:, None]
    value = tl.where(word == _WQE_OPCODE, ctrl0, 0)
    value = tl.where(word == _WQE_QPN_DS, ctrl1, value)
    value = tl.where(word == _WQE_IMM, imm, value)
    value = _with_be64(value, word, _WQE_RADDR, box)
    value = tl.where(word == _WQE_RKEY, _be32(rkey)[:, None], value)
    lkey = _be32(desc_lkey)
    value = _with_data_seg(value, word, _WQE_WRS, group_wrs * WR_BYTES, lkey, wrs_at)
    value = _with_data_seg(value, word, _WQE_SGES, sizes * SGE_BYTES, lkey, sges_at)
    at = wqe.to(tl.pointer_type(tl.uint32))[:, None] + word
    tl.store(at, value, mask=active[:, None])
    # The NIC reads the WQEs (host memory) once the doorbell rings, and the
    # register is rung with the last WQE's first 8 bytes (its ctrl0, ctrl1).
    _fence_sys()
    tl.debug_barrier()
    record = tl.load(queue + 2, mask=last, other=0).to(tl.pointer_type(tl.uint32))
    tl.store(record + 1, _be32(index + 1 & 0xFFFF), mask=last)
    tl.store(queue + 5, index + 1, mask=last)
    _fence_sys()
    tl.debug_barrier()
    register = tl.load(queue + 3, mask=last, other=0)
    ring = tl.sum(ctrl0.to(tl.uint64) | ctrl1.to(tl.uint64) << 32, 1)
    # One doorbell at a time: stores to several registers at once can merge
    # into one write that the NIC takes as a single doorbell.
    for g in range(world):
        mine = last & (gatherer == g)
        if tl.sum(mine.to(tl.int32)) > 0:
            door = tl.sum(tl.where(mine, register, 0)).to(tl.pointer_type(tl.uint64))
            tl.store(door, tl.sum(tl.where(mine, ring, 0)), cache_modifier=".wt")
            _fence_sys()


@triton.jit(do_not_specialize=["num_words", "table"])
def _place_kernel(
    owner_pos,
    starts,
    landing,
    out,
    num_words,
    table,
    ready,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    HOST: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Once table `table`'s rows have landed, copy them to their places in
    `out` (its `[..., K]` rows), zeroing dead rows; one word per lane."""
    wait_landed(ready, table)
    word = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = word < num_words
    row = word // WORDS
    item = row // K * (TABLES * K) + table * K + row % K
    packed = tl.load(owner_pos + item, mask=valid, other=-1)
    live = packed >= 0
    slot = tl.load(starts + (packed >> 32), mask=live, other=0) + (packed & 0xFFFFFFFF)
    src = landing.to(tl.pointer_type(tl.uint64)) + slot * WORDS + word - row * WORDS
    # NICs rewrite host landing between fetches: bypass caches there.
    if HOST:
        data = tl.load(src, mask=live, other=0, cache_modifier=".cv")
    else:
        data = tl.load(src, mask=live, other=0)
    tl.store(out.to(tl.pointer_type(tl.uint64)) + word, data, mask=valid)


class RdmaTransport:
    """QPs to every peer's owner proxy (requests) and from this GPU's to every
    peer's (gathers), and the buffers fetches go through."""

    def __init__(
        self,
        group,
        host_shards: bool,
        max_items: int,
        max_row_bytes: int,
        max_tables: int,
        rdma_inflight: int,
        thp_dir: str | None = None,
    ) -> None:
        self.group, self.host_shards = group, host_shards
        self.thp_dir = thp_dir
        self.rank, self.world = group.rank_in_group, group.world_size
        # Rows per table in a fetch, and in all.
        self.max_items, self.max_tables = max_items, max_tables
        total = max_items * max_tables
        self.node = node_id()
        # Bound the WRITEs in flight per NIC: many QPs for few peers, few for many.
        self.qps_per_peer = min(16, max(2, 128 // max(self.world - 1, 1)))
        self._compiled: dict[tuple, Any] = {}
        # Its own process, so it never waits for this one's GIL.
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.proc = subprocess.Popen(
            [sys.executable, proxy.__file__, str(theirs.fileno())],
            pass_fds=[theirs.fileno()],
        )
        theirs.close()
        self.sock = ours
        # Set once a request fails or the proxy exits, so every wait fails then:
        # `start` reads the host copy, kernels the HBM one.
        gpu = torch.device("cuda", torch.accelerator.current_device_index())
        self.failed = torch.zeros(1, dtype=torch.int64, device="cpu", pin_memory=True)
        self.failed_gpu = torch.zeros(1, dtype=torch.int64, device=gpu)
        self._fail_stream = torch.cuda.Stream(gpu)

        def proxy_exited() -> None:
            self.proc.wait()
            self._fail()

        threading.Thread(target=proxy_exited, daemon=True).start()
        device = pick_device()
        info = self._call(
            "init",
            device,
            self.world,
            self.rank,
            self.qps_per_peer,
            SQ_DEPTH,
            rdma_inflight,
        )
        # This GPU posts its requests itself, on one QP per gatherer.
        self.verbs = verbs.Device(device)
        cq, channel = self.verbs.create_cq(self.world * REQUEST_DEPTH, wakeable=True)
        made = [
            self.verbs.create_qp(cq, cq, REQUEST_DEPTH, 2) for _ in range(self.world)
        ]
        logger.info(
            "Engram pool RDMA proxy (pid %d) on %s, %d QPs per peer",
            self.proc.pid,
            info["device"],
            self.qps_per_peer,
        )
        requests = [qpn for _, qpn in made]
        everyone = self._all_gather(
            {**info, "requests": requests, "max_items": max_items, "node": self.node}
        )
        # Each GPU's NIC gathers its node's host shards itself (`shard`), so
        # no QPs connect GPUs of one node: their NICs may not reach each other,
        # as on rail-optimized fabrics.
        self.local_peers = {
            peer
            for peer, e in enumerate(everyone)
            if host_shards and e["node"] == self.node and peer != self.rank
        }
        remote = [
            (peer, e) for peer, e in enumerate(everyone) if peer not in self.local_peers
        ]
        # Every owner posts our WRs, so they fit every NIC and path.
        self.max_sge = min(e["max_sge"] for e in everyone)
        self.mtu = min(e["mtu"] for e in everyone)
        # Per (table, gatherer): a partial WR, and a READ and a flag WR per QP.
        rows_per_wr = self.rows_per_wr(max_row_bytes)
        # Owners keep a mailbox per requester and table, sized for that
        # requester's fetches (engines may batch differently, as P and D do).
        box_sizes = [
            proxy.mailbox_bytes(
                triton.cdiv(e["max_items"], rows_per_wr) + 1 + 2 * self.qps_per_peer,
                e["max_items"],
            )
            for e in everyone
        ]
        groups = self.world * max_tables
        self.max_wrs = triton.cdiv(total, rows_per_wr) + groups * (
            1 + 2 * self.qps_per_peer
        )
        try:
            mailboxes = self._call(
                "connect",
                [
                    [peer, e["address"], e["qpns"][self.rank]]
                    for peer, e in remote
                    if peer != self.rank
                ],
                [[peer, e["address"], e["requests"][self.rank]] for peer, e in remote],
                box_sizes,
                max_tables,
            )
            for peer, e in remote:
                gid, lid = e["address"]
                qp = made[peer][0]
                address = (bytes.fromhex(gid), lid)
                self.verbs.connect(qp, e["notices"][self.rank], address)
        except RuntimeError as err:
            near = [p for p, e in remote if e["node"] == self.node and p != self.rank]
            if not near:
                raise
            raise RuntimeError(
                f"Engram pool RDMA could not connect GPU {self.rank}'s NIC to its "
                f"node's GPUs {near}. If the NICs of a node cannot reach each "
                "other (as on rail-optimized fabrics), use host shards "
                "(cpu_offload): they need no connections within a node."
            ) from err
        peers = {qpn: peer for peer, (_, qpn) in enumerate(made)}
        threading.Thread(
            target=self._watch, args=(cq, channel, peers), daemon=True
        ).start()
        from cuda.bindings import driver as cu

        _, ordering = cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
            cu.cuDeviceGet(gpu.index)[1],
        )
        # Rows, and the flags behind them, land in HBM if this GPU sees the
        # NIC's writes there in order without a flush; else in host memory.
        landing_bytes = total * max_row_bytes
        flags_bytes = 8 * groups * self.qps_per_peer
        landing: SharedHost | CuMem
        flags: SharedHost | CuMem
        if ordering >= 100:  # CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
            landing, flags = (CuMem(n, rdma=True) for n in (landing_bytes, flags_bytes))
            self.landing = device_view(landing.map(), landing_bytes, landing)
            self.flags = device_view(flags.map(), flags_bytes, flags)
        else:
            landing = SharedHost(landing_bytes, "landing")
            flags = SharedHost(flags_bytes, "flags")
            self.landing, self.flags = landing.gpu, flags.gpu
        self.flags = self.flags.view(torch.int64)
        self.flags.zero_()
        self.landing_at, _, self.landing_rkey = self._register(
            landing, verbs.PEER_WRITE
        )
        self.flags_at, _, self.flags_rkey = self._register(flags, verbs.PEER_ORDERED)
        self.host_landing = isinstance(landing, SharedHost)
        self._keep = [landing, flags]
        boxes, rkeys, lkeys = zip(*self._all_gather(mailboxes))
        # This GPU's first mailbox at each owner, in that owner proxy's
        # addresses; one follows per table.
        self.box_bytes = box_sizes[self.rank]
        mine = sum(box_sizes[: self.rank]) * max_tables
        self.boxes = torch.tensor(
            [box + mine for box in boxes], dtype=torch.int64, device=gpu
        )
        self.box_keys = torch.tensor(lkeys, dtype=torch.int32, device=gpu)
        self.box_rkeys = torch.tensor(rkeys, dtype=torch.int32, device=gpu)
        desc_bytes = proxy.mailbox_bytes(self.max_wrs, total)
        memory = CuMem(desc_bytes, rdma=True)
        memory.map()
        self.desc = device_view(memory.address, desc_bytes, memory).view(torch.int64)
        fd = memory.dmabuf_fd()
        self.desc_lkey, _ = self.verbs.reg_dmabuf(
            fd, memory.address, desc_bytes, verbs.NIC_READ
        )
        os.close(fd)
        self.queues = torch.tensor(
            self._map_queues(made), dtype=torch.int64, device=gpu
        )
        # Peers' QPs and mailboxes must be ready before anyone requests.
        group.barrier()
        self._call("start")
        self.counts = torch.zeros(groups, dtype=torch.int32, device=gpu)
        self.starts = torch.zeros(groups, dtype=torch.int64, device=gpu)
        self.used = torch.zeros(groups, dtype=torch.int32, device=gpu)
        self.wr_starts = torch.zeros(groups, dtype=torch.int64, device=gpu)
        self.wr_counts = torch.zeros(groups, dtype=torch.int64, device=gpu)
        self.ticket = torch.zeros(1, dtype=torch.int32, device=gpu)
        self.seq = torch.zeros(1, dtype=torch.int64, device=gpu)
        # Each table's last landed seq, and `wait_landed`'s view of a fetch.
        self.landed_seq = torch.full((max_tables,), -1, dtype=torch.int64, device=gpu)
        ready = {
            READY_SEQ.value: self.seq.data_ptr(),
            READY_USED.value: self.used.data_ptr(),
            READY_FLAGS.value: self.flags.data_ptr(),
            READY_WORLD.value: self.world,
            READY_QPS.value: self.qps_per_peer,
            READY_TIMEOUT.value: TIMEOUT_NS,
            READY_LANDED.value: self.landed_seq.data_ptr(),
            READY_FAILED.value: self.failed_gpu.data_ptr(),
        }
        self.ready = torch.tensor(
            [ready[i] for i in range(len(ready))], dtype=torch.int64, device=gpu
        )
        self.owner_pos = torch.empty(total, dtype=torch.int64, device=gpu)
        self.remote_addr = torch.empty(total, dtype=torch.int64, device=gpu)
        self.keys = torch.empty(total, dtype=torch.int32, device=gpu)
        # Each row's slot in the landing buffer, -1 if dead.
        self.slots = torch.empty(total, dtype=torch.int32, device=gpu)

    def _watch(self, cq: int, channel: int, peers: dict[int, int]) -> None:
        """Flag the first failed request. Requests are unsignaled: only failed
        ones complete, as when a peer's engine is gone."""
        wc = (verbs.ibv_wc * 16)()
        while True:
            self.verbs.notify_cq(cq)
            if not (count := self.verbs.poll_cq(cq, wc, 16)):
                self.verbs.cq_event(channel)
                continue
            for c in wc[:count]:
                if c.status:
                    logger.error(
                        "Engram pool RDMA request to GPU %d failed (ibv_wc status "
                        "%d, vendor error %#x); is its engine gone?",
                        peers.get(c.qp_num, -1),
                        c.status,
                        c.vendor_err,
                    )
                    self._fail()
                    return

    def _fail(self) -> None:
        """Fail every wait. A copy engine copies the flag in, so it lands even
        while waiting kernels hold every SM."""
        self.failed.fill_(1)
        with torch.cuda.stream(self._fail_stream):
            self.failed_gpu.copy_(self.failed, non_blocking=True)

    @staticmethod
    def _map_queues(qps: list[tuple[int, int]]) -> list[list[int]]:
        """Map QPs' send queues and doorbells for this GPU; returns rows of send
        queue, WQEs, doorbell record and register, QPN and WQEs posted."""
        rows, pages = [], {}
        page = mmap.PAGESIZE
        for qp, qpn in qps:
            queue = verbs.Device.mlx5_queue(qp)
            sq, wqes = queue.sq.buf, queue.sq.wqe_cnt
            record, register = queue.dbrec, queue.bf_reg
            rows.append([sq, wqes, record, register, qpn, 0, 0, 0])
            # cudaHostRegisterMapped, | IoMemory for the register (MMIO).
            wq_bytes = wqes * verbs.MLX5_SEND_WQE_BB
            for address, size, kind in ((sq, wq_bytes, 2), (record, 8, 2)):
                for at in range(address // page, -(-(address + size) // page)):
                    pages[at * page] = kind
            pages[register // page * page] = 6
        cudart = torch.cuda.cudart()
        for address, kind in pages.items():
            if (result := cudart.cudaHostRegister(address, page, kind)).value:
                raise RuntimeError(f"cudaHostRegister of an RDMA queue: {result}")
        return rows

    def rows_per_wr(self, row_bytes: int) -> int:
        """Rows one WRITE gathers: a packet's worth, within every NIC's SGEs."""
        return max(1, min(self.max_sge, self.mtu // row_bytes))

    def _all_gather(self, value) -> list:
        return [
            json.loads(v) for v in self.group.all_gather(json.dumps(value).encode())
        ]

    def _call(self, command: str, *args, fds: list[int] | None = None):
        sock = self.sock
        socket.send_fds(sock, [json.dumps([command, *args]).encode()], fds or [])
        for fd in fds or ():
            os.close(fd)
        status, result = json.loads(sock.recv(1 << 20))
        if status != "ok":
            raise RuntimeError(f"Engram pool RDMA proxy {command} failed:\n{result}")
        return result

    def _register(self, memory: SharedHost | CuMem, access: int) -> list[int]:
        """Register host memory or HBM with the owner proxy; returns its address
        for the NIC, lkey and rkey."""
        if isinstance(memory, SharedHost):
            fd = os.dup(memory.fd)
            args = ["host", memory.nbytes, 0, access]
        else:
            fd = memory.dmabuf_fd()
            args = ["dmabuf", memory.size, memory.address, access]
        return self._call("register", *args, fds=[fd])

    def shard(self, num_bytes: int, name: str) -> tuple[torch.Tensor, list, list]:
        """Allocate and register this GPU's shard; returns it (host memory for
        host shards), its `meta` arrays and what keeps them alive. Collective."""
        registered: SharedHost | CuMem
        if self.host_shards:
            registered = host = SharedHost(num_bytes, f"table_{name}", self.thp_dir)
            local, keep = host.cpu, [host]
        else:
            registered = cumem = CuMem(num_bytes, rdma=True)
            local, keep = device_view(cumem.map(), num_bytes, cumem), []
        # No SM reads a shard: this GPU's NIC gathers its own and its node's
        # host shards (over loopback); owners gather everything else.
        address, lkey, _ = self._register(registered, verbs.NIC_READ)
        fd = registered.fd if isinstance(registered, SharedHost) else -1
        peers = self._all_gather([address, lkey, os.getpid(), fd])
        bases, keys, gatherers = [], [], []
        for rank, (address, lkey, pid, fd) in enumerate(peers):
            gatherer = rank
            if rank in self.local_peers:
                # No QP connects GPUs of one node: open the owner's memfd
                # through /proc and gather it with this GPU's NIC.
                try:
                    address, lkey = self._call("map_peer", pid, fd, num_bytes)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Engram pool GPU {self.rank} cannot open GPU {rank}'s host "
                        "shard through /proc: the engines of a node must run as one "
                        "user in one PID namespace, with ptrace_scope allowing it"
                    ) from err
                gatherer = self.rank
            bases.append(address)
            keys.append(lkey)
            gatherers.append(gatherer)
        arrays = [
            torch.tensor(values, dtype=dtype, device="cuda")
            for values, dtype in (
                (bases, torch.int64),
                (keys, torch.int32),
                (gatherers, torch.int32),
            )
        ]
        return local, [array.data_ptr() for array in arrays], keep + arrays

    def start(
        self, meta: torch.Tensor, ids: torch.Tensor, fetch
    ) -> torch.Tensor | None:
        """Start gathering rows `ids` (`[..., TABLES, K]`) of the tables in `meta`
        on the current stream; the NICs move them while the stream goes on."""
        n, tables, k = ids.numel(), meta.shape[0], ids.shape[-1]
        if self.failed.item():
            raise RuntimeError("Engram pool RDMA failed: a request or proxy failed")
        if not n:
            return None
        words = fetch.row_stride // 8
        rows_per_wr = self.rows_per_wr(fetch.row_stride)
        vg_p2 = triton.next_power_of_2(self.world * tables)
        slots = self.slots[:n].view(tables, -1)
        self._launch(
            _plan_kernel,
            (triton.cdiv(n, 256),),
            (ids, meta, self.counts, self.owner_pos, self.remote_addr, self.keys)
            + (n, self.world, self.ticket, self.starts)
            + (self.wr_starts, self.wr_counts, self.used),
            dict(
                TABLES=tables,
                K=k,
                WORDS=words,
                ROWS_PER_WR=rows_per_wr,
                QPS=self.qps_per_peer,
                VG_P2=vg_p2,
                BLOCK=256,
            ),
        )
        self._launch(
            _wr_kernel,
            (triton.cdiv(n, 128),),
            (self.owner_pos, self.counts, self.remote_addr, self.keys, self.boxes)
            + (self.box_keys, self.desc, self.seq, self.landing_at, self.landing_rkey)
            + (self.flags_at, self.flags_rkey, n, self.world, self.rank)
            + (self.ticket, self.starts, self.wr_starts, self.wr_counts, self.used)
            + (self.box_rkeys, self.desc_lkey, self.queues, slots),
            dict(
                BOX_BYTES=self.box_bytes,
                MAX_WRS=self.max_wrs,
                ROWS_PER_WR=rows_per_wr,
                QPS=self.qps_per_peer,
                VG_P2=vg_p2,
                WORDS=words,
                TABLES=tables,
                K=k,
                BLOCK=128,
            ),
        )
        return slots

    def wait(self, slots: torch.Tensor | None, table: int, out: torch.Tensor) -> None:
        """Hold the current stream until table `table`'s rows (`slots`, as
        `start` returned) have landed, and place them into `out`."""
        if slots is None:
            return
        words = out.shape[-1] // 8
        self._launch(
            _place_kernel,
            (triton.cdiv(out.numel() // 8, 1024),),
            (self.owner_pos, self.starts, self.landing, out, out.numel() // 8)
            + (table, self.ready),
            dict(
                TABLES=slots.shape[0],
                K=out.shape[-2],
                HOST=self.host_landing,
                WORDS=words,
                BLOCK=1024,
            ),
        )

    def landed(self, slots: torch.Tensor | None, table: int) -> tuple | None:
        """Where table `table`'s rows land: the landing buffer, each row's slot
        in it, and the descriptor `wait_landed` reads."""
        return None if slots is None else (self.landing, slots[table], self.ready)

    def _launch(self, kernel, grid, args: tuple, consts: dict) -> None:
        """Launch `kernel` on the current stream, compiled once per constexprs
        and tensor dtype and alignment: Triton's per-call dispatch costs ~20 us
        here."""
        tensors = [a for a in args if type(a) is torch.Tensor]
        key = (kernel, *consts.values())
        key += tuple((a.dtype, a.data_ptr() % 16 == 0) for a in tensors)
        if (compiled := self._compiled.get(key)) is None:
            compiled = kernel.warmup(*args, grid=grid, num_warps=4, **consts)
            compiled._init_handles()
            self._compiled[key] = compiled
        x, y, z = (*grid, 1, 1)[:3]
        compiled.run(
            x,
            y,
            z,
            current_stream().cuda_stream,
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *args,
            *consts.values(),
        )

    def close(self) -> None:
        self.sock.close()
        self.proc.wait()
