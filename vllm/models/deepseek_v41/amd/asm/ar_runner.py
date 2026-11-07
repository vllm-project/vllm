# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The folded all-reduces' host side: landing buffers and flags are hipMalloc'd per
rank and shared with the peers through IPC handles (exchanged over a CPU process
group), as vLLM's custom all-reduce does. `out` is written only by its own
rank, so it is a plain torch tensor (graph-pool friendly).

Every rank must make the same calls in the same order (the per-wave sequence
counters live in the flags buffer and must agree). A rank that stops calling
leaves its peers spinning until ar2.POLLS runs out; `errors()` counts those
timeouts.
"""

import ctypes

import torch

from vllm.models.deepseek_v41.amd.asm.kernels import ar2
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import Kernel

H = ar2.H


def _rt():
    """The HIP runtime through vLLM's wrapper (the library torch loaded)."""
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    return CudaRTLibrary()


def flag_state(ptr: int, nb: int, g: int) -> tuple[list, list]:
    """The last 2g dwords of a flags buffer (ar2 / eg_ag layout: ... ctr [g], err [g])
    as (ctr, err) lists. Syncs."""
    f = torch.empty(nb // 4, dtype=torch.int32, device="cuda")
    torch.accelerator.synchronize()
    _rt().cudaMemcpy(ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(ptr), nb)
    t = f[-2 * g :].tolist()
    return t[:g], t[g:]


def ipc_alloc(sizes: dict, rank: int, cpu_group, world: int = 4) -> tuple[dict, dict]:
    """hipMalloc'd, zeroed buffers {name: bytes} on every rank, shared through
    IPC handles (collective over cpu_group) → (mine {name: ptr}, all {name:
    [ptr of rank d's buffer in this process, d < world]})."""
    from vllm.distributed.device_communicators.cuda_wrapper import cudaIpcMemHandle_t

    rt = _rt()
    torch.accelerator.synchronize()
    mine, handles = {}, {}
    # plain hipMalloc (coarse-grained, cached): peers write these buffers and we read
    # them; an uncached allocation (as vLLM's custom all-reduce uses for its signals)
    # costs ~20 µs per decode step
    for n, sz in sizes.items():
        p = rt.cudaMalloc(sz)
        rt.cudaMemset(p, 0, sz)
        mine[n], handles[n] = p.value, bytes(rt.cudaIpcGetMemHandle(p))
    torch.accelerator.synchronize()
    gathered: list = [None] * world
    torch.distributed.all_gather_object(gathered, handles, group=cpu_group)
    every: dict[int, list] = {n: [] for n in sizes}
    failed: list[str] = []
    for n in sizes:
        for d in range(world):
            if d == rank:
                every[n].append(mine[n])
                continue
            hd = cudaIpcMemHandle_t.from_buffer_copy(gathered[d][n])
            try:
                every[n].append(rt.cudaIpcOpenMemHandle(hd).value)
            except RuntimeError as e:
                failed.append(f"rank {rank}: {n} of rank {d}: {e}")
    # every rank raises if any failed (no rank waits in _ipc_check's barrier)
    errs: list = [None] * world
    torch.distributed.all_gather_object(errs, failed, group=cpu_group)
    if any(errs):
        raise RuntimeError(f"IPC handles failed to open: {sum(errs, [])}")
    _ipc_check(rt, mine, every, rank, cpu_group, world)
    return mine, every


_ipc_seq = [0]


def _ipc_check(rt, mine: dict, every: dict, rank: int, cpu_group, world: int) -> None:
    """Every rank writes a signature (magic, rank, allocation #, buffer #) at
    the start of each of its new buffers; every rank reads its peers' through
    the opened IPC pointers. A mismatch means a mapping lands in some other
    allocation (peer flag writes that never arrived).
    Prints the pointers' page offsets too (suballocation). Buffers are re-zeroed."""
    import torch.distributed as dist

    _ipc_seq[0] += 1
    names = list(mine)
    sig = lambda r, i: (0x5A5A0000 | i, r, _ipc_seq[0], 0x1234)
    for i, n in enumerate(names):
        buf = (ctypes.c_int32 * 4)(*sig(rank, i))
        rt.cudaMemcpy(ctypes.c_void_p(mine[n]), ctypes.cast(buf, ctypes.c_void_p), 16)
    torch.accelerator.synchronize()
    dist.barrier(group=cpu_group)
    bad = []
    for i, n in enumerate(names):
        for d in range(world):
            if d == rank:
                continue
            buf = (ctypes.c_int32 * 4)()
            rt.cudaMemcpy(
                ctypes.cast(buf, ctypes.c_void_p), ctypes.c_void_p(every[n][d]), 16
            )
            got = tuple(v & 0xFFFFFFFF for v in buf)
            want = tuple(v & 0xFFFFFFFF for v in sig(d, i))
            if got != want:
                bad.append(f"{n}@r{d} read {[hex(v) for v in got]}")
    dist.barrier(group=cpu_group)
    for n in names:
        rt.cudaMemset(ctypes.c_void_p(mine[n]), 0, 16)
    torch.accelerator.synchronize()
    dist.barrier(group=cpu_group)
    if bad:
        raise RuntimeError(
            f"asm decode: IPC buffers don't map to the peers' (rank {rank}): "
            + "; ".join(bad)
        )


class WoBAllReduce:
    """K1's wo_b with all-reduce #1 folded in (kernels/wob_ar.py): one launch
    from wo_a's z (MXFP8 frag) to the all-reduced [M, 5120] bf16; wire "bf16"
    (the sum vLLM's custom all-reduce computes) or "fp8" (MXFP8 partials).

        # once per process, outside graph capture
        wob = WoBAllReduce(rank, cpu_group, wire="bf16")
        out = wob(w_frag, ws_frag, z, zs, m)

    Collective: every rank makes the same calls (same M) in the same order."""

    def __init__(self, rank: int, cpu_group, world: int = 4, wire: str = "bf16"):
        from vllm.models.deepseek_v41.amd.asm.kernels import wob_ar

        assert world == 4, "wob_ar is written for TP4"
        self.rank, self.wob_ar = rank, wob_ar
        sizes = dict(
            land=wob_ar.LAND_BYTES, land2=wob_ar.LAND_BYTES, flags=wob_ar.FLAG_BYTES
        )
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for d in range(world):
            self.ptrs.update(
                {
                    f"land{d}": every["land"][d],
                    f"land2_{d}": every["land2"][d],
                    f"flags{d}": every["flags"][d],
                }
            )
        self.flags_mine = mine["flags"]
        self.g, self.flag_layout = (
            wob_ar.NW,
            (wob_ar.FLAG_BYTES, 2),
        )  # for IPC diagnostics
        d = wob_ar.source(rank, wire)
        self.k = Kernel(d["name"], d["src"], d["ka"])
        self.dummy = torch.zeros(64, dtype=torch.uint8, device="cuda")
        torch.distributed.barrier(group=cpu_group)

    def __call__(
        self, w, ws, z, zs, m: int, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert 1 <= m <= 256, m
        out = (
            torch.empty(m, H, dtype=torch.bfloat16, device=z.device)
            if out is None
            else out
        )
        d = self.dummy
        self.k(
            self.wob_ar.NCTA,
            256,
            w=w,
            ws=ws,
            x=z,
            xs=zs,
            out=out,
            outs=d,
            part=d,
            cnt=d,
            cs=d,
            pos=d,
            dbg=d,
            m=m,
            **self.ptrs,
        )
        return out

    def errors(self) -> int:
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per wave (CTA·4 + wave). Syncs."""
        return flag_state(self.flags_mine, self.wob_ar.FLAG_BYTES, self.wob_ar.NW)


class CombAllReduce:
    """K2's MoE combine with all-reduce #2 folded in (kernels/comb_ar.py): one launch
    from the weighted expert rows y [M·6, 5120] bf16 (K2Runner's), the picks' ids
    [M·6] and the shared expert's output add [M, 5120] bf16 to the all-reduced
    bf16(bf16(Σ y) + add); wire as WoBAllReduce's. Collective like WoBAllReduce."""

    def __init__(self, rank: int, cpu_group, world: int = 4, wire: str = "bf16"):
        from vllm.models.deepseek_v41.amd.asm.kernels import comb_ar

        assert world == 4, "comb_ar is written for TP4"
        self.rank, self.comb_ar, self.g = rank, comb_ar, comb_ar.NW
        sizes = dict(land=4 * ar2.QMAX, land2=4 * ar2.QMAX, flags=comb_ar.FLAG_BYTES)
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for d in range(world):
            self.ptrs.update(
                {
                    f"land{d}": every["land"][d],
                    f"land2_{d}": every["land2"][d],
                    f"flags{d}": every["flags"][d],
                }
            )
        self.flags_mine = mine["flags"]
        self.flag_layout = (comb_ar.FLAG_BYTES, 2)  # for IPC diagnostics
        d = comb_ar.source(rank, wire)
        self.k = Kernel(d["name"], d["src"], d["ka"])
        torch.distributed.barrier(group=cpu_group)

    def __call__(self, y, ids, add, m: int) -> torch.Tensor:
        assert (
            1 <= m <= ar2.MAX_M
            and y.dtype == torch.bfloat16
            and ids.dtype == torch.int32
        )
        assert add.dtype == torch.bfloat16
        assert (
            y.is_contiguous()
            and ids.is_contiguous()
            and add.is_contiguous()
            and add.shape == (m, H)
        )
        out = torch.empty(m, H, dtype=torch.bfloat16, device=add.device)
        self.k(
            self.comb_ar.grid(m),
            256,
            **self.ptrs,
            out=out,
            y=y,
            ids=ids,
            add=add,
            m=m,
            y_bytes=m * 6 * H * 2,
            ids_bytes=m * 6 * 4,
            add_bytes=m * H * 2,
        )
        return out

    def errors(self) -> int:
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per wave (CTA·4 + wave). Syncs."""
        return flag_state(self.flags_mine, self.flag_layout[0], self.g)
