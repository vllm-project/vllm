# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target selection and residency checks for the specialized-decode kernels.

Every kernel here uses `tcgen05` / TMEM / block-scaled MMA,
i.e. arch-specific ("a") features of the SM100 family. They are therefore
compiled per exact compute capability (`sm_100a` on B200/GB200, `sm_103a`
on B300/GB300), never as portable PTX. `ARCHS` is the one table to extend
when another target is validated; nothing else in the package spells an arch.

Nothing in this module trusts a hard-coded SM count. `DeviceSpec` is read
from the live device; grids that spin on peers or predecessors call
`check_persistent_grid` / `check_co_resident` at build time and fail closed
(`ResidencyError`) rather than risk a deadlock at run time.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

# (major, minor) -> arch name. Only validated targets go here.
ARCHS: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",  # B200 / GB200
    (10, 3): "sm_103a",  # B300 / GB300
}

# Per-SM / per-CTA resources of the SM100 family that the kernels' static shared-memory
# and TMEM budgets were written against. Verified against the live device in
# ``device_spec`` (shared memory) where the driver reports it.
TMEM_COLUMNS = 512
RESERVED_SMEM_PER_CTA = 1024  # the driver reserves 1 KB per resident CTA


class UnsupportedArch(RuntimeError):
    pass


class ResidencyError(RuntimeError):
    """A persistent / spinning grid cannot be fully co-resident: launching it
    could deadlock, so the stage refuses to build."""


@dataclass(frozen=True)
class DeviceSpec:
    index: int
    name: str
    capability: tuple[int, int]
    arch: str
    sm_count: int
    smem_per_sm: int  # bytes, shared memory per multiprocessor
    smem_per_block_opt_in: int  # bytes, max dynamic smem a CTA can opt into
    max_threads_per_sm: int
    regs_per_sm: int


def arch_for(capability: tuple[int, int]) -> str:
    try:
        return ARCHS[tuple(capability)]  # type: ignore[index]
    except KeyError:
        raise UnsupportedArch(
            f"compute capability {capability[0]}.{capability[1]} is not a "
            f"validated target of the MiniMax-M3 specialized decode kernels "
            f"(validated: {sorted(ARCHS.values())})"
        ) from None


@functools.cache
def device_spec(index: int | None = None) -> DeviceSpec:
    import torch

    if index is None:
        index = torch.accelerator.current_device_index()
    p = torch.cuda.get_device_properties(index)
    cap = (p.major, p.minor)
    arch = arch_for(cap)
    smem_sm = int(getattr(p, "shared_memory_per_multiprocessor", 0))
    smem_opt_in = int(getattr(p, "shared_memory_per_block_optin", 0))
    if smem_sm <= 0 or smem_opt_in <= 0:
        # Older torch builds do not expose these; fall back to the driver.
        smem_sm, smem_opt_in = _driver_smem(index)
    return DeviceSpec(
        index=index,
        name=p.name,
        capability=cap,
        arch=arch,
        sm_count=int(p.multi_processor_count),
        smem_per_sm=smem_sm,
        smem_per_block_opt_in=smem_opt_in,
        max_threads_per_sm=int(getattr(p, "max_threads_per_multi_processor", 2048)),
        regs_per_sm=int(getattr(p, "regs_per_multiprocessor", 65536)),
    )


def _driver_smem(index: int) -> tuple[int, int]:
    from cuda.bindings import runtime as cudart

    def attr(a):
        err, v = cudart.cudaDeviceGetAttribute(a, index)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaDeviceGetAttribute({a}) failed: {err}")
        return int(v)

    return (
        attr(cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor),
        attr(cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin),
    )


def ctas_per_sm(
    spec: DeviceSpec,
    *,
    smem_bytes: int,
    threads: int,
    tmem_columns: int = 0,
    regs_per_thread: int = 0,
) -> int:
    """Upper bound on resident CTAs per SM of one kernel from its declared
    resources. Conservative (ignores register allocation granularity beyond a
    per-thread count); prefer the occupancy the CUDA runtime reports for a
    compiled kernel when the extension exposes it."""
    if threads <= 0:
        raise ValueError("threads must be positive")
    lim = spec.max_threads_per_sm // threads
    per_cta = smem_bytes + RESERVED_SMEM_PER_CTA
    if smem_bytes > spec.smem_per_block_opt_in:
        return 0
    lim = min(lim, spec.smem_per_sm // per_cta) if per_cta > 0 else lim
    if tmem_columns:
        lim = min(lim, TMEM_COLUMNS // tmem_columns)
    if regs_per_thread:
        lim = min(lim, spec.regs_per_sm // _regs_per_cta(regs_per_thread, threads))
    return max(lim, 0)


def check_persistent_grid(
    name: str, grid_ctas: int, occupancy_per_sm: int, spec: DeviceSpec
) -> None:
    """Fail closed unless every CTA of a grid whose CTAs wait on each other
    (or on a co-running grid) can be resident at once."""
    cap = spec.sm_count * occupancy_per_sm
    if occupancy_per_sm <= 0 or grid_ctas > cap:
        raise ResidencyError(
            f"{name}: persistent grid of {grid_ctas} CTAs needs co-residency "
            f"but {spec.name} holds {spec.sm_count} SMs x {occupancy_per_sm} "
            f"CTA/SM = {cap}"
        )


@dataclass(frozen=True)
class GridUse:
    """One grid that must be resident together with others (a spinning
    consumer and the producer it polls, launched back to back with PDL)."""

    name: str
    ctas: int
    smem_bytes: int  # static + dynamic shared memory per CTA
    threads: int
    tmem_columns: int = 0
    regs_per_thread: int = 0  # 0 = unknown (not counted)


def _regs_per_cta(regs_per_thread: int, threads: int) -> int:
    """Registers a CTA holds: per-thread count rounded up to the allocation
    unit of 8 (256 registers per warp), times the warps."""
    if regs_per_thread <= 0:
        return 0
    return -(-regs_per_thread // 8) * 8 * 32 * -(-threads // 32)


def check_co_resident(grids: list[GridUse], spec: DeviceSpec) -> None:
    """Fail closed unless all `grids` fit on the device at the same time.

    Worst-case model independent of the block scheduler: every SM may have to
    host ceil(ctas / sm_count) CTAs of each grid at once, so the per-SM sums
    of shared memory, threads, registers (where known) and TMEM columns must
    fit."""
    smem = threads = tmem = regs = 0
    for g in grids:
        k = -(-g.ctas // spec.sm_count)
        if g.smem_bytes > spec.smem_per_block_opt_in:
            raise ResidencyError(
                f"{g.name}: {g.smem_bytes} B smem per CTA exceeds the "
                f"{spec.smem_per_block_opt_in} B opt-in limit of {spec.name}"
            )
        smem += k * (g.smem_bytes + RESERVED_SMEM_PER_CTA)
        threads += k * g.threads
        tmem += k * g.tmem_columns
        regs += k * _regs_per_cta(g.regs_per_thread, g.threads)
    if (
        smem > spec.smem_per_sm
        or threads > spec.max_threads_per_sm
        or tmem > TMEM_COLUMNS
        or regs > spec.regs_per_sm
    ):
        names = ", ".join(f"{g.name}({g.ctas})" for g in grids)
        raise ResidencyError(
            f"co-residency of [{names}] not guaranteed on {spec.name} "
            f"({spec.sm_count} SMs): per-SM worst case smem {smem} B "
            f"(limit {spec.smem_per_sm}), threads {threads} "
            f"(limit {spec.max_threads_per_sm}), registers {regs} "
            f"(limit {spec.regs_per_sm}), TMEM cols {tmem} "
            f"(limit {TMEM_COLUMNS})"
        )


__all__ = [
    "ARCHS",
    "TMEM_COLUMNS",
    "DeviceSpec",
    "GridUse",
    "ResidencyError",
    "UnsupportedArch",
    "arch_for",
    "check_co_resident",
    "check_persistent_grid",
    "ctas_per_sm",
    "device_spec",
]
