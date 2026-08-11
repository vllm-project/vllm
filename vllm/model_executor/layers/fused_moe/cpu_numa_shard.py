# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decide whether to shard the CPU MoE experts across NUMA nodes, and place the
weights accordingly.

The kernel parallelises each of its two GEMM stages over the output rows of the
weight it reads, so giving a node a contiguous slice of those rows makes every
thread read only pages placed on its own node. What this module owns is the
*decision*: how many shards, or none at all.

**When this helps, and when it does nothing.** With the default
``VLLM_CPU_OMP_THREADS_BIND=auto``, ``_get_autobind_cpu_ids()`` gives each rank
the CPUs of exactly one NUMA node, so CPUs and memory already agree and there is
nothing to gain -- this declines and the kernel takes its usual path. It helps
in one configuration: a single rank whose threads span several NUMA nodes, which
is what a manual ``VLLM_CPU_OMP_THREADS_BIND=0-127`` on a multi-socket box gives
you.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

#: The microkernel's column unit (``2 * TILE_N`` in csrc/cpu/sgl-kernels/gemm.h).
#: A shard has to be a whole number of these, so it is the granularity the split
#: works in. Asserted against the kernel in tests rather than trusted here.
BLOCK_N = 32

_SYS_NODE = Path("/sys/devices/system/node")


def _parse_cpulist(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def effective_numa_nodes() -> list[int]:
    """Nodes that still have at least one CPU this process may run on.

    Deliberately not "what sysfs reports". Under ``numactl --cpunodebind=0`` the
    machine still has four nodes but the process can only use one, and sharding
    into four there puts every slice behind the same memory controller -- which
    measures roughly six times *worse* than not sharding. The mask is what
    decides.
    """
    if not _SYS_NODE.exists():
        return []
    try:
        allowed = os.sched_getaffinity(0)
    except AttributeError:  # pragma: no cover - non-Linux
        return []

    nodes = []
    for node_dir in sorted(_SYS_NODE.glob("node[0-9]*")):
        cpulist = node_dir / "cpulist"
        if not cpulist.exists():
            continue
        try:
            if _parse_cpulist(cpulist.read_text()) & allowed:
                nodes.append(int(node_dir.name[4:]))
        except (OSError, ValueError):
            continue
    return nodes


def omp_threads_are_bound_in_order() -> bool:
    """Whether OpenMP thread *i* is pinned to the *i*-th CPU of the list.

    The kernel maps thread ``ith`` to node ``ith // (nth // shards)``, which is
    only true if the runtime hands out threads in the order of the CPU list.
    ``KMP_AFFINITY=...,explicit,proclist=[...]`` and ``GOMP_CPU_AFFINITY`` both
    do that. The generic fallback vLLM sets otherwise -- ``OMP_PLACES={0,1,...}``
    with ``OMP_PROC_BIND=true`` -- defines a *single* place holding every CPU,
    which pins nothing in particular, so this returns False rather than assume.
    """
    kmp = os.environ.get("KMP_AFFINITY", "")
    if "explicit" in kmp and "proclist" in kmp:
        return True
    return bool(os.environ.get("GOMP_CPU_AFFINITY", "").strip())


def plan_shards(num_threads: int | None = None) -> int:
    """How many NUMA shards to split the MoE weights into. 1 means "don't".

    All-or-nothing on the count: if the topology does not allow one shard per
    effective node, the answer is 1 rather than some smaller divisor. Splitting
    into fewer shards than there are nodes leaves memory controllers idle while
    their cores keep reading from the others, and measures worse than the plain
    path every time it was tried.
    """
    forced = os.environ.get("VLLM_CPU_MOE_NUMA_SHARDS")
    if forced:
        try:
            value = int(forced)
        except ValueError:
            logger.warning(
                "VLLM_CPU_MOE_NUMA_SHARDS=%r is not an integer; ignoring.", forced
            )
        else:
            if value >= 1:
                return value
            logger.warning("VLLM_CPU_MOE_NUMA_SHARDS=%d is not >= 1; ignoring.", value)

    nodes = effective_numa_nodes()
    if len(nodes) <= 1:
        return 1

    if not omp_threads_are_bound_in_order():
        logger.debug(
            "CPU MoE spans %d NUMA nodes but the OpenMP threads are not bound "
            "in CPU-list order, so a thread's node cannot be derived from its "
            "index. Not sharding. Set VLLM_CPU_OMP_THREADS_BIND to pin them, "
            "or VLLM_CPU_MOE_NUMA_SHARDS to override this check.",
            len(nodes),
        )
        return 1

    if num_threads is None:
        num_threads = torch.get_num_threads()
    if num_threads % len(nodes) != 0:
        logger.debug(
            "CPU MoE spans %d NUMA nodes but %d threads do not divide evenly "
            "among them. Not sharding.",
            len(nodes),
            num_threads,
        )
        return 1

    return len(nodes)


def configure() -> int:
    """Work out the shard count, tell the kernel, and return it."""
    shards = plan_shards()
    torch.ops._C.set_cpu_moe_numa_shards(shards)
    if shards > 1:
        logger.info_once(
            "CPU MoE experts sharded across %d NUMA nodes; each node computes "
            "the output rows whose weights are placed on it.",
            shards,
        )
    return shards


def place_expert_weights(*weights: torch.Tensor) -> None:
    """Move each shard's rows onto the node that will compute them.

    A no-op when the shard count is 1, and best-effort otherwise: a weight whose
    pages did not move is slower, not wrong, so this warns instead of raising.
    """
    if torch.ops._C.cpu_moe_numa_shards() <= 1:
        return
    for weight in weights:
        if weight is None or weight.device.type != "cpu":
            continue
        torch.ops._C.place_moe_expert_weight(weight, BLOCK_N)


def configure_and_place(model: torch.nn.Module) -> int:
    """Set the shard count and move the MoE expert weights to match it.

    Called once, after the model is loaded: by then the threads are bound and
    the weights are resident, which are the two things the decision depends on.
    Doing it here rather than in each quantization method's
    ``process_weights_after_loading`` is deliberate -- the MXFP4 and FP8 CPU
    paths do not route through the experts' post-load hook at all, so hooking
    per-method would silently miss them.

    Returns the shard count, which is 1 when nothing was sharded.
    """
    shards = configure()
    if shards <= 1:
        return shards

    placed = 0
    for _, module in model.named_modules():
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if w13 is None or w2 is None:
            continue
        # Only the weights, not their scales. The scales are a 16th of the bytes
        # for MXFP4, and their row axis does not line up with the block split on
        # the FP8 block-quantized path (`scale_offset_per_block` divides by the
        # group size there), so placing them would need a second rule to get a
        # few percent of the traffic.
        place_expert_weights(w13.data, w2.data)
        placed += 1

    logger.info(
        "Placed the expert weights of %d MoE layer(s) across %d NUMA nodes.",
        placed,
        shards,
    )
    return shards
