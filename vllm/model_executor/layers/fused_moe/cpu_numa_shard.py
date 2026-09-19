# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decide whether to shard the CPU MoE experts across NUMA nodes, and place the
weights accordingly.

The kernel parallelises each of its two GEMM stages over the output rows of the
weight it reads, so giving a node a contiguous slice of those rows makes every
thread read only pages placed on its own node. What this module owns is the
*decision*: which nodes, or none at all.

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
import re
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

#: The microkernel's column unit (``2 * TILE_N`` in csrc/cpu/sgl-kernels/gemm.h).
#: A shard has to be a whole number of these, so it is the granularity the split
#: works in. Asserted against the kernel in tests rather than trusted here.
BLOCK_N = 32

#: Override the detection below with an explicit list of NUMA node ids, in the
#: order the OpenMP threads are bound to them, e.g. ``0,1,2,3``. Node ids, not a
#: count: on a machine where the usable nodes are 2 and 3, a count could only
#: mean nodes 0 and 1, which is exactly the mistake this module has to avoid.
NODES_ENV = "VLLM_CPU_MOE_NUMA_NODES"

_SYS_NODE = Path("/sys/devices/system/node")


def _parse_cpu_sequence(text: str) -> list[int]:
    """CPU ids in the order given, from the syntax both OpenMP runtimes accept.

    Order is the whole point -- these are the CPUs threads 0, 1, 2 ... land on,
    so this returns a list and not a set. ``GOMP_CPU_AFFINITY`` allows
    whitespace or commas between entries and ``lo-hi`` or ``lo-hi:stride``
    ranges; ``KMP_AFFINITY``'s ``proclist`` is the comma-separated subset.
    """
    if "{" in text:
        # A KMP proclist may group several CPUs per thread as ``{0,1}``. Nothing
        # below understands that, so decline rather than misread it.
        return []
    out: list[int] = []
    try:
        for token in text.replace(",", " ").split():
            stride = 1
            if ":" in token:
                token, _, step = token.partition(":")
                stride = int(step)
                if stride < 1:
                    return []
            if "-" in token:
                lo, _, hi = token.partition("-")
                out.extend(range(int(lo), int(hi) + 1, stride))
            else:
                out.append(int(token))
    except ValueError:
        return []
    return out


def _cpu_to_node() -> dict[int, int]:
    """Which NUMA node each CPU belongs to, straight out of sysfs."""
    mapping: dict[int, int] = {}
    if not _SYS_NODE.exists():
        return mapping
    for node_dir in sorted(_SYS_NODE.glob("node[0-9]*")):
        cpulist = node_dir / "cpulist"
        try:
            node = int(node_dir.name[len("node") :])
            for cpu in _parse_cpu_sequence(cpulist.read_text()):
                mapping[cpu] = node
        except (OSError, ValueError):
            continue
    return mapping


def omp_cpu_order() -> list[int]:
    """The CPUs the OpenMP runtime pins threads 0, 1, 2 ... to, in that order.

    Read off the environment and *not* off ``os.sched_getaffinity``, because
    vLLM's own binding (``vllm/utils/ompmultiprocessing.py``) sets these
    variables and never narrows the process mask. A rank auto-bound to one node
    still sees every CPU on the machine in its affinity mask, so the mask cannot
    tell that configuration apart from a rank deliberately spanning all of them
    -- and those two want opposite answers.

    Empty when the binding does not determine a per-thread CPU. That covers the
    generic fallback vLLM sets when it has neither libiomp nor libgomp --
    ``OMP_PLACES={0,1,...}`` with ``OMP_PROC_BIND=true``, a *single* place
    holding every CPU, which pins nothing in particular -- and the case of no
    binding at all.
    """
    kmp = os.environ.get("KMP_AFFINITY", "")
    if "explicit" in kmp:
        proclist = re.search(r"proclist\s*=\s*\[([^\]]*)\]", kmp)
        return _parse_cpu_sequence(proclist.group(1)) if proclist else []
    return _parse_cpu_sequence(os.environ.get("GOMP_CPU_AFFINITY", ""))


def shard_nodes(num_threads: int | None = None) -> list[int]:
    """The NUMA node each shard's threads run on, or ``[]`` for "do not shard".

    The kernel maps OpenMP thread ``ith`` to shard ``ith // (nth // shards)``.
    That is an assumption about how the threads are laid out, so this verifies
    it rather than hoping: the first ``nth`` entries of the CPU list have to
    fall into equal-length contiguous runs, each run entirely on one node. If
    they do, the run boundaries *are* the shard boundaries and the node of each
    run is where that shard's pages belong.

    Checking the runs is what rules out the three ways the index-to-node map
    silently stops holding: threads bound to one node's CPUs (the auto-bind
    default), ``OMP_NUM_THREADS`` shorter than the CPU list so that the first
    ``nth`` threads sit on a subset of the nodes, and a CPU list interleaved
    across nodes rather than grouped by them.
    """
    order = omp_cpu_order()
    if not order:
        logger.debug(
            "CPU MoE: the OpenMP thread binding does not fix a CPU per thread, "
            "so a thread's NUMA node cannot be derived from its index. Not "
            "sharding. Set VLLM_CPU_OMP_THREADS_BIND to pin the threads, or %s "
            "to name the nodes explicitly.",
            NODES_ENV,
        )
        return []

    if num_threads is None:
        num_threads = torch.get_num_threads()
    if num_threads < 2 or num_threads > len(order):
        # More threads than pinned CPUs means the runtime wraps the list around
        # and a thread's node stops being a function of its index.
        return []

    cpu_node = _cpu_to_node()
    try:
        nodes = [cpu_node[cpu] for cpu in order[:num_threads]]
    except KeyError:
        return []

    distinct: list[int] = []
    for node in nodes:
        if node not in distinct:
            distinct.append(node)
    shards = len(distinct)
    if shards < 2:
        return []

    # The kernel rounds the thread count down to even for m > 1
    # (``adjust_num_threads`` in csrc/cpu/sgl-kernels/common.h) and then only
    # shards if the result divides. Require both, so the policy never places
    # pages for a split the kernel will refuse to make.
    if num_threads % shards != 0 or ((num_threads >> 1) << 1) % shards != 0:
        logger.debug(
            "CPU MoE spans %d NUMA nodes but %d threads do not divide evenly "
            "among them. Not sharding.",
            shards,
            num_threads,
        )
        return []

    per_node = num_threads // shards
    for shard, node in enumerate(distinct):
        run = nodes[shard * per_node : (shard + 1) * per_node]
        if any(n != node for n in run):
            logger.debug(
                "CPU MoE threads are bound across %d NUMA nodes but not in "
                "equal contiguous runs, so a thread's node is not its index "
                "over %d. Not sharding.",
                shards,
                per_node,
            )
            return []
    return distinct


def plan_shard_nodes() -> list[int]:
    """``shard_nodes()``, or the explicit node list from the environment."""
    forced = os.environ.get(NODES_ENV, "").strip()
    if forced:
        nodes = _parse_cpu_sequence(forced)
        if len(nodes) >= 2 and len(set(nodes)) == len(nodes):
            return nodes
        if len(nodes) == 1:
            return []
        logger.warning(
            "%s=%r is not a list of two or more distinct node ids; ignoring.",
            NODES_ENV,
            forced,
        )
    return shard_nodes()


def configure() -> list[int]:
    """Work out which nodes to shard onto, tell the kernel, and return them."""
    nodes = plan_shard_nodes()
    torch.ops._C.set_cpu_moe_numa_nodes(nodes)
    if nodes:
        # A str, not the list: info_once hashes its arguments to deduplicate.
        logger.info_once(
            "CPU MoE experts sharded across NUMA nodes %s; each node computes "
            "the output rows whose weights are placed on it.",
            ",".join(str(n) for n in nodes),
        )
    return nodes


def place_expert_weights(*weights: torch.Tensor) -> None:
    """Move each shard's rows onto the node that will compute them.

    A no-op when nothing was sharded, and best-effort otherwise: a weight whose
    pages did not move is slower, not wrong, so this warns instead of raising.
    """
    if torch.ops._C.cpu_moe_numa_shards() <= 1:
        return
    for weight in weights:
        if weight is None or weight.device.type != "cpu":
            continue
        torch.ops._C.place_moe_expert_weight(weight, BLOCK_N)


def configure_and_place(model: torch.nn.Module) -> list[int]:
    """Set the shard nodes and move the MoE expert weights to match them.

    Called once, after the model is loaded: by then the threads are bound and
    the weights are resident, which are the two things the decision depends on.
    Doing it here rather than in each quantization method's
    ``process_weights_after_loading`` is deliberate -- the MXFP4 and FP8 CPU
    paths do not route through the experts' post-load hook at all, so hooking
    per-method would silently miss them.

    Returns the shard nodes, which is empty when nothing was sharded.
    """
    nodes = configure()
    if not nodes:
        return nodes

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
        "Placed the expert weights of %d MoE layer(s) across NUMA nodes %s.",
        placed,
        ",".join(str(n) for n in nodes),
    )
    return nodes
