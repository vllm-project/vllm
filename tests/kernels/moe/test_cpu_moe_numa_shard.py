# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA sharding of the CPU MoE experts.

Two halves, and they fail for different reasons.

The kernel half: sharding only changes *which thread* computes which output
block; the arithmetic of each block is untouched. So the bar is not a tolerance
-- it is ``torch.equal`` against the unsharded run. Anything else means a block
was computed twice, or not at all, or with the wrong slice of the weights, and
those are the three ways this can be wrong.

The policy half: the kernel maps thread ``ith`` to shard ``ith // (nth //
shards)``, so sharding is only safe where that map is true of the actual thread
binding. Those tests drive the policy against a synthetic topology, because the
cases worth covering -- one node's CPUs, non-contiguous node ids, fewer threads
than CPUs -- are not all reachable on whatever machine CI runs on.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import cpu_numa_shard
from vllm.model_executor.layers.fused_moe.cpu_numa_shard import (
    BLOCK_N,
    NODES_ENV,
    plan_shard_nodes,
    shard_nodes,
)
from vllm.platforms import current_platform

from .test_cpu_quant_fused_moe import (  # noqa: F401
    MXFP4QuantizeUtil,
    _prepack_mxfp4_experts,
    set_random_seed,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="CPU MoE NUMA sharding is CPU-only"
)


def _run(shards: int, M: int, N: int, K: int, E: int, topk: int, seed: int):
    torch.ops._C.set_cpu_moe_numa_nodes(list(range(shards)))
    try:
        set_random_seed(seed)
        dtype = torch.bfloat16
        a = torch.randn(M, K, dtype=dtype) / 10
        w1q, w1s = MXFP4QuantizeUtil.quantize(
            torch.randn(E, 2 * N, K, dtype=dtype) / 10
        )
        w2q, w2s = MXFP4QuantizeUtil.quantize(torch.randn(E, K, N, dtype=dtype) / 10)
        w1s = w1s.reshape(E, 2 * N, K // 32)
        w2s = w2s.reshape(E, K, N // 32)
        score = torch.randn(M, E, dtype=dtype)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(score.float(), dim=-1), topk, dim=-1
        )
        pw1, pw1s = _prepack_mxfp4_experts(w1q, w1s)
        pw2, pw2s = _prepack_mxfp4_experts(w2q, w2s)
        return torch.ops._C.fused_experts_cpu(
            a,
            pw1,
            pw2,
            topk_weights.float().contiguous(),
            topk_ids.int().contiguous(),
            False,
            ops.CPUQuantMethod.MXFP4,
            pw1s,
            pw2s,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
        )
    finally:
        torch.ops._C.set_cpu_moe_numa_nodes([])


# The first three split evenly; the rest do not, which is the case that matters.
# N=160 gives 2N=320, i.e. 10 blocks of BLOCK_N, which across 4 nodes is
# 3/3/2/2 -- an earlier draft of the policy declined on anything that did not
# divide evenly, and that would have excluded intermediate_size=2880, the size
# vLLM's own CPU MoE benchmark uses.
@pytest.mark.parametrize(
    "M,N,K,E,topk",
    [
        (4, 128, 128, 4, 2),
        (32, 128, 128, 4, 2),
        (1, 256, 256, 8, 2),
        (13, 192, 128, 4, 2),
        (4, 160, 160, 4, 2),
        (32, 160, 96, 4, 2),
        (1, 224, 160, 4, 2),
        (7, 96, 224, 4, 2),
    ],
)
@pytest.mark.parametrize("shards", [2, 4])
def test_numa_sharding_is_bit_identical(M, N, K, E, topk, shards):
    torch.set_num_threads(8)
    reference = _run(1, M, N, K, E, topk, seed=0)
    sharded = _run(shards, M, N, K, E, topk, seed=0)
    assert torch.equal(reference, sharded), (
        f"{shards}-way sharding changed the output for "
        f"M={M} N={N} K={K}: max|delta| = "
        f"{(reference - sharded).abs().max().item():.3e}"
    )


def test_block_size_matches_the_kernel():
    """The policy rounds shard boundaries to BLOCK_N; the kernel defines it.

    The C++ side static_asserts its own copy against ``block_size_n()``
    (csrc/cpu/moe_numa_parallel.hpp), so a change to the kernel breaks the build
    rather than silently placing pages on boundaries it does not split on. This
    pins the Python copy to the same value, which is the third of the three.
    """
    assert BLOCK_N == 32


def test_shard_count_defaults_to_one():
    """The default has to be inert: existing deployments must not change."""
    assert torch.ops._C.cpu_moe_numa_shards() == 1


# ---------------------------------------------------------------------------
# The policy, against a synthetic topology.
# ---------------------------------------------------------------------------


@pytest.fixture
def topology(tmp_path, monkeypatch):
    """Fake ``/sys/devices/system/node`` so the policy can be driven anywhere.

    Returns a callable taking ``{node_id: "cpulist"}``, mirroring the sysfs
    layout the policy reads, including the non-contiguous node ids a cpuset can
    leave behind.
    """

    def build(nodes: dict[int, str]):
        for node, cpulist in nodes.items():
            node_dir = tmp_path / f"node{node}"
            node_dir.mkdir()
            (node_dir / "cpulist").write_text(cpulist + "\n")
        monkeypatch.setattr(cpu_numa_shard, "_SYS_NODE", tmp_path)

    monkeypatch.delenv(NODES_ENV, raising=False)
    monkeypatch.delenv("KMP_AFFINITY", raising=False)
    monkeypatch.delenv("GOMP_CPU_AFFINITY", raising=False)
    return build


def test_policy_shards_when_threads_span_the_nodes(topology, monkeypatch):
    """The configuration this exists for: one rank across every node."""
    topology({0: "0-7", 1: "8-15", 2: "16-23", 3: "24-31"})
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in range(32)))
    assert shard_nodes(num_threads=32) == [0, 1, 2, 3]


def test_policy_declines_when_threads_are_on_one_node(topology, monkeypatch):
    """The auto-bind default, and the case that has to decline.

    ``VLLM_CPU_OMP_THREADS_BIND=auto`` gives a rank the CPUs of one node but
    leaves the *process* affinity mask untouched, so a policy that read
    ``sched_getaffinity`` would see four nodes here and shard onto all of them
    -- placing three quarters of the weights on nodes whose CPUs never run.
    """
    topology({0: "0-7", 1: "8-15", 2: "16-23", 3: "24-31"})
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in range(8)))
    assert shard_nodes(num_threads=8) == []


def test_policy_uses_the_real_node_ids(topology, monkeypatch):
    """Under ``numactl --cpunodebind=2,3`` the shards are nodes 2 and 3.

    Shard *index* and node *id* are different things, and only on the common
    machine do they coincide. Getting this wrong binds the pages to nodes 0 and
    1, which either fails or puts every page away from its threads.
    """
    topology({0: "0-7", 1: "8-15", 2: "16-23", 3: "24-31"})
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in range(16, 32)))
    assert shard_nodes(num_threads=16) == [2, 3]


def test_policy_follows_the_threads_not_the_machine(topology, monkeypatch):
    """``OMP_NUM_THREADS`` below the CPU list shards onto fewer nodes.

    32 CPUs are listed but only 16 threads run, so threads 0..15 sit on nodes 0
    and 1 and nodes 2 and 3 never execute anything. The answer is those two
    nodes: counting the machine's nodes instead would put half the weights
    where no thread will read them.
    """
    topology({0: "0-7", 1: "8-15", 2: "16-23", 3: "24-31"})
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in range(32)))
    assert shard_nodes(num_threads=16) == [0, 1]


def test_policy_declines_on_an_interleaved_cpu_list(topology, monkeypatch):
    """Round-robin across nodes: every run of threads spans every node."""
    topology({0: "0-7", 1: "8-15"})
    order = [c for pair in zip(range(0, 8), range(8, 16)) for c in pair]
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in order))
    assert shard_nodes(num_threads=16) == []


def test_policy_declines_when_threads_do_not_divide(topology, monkeypatch):
    topology({0: "0-7", 1: "8-15", 2: "16-23"})
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in range(24)))
    assert shard_nodes(num_threads=22) == []


def test_policy_reads_the_iomp_proclist(topology, monkeypatch):
    """libiomp is the default when it is preloaded, and it spells this its own
    way: an explicit ``proclist`` rather than ``GOMP_CPU_AFFINITY``."""
    topology({0: "0-7", 1: "8-15"})
    proclist = ",".join(str(c) for c in range(16))
    monkeypatch.setenv(
        "KMP_AFFINITY", f"granularity=fine,explicit,proclist=[{proclist}]"
    )
    assert shard_nodes(num_threads=16) == [0, 1]


def test_policy_declines_without_ordered_thread_binding(topology, monkeypatch):
    """Without an ordered binding a thread's node cannot be derived from its
    index, so the policy must decline rather than guess.

    ``OMP_PLACES={0,1,...}`` with ``OMP_PROC_BIND=true`` is what vLLM falls back
    to when it has neither OpenMP runtime: it names one place holding every CPU,
    so it pins nothing in particular.
    """
    topology({0: "0-7", 1: "8-15"})
    monkeypatch.setenv("OMP_PLACES", "{" + ",".join(str(c) for c in range(16)) + "}")
    monkeypatch.setenv("OMP_PROC_BIND", "true")
    assert shard_nodes(num_threads=16) == []


def test_policy_declines_on_a_grouped_proclist(topology, monkeypatch):
    """A KMP proclist may give a thread several CPUs; that is not handled."""
    topology({0: "0-7", 1: "8-15"})
    monkeypatch.setenv(
        "KMP_AFFINITY", "granularity=fine,explicit,proclist=[{0,1},{8,9}]"
    )
    assert shard_nodes(num_threads=2) == []


def test_env_override_names_nodes_not_a_count(topology, monkeypatch):
    """The escape hatch takes node ids, so it can express ``2,3``."""
    topology({0: "0-7", 1: "8-15", 2: "16-23", 3: "24-31"})
    monkeypatch.setenv("GOMP_CPU_AFFINITY", " ".join(str(c) for c in range(8)))
    assert plan_shard_nodes() == []
    monkeypatch.setenv(NODES_ENV, "2,3")
    assert plan_shard_nodes() == [2, 3]
