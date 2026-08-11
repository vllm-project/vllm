# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA sharding of the CPU MoE experts.

Sharding only changes *which thread* computes which output block; the arithmetic
of each block is untouched. So the bar is not a tolerance -- it is
``torch.equal`` against the unsharded run. Anything else means a block was
computed twice, or not at all, or with the wrong slice of the weights, and those
are the three ways this can be wrong.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.cpu_numa_shard import (
    BLOCK_N,
    plan_shards,
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
    torch.ops._C.set_cpu_moe_numa_shards(shards)
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
        torch.ops._C.set_cpu_moe_numa_shards(1)


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


def test_policy_declines_without_ordered_thread_binding(monkeypatch):
    """Without an ordered binding a thread's node cannot be derived from its
    index, so the policy must decline rather than guess."""
    monkeypatch.delenv("VLLM_CPU_MOE_NUMA_SHARDS", raising=False)
    monkeypatch.delenv("KMP_AFFINITY", raising=False)
    monkeypatch.delenv("GOMP_CPU_AFFINITY", raising=False)
    monkeypatch.setenv("OMP_PLACES", "{0,1,2,3}")
    monkeypatch.setenv("OMP_PROC_BIND", "true")
    assert plan_shards() == 1
