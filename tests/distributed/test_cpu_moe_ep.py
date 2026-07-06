# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-process CPU test for FP8 W8A16 MoE expert parallelism.

Validates the full dispatch → local-expert-skip compute → combine → TP
all-reduce chain for the AgRsAll2AllManager + CPUExpertsFp8 path on CPU
(gloo backend).

Covers:
  - TP=1, DP=6 — production CPU EP topology coverage
  - TP=2, DP=3 — component-only hybrid TP+EP plumbing coverage

Run: numactl -m 4 -N 4 .venv/bin/python -m pytest \
         tests/distributed/test_cpu_moe_ep.py -v
"""

import math
import os

import pytest
import torch
import torch.distributed as dist

from tests.distributed.cpu_mp_test_utils import (
    report_worker_failure,
    spawn_workers,
)
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (  # noqa: E402
    fused_experts_cpu_local_skip,
)

if not hasattr(torch.ops._C, "fused_experts_cpu"):
    pytest.skip("fused_experts_cpu op not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/kernels/moe/test_cpu_quant_fused_moe.py)
# ---------------------------------------------------------------------------

BLOCK_SIZE = [128, 128]
_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_SCALE = _FP8_INFO.max
FACTOR_FOR_SCALE = 1e-3


def _prepack_experts(w: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.convert_weight_packed(w)


def _make_fp8_moe_weights(E, N, K, block_size):
    block_n, block_k = block_size
    w1 = (
        (torch.randn(E, 2 * N, K) * FP8_SCALE)
        .clamp(-FP8_SCALE, FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        (torch.randn(E, K, N) * FP8_SCALE)
        .clamp(-FP8_SCALE, FP8_SCALE)
        .to(torch.float8_e4m3fn)
    )
    w1_s = (
        torch.randn(E, math.ceil(2 * N / block_n), math.ceil(K / block_k))
        * FACTOR_FOR_SCALE
    )
    w2_s = (
        torch.randn(E, math.ceil(K / block_n), math.ceil(N / block_k))
        * FACTOR_FOR_SCALE
    )
    return w1, w2, w1_s, w2_s


def _run_single_rank_reference(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, E):
    """Single-process FP8 MoE forward over all E experts (no EP masking)."""
    pw1, pw2 = _prepack_experts(w1), _prepack_experts(w2)
    return ops.fused_experts_cpu(
        a.clone(),
        pw1,
        pw2,
        topk_weight,
        topk_ids,
        False,
        ops.CPUQuantMethod.FP8_W8A16,
        w1_s,
        w2_s,
        None,
        None,
        BLOCK_SIZE,
        is_vnni=True,
    )


# ---------------------------------------------------------------------------
# Distributed environment setup
# ---------------------------------------------------------------------------


def _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port):
    """Init vLLM distributed env for TP=tp_size, DP=dp_size.

    global_rank = dp_rank * tp_size + tp_rank
    Each DP replica contributes tp_size workers; init_distributed_environment
    offsets rank by dp_rank*tp_size internally.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        _data_parallel_master_port_list=[int(dp_port)],
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=tp_size,
            rank=tp_rank,
            distributed_init_method=f"tcp://localhost:{port}",
            local_rank=rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(tp_size, 1, backend="gloo")


def _make_forward_context(dp_rank, dp_size, num_tokens_per_dp_rank):
    """Forward context with DP metadata for AgRsAll2AllManager."""
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.forward_context import set_forward_context

    class _AttnMeta:
        dp_metadata = None

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=dp_size,
        is_moe_model=True,
        data_parallel_rank=dp_rank,
    )
    return set_forward_context(
        _AttnMeta(),
        vllm_config,
        num_tokens=num_tokens_per_dp_rank,
        num_tokens_across_dp=torch.tensor(
            [num_tokens_per_dp_rank] * dp_size, dtype=torch.int
        ),
    )


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _moe_ep_worker(rank, world_size, tp_size, dp_size, port, dp_port, params, err_q):
    try:
        # Required by CpuCommunicator's SHM-group-name check on TP groups.
        os.environ.setdefault("VLLM_DIST_IDENT", f"test_cpu_moe_ep_{port}")
        _init_tp_dp_environment(rank, tp_size, dp_size, port, dp_port)

        from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_tp_group,
        )
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            determine_expert_map,
        )

        M_per_dp, N, K, E, topk, seed = params
        torch.manual_seed(seed)

        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        ep_rank = dp_rank * tp_size + tp_rank  # flattened ep rank

        # Build shared inputs (same across all ranks via fixed seed)
        # M_per_dp tokens per DP replica; total M = dp_size * M_per_dp
        total_M = dp_size * M_per_dp
        a_all = torch.randn(total_M, K, dtype=torch.bfloat16) / math.sqrt(K)
        w1, w2, w1_s, w2_s = _make_fp8_moe_weights(E, N, K, BLOCK_SIZE)
        score = torch.softmax(
            torch.randn(total_M, E, dtype=torch.bfloat16), dim=-1, dtype=torch.float32
        )
        topk_weight_all, topk_ids_all = torch.topk(score, topk)
        topk_ids_all = topk_ids_all.to(torch.int32)

        # Single-rank reference (no EP, all experts)
        ref = _run_single_rank_reference(
            a_all, w1, w2, w1_s, w2_s, topk_weight_all, topk_ids_all, E
        )

        # This rank's DP slice
        lo, hi = dp_rank * M_per_dp, (dp_rank + 1) * M_per_dp
        a_local = a_all[lo:hi].clone()
        topk_weight_local = topk_weight_all[lo:hi]
        topk_ids_local_global = topk_ids_all[lo:hi]

        # Expert placement for this EP rank
        ep_size = dp_size * tp_size  # always 6 in this test
        local_num_experts, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=E,
        )
        assert expert_map is not None

        owned = (expert_map != -1).nonzero(as_tuple=False)
        expert_lo = int(owned[0].item())
        expert_hi = expert_lo + local_num_experts
        w1_local = w1[expert_lo:expert_hi].contiguous()
        w2_local = w2[expert_lo:expert_hi].contiguous()
        w1_s_local = w1_s[expert_lo:expert_hi].contiguous()
        w2_s_local = w2_s[expert_lo:expert_hi].contiguous()

        dp_cpu_group = get_dp_group().cpu_group

        with _make_forward_context(dp_rank, dp_size, M_per_dp):
            manager = AgRsAll2AllManager(dp_cpu_group)
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                # Dispatch: all-gather over DP group → [total_M, K]
                a_gathered, tw_gathered, tid_gathered = manager.dispatch(
                    a_local.clone(),
                    topk_weight_local.clone(),
                    topk_ids_local_global.clone().long(),
                )

            # Expert compute: skip non-local selections (the shipped path).
            pw1, pw2 = _prepack_experts(w1_local), _prepack_experts(w2_local)
            expert_out = fused_experts_cpu_local_skip(
                a_gathered.clone(),
                pw1,
                pw2,
                tw_gathered,
                tid_gathered,
                expert_map,
                ops.CPUQuantMethod.FP8_W8A16,
                w1_s_local,
                w2_s_local,
                None,  # w1_zero
                None,  # w2_zero
                BLOCK_SIZE,
                None,  # w1_bias
                None,  # w2_bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )

            with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                # Combine: reduce-scatter over DP group → [M_per_dp, K]
                combined = manager.combine(expert_out)

        # TP all-reduce: sum TP-partner contributions (mirrors moe_runner.py:451)
        dist.all_reduce(combined, group=get_tp_group().device_group)

        # Gather results from all ranks at rank 0 for verification.
        # Each rank holds its DP-slice result [M_per_dp, K].
        # We need to reconstruct the full [total_M, K] and compare to ref.
        # Use TP group rank 0 per DP replica to avoid double-counting:
        # TP partners now hold identical combined tensors post-all-reduce.
        # new_group is a collective — all ranks must call it.
        tp0_ranks = [r for r in range(world_size) if r % tp_size == 0]
        tp0_group = dist.new_group(ranks=tp0_ranks, backend="gloo")

        result_full = None
        if tp_rank == 0:
            # Only TP-rank-0 workers participate in gathering
            all_slices = [torch.zeros_like(combined) for _ in range(dp_size)]
            dist.all_gather(all_slices, combined, group=tp0_group)
            result_full = torch.cat(all_slices, dim=0)

        if rank == 0:
            assert result_full is not None
            torch.testing.assert_close(
                ref.bfloat16(), result_full, atol=1e-2, rtol=1e-2
            )

        dist.barrier()

    except Exception as err:
        report_worker_failure(rank, err_q, err)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# (M_per_dp, N, K, E, topk, seed)
# E=12 divisible by EP=6 (2 experts/rank), E=10 non-divisible (ranks 0-3 hold
# 2 experts, ranks 4-5 hold 1) — mirrors the shape of 256÷6 at small scale.
_PARAMS = [
    (8, 256, 512, 12, 2, 0),
    (8, 256, 512, 10, 2, 0),
]


@pytest.mark.distributed
@pytest.mark.parametrize("params", _PARAMS, ids=["E12-div", "E10-nondiv"])
def test_cpu_moe_ep_dp6_tp1(params):
    """TP=1, DP=6: pure expert parallelism baseline (EP=6)."""
    spawn_workers(_moe_ep_worker, world_size=6, tp_size=1, dp_size=6, params=params)


@pytest.mark.distributed
@pytest.mark.parametrize("params", _PARAMS, ids=["E12-div", "E10-nondiv"])
def test_cpu_moe_ep_component_dp3_tp2(params):
    """Component-only TP=2, DP=3 hybrid tensor+expert coverage.

    This bypasses CPUPlatform's full-model topology validation and only checks
    the lower-level dispatch/combine + local-skip + TP all-reduce plumbing.
    """
    spawn_workers(_moe_ep_worker, world_size=6, tp_size=2, dp_size=3, params=params)
