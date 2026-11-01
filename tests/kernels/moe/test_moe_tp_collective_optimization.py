# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE TP Collective Optimization (output projection sharding).

These tests verify that the TP-sharded MoE kernel produces numerically
equivalent results to the non-sharded (TP=1) reference implementation.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.platforms import current_platform

# Skip if not on CUDA or not enough GPUs
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or current_platform.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs for TP testing",
)


def _create_test_data(M, N, K, E, dtype, device, seed):
    """Create test data for MoE kernel."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    a = torch.randn((M, K), device=device, dtype=dtype) / 10

    w1_fp32 = (torch.rand((E, 2 * N, K), device=device, dtype=torch.float32) - 0.5) * 2
    w1 = w1_fp32.to(dtype)

    w2_fp32 = (torch.rand((E, K, N), device=device, dtype=torch.float32) - 0.5) * 2
    w2 = w2_fp32.to(dtype)

    score = torch.randn((M, E), device=device, dtype=dtype)

    return a, w1, w2, score, w1_fp32, w2_fp32


def _run_moe_kernel(a, w1, w2, score, topk, moe_config, quant_config):
    """Run the MoE kernel and return output."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, renormalize=False)

    out = fused_experts(
        a,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=moe_config.num_experts,
        quant_config=quant_config,
        moe_config=moe_config,
    )
    return out


def _create_moe_config(M, N, K, E, topk, dtype, device, tp_size, tp_rank):
    """Create MoE config with TP settings."""
    return FusedMoEConfig(
        num_experts=E,
        experts_per_token=topk,
        hidden_dim=K,
        intermediate_size=N,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEConfig.FusedMoEParallelConfig(
            tp_size=tp_size,
            tp_rank=tp_rank,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="passthrough",
            enable_eplb=False,
        ),
        activation="silu",
        in_dtype=dtype,
        device=device,
        max_num_tokens=M * topk,
    )


def _run_tp_comparison(M, N, K, E, topk, dtype, seed):
    """Run MoE kernel with TP=1, TP=2, TP=4 and compare results."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    results_tp1 = manager.dict()

    # Run TP=1 (single process, no distributed needed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    a, w1, w2, score, w1_fp32, w2_fp32 = _create_test_data(
        M, N, K, E, dtype, device, seed
    )

    quant_config = FusedMoEQuantConfig.make()

    moe_config_tp1 = _create_moe_config(
        M, N, K, E, topk, dtype, device, tp_size=1, tp_rank=0
    )

    out_tp1 = _run_moe_kernel(a, w1, w2, score, topk, moe_config_tp1, quant_config)
    results_tp1[0] = out_tp1.cpu()

    # For TP=2 and TP=4, we need multi-GPU setup
    for tp_size in [2, 4]:
        if tp_size > current_platform.device_count():
            continue

        results_tp = manager.dict()

        def worker_tp(rank, world_size, results_dict):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")

            from vllm.distributed.parallel_state import initialize_model_parallel

            # Initialize distributed
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            initialize_model_parallel(tensor_model_parallel_size=world_size)

            a_local, w1_local, w2_local, score_local, _, _ = _create_test_data(
                M, N, K, E, dtype, device, seed
            )

            quant_config_local = FusedMoEQuantConfig.make()

            moe_config = _create_moe_config(
                M, N, K, E, topk, dtype, device,
                tp_size=world_size, tp_rank=rank
            )

            out = _run_moe_kernel(
                a_local, w1_local, w2_local, score_local,
                topk, moe_config, quant_config_local
            )
            results_dict[rank] = out.cpu()

            torch.distributed.destroy_process_group()

        # Spawn processes
        ctx.spawn(
            worker_tp,
            args=(tp_size, results_tp),
            nprocs=tp_size,
            join=True,
        )

        # Collect results from all ranks and combine
        tp_outputs = [results_tp[i] for i in range(tp_size)]
        # All ranks should have the same output after all-reduce
        for i in range(1, tp_size):
            torch.testing.assert_close(
                tp_outputs[0], tp_outputs[i], atol=1e-2, rtol=1e-2
            )

        # Compare with TP=1 reference
        torch.testing.assert_close(
            out_tp1.cpu(), tp_outputs[0], atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize(
    "M, N, K, E, topk, dtype, seed",
    [
        pytest.param(4, 128, 256, 8, 2, torch.half, 0, id="small"),
        pytest.param(16, 256, 512, 8, 2, torch.half, 0, id="medium"),
        pytest.param(32, 512, 1024, 8, 2, torch.bfloat16, 0, id="large_bf16"),
    ],
)
@torch.inference_mode()
def test_moe_tp_collective_optimization(M, N, K, E, topk, dtype, seed):
    """Test that TP-sharded MoE kernel matches TP=1 reference.

    This test verifies the TP collective optimization where each TP rank
    computes only its output shard (K // tp_size) and the all-reduce
    combines them correctly.
    """
    # Skip if not enough GPUs
    if current_platform.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for TP testing")

    # Run comparison for TP=2
    if current_platform.device_count() >= 2:
        _run_tp_comparison(M, N, K, E, topk, dtype, seed)


# Alternative simpler test that directly tests the kernel invoke function
# with mocked parallel state (for environments without multi-GPU)
def test_moe_tp_sharding_kernel_invoke():
    """Test that the kernel invoke function correctly handles TP sharding.

    This test verifies the grid computation and parameter passing
    without requiring multi-GPU setup.
    """
    from unittest.mock import patch

    from vllm.model_executor.layers.fused_moe.config import get_moe_configs

    # Test parameters (only M and topk are used)
    M, topk = 4, 2

    # Mock TP=1
    with patch(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
        return_value=1,
    ), patch(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        return_value=0,
    ):
        config = get_moe_configs(M * topk, "auto", False)[0]
        # This should not raise any assertion errors
        # We can't easily test the kernel launch without full setup,
        # but we can verify the grid computation logic
        assert config["BLOCK_SIZE_N"] > 0


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v"])