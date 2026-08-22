# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import KimiK3LatentMoETailOp
from vllm.platforms import current_platform

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
EPS = 0.1
TOP_K = 8


def _make_deferred_routed_output(
    num_tokens: int,
    device: torch.device,
) -> tuple[UnfinalizedMoEOutput, torch.Tensor]:
    num_routes = num_tokens * TOP_K
    num_permuted_rows = num_routes + 7
    expanded_output = torch.randn(
        num_routes,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.01)
    expert_weights = torch.rand(
        num_tokens,
        TOP_K,
        device=device,
        dtype=torch.bfloat16,
    )
    expert_weights.div_(expert_weights.sum(dim=-1, keepdim=True))
    expanded_idx = torch.randperm(num_permuted_rows, device=device)[:num_routes]
    gemm2_permuted = torch.empty(
        num_permuted_rows,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_permuted[expanded_idx] = expanded_output

    finalized = torch.zeros(
        num_tokens,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    expanded_output = expanded_output.view(num_tokens, TOP_K, LATENT_SIZE)
    for slot in range(TOP_K):
        finalized.add_(expanded_output[:, slot] * expert_weights[:, slot, None])

    return (
        UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_permuted,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx.to(torch.int32).view(
                num_tokens, TOP_K
            ),
        ),
        finalized,
    )


@ray.remote(num_gpus=1, max_calls=1)
def _test_latent_moe_tail_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
    )

    torch.manual_seed(0)
    rms_weight = 1 + 0.1 * torch.randn(
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    up_weight = (
        torch.randn(
            HIDDEN_SIZE,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / LATENT_SIZE**0.5
    )

    group = get_tp_group().device_group
    op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
    )
    cutedsl_warmup()

    for iteration, num_tokens in enumerate((1, 5, 8, 16, 5)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed_output = torch.randn(
            num_tokens,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(0.01)
        shared_output = torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )

        routed_reference = routed_output.clone()
        shared_reference = shared_output.clone()
        dist.all_reduce(routed_reference, group=group)
        dist.all_reduce(shared_reference, group=group)
        expected = F.linear(
            F.rms_norm(
                routed_reference,
                (LATENT_SIZE,),
                rms_weight,
                EPS,
            ),
            up_weight,
        )
        expected.add_(shared_reference)

        actual = op(
            routed_output,
            shared_output,
            rms_weight,
            up_weight,
        )
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
        assert actual.is_contiguous()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(
            routed_output,
            shared_output,
            rms_weight,
            up_weight,
        )
    graph.replay()
    torch.testing.assert_close(graph_output, expected, atol=8e-2, rtol=3e-2)


def _run_latent_moe_tail_test(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
) -> None:
    if not current_platform.is_device_capability_family(100):
        pytest.skip("K3 latent-MoE tail fusion requires SM100")
    multi_process_parallel(
        monkeypatch,
        tp_size,
        1,
        _test_latent_moe_tail_worker,
    )


@ray.remote(num_gpus=1, max_calls=1)
def _test_deferred_finalize_parity_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
    )

    torch.manual_seed(1000 + rank)
    rms_weight = 1 + 0.1 * torch.randn(
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    up_weight = (
        torch.randn(
            HIDDEN_SIZE,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / LATENT_SIZE**0.5
    )
    finalized_op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
    )
    deferred_op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
        experts_per_token=TOP_K,
    )
    cutedsl_warmup()

    for iteration, num_tokens in enumerate((1, 5, 16)):
        torch.manual_seed(2000 + 100 * iteration + rank)
        deferred_output, finalized_output = _make_deferred_routed_output(
            num_tokens,
            device,
        )
        shared_output = torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )

        expected = finalized_op(
            finalized_output,
            shared_output,
            rms_weight,
            up_weight,
        )
        actual = deferred_op(
            deferred_output,
            shared_output,
            rms_weight,
            up_weight,
        )

        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def _run_deferred_finalize_parity_test(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
) -> None:
    if not current_platform.is_device_capability_family(100):
        pytest.skip("K3 latent-MoE tail fusion requires SM100")
    multi_process_parallel(
        monkeypatch,
        tp_size,
        1,
        _test_deferred_finalize_parity_worker,
    )


@multi_gpu_test(num_gpus=8)
def test_latent_moe_tail_tp8_matches_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_latent_moe_tail_test(monkeypatch, 8)


@multi_gpu_test(num_gpus=16)
def test_latent_moe_tail_tp16_matches_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_latent_moe_tail_test(monkeypatch, 16)


@multi_gpu_test(num_gpus=8)
def test_latent_moe_tail_deferred_finalize_matches_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_deferred_finalize_parity_test(monkeypatch, 8)
