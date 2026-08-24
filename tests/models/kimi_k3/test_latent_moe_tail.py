# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.models.kimi_k3.nvidia import latent_moe_runner
from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import KimiK3LatentMoETailOp
from vllm.platforms import current_platform

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
EPS = 0.1
TOP_K = 8


def test_deferred_finalize_enabled_before_moe_kernel_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMoEConfig:
        tp_size = 8
        dp_size = 1
        ep_size = 1
        pcp_size = 1
        is_sequence_parallel = False
        hidden_dim = LATENT_SIZE
        hidden_dim_unpadded = LATENT_SIZE
        experts_per_token = 16
        defer_moe_finalize = False
        defer_moe_finalize_max_num_tokens = -1

        @property
        def use_deferred_moe_finalize(self) -> bool:
            return self.defer_moe_finalize

    moe_config = FakeMoEConfig()
    quant_method = SimpleNamespace(
        experts_cls=TrtLlmMxfp4ExpertsMonolithic,
        moe_kernel=None,
    )
    norm_weight = torch.empty(LATENT_SIZE, dtype=torch.bfloat16)
    transform = SimpleNamespace(
        norm=SimpleNamespace(weight=norm_weight, variance_epsilon=EPS),
        up_proj=SimpleNamespace(
            weight=SimpleNamespace(shape=(HIDDEN_SIZE, LATENT_SIZE))
        ),
    )

    def fake_runner_init(runner, *args, **kwargs) -> None:
        runner.moe_config = moe_config
        runner.routed_experts = SimpleNamespace(quant_method=quant_method)
        runner._shared_experts = object()
        runner.routed_output_transform = transform

    initialized_with: dict[str, object] = {}
    tail_op = SimpleNamespace(contract=SimpleNamespace(max_num_tokens=128))

    def fake_tail_initialize(**kwargs):
        initialized_with.update(kwargs)
        return tail_op

    monkeypatch.setattr(MoERunner, "__init__", fake_runner_init)
    monkeypatch.setattr(latent_moe_runner.torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(
        latent_moe_runner,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: capability == 100,
        ),
    )
    monkeypatch.setattr(
        latent_moe_runner,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(use_ubatching=False),
            model_config=SimpleNamespace(enable_sleep_mode=False),
        ),
    )
    monkeypatch.setattr(KimiK3LatentMoETailOp, "initialize", fake_tail_initialize)

    latent_moe_runner.LatentMoERunner()

    assert moe_config.defer_moe_finalize
    assert moe_config.defer_moe_finalize_max_num_tokens == 128
    assert initialized_with["experts_per_token"] == 16


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

    # DeepGEMM MegaMoE has already combined and normalized the routed latent,
    # so every TP rank owns the same routed tensor while the shared tensor is a
    # rank-local partial.
    torch.manual_seed(3000)
    routed_normalized = torch.randn(
        16,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(3100 + rank)
    shared_partial = torch.randn(
        16,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    shared_reference = shared_partial.clone()
    dist.all_reduce(shared_reference, group=group)
    expected_normalized = F.linear(routed_normalized, up_weight)
    expected_normalized.add_(shared_reference)

    actual_normalized = op.from_normalized_replicated_routed(
        routed_normalized,
        shared_partial,
        up_weight,
    )
    torch.testing.assert_close(
        actual_normalized,
        expected_normalized,
        atol=8e-2,
        rtol=3e-2,
    )

    normalized_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(normalized_graph):
        normalized_graph_output = op.from_normalized_replicated_routed(
            routed_normalized,
            shared_partial,
            up_weight,
        )
    normalized_graph.replay()
    torch.testing.assert_close(
        normalized_graph_output,
        expected_normalized,
        atol=8e-2,
        rtol=3e-2,
    )

    # DeepGEMM can scatter every rank's shared partial directly into the
    # destination rank's symmetric Lamport workspace. The up-projection
    # epilogue reduces the already-local fragments without an intermediate
    # full shared output or standalone consumer kernel.
    published_workspace, published_flags, _ = op.published_shared_workspace()
    torch.manual_seed(3200)
    routed_published = torch.randn(
        16,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(3300 + rank)
    shared_published = torch.randn(
        16,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    published_sources = [torch.empty_like(shared_published) for _ in range(tp_size)]
    dist.all_gather(published_sources, shared_published, group=group)
    local_hidden_size = HIDDEN_SIZE // tp_size

    def populate_published_generation(generation: int) -> None:
        local_workspace = published_workspace[generation, :16]
        for source_rank, source in enumerate(published_sources):
            local_workspace[:, source_rank].copy_(
                source[:, rank * local_hidden_size : (rank + 1) * local_hidden_size]
            )

    published_generation = int(published_flags[0].item())
    populate_published_generation(published_generation)
    shared_published_reference = shared_published.clone()
    dist.all_reduce(shared_published_reference, group=group)
    expected_published = F.linear(routed_published, up_weight)
    expected_published.add_(shared_published_reference)

    actual_published = op.from_normalized_replicated_routed_published(
        routed_published,
        up_weight,
    )
    torch.testing.assert_close(
        actual_published,
        expected_published,
        atol=8e-2,
        rtol=3e-2,
    )

    # Populate all generations with the same source so capture and replay can
    # exercise device-side generation rotation without a host synchronization.
    for generation in range(published_workspace.shape[0]):
        populate_published_generation(generation)
    published_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(published_graph):
        published_graph_output = op.from_normalized_replicated_routed_published(
            routed_published,
            up_weight,
        )
    published_graph.replay()
    torch.testing.assert_close(
        published_graph_output,
        expected_published,
        atol=8e-2,
        rtol=3e-2,
    )


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
