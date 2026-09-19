# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_shared_output_ar_uses_one_slot_without_ubatching() -> None:
    assert latent_moe_runner._num_shared_output_ar_slots(0) == 1
    assert latent_moe_runner._num_shared_output_ar_slots(1) == 1
    assert latent_moe_runner._num_shared_output_ar_slots(2) == 2


def test_shared_output_reduction_overlaps_routed_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latent_moe_runner.envs, "VLLM_KIMI_K3_SHARED_OUTPUT_AR_OVERLAP", True
    )
    hidden = torch.ones(2, 2)
    shared_input = torch.full((2, 3), 2.0)
    shared = MagicMock()
    shared.output = shared_input + 1
    reducer = MagicMock(side_effect=lambda value, *, slot: value * 2)
    reducer.supports.return_value = True
    execute = MagicMock(side_effect=lambda main, aux, *_: (main(), aux()))
    monkeypatch.setattr(latent_moe_runner, "maybe_execute_in_parallel", execute)
    monkeypatch.setattr(latent_moe_runner, "dbo_current_ubatch_id", lambda: 1)
    monkeypatch.setattr(latent_moe_runner, "aux_stream", lambda: object())

    runner = object.__new__(latent_moe_runner.LatentMoERunner)
    runner.enable_k3_latent_moe_tail_fusion = False
    runner._shared_output_ar = reducer
    runner._shared_output_ar_events = ((object(), object()),) * 2
    runner._shared_experts = shared
    runner.routed_experts = SimpleNamespace(forward_monolithic=lambda x, **_: x * 3)

    shared_out, routed_out = runner._apply_quant_method(
        hidden, torch.empty(2, 1), shared_input
    )

    execute.assert_called_once()
    shared.assert_called_once_with(
        shared_input, latent_moe_runner.SharedExpertsOrder.NO_OVERLAP
    )
    assert reducer.call_args.args[0] is shared.output
    assert reducer.call_args.kwargs == {"slot": 1}
    torch.testing.assert_close(shared_out, torch.full_like(shared_input, 6))
    torch.testing.assert_close(routed_out, hidden * 3)


def test_shared_output_ar_defers_to_fused_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latent_moe_runner.envs, "VLLM_KIMI_K3_SHARED_OUTPUT_AR_OVERLAP", True
    )
    shared_input = torch.empty(2, 3)
    reducer = MagicMock()
    reducer.supports.return_value = True

    runner = object.__new__(latent_moe_runner.LatentMoERunner)
    runner._shared_output_ar = reducer
    runner.enable_k3_latent_moe_tail_fusion = True
    runner._k3_latent_moe_tail_op = SimpleNamespace(
        contract=SimpleNamespace(max_num_tokens=2)
    )
    assert not runner._overlap_shared_ar(shared_input)
    runner._k3_latent_moe_tail_op.contract.max_num_tokens = 1
    assert runner._overlap_shared_ar(shared_input)
    monkeypatch.setattr(
        latent_moe_runner.envs, "VLLM_KIMI_K3_SHARED_OUTPUT_AR_OVERLAP", False
    )
    assert not runner._overlap_shared_ar(shared_input)


def test_pre_reduced_shared_output_skips_second_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = torch.full((2, 3), 5.0)
    routed = torch.full((2, 2), 2.0)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    all_reduce = MagicMock(side_effect=lambda value: value * 10)
    monkeypatch.setattr(
        latent_moe_runner, "tensor_model_parallel_all_reduce", all_reduce
    )

    runner = object.__new__(latent_moe_runner.LatentMoERunner)
    runner.routed_output_transform = SimpleNamespace(
        norm=None, up_proj=SimpleNamespace(weight=weight)
    )
    runner._maybe_reduce_final_output = lambda value, *args, **kwargs: value

    result = runner._pre_reduced_shared_tail(routed, shared, None)

    torch.testing.assert_close(result, torch.mm(routed * 10, weight.t()) + shared)
    all_reduce.assert_called_once_with(routed)


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
            current_device=lambda: torch.device("cuda"),
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


def _make_bf16_top16_deferred_output(
    num_tokens: int,
    device: torch.device,
    *,
    drop_last_route: bool = False,
) -> tuple[UnfinalizedMoEOutput, torch.Tensor]:
    top_k = 16
    num_routes = num_tokens * top_k
    num_permuted_rows = num_routes + 7
    expanded_output = torch.randn(
        num_routes,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.01)
    expert_weights = torch.rand(
        num_tokens,
        top_k,
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

    expanded_output = expanded_output.view(num_tokens, top_k, LATENT_SIZE)
    expanded_idx = expanded_idx.view(num_tokens, top_k)
    if drop_last_route:
        expanded_output[:, -1].zero_()
        expanded_idx[:, -1] = -1
    finalized = (
        (expanded_output.float() * expert_weights[:, :, None].float())
        .sum(1)
        .to(torch.bfloat16)
    )
    return (
        UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_permuted,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx.to(torch.int32),
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

    # M=65 crosses two 32-CTA token waves and reuses the first DSM slot.
    for iteration, num_tokens in enumerate((1, 5, 8, 16, 33, 65, 5)):
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
    bf16_deferred_op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
        experts_per_token=16,
    )
    cutedsl_warmup()

    for iteration, num_tokens in enumerate((1, 5, 16, 33, 65)):
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

    for iteration, num_tokens in enumerate((1, 2, 4, 5, 16)):
        torch.manual_seed(3000 + 100 * iteration + rank)
        deferred_output, finalized_output = _make_bf16_top16_deferred_output(
            num_tokens,
            device,
            drop_last_route=num_tokens == 5,
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
        actual = bf16_deferred_op(
            deferred_output,
            shared_output,
            rms_weight,
            up_weight,
        )

        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
        if num_tokens == 4:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_actual = bf16_deferred_op(
                    deferred_output,
                    shared_output,
                    rms_weight,
                    up_weight,
                )
            graph.replay()
            torch.testing.assert_close(
                graph_actual,
                expected,
                atol=8e-2,
                rtol=3e-2,
            )


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
