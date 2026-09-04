# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.warmup.triton_moe_warmup as warmup_mod
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.warmup.triton_moe_warmup import (
    _naive_block_assignment_max_m,
    _triton_experts,
    _warmup_m_values,
    triton_moe_warmup,
)


def _routed_experts(experts: object, n: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(impl=SimpleNamespace(fused_experts=experts))
        ),
        w13_weight=torch.empty((4, n, 8)),
        w2_weight=torch.empty((4, 8, n // 2)),
    )


def _moe_runner(routed_experts: object) -> MoERunner:
    # MoERunner is an nn.Module whose __init__ builds a full MoE layer, and
    # `_quant_method` is a property forwarding to routed_experts; populate
    # __dict__ directly to keep the isinstance checks under test real.
    runner = object.__new__(MoERunner)
    object.__setattr__(runner, "__dict__", {"routed_experts": routed_experts})
    return runner


def _triton_experts_instance() -> TritonExperts:
    return object.__new__(TritonExperts)


def _worker(modules: list[object], *, is_pooling_model: bool = False):
    return SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=lambda: iter(modules)),
        model_runner=SimpleNamespace(is_pooling_model=is_pooling_model),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
    )


def test_warmup_m_values_end_at_the_token_budget() -> None:
    # The budget must be warmed even when it is off-grid, and nothing may run
    # past it.
    assert max(_warmup_m_values(40, 8)) == 40
    assert 40 in _warmup_m_values(40, 8)
    assert _warmup_m_values(1, 0) == [1]


def test_warmup_m_values_pair_every_tile_config_with_both_parities() -> None:
    # EM and num_valid_tokens scale with M, and Triton keys their
    # 16-divisibility separately. Every M that can select a tile config needs
    # an odd and an even representative, or half the variants stay uncompiled.
    values = _warmup_m_values(4096, naive_max_m=8)
    for m in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048):
        assert m in values, m
        assert (m + 1) in values, m + 1
    # Both sides of the naive block assignment cutoff, with both parities.
    assert {8, 9, 10}.issubset(values)


def test_naive_block_assignment_cutoff_matches_the_kernel_condition() -> None:
    # `num_tokens * top_k * 4 <= global_num_experts`, and expert parallelism
    # disables the naive path entirely.
    assert _naive_block_assignment_max_m(256, 8, None) == 8
    assert _naive_block_assignment_max_m(64, 8, None) == 2
    assert _naive_block_assignment_max_m(256, 8, torch.empty(256)) == 0


def test_triton_experts_found_through_deepgemm_fallback() -> None:
    triton = _triton_experts_instance()
    fallback = object.__new__(TritonOrDeepGemmExperts)
    object.__setattr__(fallback, "__dict__", {"fallback_experts": triton})

    assert _triton_experts(_moe_runner(_routed_experts(triton))) is triton
    assert _triton_experts(_moe_runner(_routed_experts(fallback))) is triton


def test_triton_experts_skips_layers_that_never_reach_triton() -> None:
    # A non-MoE module, a MoE layer whose kernel is not built yet, and a MoE
    # layer on a non-Triton expert implementation are all skipped.
    assert _triton_experts(SimpleNamespace()) is None

    unbuilt = _moe_runner(
        SimpleNamespace(quant_method=SimpleNamespace(moe_kernel=None))
    )
    assert _triton_experts(unbuilt) is None

    assert _triton_experts(_moe_runner(_routed_experts(object()))) is None


def test_warmup_runs_once_per_weight_shape(monkeypatch) -> None:
    triton = _triton_experts_instance()
    modules: list[object] = [
        _moe_runner(_routed_experts(triton, n=4)),
        _moe_runner(_routed_experts(triton, n=4)),
        _moe_runner(_routed_experts(triton, n=8)),
        SimpleNamespace(),
    ]

    warmed: list[int] = []
    synchronized: list[bool] = []
    monkeypatch.setattr(
        warmup_mod, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    monkeypatch.setattr(
        warmup_mod,
        "_warmup_expert_gemms",
        lambda experts, layer, max_tokens: warmed.append(layer.w13_weight.size(1)),
    )
    monkeypatch.setattr(
        warmup_mod.torch.accelerator, "synchronize", lambda: synchronized.append(True)
    )

    triton_moe_warmup(_worker(modules))

    assert warmed == [4, 8]
    assert synchronized == [True]


def test_warmup_skipped_for_pooling_models_and_non_cuda(monkeypatch) -> None:
    runner = _moe_runner(_routed_experts(_triton_experts_instance()))
    calls: list[object] = []
    monkeypatch.setattr(
        warmup_mod, "_warmup_expert_gemms", lambda *args: calls.append(args)
    )

    monkeypatch.setattr(
        warmup_mod, "current_platform", SimpleNamespace(is_cuda_alike=lambda: False)
    )
    triton_moe_warmup(_worker([runner]))

    monkeypatch.setattr(
        warmup_mod, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    triton_moe_warmup(_worker([runner], is_pooling_model=True))

    assert calls == []


def test_warmup_stops_on_out_of_memory(monkeypatch) -> None:
    triton = _triton_experts_instance()
    modules: list[object] = [
        _moe_runner(_routed_experts(triton, n=4)),
        _moe_runner(_routed_experts(triton, n=8)),
    ]
    attempts: list[int] = []

    def raise_oom(experts, layer, max_tokens):
        attempts.append(layer.w13_weight.size(1))
        raise torch.cuda.OutOfMemoryError("out of memory")

    monkeypatch.setattr(
        warmup_mod, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    monkeypatch.setattr(warmup_mod, "_warmup_expert_gemms", raise_oom)
    monkeypatch.setattr(warmup_mod.torch.accelerator, "synchronize", lambda: None)

    # OOM must not fail startup, and must not keep trying larger shapes.
    triton_moe_warmup(_worker(modules))

    assert attempts == [4]
