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
    _sample_local_expert_ids,
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
    assert max(_warmup_m_values(40, 8, 8)) == 40
    assert 40 in _warmup_m_values(40, 8, 8)
    # A budget of one token leaves no room for a companion.
    assert _warmup_m_values(1, 0, 8) == [1]


def test_warmup_m_values_cover_both_divisibility_classes() -> None:
    # num_valid_tokens is M * top_k and Triton keys its 16-divisibility
    # separately, so every M that can select a tile config needs a divisible
    # and a non-divisible representative. Pairing on M parity alone is not
    # enough: at top_k=1 neither 24 nor 25 is divisible.
    for top_k in (1, 2, 3, 5, 8):
        values = _warmup_m_values(4096, naive_max_m=8, top_k=top_k)
        for m in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048):
            assert m in values, (top_k, m)
            warmed = [v for v in values if abs(v - m) <= 16]
            assert any((v * top_k) % 16 == 0 for v in warmed), (top_k, m)
            assert any((v * top_k) % 16 != 0 for v in warmed), (top_k, m)


def test_warmup_m_values_cover_the_naive_block_assignment_cutoff() -> None:
    # The tile config just above the cutoff is shared by too few grid entries
    # to be reached by the grid alone.
    values = _warmup_m_values(4096, naive_max_m=8, top_k=8)
    assert {8, 9}.issubset(values)


def test_warmup_m_values_fall_back_downward_at_the_budget() -> None:
    # At the token budget there is no larger neighbour, so the companion has
    # to come from below or the non-divisible variant is never compiled.
    values = _warmup_m_values(4096, naive_max_m=8, top_k=8)
    assert 4096 in values
    assert 4095 in values


def test_warmup_m_values_when_no_second_divisibility_class_exists() -> None:
    # With top_k a multiple of 16, M * top_k is always divisible; there is no
    # second variant to warm and the search must not invent one.
    values = _warmup_m_values(64, naive_max_m=4, top_k=16)
    assert all((m * 16) % 16 == 0 for m in values)
    assert values == sorted(set(values))


def test_sampled_expert_ids_are_global_when_not_expert_parallel() -> None:
    ids = _sample_local_expert_ids((16, 4), 8, None, torch.device("cpu"))
    assert ids.shape == (16, 4)
    assert ids.dtype == torch.int32
    assert int(ids.min()) >= 0
    assert int(ids.max()) < 8


def test_sampled_expert_ids_stay_on_this_rank_under_expert_parallelism() -> None:
    # topk_ids carries global ids; a rank whose slice does not start at 0 would
    # warm nothing if the local id range were sampled instead, because
    # moe_align_block_size drops ids the map sends to -1.
    global_num_experts, local_num_experts, start = 256, 32, 32  # rank 1, linear
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    expert_map[start : start + local_num_experts] = torch.arange(
        local_num_experts, dtype=torch.int32
    )

    ids = _sample_local_expert_ids(
        (64, 8), global_num_experts, expert_map, torch.device("cpu")
    )

    assert ids.dtype == torch.int32
    assert bool((expert_map[ids.long()] >= 0).all())
    assert int(ids.min()) >= start
    assert int(ids.max()) < start + local_num_experts


def test_no_sampled_ids_when_the_rank_owns_no_expert() -> None:
    # Returning a tensor here would hand uninitialised ids to `apply` as global
    # expert ids, which can index expert_map out of bounds; the caller has to
    # skip the warmup instead.
    expert_map = torch.full((16,), -1, dtype=torch.int32)
    assert _sample_local_expert_ids((4, 2), 16, expert_map, torch.device("cpu")) is None


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
