# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route selection and per-call dispatch of DeepSeek-V4.1's amd_kernels routes.

CPU only: these guard the decisions that must agree across ranks or across
the target and draft models, not the kernels."""

import types

import pytest
import torch

from vllm.config.kernel import AmdKernelsConfig, KernelConfig
from vllm.models.deepseek_v41.amd import (
    akr_config,
    akr_experts,
    akr_mhc,
    akr_moe,
    akr_warmup,
)

ALL = frozenset(akr_config.ROUTES)


@pytest.fixture
def routes(monkeypatch):
    """Set the process's routes (as `routes()` would after model build)."""

    def set_routes(on=ALL):
        monkeypatch.setattr(akr_config, "_routes", frozenset(on))
        monkeypatch.setattr(akr_config, "_failed", {})

    set_routes()
    return set_routes


def test_config_validates_routes_and_keeps_warmup_out_of_compile_hash():
    cfg = KernelConfig(amd_kernels={"disabled_routes": ["fused_ep8", "gemm"]})
    assert cfg.amd_kernels.disabled_routes == ["fused_ep8", "gemm"]
    with pytest.raises(ValueError):
        KernelConfig(amd_kernels={"disabled_routes": ["no_such_route"]})
    on, off = AmdKernelsConfig(), AmdKernelsConfig(prefill_warmup=False)
    assert on.compile_factors() == off.compile_factors()
    assert on.compile_factors() != AmdKernelsConfig(indexer_max_ctx=4).compile_factors()


def test_minimax_routes_unchanged():
    from vllm.models.minimax_m3.amd import akr_config as minimax

    assert "fused_ep8" not in minimax.ROUTES and "gemm" in minimax.ROUTES
    assert set(akr_config.GATES) == ALL


def test_resolve_routes():
    assert akr_config.resolve_routes(True, set(), True) == ALL
    assert akr_config.resolve_routes(False, set(), True) == frozenset()
    assert akr_config.resolve_routes(True, set(), False) == frozenset()
    on = akr_config.resolve_routes(True, {"wo_a"}, True)
    assert "mhc_wo_b" not in on and "mhc_seam" in on and "mhc_ffn" in on
    on = akr_config.resolve_routes(True, {"mhc_seam"}, True)
    assert not {"mhc_wo_b", "mhc_ffn"} & on and "mhc_tail" in on
    on = akr_config.resolve_routes(True, {"shared_expert"}, True)
    assert "mhc_ffn" not in on and "mhc_wo_b" in on


@pytest.mark.parametrize(
    "route,last_on",
    [
        ("mhc_seam", 384),
        ("mhc_wo_b", 128),
        ("mhc_ffn", 192),
        ("sparse_mla", 768),
        ("fused_ep8", 1024),
        ("markov", 160),
    ],
)
def test_row_gates(routes, route, last_on):
    assert akr_config.serves(route, 1) and akr_config.serves(route, last_on)
    assert not akr_config.serves(route, last_on + 1)
    akr_config.fail(route, "test")
    assert not akr_config.serves(route, 1)


def test_lm_head_and_main_proj_rows(routes):
    lm_head = [m for m in range(16, 1025, 16) if akr_config.serves("lm_head", m)]
    assert lm_head == [128, 496, 512]
    assert not akr_config.serves("main_proj", 639)
    assert akr_config.serves("main_proj", 640)


def test_indexer_bound():
    mml = 1 << 20
    assert akr_config.indexer_bound(mml, 1) == mml
    assert akr_config.indexer_bound(mml, 2) == mml // 2
    assert akr_config.indexer_bound(mml, 1, override=4096) == 4096
    assert akr_config.indexer_bound(2 * mml, 1) > akr_config.INDEXER_MAX_ENTRIES
    # long dense rows only from 128 decode rows; candidate consumers always
    assert not akr_config.indexer_rows_ok(mml, 64, candidate_consumer=False)
    assert akr_config.indexer_rows_ok(mml, 128, candidate_consumer=False)
    assert akr_config.indexer_rows_ok(mml, 6, candidate_consumer=True)
    assert akr_config.indexer_rows_ok(262144, 6, candidate_consumer=False)


def test_dp_rows_is_the_dp_group_max(monkeypatch):
    import vllm.forward_context as fc

    dp = types.SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([3, 700, 5]))
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: types.SimpleNamespace(dp_metadata=dp)
    )
    assert akr_config.dp_rows(3) == 700
    dp.num_tokens_across_dp_cpu = torch.tensor([3, 1100, 5])
    assert akr_config.dp_rows(3) == 1100


def _experts(**kw):
    base = dict(
        fuse_shared_expert=False,
        ep_size=8,
        num_local_experts=48,
        top_k=6,
        swiglu_limit=10.0,
    )
    return types.SimpleNamespace(**{**base, **kw})


def test_fused_ep8_eligibility(monkeypatch):
    """vLLM-side reasons first, then the entry's blocker (shapes, ranks), asked
    for the kernel's row limit so that no rank declines a step on its own."""
    asked = []

    def blocker(experts, max_tokens):
        asked.append(max_tokens)
        return None if experts.top_k == 6 else "not covered"

    monkeypatch.setattr(akr_experts.akr_kernels, "fused_ep_unsupported_reason", blocker)
    assert akr_experts.unsupported_reason(_experts()) is None
    assert asked == [akr_experts.MAX_ROWS]
    assert akr_experts.unsupported_reason(_experts(top_k=3)) == "not covered"
    for kw in ({"fuse_shared_expert": True}, {"swiglu_limit": 7.0}):
        assert akr_experts.unsupported_reason(_experts(**kw)) is not None, kw


def test_fused_ep8_branch_follows_the_dp_max(routes, monkeypatch):
    """Every rank must take the same branch: a rank with 3 rows falls back to
    MegaMoE when another rank of its DP group has more than 1024."""
    calls: list[int] = []

    def fused_ep_moe(*args, rows):
        calls.append(rows)
        return "fused"

    monkeypatch.setattr(akr_experts.akr_kernels, "fused_ep_moe", fused_ep_moe)
    dp_max = {"rows": 700}
    monkeypatch.setattr(akr_config, "dp_rows", lambda m: max(m, dp_max["rows"]))
    layer = akr_experts.FusedEp8(ep=object(), weights=object())
    x = torch.empty(3, 8)
    assert layer.forward(x, None, None, None) == "fused" and calls == [700]
    dp_max["rows"] = 1100
    assert layer.forward(x, None, None, None) is None
    assert not layer.serves(3)
    routes(ALL - {"fused_ep8"})
    dp_max["rows"] = 1
    assert layer.forward(x, None, None, None) is None


def test_shared_expert_plan(routes):
    assert akr_moe.shared_plan(769) is None
    kinds = {m: akr_moe.shared_plan(m) for m in (1, 32, 33, 96, 97, 192, 768)}
    assert kinds == {
        1: "swiglu",
        32: "swiglu",
        33: "stock",
        96: "stock",
        97: "swiglu",
        192: "swiglu",
        768: "linear",
    }
    codes = {m for m in range(1, 1025) if akr_moe.shared_takes_codes(m)}
    assert codes == set(range(1, 17)) | set(range(33, 97))
    routes(ALL - {"shared_expert"})
    assert not akr_moe.shared_takes_codes(48)


def test_seam_codes_requested_for_the_fused_experts(routes):
    experts = types.SimpleNamespace(akr=types.SimpleNamespace(serves=lambda m: m <= 4))
    moe = types.SimpleNamespace(experts=experts, shared_experts=None)
    assert akr_moe.wants_x_quant(moe, 4) and not akr_moe.wants_x_quant(moe, 5)
    moe.shared_experts = types.SimpleNamespace(akr=object())
    assert akr_moe.wants_x_quant(moe, 48) and not akr_moe.wants_x_quant(moe, 200)


def test_moe_runs_shared_expert_serially_with_routes_on(monkeypatch):
    """With `akr` set, the shared expert must not overlap MegaMoE on the aux
    stream (both persistent kernels stall); without it the stock overlap stays."""
    from vllm.models.deepseek_v4.amd import model as v4

    order: list[str] = []
    overlapped: list[bool] = []

    def routed(x, input_ids, x_quant):
        order.append("routed")
        return x

    def shared(x, x_quant):
        order.append("shared")
        return x

    def in_parallel(fn0, fn1, **kwargs):
        overlapped.append(True)
        return fn0(), fn1()

    monkeypatch.setattr(v4, "maybe_execute_in_parallel", in_parallel)
    moe = types.SimpleNamespace(
        akr=None,
        shared_experts=shared,
        shared_experts_events=(object(), object()),
        shared_experts_stream=object(),
        _forward_mega_moe_routed=routed,
    )
    x = torch.ones(2, 4)
    v4.DeepseekV4MoE._forward_mega_moe(moe, x, None)
    assert overlapped == [True]
    moe.akr = akr_moe.Router("router", None, moe)
    order.clear()
    v4.DeepseekV4MoE._forward_mega_moe(moe, x, None)
    assert overlapped == [True] and order == ["routed", "shared"]


def test_draft_post_load_keeps_the_target_router(routes, monkeypatch):
    """The DSpark draft (128 experts) has no 384-expert layer; its post_load
    must not take the target's router route out (the failure is process-wide)."""
    draft = types.SimpleNamespace(
        model=types.SimpleNamespace(markov_head=object()), modules=lambda: []
    )
    akr_moe.post_load(draft)
    assert akr_config.failed() == {}
    target = types.SimpleNamespace(model=types.SimpleNamespace(), modules=lambda: [])
    akr_moe.post_load(target)
    assert {"router", "shared_expert"} <= set(akr_config.failed())


def _model(end=40, replay=None):
    layers = [
        types.SimpleNamespace(akr_mhc=types.SimpleNamespace(ready=True), engram=None)
        for _ in range(end)
    ]
    layers[1].engram = object()
    return types.SimpleNamespace(
        layers=layers,
        end_layer=end,
        use_sequence_parallel=False,
        decoder_replay_layers=replay,
        decoder_replay_start=end if replay is None else 20,
        akr_tail_ready=True,
    )


def test_aux_hidden_state_deferral(routes):
    layer_ids = range(40)
    model = _model()
    assert akr_mhc.defer_aux(model, 37, layer_ids, 192)
    assert akr_mhc.defer_aux(model, 38, layer_ids, 1024)
    # Engram injects between the post-mix and the seam
    assert not akr_mhc.defer_aux(model, 1, layer_ids, 192)
    assert akr_mhc.defer_aux(model, 40, layer_ids, 192)
    assert not akr_mhc.defer_aux(model, 40, layer_ids, 2048)
    assert not akr_mhc.defer_aux(_model(replay=object()), 40, layer_ids, 192)
    model.use_sequence_parallel = True
    assert not akr_mhc.defer_aux(model, 37, layer_ids, 192)
    routes(ALL - {"mhc_seam"})
    assert not akr_mhc.defer_aux(_model(), 37, layer_ids, 192)


def test_mhc_step_plan_is_consumed_once():
    mhc = akr_mhc.AkrMhc.__new__(akr_mhc.AkrMhc)
    mhc._step = akr_mhc.Step(False, None, False)
    nxt = object()
    mhc.plan_step(want_aux=True, next_layer=nxt, chain_aux=True)
    assert mhc.begin_step() == akr_mhc.Step(True, nxt, True)
    assert mhc.begin_step() == akr_mhc.Step(False, None, False)


def test_prefill_warmup_sizes():
    # one size per AITER GEMM bucket above the captured sizes
    sizes = akr_warmup.gemm_sizes(32768, 1024)
    assert sizes[0] > 1024 and sizes[-1] == 32768
    assert len(sizes) == len(set(sizes)) == 272
    assert akr_warmup.model_sizes(32768, 1024)[0] == 1536
    assert akr_warmup.model_sizes(8192, 1024)[-1] == 8192
    assert akr_warmup.model_sizes(1024, 1024) == []


def test_serving_transient_floor(monkeypatch):
    """With routes on, KV cache sizing keeps at least `serving_transient_gib`
    free for serving transients; a larger measured value is kept."""
    from vllm.models.deepseek_v41.amd import akr_hooks

    gib = 1 << 30
    kv_cache = types.SimpleNamespace(torch_reserve_growth_bytes=int(10.65 * gib))
    worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(extensible_kv_cache=kv_cache),
        vllm_config=types.SimpleNamespace(kernel_config=KernelConfig()),
    )
    monkeypatch.setattr(akr_hooks, "loaded", False)
    akr_warmup.reserve_serving_transient(worker)
    assert kv_cache.torch_reserve_growth_bytes == int(10.65 * gib)
    monkeypatch.setattr(akr_hooks, "loaded", True)
    akr_warmup.reserve_serving_transient(worker)
    assert kv_cache.torch_reserve_growth_bytes == 20 * gib
    kv_cache.torch_reserve_growth_bytes = 25 * gib
    akr_warmup.reserve_serving_transient(worker)
    assert kv_cache.torch_reserve_growth_bytes == 25 * gib
    worker.vllm_config.kernel_config = KernelConfig(
        amd_kernels={"serving_transient_gib": 0}
    )
    kv_cache.torch_reserve_growth_bytes = gib
    akr_warmup.reserve_serving_transient(worker)
    assert kv_cache.torch_reserve_growth_bytes == gib
