# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import operator
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import fx

# Registers vllm::moe_forward_shared.
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
from vllm.compilation.passes.fusion.xpu_moe_shared_fusion import (
    XpuMoESharedFusionPass,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsFp8

HIDDEN = 2048
MOE = torch.ops.vllm.moe_forward_shared.default
# Schema of the op as registered by vllm._xpu_ops (torch >= 2.11: LayerName).
FUSED_SCHEMA = (
    "vllm::xpu_moe_shared_fused(Tensor hidden_states, Tensor router_logits, "
    "{layer_name_type} layer_name) -> Tensor"
)


def _xpu_ops_or_skip():
    pytest.importorskip("vllm_xpu_kernels")
    import vllm._xpu_ops as xpu_ops_mod

    return xpu_ops_mod


def _fused_target():
    """The rewrite target; registered by vllm._xpu_ops on XPU, or a stand-in
    op with the same signature where that module cannot be imported."""
    if not hasattr(torch.ops.vllm, "xpu_moe_shared_fused"):
        from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op

        def _impl(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            layer_name: LayerNameType,
        ) -> torch.Tensor:
            return torch.empty_like(hidden_states)

        direct_register_custom_op(
            op_name="xpu_moe_shared_fused", op_func=_impl, fake_impl=_impl
        )
    return torch.ops.vllm.xpu_moe_shared_fused.default


# Use the real registration when vllm._xpu_ops is importable.
with contextlib.suppress(ImportError):
    import vllm._xpu_ops  # noqa: F401
FUSED = _fused_target()


# ---------------------------------------------------------------------------
# (a) graph rewrite
# ---------------------------------------------------------------------------
def _moe_layer(g, x, layer, *, router_in=None, input_ids=None):
    moe = g.call_function(
        MOE, (x, x if router_in is None else router_in, x, input_ids, layer, 0)
    )
    shared = g.call_function(operator.getitem, (moe, 0))
    routed = g.call_function(operator.getitem, (moe, 1))
    add = g.call_function(torch.ops.aten.add.Tensor, (shared, routed))
    add.meta["val"] = x.meta["val"]
    out = g.call_function(torch.ops.aten.mul.Tensor, (add, 2.0))
    out.meta["val"] = x.meta["val"]
    return moe, shared, routed, add, out


def _graph(num_layers=2, **kw):
    g = fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = torch.empty(4, HIDDEN, dtype=torch.float16, device="meta")
    extra = []
    for i in range(num_layers):
        layer = g.placeholder(f"layer_{i}")
        nodes = _moe_layer(g, x, layer, **kw)
        extra.append(nodes)
        x = nodes[-1]
    g.output(x)
    return g, extra


def _count(g, target):
    return sum(1 for n in g.nodes if n.op == "call_function" and n.target is target)


@pytest.fixture
def fusion_pass(monkeypatch):
    monkeypatch.setattr(
        XpuMoESharedFusionPass,
        "_all_moe_layers_supported",
        staticmethod(lambda config: True),
    )
    p = XpuMoESharedFusionPass(VllmConfig())
    # VllmConfig() has no model; force the model-level gate on.
    p.enabled = True
    return p


def test_rewrites_every_layer(fusion_pass):
    g, layers = _graph(num_layers=2)
    fusion_pass(g)
    assert _count(g, MOE) == 0
    assert _count(g, FUSED) == 2
    for *_, out in layers:
        (src, _) = out.args
        assert src.target is FUSED


def test_range_gating(fusion_pass):
    assert fusion_pass.is_applicable_for_range(Range(start=1, end=8))
    assert not fusion_pass.is_applicable_for_range(Range(start=9, end=4096))
    fusion_pass.enabled = False
    assert not fusion_pass.is_applicable_for_range(Range(start=1, end=8))


def test_disabled_when_a_layer_is_unsupported(monkeypatch):
    runners = {"l0": "ok", "l1": "bad"}
    import vllm.compilation.passes.fusion.xpu_moe_shared_fusion as mod

    monkeypatch.setattr(mod, "get_layers_from_vllm_config", lambda c, t: runners)
    # Stand-in for vllm._xpu_ops so this runs without vllm_xpu_kernels.
    fake_xpu_ops = ModuleType("vllm._xpu_ops")
    fake_xpu_ops.xpu_moe_shared_fused_unsupported_reason = lambda r: (
        None if r == "ok" else "bad"
    )
    monkeypatch.setitem(sys.modules, "vllm._xpu_ops", fake_xpu_ops)
    assert not XpuMoESharedFusionPass._all_moe_layers_supported(VllmConfig())
    runners.pop("l1")
    assert XpuMoESharedFusionPass._all_moe_layers_supported(VllmConfig())
    runners.clear()
    assert not XpuMoESharedFusionPass._all_moe_layers_supported(VllmConfig())


def _structure_before_after(g, fusion_pass):
    before = [(n.op, n.target) for n in g.nodes]
    fusion_pass(g)
    return before, [(n.op, n.target) for n in g.nodes]


def test_router_in_not_hidden_unchanged(fusion_pass):
    g = fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = torch.empty(4, HIDDEN, dtype=torch.float16, device="meta")
    logits = g.placeholder("logits")
    layer = g.placeholder("layer")
    *_, out = _moe_layer(g, x, layer, router_in=logits)
    g.output(out)
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


def test_input_ids_unchanged(fusion_pass):
    g = fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = torch.empty(4, HIDDEN, dtype=torch.float16, device="meta")
    ids = g.placeholder("ids")
    layer = g.placeholder("layer")
    *_, out = _moe_layer(g, x, layer, input_ids=ids)
    g.output(out)
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


def test_extra_getitem_user_unchanged(fusion_pass):
    g, layers = _graph(num_layers=1)
    _, shared, _, _, out = layers[0]
    with g.inserting_after(out):
        g.call_function(torch.ops.aten.neg.default, (shared,))
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


def test_non_add_combine_unchanged(fusion_pass):
    g, layers = _graph(num_layers=1)
    _, _, _, add, _ = layers[0]
    add.target = torch.ops.aten.sub.Tensor
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


def test_wrong_dtype_unchanged(fusion_pass):
    g, _ = _graph(num_layers=1)
    x = next(iter(g.nodes))
    x.meta["val"] = torch.empty(4, HIDDEN, dtype=torch.bfloat16, device="meta")
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


# ---------------------------------------------------------------------------
# (b) preconditions
# ---------------------------------------------------------------------------
I_LOCAL = 256
E = 256


def _fp8(*shape, device="meta"):
    return torch.empty(*shape, dtype=torch.float8_e4m3fn, device=device)


def _xpu_experts(w1_scale, w2_scale):
    # Only the attributes unsupported_reason() / _get_plan() read.
    experts = XPUExpertsFp8.__new__(XPUExpertsFp8)
    experts.quant_config = SimpleNamespace(w1_scale=w1_scale, w2_scale=w2_scale)
    return experts


def _stub_runner(w13, w2, s13, s2, sw13, ss13, sw2, ss2, gate_w):
    parallel = SimpleNamespace(
        enable_eplb=False,
        use_ep=False,
        ep_size=1,
        dp_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
    )
    moe_config = SimpleNamespace(
        moe_parallel_config=parallel,
        is_lora_enabled=False,
        has_bias=False,
        in_dtype=torch.float16,
        experts_per_token=8,
    )
    quant_method = SimpleNamespace(
        moe_kernel=SimpleNamespace(fused_experts=_xpu_experts(s13, s2)),
        topk_indices_dtype=None,
    )
    mlp = SimpleNamespace(
        act_fn=SiluAndMul.__new__(SiluAndMul),
        gate_up_proj=SimpleNamespace(weight=sw13, weight_scale=ss13, bias=None),
        down_proj=SimpleNamespace(weight=sw2, weight_scale=ss2, bias=None),
        expert_gate=SimpleNamespace(weight=gate_w, bias=None),
    )
    return SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        gate=object(),
        shared_expert_gate=None,
        shared_experts=SimpleNamespace(_layer=mlp),
        routed_scaling_factor=1.0,
        activation=MoEActivation.SILU,
        expert_map=None,
        moe_config=moe_config,
        routed_experts=SimpleNamespace(
            quant_method=quant_method,
            apply_router_weight_on_input=False,
            w13_weight=w13,
            w2_weight=w2,
        ),
    )


def _base_runner():
    return _stub_runner(
        _fp8(E, HIDDEN, 2 * I_LOCAL),
        _fp8(E, I_LOCAL, HIDDEN),
        torch.empty(E),
        torch.empty(E),
        _fp8(HIDDEN, 2 * I_LOCAL),
        torch.empty(1),
        _fp8(I_LOCAL, HIDDEN),
        torch.empty(1),
        torch.empty(1, HIDDEN, dtype=torch.float16),
    )


@pytest.fixture
def xpu_ops_mod():
    return _xpu_ops_or_skip()


@pytest.fixture
def fake_kernel_interface(monkeypatch, xpu_ops_mod):
    mod = ModuleType("vllm_xpu_kernels.moe_shared_fused_interface")
    mod.supports = lambda *args: True
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(xpu_ops_mod, "xpu_moe_shared_fused_available", lambda: True)
    return mod


def _reason(runner):
    import vllm._xpu_ops as xpu_ops_mod

    return xpu_ops_mod.xpu_moe_shared_fused_unsupported_reason(runner)


def _set(obj, path, value):
    *parents, attr = path.split(".")
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, attr, value)


def test_base_case_supported(fake_kernel_interface):
    assert _reason(_base_runner()) is None


MLP = "shared_experts._layer"


@pytest.mark.parametrize(
    "path,value",
    [
        ("activation", MoEActivation.GELU),
        (f"{MLP}.act_fn", object()),
        (f"{MLP}.expert_gate", None),
        ("gate", None),
        ("shared_expert_gate", object()),
        ("routed_scaling_factor", 2.5),
        ("routed_experts.apply_router_weight_on_input", True),
        ("moe_config.is_lora_enabled", True),
        ("moe_config.moe_parallel_config.enable_eplb", True),
        ("moe_config.moe_parallel_config.use_ep", True),
        ("moe_config.moe_parallel_config.ep_size", 2),
        ("expert_map", torch.empty(E)),
        ("moe_config.moe_parallel_config.dp_size", 2),
        ("moe_config.moe_parallel_config.pcp_size", 2),
        ("moe_config.moe_parallel_config.is_sequence_parallel", True),
        ("moe_config.has_bias", True),
        (f"{MLP}.gate_up_proj.bias", torch.empty(2 * I_LOCAL)),
        (f"{MLP}.down_proj.bias", torch.empty(HIDDEN)),
        ("routed_experts.w13_weight", torch.empty(E, HIDDEN, 2 * I_LOCAL)),
        (f"{MLP}.gate_up_proj.weight_scale", torch.empty(2)),
        (f"{MLP}.down_proj.weight", _fp8(2 * I_LOCAL, HIDDEN)),
        (f"{MLP}.expert_gate.weight", torch.empty(1, HIDDEN)),
    ],
)
def test_each_precondition(fake_kernel_interface, path, value):
    runner = _base_runner()
    _set(runner, path, value)
    assert _reason(runner) is not None


def test_routed_scale_not_per_expert(fake_kernel_interface):
    runner = _base_runner()
    experts = runner.routed_experts.quant_method.moe_kernel.fused_experts
    experts.quant_config.w1_scale = torch.empty(E, 2)
    assert _reason(runner) is not None


def test_no_shared_layer_attribute(fake_kernel_interface):
    runner = _base_runner()
    runner.shared_experts = SimpleNamespace()
    assert _reason(runner) is not None
    runner.shared_experts = None
    assert _reason(runner) is not None


def test_experts_not_xpu_fp8(fake_kernel_interface):
    runner = _base_runner()
    runner.routed_experts.quant_method.moe_kernel.fused_experts = SimpleNamespace(
        quant_config=SimpleNamespace(w1_scale=torch.empty(E), w2_scale=torch.empty(E))
    )
    assert _reason(runner) is not None
    runner.routed_experts.quant_method.moe_kernel = None
    assert _reason(runner) is not None


def test_kernel_rejects_configuration(fake_kernel_interface):
    fake_kernel_interface.supports = lambda *args: False
    assert _reason(_base_runner()) is not None


def test_kernel_unavailable(monkeypatch, xpu_ops_mod):
    monkeypatch.setattr(xpu_ops_mod, "xpu_moe_shared_fused_available", lambda: False)
    assert _reason(_base_runner()) is not None


def test_op_registered_by_xpu_ops(xpu_ops_mod):
    if not xpu_ops_mod.xpu_moe_shared_fused_available():
        pytest.skip("vllm-xpu-kernels build without the fused op")
    schema = torch.ops.vllm.xpu_moe_shared_fused.default._schema
    moe_schema = torch.ops.vllm.moe_forward_shared.default._schema
    # layer_name must have the same type as in moe_forward_shared, so the
    # pass can forward that graph argument unchanged.
    layer_name_type = moe_schema.arguments[4].type
    assert schema.arguments[2].type == layer_name_type
    assert str(schema) == FUSED_SCHEMA.format(layer_name_type=layer_name_type)


# ---------------------------------------------------------------------------
# (c) numerics on XPU: the full op with a stub runner
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_op_matches_reference(monkeypatch, xpu_ops_mod, m):
    if not (torch.xpu.is_available() and xpu_ops_mod.xpu_moe_shared_fused_available()):
        pytest.skip("needs XPU fused MoE op")
    dev = "xpu"
    torch.manual_seed(0)

    def q8(w):
        if w.dim() == 3:
            s = w.abs().amax(dim=(1, 2)) / 448.0
            return (w / s[:, None, None]).to(torch.float8_e4m3fn), s.float()
        s = (w.abs().max() / 448.0).reshape(1)
        return (w / s).to(torch.float8_e4m3fn), s.float()

    w13, s13 = q8(torch.randn(E, HIDDEN, 2 * I_LOCAL, device=dev) / 32)
    w2, s2 = q8(torch.randn(E, I_LOCAL, HIDDEN, device=dev) / 32)
    sw13, ss13 = q8(torch.randn(HIDDEN, 2 * I_LOCAL, device=dev) / 32)
    sw2, ss2 = q8(torch.randn(I_LOCAL, HIDDEN, device=dev) / 32)
    gate_w = (torch.randn(1, HIDDEN, device=dev) / 32).half()
    router_w = (torch.randn(E, HIDDEN, device=dev) / 32).half()
    runner = _stub_runner(w13, w2, s13, s2, sw13, ss13, sw2, ss2, gate_w)

    def route(hidden_states, router_logits, topk_indices_dtype, input_ids):
        probs = torch.softmax(router_logits.float(), dim=-1)
        tw, ti = torch.topk(probs, 8, dim=-1)
        return tw / tw.sum(-1, keepdim=True), ti.to(torch.int32)

    runner.gate = lambda x: (torch.nn.functional.linear(x, router_w), None)
    runner.router = SimpleNamespace(select_experts=route)
    monkeypatch.setattr(moe_runner, "get_layer_from_name", lambda name: runner)

    x = (torch.randn(m, HIDDEN, device=dev) / 4).half()
    out = torch.ops.vllm.xpu_moe_shared_fused(x, x, "layer").float()
    assert getattr(runner, xpu_ops_mod._XPU_MOE_SHARED_FUSED_PLAN_ATTR) is not None

    def deq(w, s):
        return w.float() * (s[:, None, None] if w.dim() == 3 else s)

    W13, W2, SW13, SW2 = deq(w13, s13), deq(w2, s2), deq(sw13, ss13), deq(sw2, ss2)
    tw, ti = route(x, runner.gate(x)[0], None, None)
    xf = x.float()
    ref = torch.zeros_like(xf)
    for i in range(m):
        for k in range(8):
            e = int(ti[i, k])
            h = xf[i] @ W13[e]
            a = torch.nn.functional.silu(h[:I_LOCAL]) * h[I_LOCAL:]
            ref[i] += tw[i, k] * (a.half().float() @ W2[e])
    h = xf @ SW13
    a = torch.nn.functional.silu(h[:, :I_LOCAL]) * h[:, I_LOCAL:]
    ref += torch.sigmoid(xf @ gate_w.float().t()) * (a.half().float() @ SW2)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
