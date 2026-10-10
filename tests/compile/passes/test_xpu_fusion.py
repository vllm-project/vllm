# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import operator
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import fx

# Registers vllm::moe_forward_shared.
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.xpu_fusion import (
    XpuFp8GemmPairFusionPass,
    XpuMoESharedFusionPass,
    XpuNormFp8GemmFusionPass,
    XpuQkvNormRopeFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsFp8
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType

# The MoE + shared-expert graph-rewrite and precondition tests also run off
# XPU; everything else needs it.
xpu_only = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

# Only config.json is read (GDN head geometry, rms_norm_eps, rope_parameters).
MODEL = os.environ.get("VLLM_TEST_QWEN36_MOE_MODEL", "Qwen/Qwen3.6-35B-A3B")


# ---------------------------------------------------------------------------
# XpuFp8GemmPairFusionPass
# ---------------------------------------------------------------------------
@pytest.fixture
def pair_pass():
    import vllm._xpu_ops  # noqa: F401

    p = XpuFp8GemmPairFusionPass(VllmConfig())
    if not p.enabled:
        pytest.skip("vllm-xpu-kernels without fp8_gemm_w8a16_pair")
    return p


def _meta(*shape, dtype=torch.float16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _pair_graph(n_gemms=2, bias=False, block_scale=False):
    g = fx.Graph()
    a = g.placeholder("a")
    a.meta["val"] = _meta(1, 2048)
    sizes = (6144, 32, 128)[:n_gemms]
    params = []
    for i, n in enumerate(sizes):
        w, s = g.placeholder(f"w{i}"), g.placeholder(f"s{i}")
        w.meta["val"] = _meta(n, 2048, dtype=torch.float8_e4m3fn).t()
        s.meta["val"] = (
            _meta(16, 16, dtype=torch.float32)
            if block_scale
            else _meta(1, dtype=torch.float32)
        )
        params.append((w, s))
    outs = []
    for n, (w, s) in zip(sizes, params):
        args = (a, w, s, a) if bias else (a, w, s, None)
        mm = g.call_function(torch.ops._xpu_C.fp8_gemm_w8a16.default, args)
        mm.meta["val"] = _meta(1, n)
        outs.append(mm)
    g.output(tuple(outs))
    return g


@xpu_only
def test_pairs_two_gemms_sharing_input(pair_pass):
    g = _pair_graph()
    pair_pass(g)
    assert pair_pass.matched_count == 1
    assert _count(g, torch.ops._xpu_C.fp8_gemm_w8a16_pair.default) == 1
    assert _count(g, torch.ops._xpu_C.fp8_gemm_w8a16.default) == 0


@xpu_only
@pytest.mark.parametrize("kw", [{"n_gemms": 3}, {"bias": True}, {"block_scale": True}])
def test_other_cases_unchanged(pair_pass, kw):
    g = _pair_graph(**kw)
    pair_pass(g)
    assert pair_pass.matched_count == 0


@xpu_only
def test_pair_decode_ranges_only(pair_pass):
    assert pair_pass.is_applicable_for_range(Range(start=1, end=8))
    assert not pair_pass.is_applicable_for_range(Range(start=9, end=4096))


# ---------------------------------------------------------------------------
# XpuMoESharedFusionPass
# ---------------------------------------------------------------------------
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
    if not hasattr(torch.ops.vllm, "xpu_moe_shared_fused_resadd_norm"):
        from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op

        def _norm_impl(
            x: torch.Tensor,
            residual: torch.Tensor,
            norm_weight: torch.Tensor,
            eps: float,
            layer_name: LayerNameType,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty_like(residual)

        direct_register_custom_op(
            op_name="xpu_moe_shared_fused_resadd_norm",
            op_func=_norm_impl,
            fake_impl=_norm_impl,
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


def _moe_graph(num_layers=2, dtype=torch.float16, **kw):
    g = fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = torch.empty(4, HIDDEN, dtype=dtype, device="meta")
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rewrites_every_layer(fusion_pass, dtype):
    g, layers = _moe_graph(num_layers=2, dtype=dtype)
    fusion_pass(g)
    assert _count(g, MOE) == 0
    assert _count(g, FUSED) == 2
    for *_, out in layers:
        (src, _) = out.args
        assert src.target is FUSED


def _norm_moe_graph(extra_normed_user=False, dtype=torch.float16):
    """Graph: res' , h = fused_add_rms_norm(x, res, w.float() + 1, eps); moe(h);
    returns the graph and the node using res'."""
    g = fx.Graph()
    val = torch.empty(4, HIDDEN, dtype=dtype, device="meta")
    x, res, w, layer = (g.placeholder(n) for n in ("x", "res", "w", "layer"))
    x.meta["val"], res.meta["val"] = val, val
    w.meta["val"] = torch.empty(HIDDEN, dtype=dtype, device="meta")
    wf = g.call_function(
        torch.ops.prims.convert_element_type.default, (w, torch.float32)
    )
    w1 = g.call_function(torch.ops.aten.add.Tensor, (wf, 1.0))
    norm = g.call_function(
        torch.ops.vllm_ir.fused_add_rms_norm.default, (x, res, w1, 1e-6)
    )
    h = g.call_function(operator.getitem, (norm, 0))
    h.meta["val"] = val
    new_res = g.call_function(operator.getitem, (norm, 1))
    new_res.meta["val"] = val
    *_, out = _moe_layer(g, h, layer)
    res_user = g.call_function(torch.ops.aten.mul.Tensor, (new_res, 3.0))
    outs = [out, res_user]
    if extra_normed_user:
        outs.append(h)
    g.output(tuple(outs))
    return g, res_user


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_input_norm_fused(fusion_pass, dtype):
    if not hasattr(torch.ops.vllm, "xpu_moe_shared_fused_resadd_norm"):
        pytest.skip("needs vllm._xpu_ops")
    fusion_pass.fuse_input_norm = True
    g, res_user = _norm_moe_graph(dtype=dtype)
    fusion_pass(g)
    fused = torch.ops.vllm.xpu_moe_shared_fused_resadd_norm.default
    assert _count(g, fused) == 1
    assert _count(g, torch.ops.vllm_ir.fused_add_rms_norm.default) == 0
    assert _count(g, MOE) == 0
    node = next(n for n in g.nodes if n.target is fused)
    x, res, w, eps, _ = node.args
    assert (x.name, res.name, w.name, eps) == ("x", "res", "w", 1e-6)
    # The new residual comes from the fused op.
    src = res_user.args[0]
    assert src.target is operator.getitem and src.args == (node, 1)


def test_input_norm_with_other_user_not_fused(fusion_pass):
    if not hasattr(torch.ops.vllm, "xpu_moe_shared_fused_resadd_norm"):
        pytest.skip("needs vllm._xpu_ops")
    fusion_pass.fuse_input_norm = True
    g, _ = _norm_moe_graph(extra_normed_user=True)
    fusion_pass(g)
    assert _count(g, torch.ops.vllm_ir.fused_add_rms_norm.default) == 1
    assert _count(g, FUSED) == 1


def test_range_gating(fusion_pass):
    assert fusion_pass.is_applicable_for_range(Range(start=1, end=8))
    assert not fusion_pass.is_applicable_for_range(Range(start=9, end=4096))
    fusion_pass.enabled = False
    assert not fusion_pass.is_applicable_for_range(Range(start=1, end=8))


def test_disabled_when_a_layer_is_unsupported(monkeypatch):
    runners = {"l0": "ok", "l1": "bad"}
    import vllm.compilation.passes.fusion.xpu_fusion as mod

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


@pytest.mark.parametrize(
    "linear,moe,targets,expected",
    [
        ("tensor", "tensor", None, True),
        ("tensor", "block", None, False),
        (None, "tensor", None, False),
        ("tensor", "tensor", {"re:.*": "fp8_per_tensor_static"}, False),
    ],
)
def test_fp8_per_tensor_online_quantization(linear, moe, targets, expected):
    # --quantization fp8 on an unquantized checkpoint is online quantization.
    from vllm.compilation.passes.fusion.xpu_fusion import _is_fp8_per_tensor
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Static128BlockSym,
        kFp8StaticTensorSym,
    )

    keys = {"tensor": kFp8StaticTensorSym, "block": kFp8Static128BlockSym}

    def spec(name):
        return None if name is None else QuantSpec(weight=keys[name])

    args = SimpleNamespace(linear=spec(linear), moe=spec(moe), targets=targets)
    quant = SimpleNamespace(get_name=lambda: "online", args=args)
    assert _is_fp8_per_tensor(SimpleNamespace(quant_config=quant)) is expected


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
    g, layers = _moe_graph(num_layers=1)
    _, shared, _, _, out = layers[0]
    with g.inserting_after(out):
        g.call_function(torch.ops.aten.neg.default, (shared,))
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


def test_non_add_combine_unchanged(fusion_pass):
    g, layers = _moe_graph(num_layers=1)
    _, _, _, add, _ = layers[0]
    add.target = torch.ops.aten.sub.Tensor
    before, after = _structure_before_after(g, fusion_pass)
    assert before == after


def test_wrong_dtype_unchanged(fusion_pass):
    g, _ = _moe_graph(num_layers=1)
    x = next(iter(g.nodes))
    x.meta["val"] = torch.empty(4, HIDDEN, dtype=torch.float32, device="meta")
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


def _stub_runner(w13, w2, s13, s2, sw13, ss13, sw2, ss2, gate_w, router_w=None):
    if router_w is None:
        router_w = torch.empty(E, HIDDEN, dtype=gate_w.dtype, device="meta")
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
        in_dtype=gate_w.dtype,
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
        gate=SimpleNamespace(weight=router_w, bias=None),
        router=FusedTopKRouter(top_k=8, global_num_experts=E, renormalize=True),
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


def _base_runner(dtype=torch.float16):
    return _stub_runner(
        _fp8(E, HIDDEN, 2 * I_LOCAL),
        _fp8(E, I_LOCAL, HIDDEN),
        torch.empty(E),
        torch.empty(E),
        _fp8(HIDDEN, 2 * I_LOCAL),
        torch.empty(1),
        _fp8(I_LOCAL, HIDDEN),
        torch.empty(1),
        torch.empty(1, HIDDEN, dtype=dtype),
    )


@pytest.fixture
def xpu_ops_mod():
    return _xpu_ops_or_skip()


@pytest.fixture
def fake_kernel_interface(monkeypatch, xpu_ops_mod):
    mod = ModuleType("vllm_xpu_kernels.moe_shared_fused_interface")
    mod.supports = lambda *args: True
    mod.router_supports = lambda *args: True
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_base_case_supported(fake_kernel_interface, dtype):
    assert _reason(_base_runner(dtype=dtype)) is None


@pytest.mark.parametrize(
    "path", ["gate.weight", "shared_experts._layer.expert_gate.weight"]
)
def test_bf16_runner_rejects_fp16_gate(fake_kernel_interface, path):
    runner = _base_runner(dtype=torch.bfloat16)
    old = (
        runner.gate.weight
        if path == "gate.weight"
        else runner.shared_experts._layer.expert_gate.weight
    )
    _set(runner, path, old.to(torch.float16))
    assert _reason(runner) is not None


@pytest.mark.parametrize("input_name", ["x", "res", "w"])
def test_input_norm_mixed_dtype_keeps_unfused_norm(fusion_pass, input_name):
    graph, _ = _norm_moe_graph(dtype=torch.bfloat16)
    value = next(node for node in graph.nodes if node.name == input_name)
    value.meta["val"] = value.meta["val"].to(torch.float16)
    fusion_pass.fuse_input_norm = True
    fusion_pass(graph)
    assert _count(graph, torch.ops.vllm_ir.fused_add_rms_norm.default) == 1
    assert _count(graph, FUSED) == 1


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
        ("router.scoring_func", "sigmoid"),
        ("router", SimpleNamespace(scoring_func="softmax", eplb_state=None)),
        ("gate.bias", torch.empty(E)),
        ("gate.weight", _fp8(E, HIDDEN)),
        ("gate.weight", torch.empty(E, 2 * HIDDEN, dtype=torch.float16)),
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


def test_router_kernel_rejects_configuration(fake_kernel_interface):
    fake_kernel_interface.router_supports = lambda *args: False
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
    runner = _stub_runner(w13, w2, s13, s2, sw13, ss13, sw2, ss2, gate_w, router_w)
    captured = []
    runner.router.set_capture_fn(captured.append)

    def route(x):
        # F.linear router + softmax top-k, renormalized.
        probs = torch.softmax(torch.nn.functional.linear(x, router_w).float(), -1)
        tw, ti = torch.topk(probs, 8, dim=-1)
        return tw / tw.sum(-1, keepdim=True), ti.to(torch.int32)

    monkeypatch.setattr(moe_runner, "get_layer_from_name", lambda name: runner)

    x = (torch.randn(m, HIDDEN, device=dev) / 4).half()
    out = torch.ops.vllm.xpu_moe_shared_fused(x, x, "layer").float()
    assert getattr(runner, xpu_ops_mod._XPU_MOE_SHARED_FUSED_PLAN_ATTR) is not None

    def deq(w, s):
        return w.float() * (s[:, None, None] if w.dim() == 3 else s)

    W13, W2, SW13, SW2 = deq(w13, s13), deq(w2, s2), deq(sw13, ss13), deq(sw2, ss2)
    tw, ti = route(x)
    assert len(captured) == 1
    assert torch.equal(captured[0].sort(-1).values, ti.sort(-1).values)
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


# ---------------------------------------------------------------------------
# XpuNormFp8GemmFusionPass
# ---------------------------------------------------------------------------
def _norm_config():
    return VllmConfig(
        model_config=ModelConfig(
            model=MODEL, dtype=torch.float16, trust_remote_code=True
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["none"],
            pass_config=PassConfig(fuse_xpu_norm_fp8_gemm=True, eliminate_noops=True),
        ),
    )


def _norm_ops_available():
    import vllm._xpu_ops  # noqa: F401

    return hasattr(torch.ops._xpu_C, "gated_rmsnorm_fp8_gemm")


class GdnOutput(torch.nn.Module):
    """RMSNormGated + out_proj, as QwenGatedDeltaNetAttention part 3 (TP1)."""

    def __init__(self, heads=32, head_dim=128, flat_rows=True):
        super().__init__()
        # flat_rows: norm on (T * H, D) rows; otherwise on the (T, H, D)
        # tensors directly.
        self.flat_rows = flat_rows
        self.norm = RMSNormGated(head_dim, eps=1e-6, norm_before_gate=True)
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)
        w = (torch.randn(2048, heads * head_dim) * 0.05).to(torch.float8_e4m3fn)
        self.register_buffer("w_t", w.t(), persistent=False)
        self.register_buffer("scale", torch.tensor([0.02]), persistent=False)

    def forward(self, core_attn_out, z):
        if self.flat_rows:
            z_shape = z.shape
            x = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            y = self.norm(x, z.reshape(-1, z.shape[-1]))
            y = y.reshape(z_shape).flatten(-2)
        else:
            y = self.norm(core_attn_out, z).flatten(-2)
        return torch.ops._xpu_C.fp8_gemm_w8a16.default(y, self.w_t, self.scale, None)


@xpu_only
@pytest.mark.parametrize("flat_rows", [True, False])
def test_gated_norm_fused(flat_rows):
    if not _norm_ops_available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    torch.set_default_device("xpu")
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(0)
    vllm_config = _norm_config()
    # Inference graph (as in vLLM): no autograd decompositions.
    with set_current_vllm_config(vllm_config), torch.inference_mode():
        model = GdnOutput(flat_rows=flat_rows)
        fusion = XpuNormFp8GemmFusionPass(vllm_config)
        noop, cleanup = NoOpEliminationPass(vllm_config), PostCleanupPass(vllm_config)
        x = torch.randn(1, 32, 128)
        z = torch.randn(1, 32, 128)
        ref = torch.compile(model, backend=TestBackend(noop, cleanup))(x, z)
        out = torch.compile(model, backend=TestBackend(noop, fusion, cleanup))(x, z)
        assert fusion.gated_count == 1
        torch.testing.assert_close(out, ref, atol=3e-3, rtol=2e-2)


def _resadd_graph(pair=False, extra_user=False):
    g = fx.Graph()
    val = torch.empty(1, 2048, dtype=torch.float16, device="meta")
    x, res, w = (g.placeholder(n) for n in ("x", "res", "w"))
    x.meta["val"], res.meta["val"] = val, val
    w.meta["val"] = torch.empty(2048, dtype=torch.float16, device="meta")
    wf = g.call_function(
        torch.ops.prims.convert_element_type.default, (w, torch.float32)
    )
    w1 = g.call_function(torch.ops.aten.add.Tensor, (wf, 1.0))
    norm = g.call_function(
        torch.ops.vllm_ir.fused_add_rms_norm.default, (x, res, w1, 1e-6)
    )
    h = g.call_function(operator.getitem, (norm, 0))
    new_res = g.call_function(operator.getitem, (norm, 1))
    b1, s1, b2, s2 = (g.placeholder(n) for n in ("b1", "s1", "b2", "s2"))
    if pair:
        mm = g.call_function(
            torch.ops._xpu_C.fp8_gemm_w8a16_pair.default, (h, b1, s1, b2, s2)
        )
        outs = [g.call_function(operator.getitem, (mm, i)) for i in (0, 1)]
    else:
        mm = g.call_function(torch.ops._xpu_C.fp8_gemm_w8a16.default, (h, b1, s1, None))
        mm.meta["val"] = val
        outs = [mm]
    outs.append(new_res)
    if extra_user:
        outs.append(h)
    g.output(tuple(outs))
    return g


@xpu_only
@pytest.mark.parametrize("pair", [False, True])
def test_resadd_norm_fused(pair):
    if not _norm_ops_available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    fusion = XpuNormFp8GemmFusionPass(VllmConfig())
    g = _resadd_graph(pair)
    fusion(g)
    assert fusion.resadd_count == 1
    target = (
        torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm_pair.default
        if pair
        else torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm.default
    )
    assert any(n.target is target for n in g.nodes)
    assert not any(
        n.target is torch.ops.vllm_ir.fused_add_rms_norm.default for n in g.nodes
    )


@xpu_only
def test_resadd_norm_with_other_user_unchanged():
    if not _norm_ops_available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    fusion = XpuNormFp8GemmFusionPass(VllmConfig())
    g = _resadd_graph(extra_user=True)
    fusion(g)
    assert fusion.resadd_count == 0


@xpu_only
def test_norm_decode_ranges_only():
    if not _norm_ops_available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    fusion = XpuNormFp8GemmFusionPass(VllmConfig())
    assert fusion.is_applicable_for_range(Range(start=1, end=8))
    assert not fusion.is_applicable_for_range(Range(start=9, end=4096))


# ---------------------------------------------------------------------------
# XpuQkvNormRopeFusionPass
# ---------------------------------------------------------------------------
FUSED_OP = "_xpu_C.qkv_split_norm_rope"

ROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}


class GatedQkvModel(torch.nn.Module):
    """The unfused gated QKV post-processing of Qwen3NextAttention."""

    def __init__(self, num_heads, num_kv_heads, head_dim, vllm_config, dtype):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.3.self_attn.attn",
            attn_type=AttentionType.DECODER,
        )
        self.q_norm = GemmaRMSNorm(head_dim, eps=1e-6)
        self.k_norm = GemmaRMSNorm(head_dim, eps=1e-6)
        with torch.no_grad():
            self.q_norm.weight.normal_(0, 0.1)
            self.k_norm.weight.normal_(0, 0.1)
        self.rotary_emb = get_rope(
            head_size=head_dim,
            max_position=4096,
            rope_parameters=ROPE_PARAMETERS,
            dtype=dtype,
        )

    def forward(self, qkv, positions):
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        # As Attention.forward does before the attention op.
        return (
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            torch.sigmoid(gate),
        )


class TritonGatedQkvModel(GatedQkvModel):
    """Qwen3NextAttention with the fused Triton q/k norm + RoPE + gate kernel."""

    def forward(self, qkv, positions):
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q, k, gate = fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.q_norm.variance_epsilon,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.rotary_dim,
            norm_beta=1.0,
            mrope_section=self.rotary_emb.mrope_section
            if positions.ndim == 2
            else None,
        )
        return (
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            torch.sigmoid(gate),
        )


def _has_op(graph, name):
    # The fused op sits inside auto_functionalized(op, ...).
    return any(
        n.op == "call_function"
        and (name in str(n.target) or (n.args and name in str(n.args[0])))
        for n in graph.nodes
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qkv_fusion_keeps_rules_for_target_and_draft_heads(monkeypatch, dtype):
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
    from torch._inductor.pattern_matcher import PatternMatcherPass
    from torch._subclasses.fake_tensor import FakeTensorMode

    import vllm.compilation.passes.fusion.xpu_fusion as fusion_module
    from vllm.config import DeviceConfig

    if not hasattr(torch.ops._xpu_C, "qkv_split_norm_rope"):
        pytest.skip("qkv_split_norm_rope not available")
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    matcher = PatternMatcherPass()
    traced_rules = []
    register = fusion_module.pm.register_replacement

    def record(search, replacement, inputs, trace, *args, **kwargs):
        result = register(search, replacement, inputs, trace, *args, **kwargs)
        traced_rules.append((search, inputs, trace))
        return result

    monkeypatch.setattr(fusion_module.pm, "register_replacement", record)
    with (
        FakeTensorMode(),
        set_current_vllm_config(config),
        config.kernel_config.ir_op_priority.set_priority(),
    ):
        for heads in (4, 8):
            pattern = fusion_module.XpuGatedQkvNormRopePattern(
                num_heads=heads,
                num_kv_heads=1,
                eps=1e-6,
                rope=fusion_module._RopeSpec(
                    head_dim=256,
                    rotary_dim=64,
                    mrope_section=[11, 11, 10],
                    mrope_interleaved=True,
                ),
                mrope_positions=False,
                dtype=dtype,
                config=config,
            )
            inputs = [
                torch.empty(5, 2 * heads * 256 + 512, dtype=dtype, device="cpu"),
                torch.empty(5, dtype=torch.int64, device="cpu"),
                torch.empty(256, dtype=dtype, device="cpu"),
                torch.empty(256, dtype=dtype, device="cpu"),
                torch.empty(4096, 64, dtype=dtype, device="cpu"),
            ]
            monkeypatch.setattr(pattern, "get_inputs", lambda inputs=inputs: inputs)
            pattern.register(matcher)

        for heads, (search, inputs, trace) in zip((4, 8), traced_rules[::2]):
            graph = trace(search, inputs)
            assert matcher.apply(graph) == 1
            fused = next(
                node for node in graph.graph.nodes if node.target == auto_functionalized
            )
            assert fused.kwargs["num_q_heads"] == heads
            assert fused.kwargs["head_dim"] == 256


@xpu_only
@pytest.mark.parametrize("model_cls", [GatedQkvModel, TritonGatedQkvModel])
@pytest.mark.parametrize("mrope", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 5])
def test_xpu_qkv_norm_rope_fusion(model_cls, mrope, num_tokens):
    if not hasattr(torch.ops._xpu_C, "qkv_split_norm_rope"):
        pytest.skip("qkv_split_norm_rope not available")
    dtype = torch.float16
    torch.set_default_device("xpu")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=MODEL, dtype=dtype, trust_remote_code=True),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_xpu_qkv_norm_rope=True, eliminate_noops=True),
        ),
    )
    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = model_cls(8, 1, 256, vllm_config, dtype)
        fusion = XpuQkvNormRopeFusionPass(vllm_config)
        noop, cleanup = NoOpEliminationPass(vllm_config), PostCleanupPass(vllm_config)
        backend = TestBackend(noop, fusion, cleanup)
        backend_ref = TestBackend(noop, cleanup)
        T = num_tokens
        qkv = torch.randn(T, 2 * model.q_size + 2 * model.kv_size) * 3
        pos = torch.randint(0, 4096, (3, T) if mrope else (T,))
        args, args_ref = (qkv.clone(), pos.clone()), (qkv.clone(), pos.clone())
        for q_in, p_in in (args, args_ref):
            # Dynamic token count, as in vLLM's compiled graphs.
            torch._dynamo.mark_dynamic(q_in, 0)
            torch._dynamo.mark_dynamic(p_in, p_in.dim() - 1)
        ref = torch.compile(model, backend=backend_ref)(*args_ref)
        out = torch.compile(model, backend=backend)(*args)
        assert fusion.matched_count == 1
        assert _has_op(backend.graph_post_pass, "qkv_split_norm_rope")
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)


@xpu_only
def test_xpu_qkv_norm_rope_fusion_decode_ranges_only():
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=MODEL, dtype=torch.float16, trust_remote_code=True
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_xpu_qkv_norm_rope=True),
        ),
    )
    with set_current_vllm_config(vllm_config):
        fusion = XpuQkvNormRopeFusionPass(vllm_config)
    assert fusion.is_applicable_for_range(Range(start=1, end=8))
    assert not fusion.is_applicable_for_range(Range(start=9, end=4096))
