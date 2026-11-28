# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    get_moe_padding_mask,
)
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.models.deepseek_v4.amd import mega_moe as mega_moe_module
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
)
from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import aux_stream

HIDDEN = 64
INTER = 64


def _make_experts(
    monkeypatch,
    *,
    ep_rank=0,
    ep_size=1,
    num_experts=2,
    intermediate_size=INTER,
    max_num_batched_tokens=3000,
    swiglu_limit=None,
    fuse_shared_expert=False,
):
    monkeypatch.setattr(
        mega_moe_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            rank_in_group=ep_rank, world_size=ep_size, cpu_group="ep"
        ),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
    )
    return DeepseekV4AiterMegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=intermediate_size,
        swiglu_limit=swiglu_limit,
        fuse_shared_expert=fuse_shared_expert,
    )


@pytest.mark.cpu_test
def test_experts_load_into_local_slots(monkeypatch):
    model = torch.nn.Module()
    model.ffn = torch.nn.Module()
    model.ffn.experts = _make_experts(monkeypatch, ep_rank=1, ep_size=2, num_experts=4)
    params = dict(model.named_parameters())
    mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=4,
        routed_experts_prefix="",
    )

    shapes = {
        ("w1", "weight"): (INTER, HIDDEN // 2),
        ("w3", "weight"): (INTER, HIDDEN // 2),
        ("w2", "weight"): (HIDDEN, INTER // 2),
        ("w1", "weight_scale"): (INTER, HIDDEN // 32),
        ("w3", "weight_scale"): (INTER, HIDDEN // 32),
        ("w2", "weight_scale"): (HIDDEN, INTER // 32),
    }
    shard_value = {"w1": 1, "w3": 2, "w2": 3}
    loaded = {}
    for expert in range(4):
        for (shard, suffix), shape in shapes.items():
            name = f"ffn.experts.{expert}.{shard}.{suffix}"
            weight = torch.full(
                shape, 10 * expert + shard_value[shard], dtype=torch.uint8
            )
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params[mapped]
                if param.weight_loader(
                    param,
                    weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded[name] = mapped
                    break

    assert sorted({n.split(".")[2] for n in loaded}) == ["2", "3"]
    assert len(loaded) == 2 * len(shapes)
    for slot, expert in enumerate((2, 3)):
        for suffix in ("weight", "weight_scale"):
            w13 = params[f"ffn.experts.w13_{suffix}"][slot]
            w2 = params[f"ffn.experts.w2_{suffix}"][slot]
            assert torch.all(w13[:INTER] == 10 * expert + 1)
            assert torch.all(w13[INTER:] == 10 * expert + 2)
            assert torch.all(w2 == 10 * expert + 3)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens_across_dp, expected",
    [([4, 128], "decode"), ([4, 129], "prefill")],
)
def test_runtime_selected_by_max_tokens_across_dp(
    monkeypatch, tokens_across_dp, expected
):
    experts = _make_experts(monkeypatch)
    experts._runtimes = [(128, "decode"), (4096, "prefill")]
    monkeypatch.setattr(mega_moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_moe_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp)
            )
        ),
    )
    # Every rank must take the same runtime, so a small local batch still gets
    # the prefill runtime when another DP rank is over the decode capacity.
    assert experts.select_runtime(4) == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "moe_backend, targets, fused",
    [
        ("aiter_mega_moe", None, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp8"}, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp4"}, True),
        ("aiter", {"*shared_experts*": "mxfp4"}, False),
    ],
)
def test_mega_moe_shared_expert_fusion_requires_mxfp4_target(
    moe_backend, targets, fused
):
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    quant_config._resolved_expert_dtype = "fp4"
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}
    if targets is not None:
        quant_config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets=targets)
        )
    compatible, _ = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.3.ffn.experts",
        "model.layers.3.ffn.shared_experts",
        moe_backend=moe_backend,
    )
    assert compatible is fused


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_shared_expert_route_appended(monkeypatch):
    experts = _make_experts(
        monkeypatch, ep_rank=3, ep_size=8, num_experts=384, fuse_shared_expert=True
    )
    ids = torch.randint(0, 384, (37, 2), device="cuda", dtype=torch.int32)
    ids[::5, 1] = -1
    weights = torch.rand(37, 2, device="cuda")
    out_weights, out_ids = experts.append_shared_expert(weights, ids)

    shared = torch.full((37, 1), 3 * 49 + 48, device="cuda", dtype=torch.int32)
    expected_ids = torch.cat([torch.where(ids >= 0, ids + ids // 48, ids), shared], 1)
    torch.testing.assert_close(out_ids, expected_ids)
    torch.testing.assert_close(
        out_weights, torch.cat([weights, torch.ones(37, 1, device="cuda")], 1)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("shared_stream", [False, True])
def test_mega_moe_routing_matches_reference(
    monkeypatch, hash_routing, num_tokens, shared_stream
):
    num_experts, top_k, vocab = 16, 4, 64
    scaling = 2.5
    device = "cuda"
    torch.manual_seed(0)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(num_experts, HIDDEN, device=device) * 0.1
            self.e_score_correction_bias = (
                None if hash_routing else torch.randn(num_experts, device=device)
            )
            self.tid2eid = (
                torch.stack(
                    [
                        torch.randperm(num_experts, device=device)[:top_k]
                        for _ in range(vocab)
                    ]
                ).to(torch.int32)
                if hash_routing
                else None
            )
            self.bias_vl = None

        def forward(self, x):
            return x.float() @ self.weight.t(), None

    captured = {}

    def experts(x, topk_weights, topk_ids):
        captured["weights"], captured["ids"] = topk_weights, topk_ids
        return 2 * x

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = Gate()
    moe.experts = experts
    moe.shared_experts = lambda x: 3 * x
    moe.shared_experts_stream = aux_stream() if shared_stream else None
    moe.shared_experts_events = (
        (torch.cuda.Event(), torch.cuda.Event()) if shared_stream else None
    )
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = top_k
    moe.renormalize = True
    moe.routed_scaling_factor = scaling
    moe.image_sentinel_lo = 0

    x = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab, (num_tokens,), device=device)
    out = moe._forward_mega_moe(x, input_ids)

    ids, weights = captured["ids"], captured["weights"]
    assert ids.dtype == torch.int32 and ids.is_contiguous()
    assert weights.dtype == torch.float32 and weights.is_contiguous()

    scores = torch.nn.functional.softplus(x.float() @ moe.gate.weight.t()).sqrt()
    if hash_routing:
        expected_ids = moe.gate.tid2eid[input_ids].long()
    else:
        expected_ids = (scores + moe.gate.e_score_correction_bias).topk(top_k).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= scaling / expected_weights.sum(dim=-1, keepdim=True)

    order = ids.long().sort(dim=-1).indices
    expected_order = expected_ids.sort(dim=-1).indices
    torch.testing.assert_close(
        ids.long().gather(1, order), expected_ids.gather(1, expected_order)
    )
    torch.testing.assert_close(
        weights.gather(1, order),
        expected_weights.gather(1, expected_order),
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(out, 5 * x)


def _make_router_gate(hidden=5120, num_experts=384, top_k=6):
    """GateLinear as DeepseekV4MoE builds it, loaded with random weights."""
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = GateLinear(
        hidden,
        num_experts,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        gate.weight.normal_(std=0.02)
    gate.quant_method.process_weights_after_loading(gate)
    gate.e_score_correction_bias = torch.nn.Parameter(
        torch.randn(num_experts, device="cuda") * 0.1, requires_grad=False
    )
    gate.tid2eid = None
    gate.bias_vl = None
    return gate


def _make_routing_moe(gate, top_k=6):
    """The router half of a DeepSeek-V4.1 MegaMoE layer (the deepseek_v41
    subclass: no hash layers), set up as its constructor does. `__new__`
    leaves out the MegaMoE experts, which need an EP + MoRI config and every
    expert's weights; a test passes its own `experts` to capture the routing.
    """
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE as DeepseekV41MoE

    moe = DeepseekV41MoE.__new__(DeepseekV41MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = gate
    moe.use_mega_moe = True
    moe.text_only = True
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = top_k
    moe.renormalize = True
    moe.routed_scaling_factor = 1.5
    moe.image_sentinel_lo = 0
    return moe


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize(
    "env",
    [
        {"VLLM_ROCM_USE_AITER": "1"},
        {"VLLM_ROCM_USE_AITER": "0", "VLLM_ROCM_USE_AITER_FLYDSL_ROUTER": "1"},
        {
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_FLYDSL_ROUTER": "1",
            "VLLM_BATCH_INVARIANT": "1",
        },
    ],
)
def test_flydsl_router_stays_off(monkeypatch, env):
    """Off by default, with AITER off, and under batch invariance (its k-split
    depends on the row count)."""
    from vllm._aiter_ops import rocm_aiter_ops

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    rocm_aiter_ops.refresh_env_variables()
    try:
        assert not rocm_aiter_ops.is_flydsl_router_enabled()
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


@pytest.fixture
def flydsl_router_env(monkeypatch, dist_init):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_ROUTER", "1")
    rocm_aiter_ops.refresh_env_variables()
    if not rocm_aiter_ops.is_flydsl_router_enabled():
        pytest.skip("needs gfx950 and an AITER with the FlyDSL MoE router")
    torch.manual_seed(0)
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize(
    "case, served",
    [
        ("language_model_only", True),
        ("with_images", False),
        ("not_mega_moe", False),
        ("hash_routing", False),
        ("no_bias", False),
        ("top_3", False),
    ],
)
def test_flydsl_router_layer_eligibility(flydsl_router_env, case, served):
    """DeepSeek-V4.1's layers (sqrtsoftplus, biased, `bias_vl`, no hash table)
    get the FlyDSL gate only where no image input can reach the router; other
    MoE layers keep the GEMM + top-k router."""
    gate = _make_router_gate()
    gate.bias_vl = torch.nn.Parameter(
        torch.zeros(384, device="cuda"), requires_grad=False
    )
    moe = _make_routing_moe(gate, top_k=3 if case == "top_3" else 6)
    moe.image_sentinel_lo = 129264
    if case == "with_images":
        moe.text_only = False
    elif case == "not_mega_moe":
        moe.use_mega_moe = False
    elif case == "hash_routing":
        gate.tid2eid = torch.nn.Parameter(
            torch.zeros(16, 6, dtype=torch.int32, device="cuda"),
            requires_grad=False,
        )
    elif case == "no_bias":
        gate.e_score_correction_bias = None
    moe.process_weights_after_loading()
    if served:
        assert moe.flydsl_router_gate.shape == gate.weight.shape
        assert not torch.equal(moe.flydsl_router_gate, gate.weight)
    else:
        assert moe.flydsl_router_gate is None


@pytest.fixture
def router_layers(flydsl_router_env):
    """A FlyDSL-routed layer and a stock layer (hook not run, as with the flag
    off) on the same DeepSeek-V4.1-Flash gate: hidden 5120, 384 experts."""
    gate = _make_router_gate()
    flydsl, stock = _make_routing_moe(gate), _make_routing_moe(gate)
    flydsl.process_weights_after_loading()
    assert flydsl.flydsl_router_gate is not None
    assert stock.flydsl_router_gate is None
    return flydsl, stock


def _route(moe, x):
    captured = {}

    def experts(x, topk_weights, topk_ids):
        captured["weights"], captured["ids"] = topk_weights, topk_ids
        return x

    moe.experts = experts
    moe._forward_mega_moe_routed(x, None)
    return captured["weights"], captured["ids"]


def _flydsl_topk(moe, x):
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.flydsl_router_topk(
        x,
        moe.flydsl_router_gate,
        moe.gate.e_score_correction_bias,
        moe.routed_scaling_factor,
        get_moe_padding_mask(x.shape[0]),
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("num_tokens", [1, 37, 768, 769])
def test_flydsl_router_matches_gemm_topk_router(router_layers, num_tokens):
    """The FlyDSL router serves up to 768 rows with the routing of GateLinear +
    `fused_topk_bias`, padding rows included; larger batches fall back."""
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context

    flydsl, stock = router_layers
    x = torch.randn(num_tokens, 5120, device="cuda", dtype=torch.bfloat16)
    is_padding = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    is_padding[3::4] = True
    with set_forward_context(None, VllmConfig(), is_padding=is_padding):
        ref_weights, ref_ids = _route(stock, x)
        weights, ids = _route(flydsl, x)
        direct = _flydsl_topk(flydsl, x)

    # Which path ran: the FlyDSL op's output, or GateLinear + top-k's.
    if num_tokens > 768:
        assert direct is None
        assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)
        return
    assert torch.equal(ids, direct[1]) and torch.equal(weights, direct[0])
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert torch.all(ids[is_padding] == -1) and torch.all(weights[is_padding] == 0)
    # Logits differ in the last fp32 bits (k-split order): skip rows whose 6th
    # and 7th candidates (score + bias) are within 1e-4.
    logits, _ = stock.gate(x)
    biased = torch.nn.functional.softplus(logits).sqrt()
    biased += stock.gate.e_score_correction_bias
    top7 = biased.topk(7, dim=-1).values
    rows = ~is_padding & (top7[:, 5] - top7[:, 6] > 1e-4)
    assert rows.sum() >= rows.numel() // 2
    order, ref_order = ids[rows].sort(-1).indices, ref_ids[rows].sort(-1).indices
    torch.testing.assert_close(
        ids[rows].gather(1, order), ref_ids[rows].gather(1, ref_order)
    )
    torch.testing.assert_close(
        weights[rows].gather(1, order),
        ref_weights[rows].gather(1, ref_order),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_flydsl_router_declined_layout_falls_back(router_layers):
    """Hidden states the kernel rejects (not 16 B aligned) are routed by
    GateLinear + `fused_topk_bias`, exactly as on a stock layer."""
    flydsl, stock = router_layers
    x = torch.randn(37, 5121, device="cuda", dtype=torch.bfloat16)[:, 1:]
    assert _flydsl_topk(flydsl, x) is None
    weights, ids = _route(flydsl, x)
    ref_weights, ref_ids = _route(stock, x)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_flydsl_router_graph_replay(router_layers):
    """Captured once, the FlyDSL router replays on new inputs as it runs
    eagerly (bitwise; the kernel has a fixed summation order)."""
    flydsl, _ = router_layers
    x = torch.randn(64, 5120, device="cuda", dtype=torch.bfloat16)
    _route(flydsl, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = _route(flydsl, x)
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        graph.replay()
        weights, ids = _flydsl_topk(flydsl, x)
        assert torch.equal(graph_ids, ids) and torch.equal(graph_weights, weights)
