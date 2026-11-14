# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper MoE kernels (fused router, routed plus shared experts) against
vLLM's router, the stock Triton FP8 MoE and an FP64 reference, at the TP4 rank
shapes of GLM-5.3-Flash."""

import pytest
import torch

from vllm.platforms import current_platform

EXPERTS, HIDDEN, INTER, TOP_K, LIMIT, SCALE = 288, 4096, 512, 8, 10.0, 2.5
FP8 = torch.float8_e4m3fn
# Token counts: plain decode, MTP5 verify steps of 1-4 requests, the row cap.
TOKENS = (1, 6, 12, 18, 24, 32)

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires SM90"
)

# One verify step's routing (24 tokens) captured from GLM-5.3-Flash on TP4.
REAL_ROUTES = [
    [81, 163, 123, 55, 34, 140, 199, 119], [92, 176, 56, 278, 192, 208, 38, 212],
    [10, 15, 271, 5, 224, 85, 247, 273], [79, 162, 119, 213, 36, 269, 198, 44],
    [173, 25, 83, 271, 47, 169, 18, 177], [118, 211, 170, 199, 25, 9, 189, 32],
    [267, 13, 83, 96, 230, 164, 246, 78], [48, 205, 201, 10, 128, 148, 210, 237],
    [127, 10, 269, 226, 171, 212, 112, 71], [267, 264, 43, 265, 276, 209, 25, 163],
    [91, 114, 92, 238, 59, 89, 58, 227], [39, 249, 188, 163, 165, 266, 187, 13],
    [74, 17, 83, 138, 85, 110, 111, 140], [267, 13, 140, 164, 96, 85, 78, 233],
    [75, 268, 140, 111, 240, 170, 78, 143], [205, 58, 199, 79, 140, 151, 284, 229],
    [77, 99, 60, 140, 129, 2, 171, 28], [146, 178, 70, 258, 275, 140, 30, 234],
    [83, 20, 271, 222, 179, 14, 225, 101], [254, 191, 31, 140, 212, 20, 97, 111],
    [223, 17, 78, 247, 138, 85, 126, 245], [133, 212, 47, 266, 143, 160, 249, 209],
    [77, 60, 99, 140, 259, 85, 78, 2], [223, 140, 258, 247, 245, 138, 146, 228],
]  # fmt: skip


def _weights(seed: int):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.moe import MoeWeights

    g = torch.Generator(device="cuda").manual_seed(seed)

    def fp8(*shape):
        return (torch.randn(*shape, generator=g, device="cuda") * 8.0).to(FP8)

    def scales(*shape):
        return (0.5 + torch.rand(*shape, generator=g, device="cuda")) * 2e-2

    return MoeWeights(
        gate=(torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * 0.02).to(
            torch.bfloat16
        ),
        bias=torch.randn(EXPERTS, generator=g, device="cuda") * 0.05,
        w13=fp8(EXPERTS, 2 * INTER, HIDDEN),  # [gate; up]
        w13_scale=scales(EXPERTS, 2 * INTER // 128, HIDDEN // 128),
        w2=fp8(EXPERTS, HIDDEN, INTER),
        w2_scale=scales(EXPERTS, HIDDEN // 128, INTER // 128),
        shared13=fp8(2 * INTER, HIDDEN),
        shared13_scale=scales(2 * INTER // 128, HIDDEN // 128),
        shared2=fp8(HIDDEN, INTER),
        shared2_scale=scales(HIDDEN // 128, INTER // 128),
    )


def _inputs(w, m: int, routing: str, seed: int):
    """BF16 tokens and the weights that route them: "uniform" (random gate),
    "skewed" (a bias that sends every token to 16 experts) or "real" (a gate
    with an identity block that selects `REAL_ROUTES` exactly)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, generator=g, device="cuda") * 0.5
    if routing == "skewed":
        bias = w.bias.clone()
        bias[100:116] += 5.0
        w = w._replace(bias=bias)
    elif routing == "real":
        routes = (REAL_ROUTES * 2)[:m]
        gate = torch.zeros(EXPERTS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        gate[:, :EXPERTS] = torch.eye(EXPERTS, dtype=torch.bfloat16, device="cuda")
        u = torch.rand(m, EXPERTS, generator=g, device="cuda")
        chosen = torch.zeros(m, EXPERTS, dtype=torch.bool, device="cuda")
        chosen.scatter_(1, torch.tensor(routes, device="cuda"), True)
        # Logit margin >= 1.5 between chosen and other experts.
        x[:, :EXPERTS] = torch.where(chosen, 1.0 + 0.5 * u, -1.0 + 0.5 * u)
        w = w._replace(gate=gate, bias=torch.zeros_like(w.bias))
    return w, x.to(torch.bfloat16)


def _stock_router(x, w):
    """The stock router's kernels: the gate GEMM GateLinear picks, then
    grouped top-k (one group, sigmoid, renormalized, scaled)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    if x.shape[0] <= 16:
        logits = ll_bf16_gemm(x, w.gate)
    else:
        logits = torch.mm(x, w.gate.T, out_dtype=torch.float32)
    weights, ids = ops.grouped_topk(logits, 1, 1, TOP_K, True, SCALE, w.bias, 1, None)
    return ids, weights


def _quantize(x, block: int, ue8m0: bool):
    """FP8 per `block` along the last dim, dequantized (the stock scale rule)."""
    blocks = x.double().unflatten(-1, (-1, block))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax.float() * (1.0 / 448.0)).double()
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return ((blocks / scale).float().to(FP8).double() * scale).flatten(-2)


def _dequant(w, s):
    return w.double() * s.double().repeat_interleave(128, -2).repeat_interleave(128, -1)


def _expert(w13, s13, w2, s2, x):
    """FP64 expert with exact dots, the clamped SwiGLU, no hidden quant."""
    gate_up = x @ _dequant(w13, s13).T
    gate = gate_up[:, :INTER].clamp(max=LIMIT)
    up = gate_up[:, INTER:].clamp(-LIMIT, LIMIT)
    return (gate * torch.sigmoid(gate) * up) @ _dequant(w2, s2).T


def _reference(x, w, ids, weights, ue8m0=False, shared=True, quantize=True):
    """FP64 MoE output for given routing; `quantize`: the input FP8 per 128 with
    HopperMoe's scale rule (with `ue8m0`, the stock rule)."""
    xq = _quantize(x, 128, ue8m0) if quantize else x.double()
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64, device=x.device)
    for e in ids.unique().tolist():
        rows, ks = (ids == e).nonzero(as_tuple=True)
        y = _expert(w.w13[e], w.w13_scale[e], w.w2[e], w.w2_scale[e], xq[rows])
        out[rows] += weights[rows, ks].double()[:, None] * y
    if shared:
        out += _expert(w.shared13, w.shared13_scale, w.shared2, w.shared2_scale, xq)
    return out


_stock: dict = {}


def _stock_experts(w, x, ids, weights):
    """The stock Triton FP8 routed experts (block 128) of the Hopper layer."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        make_fp8_moe_kernel,
        make_fp8_moe_quant_config,
        select_fp8_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    config = VllmConfig()
    with set_current_vllm_config(config):
        key = id(w.w13)
        if key not in _stock:
            if not _stock:
                init_workspace_manager(x.device)
            moe = FusedMoEConfig(
                num_experts=EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN,
                intermediate_size=INTER, num_local_experts=EXPERTS,
                num_logical_experts=EXPERTS, activation=MoEActivation.SILU,
                device=x.device, routing_method=RoutingMethodType.TopK,
                moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                in_dtype=torch.bfloat16, moe_backend="auto", swiglu_limit=LIMIT,
            )  # fmt: skip
            backend, cls = select_fp8_moe_backend(
                moe, kFp8Static128BlockSym, kFp8Dynamic128Sym
            )
            quant = make_fp8_moe_quant_config(
                backend, w.w13_scale, w.w2_scale, None, None, block_shape=[128, 128],
                per_act_token_quant=False, swiglu_limit=LIMIT,
            )  # fmt: skip
            _stock[key] = make_fp8_moe_kernel(quant, moe, cls, backend)
        return _stock[key].apply(
            x, w.w13, w.w2, topk_weights=weights, topk_ids=ids.long(),
            activation=MoEActivation.SILU, global_num_experts=EXPERTS,
            expert_map=None, apply_router_weight_on_input=False,
        )  # fmt: skip


def _rel(a, b):
    a, b = a.double(), b.double()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


@pytest.fixture(scope="module")
def weights():
    return _weights(0)


def _moe(w, ue8m0=False):
    """HopperMoe as `HopperMoe.of` builds it (exact activation scales) unless
    `ue8m0`."""
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.moe import HopperMoe

    return HopperMoe(w, SCALE, ue8m0)


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "real"])
def test_router_matches_stock(weights, m, routing):
    w, x = _inputs(weights, m, routing, seed=m)
    moe = _moe(w)
    moe(x)
    ids, tw = (t.clone() for t in moe.routing(m))
    stock_ids, stock_tw = _stock_router(x, w)
    # Same experts in the same order; weights within the sigmoid's last bits.
    assert torch.equal(ids.long(), stock_ids.long())
    torch.testing.assert_close(tw, stock_tw, rtol=2e-5, atol=1e-6)
    if routing == "real":
        expected = torch.tensor((REAL_ROUTES * 2)[:m], device="cuda")
        assert torch.equal(ids.long().sort(1).values, expected.sort(1).values)


def test_router_exact_ties(weights):
    """Experts 2k + 1 copy expert 2k's gate row and bias: the lower id wins."""
    gate = weights.gate.view(EXPERTS // 2, 2, HIDDEN).clone()
    gate[:, 1] = gate[:, 0]
    bias = weights.bias.view(EXPERTS // 2, 2).clone()
    bias[:, 1] = bias[:, 0]
    w = weights._replace(gate=gate.view(EXPERTS, HIDDEN), bias=bias.view(EXPERTS))
    for m in (6, 24):
        _, x = _inputs(w, m, "uniform", seed=100 + m)
        moe = _moe(w)
        moe(x)
        ids, _ = moe.routing(m)
        stock_ids, _ = _stock_router(x, w)
        assert torch.equal(ids.long(), stock_ids.long())


def _stock_shared(w, x):
    """The stock shared expert's numerics: input quant, block-FP8 GEMM to BF16,
    clamped SwiGLU in BF16, quant, block-FP8 GEMM."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        w8a8_triton_block_scaled_mm,
    )

    def linear(a, weight, scale):
        q, s = per_token_group_quant_fp8(a, 128, use_ue8m0=True)
        return w8a8_triton_block_scaled_mm(
            q, weight, s, scale, [128, 128], torch.bfloat16
        )

    gate_up = linear(x, w.shared13, w.shared13_scale)
    act = torch.empty(x.shape[0], INTER, dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(act, gate_up, LIMIT, 1.0, 0.0)
    return linear(act, w.shared2, w.shared2_scale)


# The FP64 reference has no hidden quantization: both paths quantize the hidden
# activations to FP8 (per 64 here, per 128 after a BF16 round in stock), which
# alone costs about 3% relative L2 on these weights. Against stock the
# reference does not quantize the input either, since the two paths quantize it
# with different scale rules; that adds about 2%.
MAX_ERROR = 0.04
MAX_ERROR_UNQUANTIZED = 0.06


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "real"])
def test_routed_experts_no_worse_than_stock(weights, m, routing):
    """With a zero shared expert: the routed sum against FP64 and the stock
    Triton FP8 experts on the same routing."""
    w, x = _inputs(weights, m, routing, seed=1000 + m)
    w = w._replace(shared13=torch.zeros_like(w.shared13))
    moe = _moe(w)
    out = moe(x).clone()
    ids, tw = (t.clone() for t in moe.routing(m))
    reference = _reference(x, w, ids.long(), tw, shared=False, quantize=False)
    error = _rel(out, reference)
    stock_error = _rel(_stock_experts(w, x, ids, tw), reference)
    assert torch.isfinite(out).all()
    assert error <= min(stock_error, MAX_ERROR_UNQUANTIZED), (error, stock_error)


@pytest.mark.parametrize("m", (1, 6, 24, 32))
def test_shared_expert_no_worse_than_stock(weights, m):
    """With zero routed down projections: the shared expert alone."""
    w, x = _inputs(weights, m, "uniform", seed=1500 + m)
    w = w._replace(w2=torch.zeros_like(w.w2))
    out = _moe(w)(x)
    reference = _expert(
        w.shared13, w.shared13_scale, w.shared2, w.shared2_scale, x.double()
    )
    error, stock_error = _rel(out, reference), _rel(_stock_shared(w, x), reference)
    assert error <= min(stock_error, MAX_ERROR_UNQUANTIZED), (error, stock_error)


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "real"])
def test_moe_against_reference(weights, m, routing):
    """The whole layer against FP64, deterministic and finite."""
    w, x = _inputs(weights, m, routing, seed=2500 + m)
    moe = _moe(w)
    out = moe(x).clone()
    ids, tw = (t.clone() for t in moe.routing(m))
    assert torch.isfinite(out).all()
    assert _rel(out, _reference(x, w, ids.long(), tw)) < MAX_ERROR
    assert torch.equal(out, moe(x))


def test_moe_with_ue8m0(weights):
    """Power-of-two activation scales (the stock rule under DeepGEMM E8M0)."""
    for m in (6, 24):
        _, x = _inputs(weights, m, "uniform", seed=2000 + m)
        moe = _moe(weights, ue8m0=True)
        out = moe(x)
        ids, tw = moe.routing(m)
        reference = _reference(x, weights, ids.long(), tw, ue8m0=True)
        assert _rel(out, reference) < MAX_ERROR


def _pack_ue8m0(scale, n: int, k: int):
    """DeepGEMM's packed UE8M0 layout of power-of-two FP32 `[n / 128, k / 128]`
    block scales: int32 `[n, k / 512]`, byte g % 4 of column g // 4 for K group g."""
    exponents = ((scale.view(torch.int32) >> 23) & 0xFF).long()
    assert k % 512 == 0
    rows = exponents.repeat_interleave(128, 0).view(n, k // 512, 4)
    shifts = 8 * torch.arange(4, device=scale.device)
    packed = (rows << shifts).sum(-1)
    return (packed - (packed >= 2**31).long() * 2**32).to(torch.int32)


def _stock_layer(w, backend=None):
    """A stand-in for a processed stock `Glm5NextMoE` with the attributes
    `HopperMoe.of` reads, shared expert scales packed as the stock load leaves
    them, and its text config."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    def linear(weight, scale):
        n, k = weight.shape
        return SimpleNamespace(weight=weight, weight_scale_inv=_pack_ue8m0(scale, n, k))

    routed = SimpleNamespace(
        w13_weight=w.w13, w13_weight_scale_inv=w.w13_scale,
        w2_weight=w.w2, w2_weight_scale_inv=w.w2_scale,
        quant_method=SimpleNamespace(fp8_backend=backend or Fp8MoeBackend.TRITON),
    )  # fmt: skip
    bias = torch.nn.Parameter(w.bias, requires_grad=False)
    moe = SimpleNamespace(
        gate=SimpleNamespace(
            weight=w.gate, e_score_correction_bias=bias, out_dtype=torch.float32
        ),
        experts=SimpleNamespace(routed_experts=routed),
        shared_experts=SimpleNamespace(
            gate_up_proj=linear(w.shared13, w.shared13_scale),
            down_proj=linear(w.shared2, w.shared2_scale),
        ),
        routed_scaling_factor=SCALE,
    )
    config = SimpleNamespace(
        n_routed_experts=EXPERTS, num_experts_per_tok=TOP_K, hidden_size=HIDDEN,
        scoring_func="sigmoid", topk_method="noaux_tc", n_group=1, topk_group=1,
        norm_topk_prob=True, swiglu_limit=LIMIT, hidden_act="silu",
    )  # fmt: skip
    return moe, config


def _pow2_shared(w):
    def pow2(s):
        return torch.exp2(torch.round(torch.log2(s)))

    return w._replace(
        shared13_scale=pow2(w.shared13_scale), shared2_scale=pow2(w.shared2_scale)
    )


def test_of_matches_direct(weights):
    """`HopperMoe.of` on a stock layer (packed UE8M0 shared expert scales) runs
    like `HopperMoe` built from the same tensors with FP32 scales."""
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.moe import (
        HopperMoe,
        _fp32_scales,
    )

    w = _pow2_shared(weights)
    moe, config = _stock_layer(w)
    shared13 = moe.shared_experts.gate_up_proj
    assert torch.equal(
        _fp32_scales(shared13.weight_scale_inv, shared13.weight), w.shared13_scale
    )
    built = HopperMoe.of(moe, config)
    assert built is not None
    direct = _moe(w)
    for m in (6, 24):
        _, x = _inputs(w, m, "uniform", seed=6000 + m)
        assert torch.equal(built(x).clone(), direct(x))


def test_unsupported_layouts(weights):
    """`HopperMoe.of` refuses what the kernels cannot read, with a reason."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.moe import (
        HopperMoe,
        unsupported,
    )

    w = _pow2_shared(weights)
    moe, config = _stock_layer(w)
    assert unsupported(moe, config) == []
    # FlashInfer reorders w13 to [up; gate] in place of the same shapes.
    moe, config = _stock_layer(w, Fp8MoeBackend.FLASHINFER_CUTLASS)
    assert unsupported(moe, config) == [
        "FP8 MoE backend FLASHINFER_CUTLASS (expects TRITON)"
    ]
    assert HopperMoe.of(moe, config) is None
    moe, config = _stock_layer(w._replace(w13=w.w13[: EXPERTS // 2]))
    (reason,) = unsupported(moe, config)
    assert reason.startswith("w13_weight")
    moe, config = _stock_layer(w)
    config.n_routed_experts = 256
    assert unsupported(moe, config) == ["n_routed_experts=256 (expects 288)"]


def test_moe_extreme_inputs(weights):
    moe = _moe(weights)
    # Zero tokens: zero FP8 input, every expert output is exactly 0.
    zero = torch.zeros(6, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(moe(zero), torch.zeros_like(zero))
    # One large token among small ones: per-token scales keep the small ones
    # exact to the usual error. The large one saturates the SwiGLU clamp, so all
    # its hidden values sit near 100 and round to the same FP8 value (a tie at
    # 400 x 0.25, down to 384, in stock as well): its error does not average
    # out and is bounded by half an FP8 step (1 / 24) plus the input's error.
    _, x = _inputs(weights, 12, "uniform", seed=3000)
    x[3] *= 1000
    out = moe(x)
    ids, tw = moe.routing(12)
    reference = _reference(x, weights, ids.long(), tw)
    assert torch.isfinite(out).all()
    for t in (0, 11):
        assert _rel(out[t], reference[t]) < MAX_ERROR
    assert _rel(out[3], reference[3]) < 1 / 24 + 0.02


def test_moe_cuda_graph():
    """Two layers sharing the workspace in one graph: each replay equals eager."""
    first, second = _moe(_weights(1)), _moe(_weights(2))
    for m in (6, 24):
        xa = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        xb = torch.zeros_like(xa)
        first.prepare([m])
        second.prepare([m])
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            ya = first(xa)
            yb = second(xb)
        for seed in range(3):
            g = torch.Generator(device="cuda").manual_seed(4000 + seed)
            x = torch.randn(m, HIDDEN, generator=g, device="cuda") * 0.5
            x = x.to(torch.bfloat16)
            xa.copy_(x)
            xb.copy_(x.flip(0))
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(ya, first(xa))
            assert torch.equal(yb, second(xb))


def _late_writer(words: int, delay_ns: int):
    """A kernel that releases its programmatic dependents at entry and copies
    `words` 16-byte words src -> dst only after `delay_ns`, as the backbone
    boundary kernel releases the MoE router before it writes the MoE input."""
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import T, dsl_user_op

    from vllm.models.glm5next.nvidia.specialized_decode.hopper import _moe_ptx as P

    @dsl_user_op
    def now(*, loc=None, ip=None):
        return Int64(
            P._asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip, True)
        )

    class LateWriter:
        @cute.jit
        def __call__(self, src: Int64, dst: Int64, stream):
            self.kernel(src, dst).launch(
                grid=[4, 1, 1], block=[128, 1, 1], stream=stream, use_pdl=False
            )

        @cute.kernel
        def kernel(self, src: Int64, dst: Int64):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            P.griddep_launch()
            start = now()
            while now() - start < Int64(delay_ns):
                P.nanosleep(200)
            i = bid * 128 + tid
            while i < Int32(words):
                v = P.ldg_cg_f32x4(src + Int64(i) * 16)
                P.stg_u32x4(dst + Int64(i) * 16, *(P.f32_bits(e) for e in v))
                i = i + 4 * 128

    return cute.compile(LateWriter(), Int64(0), Int64(0), make_fake_stream())


@pytest.mark.parametrize("m", (1, 6, 24))
def test_moe_after_early_release(weights, m):
    """The router as the programmatic dependent of a kernel that releases it
    before writing the MoE input: eager and graph replays equal the layer on
    the final input (the input is read only after `griddepcontrol.wait`, and
    coherently)."""
    from cutlass import Int64

    from vllm.models.glm5next.nvidia.specialized_decode.cutedsl.host import stream

    moe = _moe(weights)
    moe.prepare([m])
    g = torch.Generator(device="cuda").manual_seed(5000 + m)
    old = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 0.5).bfloat16()
    new = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 0.5).bfloat16()
    x = new.clone()
    want = moe(x).clone()
    writer = _late_writer(m * HIDDEN * 2 // 16, 50_000)

    def step():
        x.copy_(old)
        moe(x)  # an earlier layer reads the same buffer
        writer(Int64(new.data_ptr()), Int64(x.data_ptr()), stream(x.device))
        return moe(x)

    for _ in range(5):
        out = step()
        torch.accelerator.synchronize()
        assert torch.equal(out, want)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=side):
        out = step()
    for _ in range(5):
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, want)
