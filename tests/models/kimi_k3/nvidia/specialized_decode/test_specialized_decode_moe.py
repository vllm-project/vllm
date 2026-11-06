# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode (decode_backend=latency_specialized): the MoE routing
and SiTU arithmetic bit for bit against the stock ops (FlashInfer trtllm-gen routing,
vLLM's SituAndMul as the deployment dispatches it: the _C situ_and_mul kernel)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

EXPERTS, TOPK, SHARED = 896, 16, 768  # TP8 shared expert width
BETA, LINEAR_BETA = 4.0, 25.0


def stock_routing(logits, bias, scale):
    """FlashInfer trtllm-gen DeepSeekV3 routing (one group), as the stock MoE runs
    it: (ids, BF16 weights) per token, sorted by expert id."""
    from flashinfer.fused_moe import trtllm_gen_routing
    from flashinfer.tllm_enums import RoutingMethodType

    out = trtllm_gen_routing(
        logits,
        bias,
        RoutingMethodType.DeepSeekV3,
        TOPK,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
    )
    return by_expert(out.topk_ids[:, :TOPK], out.topk_weights[:, :TOPK])


def by_expert(ids, weights):
    order = ids.long().argsort(dim=-1)
    return ids.long().gather(-1, order), weights.gather(-1, order).view(torch.int16)


def route_logits(batch, kind, gen):
    """Router logits and bias. ``near_tie``: 48 experts per token whose biased
    sigmoids straddle the 16th place within a few ULPs (exact ties included)."""
    logits = torch.randn(batch, EXPERTS, generator=gen, device="cuda") * 2
    bias = torch.randn(EXPERTS, generator=gen, device="cuda") * 0.1
    if kind == "near_tie":
        bias[:] = 0.05
        for t in range(batch):
            chosen = torch.randperm(EXPERTS, generator=gen, device="cuda")[:48]
            ulps = torch.randint(-4, 5, (48,), generator=gen, device="cuda")
            base = torch.tensor(3.0, device="cuda").view(torch.int32)
            logits[t, chosen] = (base + ulps.int()).view(torch.float32)
    # -0.0 is the mailbox's unpublished sentinel; the router GEMM never stores it.
    return torch.where(logits == 0, torch.zeros_like(logits), logits), bias


@pytest.mark.parametrize("batch", [2, 3, 8, 9, 16, 33, 64])
@pytest.mark.parametrize("kind", ["random", "near_tie"])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_route_matches_flashinfer_routing(batch, kind, scale):
    """RouteSitu's route role (B 2-64; the polled logits the front GEMM publishes)
    picks the stock experts with bit-identical BF16 weights."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import (
        WIDTH,
        compile_route,
    )

    gen = torch.Generator(device="cuda").manual_seed(batch * 7 + len(kind))
    device = torch.accelerator.current_device_index()
    route = compile_route(batch, SHARED, True, (True, False, False), device, True)
    bf16, f32 = torch.bfloat16, torch.float32
    mailbox = torch.zeros(1, 64, WIDTH, dtype=bf16, device="cuda")
    ids = torch.full((batch, TOPK), -1, dtype=torch.int32, device="cuda")
    weights = torch.zeros(batch, TOPK, dtype=bf16, device="cuda")
    unused = (
        torch.empty(batch, EXPERTS, dtype=f32, device="cuda"),
        torch.empty(batch, 3584, dtype=bf16, device="cuda"),
        torch.empty(batch, 2 * SHARED, dtype=f32, device="cuda"),
        torch.empty(batch, SHARED, dtype=bf16, device="cuda"),
    )
    for trial in range(4):
        logits, bias = route_logits(batch, kind, gen)
        mailbox[0, :batch, 3584:].view(f32).copy_(logits)
        args = (to_cute(unused[0]), to_cute(bias), to_cute(ids), to_cute(weights))
        args += (to_cute(mailbox), *map(to_cute, unused[1:]))
        scalars = map(cutlass.Float32, (scale, BETA, LINEAR_BETA))
        route(*args, *scalars, stream(torch.device("cuda", device)))
        got = by_expert(ids, weights)
        want = stock_routing(logits, bias, scale)
        assert torch.equal(got[0], want[0]), f"trial {trial}: expert ids differ"
        assert torch.equal(got[1], want[1]), f"trial {trial}: weights differ"


@pytest.mark.parametrize("batch", [5, 8])
def test_route_early_release_is_exact(batch):
    """TP8 B5-8 routing releases the persistent experts at entry
    (``front.EARLY_ROUTE``): only the PDL trigger moves, so ids and BF16 weights equal
    the late-release kernel's and the stock routing's."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import (
        EARLY_ROUTE,
        WIDTH,
        compile_route,
    )

    assert batch_table(EARLY_ROUTE[8]).get(batch)
    gen = torch.Generator(device="cuda").manual_seed(batch)
    device = torch.accelerator.current_device_index()
    roles = (True, False, False)
    codes = [
        compile_route(batch, SHARED, True, roles, device, True, False, early)
        for early in (True, False)
    ]
    bf16, f32 = torch.bfloat16, torch.float32
    mailbox = torch.zeros(1, 64, WIDTH, dtype=bf16, device="cuda")
    unused = (
        torch.empty(batch, EXPERTS, dtype=f32, device="cuda"),
        torch.empty(batch, 3584, dtype=bf16, device="cuda"),
        torch.empty(batch, 2 * SHARED, dtype=f32, device="cuda"),
        torch.empty(batch, SHARED, dtype=bf16, device="cuda"),
    )
    for kind in ("random", "near_tie"):
        logits, bias = route_logits(batch, kind, gen)
        got = []
        for code in codes:  # each call re-arms the polled cells: publish again
            mailbox[0, :batch, 3584:].view(f32).copy_(logits)
            ids = torch.full((batch, TOPK), -1, dtype=torch.int32, device="cuda")
            weights = torch.zeros(batch, TOPK, dtype=bf16, device="cuda")
            args = (to_cute(unused[0]), to_cute(bias), to_cute(ids), to_cute(weights))
            args += (to_cute(mailbox), *map(to_cute, unused[1:]))
            scalars = map(cutlass.Float32, (2.5, BETA, LINEAR_BETA))
            code(*args, *scalars, stream(torch.device("cuda", device)))
            got.append((ids.clone(), weights.view(torch.int16).clone()))
        assert torch.equal(got[0][0], got[1][0]) and torch.equal(got[0][1], got[1][1])
        want = stock_routing(logits, bias, 2.5)
        assert torch.equal(by_expert(got[0][0], got[0][1].view(bf16))[0], want[0])


def bf16_patterns():
    return (
        torch.arange(65536, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )


def situ_grid(ups):
    """[len(ups), 2 * 65536] gate/up rows: every BF16 gate against each up."""
    gates = bf16_patterns().unsqueeze(0).expand(ups.numel(), -1)
    return torch.cat([gates, ups.view(-1, 1).expand(-1, 65536)], dim=1).contiguous()


def sampled_ups(count, gen):
    patterns = bf16_patterns()
    finite = patterns[torch.isfinite(patterns.float())]
    picks = finite[
        torch.randint(0, finite.numel(), (count,), generator=gen, device="cuda")
    ]
    special = [0.0, -0.0, 1.0, -1.0, 25.0, -100.0, 1e-30, float("inf"), float("-inf")]
    special = torch.tensor(special, dtype=torch.bfloat16, device="cuda")
    return torch.cat([picks, special])


# (beta, linear_beta): Kimi K3's, up unclipped (linear_beta unset), and
# non-power-of-two reciprocals.
SITU_PARAMS = [(BETA, LINEAR_BETA), (BETA, None), (2.5, 7.0), (3.0, None)]


def stock_situ(x, beta=BETA, linear_beta=LINEAR_BETA):
    """The ``_C`` situ_and_mul kernel, which SituAndMul runs in the deployment (the
    custom op enabled: custom_ops "all" under compilation mode NONE)."""
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=x.dtype, device=x.device)
    lb = -1.0 if linear_beta is None else linear_beta
    torch.ops._C.situ_and_mul(out, x, beta, lb)
    return out


def test_stock_situ_dispatch_is_c_kernel():
    """Under the production config (compilation mode NONE, as the breakable CUDA graph
    forces) SituAndMul resolves custom_ops "all" and runs the _C kernel."""
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.activation import SituAndMul

    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    assert "all" in config.compilation_config.custom_ops
    with set_current_vllm_config(config):
        op = SituAndMul(BETA, LINEAR_BETA)
        assert op._forward_method == op.forward_cuda
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = situ_grid(sampled_ups(16, gen))
        assert_same_bits(op(x), stock_situ(x))


def assert_same_bits(got, want):
    nan = torch.isnan(want.float()) & torch.isnan(got.float())
    bad = (got.view(torch.int16) != want.view(torch.int16)) & ~nan
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} SiTU outputs differ"


@pytest.mark.parametrize("params", SITU_PARAMS, ids=lambda p: f"b{p[0]}-lb{p[1]}")
@pytest.mark.parametrize("site", ["dense", "route_situ", "tables"])
def test_situ_matches_stock(site, params):
    """Every BF16 gate against sampled ups: the dense layer's activation, the
    shared expert's SiTU in RouteSitu (B 2-8) and the TP8 B1 lookup tables equal
    the _C situ_and_mul kernel bit for bit."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        architecture,
        dynamic_view,
        stream,
        to_cute,
    )

    gen = torch.Generator(device="cuda").manual_seed(1)
    ups = sampled_ups(128 if site == "route_situ" else 1024, gen)
    x = situ_grid(ups)
    beta, linear_beta = params
    want = stock_situ(x, beta, linear_beta)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    lb = -1.0 if linear_beta is None else linear_beta
    scalars = (cutlass.Float32(beta), cutlass.Float32(lb))
    if site == "dense":
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.dense import (
            _compile_activation,
        )

        got = torch.empty_like(want)
        code = _compile_activation(65536, 65536, architecture(dev))
        code(dynamic_view(x), dynamic_view(got), *scalars, stream(dev))
    elif site == "tables":
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.small_batch.chain import (  # noqa: E501
            _launch,
        )

        table = torch.empty(131072, dtype=torch.float32, device=dev)
        code, args = _launch("tables", (table,))
        code(*args, *scalars, stream(dev))
        gates = x[:, :65536].view(torch.int16).long() & 0xFFFF
        up_rows = x[:, 65536:].view(torch.int16).long() & 0xFFFF
        got = (table[gates] * table[65536 + up_rows]).to(torch.bfloat16)
    else:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import (
            WIDTH,
            compile_route,
        )

        batch, f32, bf16 = 8, torch.float32, torch.bfloat16
        situ = compile_route(batch, SHARED, True, (False, False, True), dev.index)
        pairs = x.view(-1, 2, 65536).transpose(1, 2).reshape(-1, 2)  # (gate, up)
        chunk = batch * SHARED
        count = pairs.shape[0]
        pairs = torch.cat([pairs, pairs[: -count % chunk]])
        gate_up = torch.empty(batch, 2 * SHARED, dtype=f32, device=dev)
        hidden = torch.empty(batch, SHARED, dtype=bf16, device=dev)
        unused = (
            torch.empty(batch, EXPERTS, dtype=f32, device=dev),
            torch.empty(EXPERTS, dtype=f32, device=dev),
            torch.empty(batch, TOPK, dtype=torch.int32, device=dev),
            torch.empty(batch, TOPK, dtype=bf16, device=dev),
            torch.empty(1, 64, WIDTH, dtype=bf16, device=dev),
            torch.empty(batch, 3584, dtype=bf16, device=dev),
        )
        args = (*map(to_cute, unused), to_cute(gate_up), to_cute(hidden))
        out = []
        for start in range(0, pairs.shape[0], chunk):
            part = pairs[start : start + chunk].float().view(batch, SHARED, 2)
            gate_up[:, :SHARED], gate_up[:, SHARED:] = part[..., 0], part[..., 1]
            situ(*args, cutlass.Float32(1.0), *scalars, stream(dev))
            out.append(hidden.flatten().clone())
        got = torch.cat(out)[:count].view(-1, 65536)
    assert_same_bits(got, want)


HIDDEN = 3584  # routed latent width
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4(w):
    """(packed E2M1, UE8M0 bytes) of BF16 ``w``, asserting the values are exact."""
    from flashinfer import fp4_quantize

    one = torch.tensor(1.0, device="cuda")
    q, s = fp4_quantize(w, one, 32, sf_use_ue8m0=True, is_sf_swizzled_layout=False)
    q, s = q.reshape(*w.shape[:-1], -1), s.view(torch.uint8).reshape(*w.shape[:-1], -1)
    table = torch.tensor(E2M1, device="cuda")
    table = torch.cat([table, -table])
    codes = torch.stack([q & 15, q >> 4], -1).flatten(-2).long()
    scales = torch.exp2(s.float() - 127).repeat_interleave(32, -1)
    assert torch.equal(table[codes] * scales, w.float())
    return q, s


_WEIGHTS: dict = {}


def selection_experts(intermediate, real, paired=False):
    """Stock trtllm-gen weights of 896 identical experts whose first ``real`` FC1
    gate/up rows each tap two latent elements (the rest are the TP padding: zero) and
    whose FC2 row o takes hidden element o % real with weight 1. Random taps carry
    E2M1 x 2^k weights; ``paired``: gate row j sums latent 2j and 2j + 1, up row j
    latent 768 + 2j and 769 + 2j. With ``exact_latent`` / ``paired_latent`` inputs
    gate/up are exact in FP32 and the unweighted expert output is the
    MXFP8-dequantized SiTU hidden, exactly, in any accumulation order."""
    key = intermediate, paired
    if key in _WEIGHTS:
        return _WEIGHTS[key]
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_weight_to_mxfp4_moe_kernel_format,
    )

    gen = torch.Generator(device="cuda").manual_seed(intermediate)
    rows = 2 * intermediate
    blocks = torch.stack(
        [
            torch.randperm(HIDDEN // 32, generator=gen, device="cuda")[:2]
            for _ in range(rows)
        ]
    )
    taps = blocks * 32 + torch.randint(
        0, 32, blocks.shape, generator=gen, device="cuda"
    )
    values = torch.tensor(E2M1[1:], device="cuda")
    values = values[torch.randint(0, 7, taps.shape, generator=gen, device="cuda")]
    values *= torch.exp2(torch.randint(-2, 2, taps.shape, generator=gen, device="cuda"))
    values *= torch.randint(0, 2, taps.shape, generator=gen, device="cuda") * 2 - 1
    w13 = torch.zeros(rows, HIDDEN, device="cuda").scatter_(1, taps, values)
    if paired:
        j = torch.arange(intermediate, device="cuda")
        w13.zero_()
        w13[j, 2 * j] = w13[j, 2 * j + 1] = 1
        w13[intermediate + j, 768 + 2 * j] = w13[intermediate + j, 769 + 2 * j] = 1
    w13.view(2, intermediate, HIDDEN)[:, real:] = 0
    w2 = torch.zeros(HIDDEN, intermediate, device="cuda")
    w2[torch.arange(HIDDEN), torch.arange(HIDDEN) % real] = 1.0
    (q13, s13), (q2, s2) = mxfp4(w13.bfloat16()), mxfp4(w2.bfloat16())
    expand = lambda t: t.unsqueeze(0).expand(EXPERTS, -1, -1).contiguous()
    weights = convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        torch.nn.Module(),
        *map(expand, (q13, q2, s13, s2)),
        _cache_permute_indices={},
    )[:4]
    _WEIGHTS.clear()
    _WEIGHTS[key] = weights
    return weights


def exact_latent(batch, gen):
    """Latent rows of +-(1..7)/4 x 2^k, k in [-5, 1]: exact in MXFP8, so FC1 sums of
    two taps (at most 22 significant bits) are exact in FP32."""
    shape = (batch, HIDDEN)
    mantissa = torch.randint(1, 8, shape, generator=gen, device="cuda") / 4
    exponent = torch.randint(-5, 2, shape, generator=gen, device="cuda")
    sign = torch.randint(0, 2, shape, generator=gen, device="cuda") * 2 - 1
    return (sign * mantissa * torch.exp2(exponent)).bfloat16()


def paired_latent(gate, up):
    """Latent rows whose ``paired`` FC1 outputs are exactly the BF16 ``gate`` and
    ``up`` [B, I]: each value split into its top four significant bits and the rest,
    both exact in MXFP8 when a 32-element block spans few binades."""

    def split(v):
        high = (v.view(torch.int16) & ~0xF).view(torch.bfloat16)
        return high, (v.float() - high.float()).bfloat16()

    x = torch.zeros(gate.shape[0], HIDDEN, dtype=torch.bfloat16, device="cuda")
    (gh, gl), (uh, ul) = split(gate), split(up)
    x[:, 0:768:2], x[:, 1:768:2], x[:, 768:1536:2], x[:, 769:1536:2] = gh, gl, uh, ul
    return x


def stock_expert_rows(weights, x, intermediate):
    """trtllm_fp4_block_scale_moe (MXFP8 x MXFP4, SiTU) as the stock MoE runs it,
    unfinalized: one unweighted GEMM2 row per token (its 16 routes are equal)."""
    from flashinfer import mxfp8_quantize, trtllm_fp4_block_scale_moe
    from flashinfer.tllm_enums import RoutingMethodType

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    w13, w2, s13, s2 = weights
    tokens = x.shape[0]
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    xs = xs.view(torch.float8_e4m3fn).reshape(tokens, -1)
    full = lambda v: torch.full((EXPERTS,), v, device="cuda")
    gemm2, _, index = trtllm_fp4_block_scale_moe(
        routing_logits=torch.randn(tokens, EXPERTS, device="cuda"),
        routing_bias=torch.zeros(EXPERTS, device="cuda"),
        hidden_states=xq,
        hidden_states_scale=xs,
        gemm1_weights=w13,
        gemm1_weights_scale=s13,
        gemm1_bias=None,
        gemm1_alpha=full(BETA),
        gemm1_beta=full(LINEAR_BETA),
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=EXPERTS,
        top_k=TOPK,
        n_group=1,
        topk_group=1,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.DeepSeekV3.value,
        do_finalize=False,
        activation_type=activation_to_flashinfer_int(MoEActivation.SITU),
    )
    rows = gemm2[index.view(tokens, TOPK).long()]
    assert (rows == rows[:, :1]).all()
    return rows[:, 0]


@pytest.mark.parametrize(
    "tp,batch",
    [(8, 1), (8, 2), (8, 4), (8, 8), (8, 9), (8, 16), (8, 64)]
    + [(16, 1), (16, 2), (16, 4), (16, 16), (16, 64)]
    # The wide engine (B > 64): B640 gives experts ~11 tokens, so several units.
    + [(8, 80), (8, 200), (8, 640)],
)
def test_routed_experts_match_trtllm_gen(tp, batch):
    """The routed experts' FC1 epilogue (SiTU, MXFP8 E8M0 scale and E4M3 hidden)
    against the stock trtllm-gen kernel through exact selection weights: every
    expert row equals the stock unweighted GEMM2 row bit for bit (TP8 B1 and TP16
    B1-2: the small-batch cluster experts; B > 64: the wide engine on its own
    scratch; otherwise the persistent experts)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_runtime import (  # noqa: E501
        PersistentExperts,
        scratch,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.small_batch.chain import (  # noqa: E501
        _launch,
    )

    intermediate, real = (384, 384) if tp == 8 else (256, 192)  # TP16 pads 192
    weights = selection_experts(intermediate, real)
    w13, w2, s13, s2 = weights
    experts = PersistentExperts(w13.view(torch.uint8), s13, w2.view(torch.uint8), s2)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    cluster = batch == 1 if tp == 8 else batch <= 2
    out = torch.empty(batch * TOPK, HIDDEN, dtype=torch.bfloat16, device=dev)
    if not cluster:
        experts.prepare(batch, scratch(tp, dev, wide=batch > 64), out)
    gen = torch.Generator(device="cuda").manual_seed(tp * 100 + batch)
    mismatches, total = 0, 0
    for _ in range(max(1, 64 // batch)):
        x = exact_latent(batch, gen)
        ids = torch.stack(
            [
                torch.randperm(EXPERTS, generator=gen, device="cuda")[:TOPK]
                for _ in range(batch)
            ]
        ).int()
        if cluster:
            scales = tuple(
                experts.weights[i].view(torch.uint8).reshape(-1) for i in (1, 3)
            )
            out.view(torch.int16).fill_(-1)
            counter = torch.zeros(1, dtype=torch.int32, device=dev)
            if tp == 8:
                tensors = (
                    x[0].contiguous(),
                    ids.flatten(),
                    experts.maps[256:640].view(3, 128),
                )
                code, args = _launch("experts_tp8", tensors + (*scales, out, counter))
            else:
                tensors = (
                    x.reshape(-1),
                    ids.flatten(),
                    experts.maps[:256].view(2, 128),
                )
                code, args = _launch("experts_tp16", tensors + (*scales, out), batch)
            code(*args, *experts.scalars, stream(dev))
        else:
            experts.run(x, ids)
        want = stock_expert_rows(weights, x, intermediate)
        got = out.view(batch, TOPK, HIDDEN)
        bad = got.view(torch.int16) != want.unsqueeze(1).view(torch.int16)
        mismatches += int(bad.sum())
        total += bad.numel()
    assert mismatches == 0, f"{mismatches} of {total} expert outputs differ"


@pytest.mark.parametrize("batch", [5, 8, 20, 60])
def test_routed_experts_prefetch_ahead_is_exact(batch):
    """Experts prepared ``ahead`` (their producers L2-prefetch a table of next
    weights after the last weight stage) give the same rows, bit for bit, as the
    stock trtllm-gen kernel; the table mixes real ranges, an unaligned tail and
    zero (skipped) rows."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts import (
        PREFETCH_ROWS,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_runtime import (  # noqa: E501
        PersistentExperts,
        scratch,
    )

    intermediate = real = 384
    weights = selection_experts(intermediate, real)
    w13, w2, s13, s2 = weights
    experts = PersistentExperts(w13.view(torch.uint8), s13, w2.view(torch.uint8), s2)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    out = torch.empty(batch * TOPK, HIDDEN, dtype=torch.bfloat16, device=dev)
    experts.prepare(batch, scratch(8, dev), out, ahead=True)
    assert experts.ahead[batch]
    table = torch.zeros(PREFETCH_ROWS, 2, dtype=torch.int64, device=dev)
    table[0] = torch.tensor([w13.data_ptr(), 64 << 20])
    table[1] = torch.tensor([w2.data_ptr() + 4096, (5 << 20) + 37])
    table[3] = torch.tensor([s13.data_ptr(), s13.numel()])
    gen = torch.Generator(device="cuda").manual_seed(7000 + batch)
    for _ in range(max(2, 64 // batch)):
        x = exact_latent(batch, gen)
        ids = torch.stack(
            [
                torch.randperm(EXPERTS, generator=gen, device="cuda")[:TOPK]
                for _ in range(batch)
            ]
        ).int()
        experts.run(x, ids, None, table)
        want = stock_expert_rows(weights, x, intermediate)
        got = out.view(batch, TOPK, HIDDEN)
        assert torch.equal(
            got.view(torch.int16), want.unsqueeze(1).expand_as(got).view(torch.int16)
        )


def random_experts(seed):
    """Random MXFP4 expert bytes (E8M0 scales near 2^-4) in the kernel's layout: every
    weight bit and scale reaches the MMAs, unlike ``selection_experts``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def u8(shape, lo=0, hi=256):
        values = torch.randint(lo, hi, shape, generator=gen, device="cuda")
        return values.to(torch.uint8)

    w13 = u8((EXPERTS, 768, HIDDEN // 2))
    s13 = u8((EXPERTS, 768, HIDDEN // 32), 122, 125)
    w2 = u8((EXPERTS, HIDDEN, 192))
    s2 = u8((EXPERTS, HIDDEN, 12), 122, 125)
    return w13, s13, w2, s2


@pytest.mark.parametrize("batch", [4, 5, 8])
def test_routed_experts_static_consumer_is_exact(batch, monkeypatch):
    """The streamed B<=8 engine's MMA warp walking the producer's schedule
    (``consume_static``), with static or claimed FC2 pairs, gives the
    slot-metadata consumer's rows bit for bit, on
    random weights and activations (every weight bit, scale and FP32 accumulation
    order reaches the outputs), over several calls of one prepared engine."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    weights = random_experts(batch)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    outs = []
    # The metadata consumer, the static walk over static FC2 slices, and the
    # static walk over dynamically claimed FC2 pairs (``DYNAMIC``).
    for policy, claimed in (("", ""), (er.STATIC, ""), (er.STATIC, er.DYNAMIC)):
        monkeypatch.setattr(er, "STATIC", policy)
        monkeypatch.setattr(er, "DYNAMIC", claimed)
        assert er.static(batch) == bool(policy)
        assert er.dynamic(batch) == bool(claimed)
        experts = er.PersistentExperts(*weights)
        out = torch.empty(batch * TOPK, HIDDEN, dtype=torch.bfloat16, device=dev)
        experts.prepare(batch, er.scratch(8, dev), out)
        gen = torch.Generator(device="cuda").manual_seed(500 + batch)
        rows = []
        for _ in range(3):
            x = torch.randn(batch, HIDDEN, generator=gen, device="cuda").bfloat16()
            ids = torch.stack(
                [
                    torch.randperm(EXPERTS, generator=gen, device="cuda")[:TOPK]
                    for _ in range(batch)
                ]
            ).int()
            experts.run(x, ids)
            rows.append(out.clone())
        outs.append(torch.stack(rows))
    assert outs[0].isfinite().all() and outs[0].abs().amax() > 0
    for out in outs[1:]:
        assert torch.equal(outs[0].view(torch.int16), out.view(torch.int16))


def binade_values(low, high):
    """Every BF16 value with magnitude in [2^low, 2^high), both signs."""
    values = bf16_patterns()
    magnitude = values.float().abs()
    return values[(magnitude >= 2.0**low) & (magnitude < 2.0**high)]


def test_routed_situ_grid_matches_trtllm_gen():
    """Every BF16 gate in +-[0.5, 1) against every BF16 up in +-[2, 4) through the
    TP8 persistent experts and the stock kernel, bit for bit: separates SiTU product
    orders (e.g. (beta * tanh * sigmoid) * (lb * tanh) from (beta * lb * tanh) *
    sigmoid * tanh, one rounding apart at g = -0.796875, u = 2.4375)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_runtime import (  # noqa: E501
        PersistentExperts,
        scratch,
    )

    batch, intermediate = 64, 384
    weights = selection_experts(intermediate, intermediate, paired=True)
    w13, w2, s13, s2 = weights
    experts = PersistentExperts(w13.view(torch.uint8), s13, w2.view(torch.uint8), s2)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    out = torch.empty(batch * TOPK, HIDDEN, dtype=torch.bfloat16, device=dev)
    experts.prepare(batch, scratch(8, dev), out)
    gates, ups = binade_values(-1, 0), binade_values(1, 2)
    gate = gates.repeat_interleave(ups.numel())
    up = ups.repeat(gates.numel())
    chunk = batch * intermediate
    count = gate.numel()
    gate, up = (torch.cat([v, v[: -count % chunk]]) for v in (gate, up))
    ids = torch.arange(TOPK, dtype=torch.int32, device=dev).repeat(batch, 1)
    mismatches = 0
    for start in range(0, gate.numel(), chunk):
        g = gate[start : start + chunk].view(batch, intermediate)
        u = up[start : start + chunk].view(batch, intermediate)
        x = paired_latent(g, u)
        experts.run(x, ids)
        want = stock_expert_rows(weights, x, intermediate)
        got = out.view(batch, TOPK, HIDDEN)
        mismatches += int(
            (got.view(torch.int16) != want.unsqueeze(1).view(torch.int16)).sum()
        )
    assert mismatches == 0, f"{mismatches} expert outputs differ"


@pytest.mark.parametrize("batch", [65, 200])
def test_wide_finalize_matches_reference(batch):
    """The wide MoE's top16 sum: FP32 accumulation of the routing-weighted BF16
    rows in slot order, one BF16 rounding (to within the FMA-vs-mul/add rounding
    of the FP32 partial sums)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.wide import (
        compile_finalize,
    )

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    gen = torch.Generator(device="cuda").manual_seed(batch)
    rows = torch.randn(batch * TOPK, HIDDEN, generator=gen, device="cuda").bfloat16()
    weights = torch.rand(batch, TOPK, generator=gen, device="cuda").bfloat16()
    out = torch.empty(batch, HIDDEN, dtype=torch.bfloat16, device="cuda")
    code = compile_finalize(batch, dev.index)
    code(to_cute(rows), to_cute(weights), to_cute(out), stream(dev))
    acc = torch.zeros(batch, HIDDEN, device="cuda")
    for k in range(TOPK):
        acc += weights[:, k : k + 1].float() * rows.view(batch, TOPK, -1)[:, k].float()
    want = acc.bfloat16()
    ulp = (want.float().abs() * 2**-7).clamp_min(2**-126)
    assert ((out.float() - want.float()).abs() <= ulp).all()
    assert (out == want).float().mean() > 0.99
