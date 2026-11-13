# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 MoE kernels vs. dequantized torch references."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("Inkling2 MoE requires Blackwell", allow_module_level=True)

import vllm.models.inkling2.moe as moe_module  # noqa: E402
import vllm.models.inkling2.ops.trtllm_bmm as trtllm_bmm  # noqa: E402
from vllm.models.inkling2.moe import Inkling2MoE  # noqa: E402
from vllm.models.inkling2.ops.activation import (  # noqa: E402
    _div_rn,
    _sigmoid_approx,
    swiglu_quantize,
)
from vllm.models.inkling2.ops.block_scales import (  # noqa: E402
    shuffle_rows,
    swizzle_block_scales,
)
from vllm.models.inkling2.ops.linear import linear_k_groups  # noqa: E402
from vllm.models.inkling2.ops.mxfp8 import mxfp8_quantize  # noqa: E402
from vllm.models.inkling2.ops.nvfp4 import nvfp4_quantize  # noqa: E402
from vllm.models.inkling2.ops.trtllm_bmm import ExpertPlan  # noqa: E402
from vllm.triton_utils import tl, triton  # noqa: E402

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1, device=values.device)
    v = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    return v.flatten(-2) * scales.float().repeat_interleave(16, -1)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    exp2 = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return values.float() * exp2


def unswizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    *lead, rows, cols = scales.shape
    tiles = scales.reshape(-1, rows // 128, cols // 4, 32, 4, 4)
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(scales.shape)


def unshuffle_rows(x: torch.Tensor) -> torch.Tensor:
    *lead, rows, cols = x.shape
    return x.reshape(-1, rows // 32, 4, 8, cols).transpose(2, 3).reshape(x.shape)


def r8c4_block_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, cols = scales.shape
    tiles = scales.reshape(rows // 8, 8, cols // 4, 4).permute(0, 2, 1, 3)
    return tiles.reshape(scales.shape).contiguous()


def random_nvfp4(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device="cuda")
    scales = torch.rand(*shape[:-1], shape[-1] // 16, device="cuda") + 0.5
    return values.to(torch.uint8), scales.to(torch.float8_e4m3fn)


def random_mxfp8(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(120, 128, (*shape[:-1], shape[-1] // 32), device="cuda")
    return values, scales.to(torch.uint8)


def test_nvfp4_quantize_picks_better_of_four_and_six():
    """Each 16-block keeps whichever of the 6.0/4.0 scale candidates
    reconstructs it with the smaller absolute error (fouroversix_mae)."""
    torch.manual_seed(0)
    x = (torch.randn(8, 512, device="cuda") * 3).bfloat16()
    values, scales, amax = nvfp4_quantize(x)
    torch.testing.assert_close(amax, x.float().abs().amax(-1))

    blocks = x.float().view(8, 32, 16)
    outer = (amax / 1536)[:, None]
    block_max = blocks.abs().amax(-1)
    candidates = [
        (block_max * (1536 / amax[:, None]) / d).clamp(max=448).to(torch.float8_e4m3fn)
        for d in (6.0, 4.0)
    ]
    chosen = scales.float()
    assert ((chosen == candidates[0].float()) | (chosen == candidates[1].float())).all()

    recon = dequant_nvfp4(values, scales).view(8, 32, 16) * outer[..., None]
    error = (recon - blocks).abs().sum(-1)
    for cand in candidates:
        # Re-quantizing with the other candidate is never strictly better.
        q = blocks / (cand.float() * outer)[..., None]
        grid = torch.tensor(E2M1[:8], device="cuda")
        rounded = grid[(q.abs()[..., None] - grid).abs().argmin(-1)] * q.sign()
        cand_error = (rounded * cand.float()[..., None] * outer[..., None]) - blocks
        assert (error <= cand_error.abs().sum(-1) * (1 + 1e-6) + 1e-12).all()


def test_mxfp8_quantize_rounds_scale_up_to_power_of_two():
    torch.manual_seed(0)
    x = (torch.randn(8, 512, device="cuda") * 5).bfloat16()
    values, scales = mxfp8_quantize(x)
    block_amax = x.float().view(8, 16, 32).abs().amax(-1)
    expected = torch.ceil(torch.log2(block_amax / 448)) + 127
    torch.testing.assert_close(scales.float(), expected)
    torch.testing.assert_close(
        dequant_mxfp8(values, scales), x.float(), atol=0, rtol=2**-4
    )


@triton.jit
def _div_kernel(X, Out, Ref, CAP: tl.constexpr):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    x = tl.load(X + offs)
    tl.store(Out + offs, _div_rn(x, CAP))
    tl.store(Ref + offs, tl.div_rn(x, CAP))


@pytest.mark.parametrize("cap", [4.0, 25.0])
def test_softcap_division_rounds_correctly(cap: float):
    """The softcap's division matches IEEE division bitwise on every bf16
    input (the GEMM outputs), including signed zeros and infinities."""
    x = torch.arange(1 << 16, dtype=torch.int32, device="cuda") << 16
    x = x.view(torch.float32)
    out, ref = torch.empty_like(x), torch.empty_like(x)
    _div_kernel[(x.numel() // 1024,)](x, out, ref, cap, enable_fp_fusion=False)
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


@triton.jit
def _situ_reference_kernel(
    X, Gammas, Y, N: tl.constexpr, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * 256 + tl.arange(0, 256)
    gate = tl.load(X + row * 2 * N + 2 * cols).to(tl.float32)
    up = tl.load(X + row * 2 * N + 2 * cols + 1).to(tl.float32)
    gate_cap = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [tl.div_rn(gate, GATE_CAP)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    up_cap = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [tl.div_rn(up, UP_CAP)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    y = ((GATE_CAP * gate_cap) * _sigmoid_approx(gate)) * (UP_CAP * up_cap)
    tl.store(Y + row * N + cols, (y * tl.load(Gammas + row)).to(tl.bfloat16))


def _unswizzled_scales(scales: torch.Tensor, rows: torch.Tensor, tile: int):
    """Row-major block scales of `rows` of a layout for `tile`-row tiles."""
    cols = scales.shape[1]
    r, c = rows[:, None], torch.arange(cols, device=rows.device)[None, :]
    if tile >= 128:
        tiles = (r // 128) * (cols // 4) + c // 4
        offsets = tiles * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + c % 4
    else:
        offsets = ((r // 8) * (cols // 4) + c // 4) * 32 + (r % 8) * 4 + c % 4
    return scales.view(torch.uint8).flatten()[offsets]


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tile", [8, 128])
def test_situ_quantize_matches_separate_quantization(fmt: str, tile: int):
    """The fused SiTU quantization equals quantizing the bf16 SiTU output:
    bitwise values and scales, and for NVFP4 the row amax times the weight
    scale as the down-projection row scale. MXFP8 covers both the per-column
    block (few rows) and the whole-row programs."""
    torch.manual_seed(0)
    n, num_rows, num_experts = 2304, 5 * tile, 4
    x = (torch.randn(num_rows, 2 * n, device="cuda") * 4).bfloat16()
    gammas = torch.rand(num_rows, device="cuda")
    routed = torch.randperm(num_rows, device="cuda")[: num_rows - 5].int()
    num_routed = torch.tensor([routed.numel()], dtype=torch.int32, device="cuda")
    cta_expert = torch.randint(0, num_experts, (5,), device="cuda").int()
    nvfp4 = fmt == "nvfp4"
    w_amax = torch.rand(num_experts, device="cuda") + 0.5 if nvfp4 else None
    w_denominator = torch.tensor([2688.0], device="cuda") if nvfp4 else None
    values, scales, row_scales = swiglu_quantize(
        x,
        gammas,
        4.0,
        25.0,
        (routed, num_routed, cta_expert),
        tile,
        w_amax,
        w_denominator,
    )

    y = torch.empty((num_rows, n), dtype=torch.bfloat16, device="cuda")
    _situ_reference_kernel[(num_rows, n // 256)](
        x, gammas, y, n, 4.0, 25.0, enable_fp_fusion=False
    )
    rows = routed.long()
    if w_amax is not None:
        ref_values, ref_scales, amax = nvfp4_quantize(y[rows])
        assert row_scales is not None
        expert = cta_expert[rows // tile].long()
        ref_row_scales = (amax * w_amax[expert]) * (1.0 / (1536.0 * 2688.0))
        torch.testing.assert_close(row_scales[rows], ref_row_scales, atol=0, rtol=0)
    else:
        ref_values, ref_scales = mxfp8_quantize(y[rows])
    assert torch.equal(values[rows].view(torch.uint8), ref_values.view(torch.uint8))
    assert torch.equal(
        _unswizzled_scales(scales, rows, tile), ref_scales.view(torch.uint8)
    )


def test_shuffle_rows_matches_flashinfer():
    from flashinfer import shuffle_matrix_a

    x = torch.randint(0, 256, (256, 64), dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(shuffle_rows(x), shuffle_matrix_a(x, 128))
    torch.testing.assert_close(unshuffle_rows(shuffle_rows(x)), x)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tile", [8, 128])
def test_bmm_matches_dequantized_matmul(fmt: str, tile: int):
    """Expert GEMM over tile-padded expert segments equals an fp64 matmul of
    the dequantized operands, scaled per row (NVFP4) and rounded to bf16."""
    torch.manual_seed(0)
    nvfp4 = fmt == "nvfp4"
    counts, n, k = [3, 0, 17, 1], 256, 1024
    random_q = random_nvfp4 if nvfp4 else random_mxfp8
    weight, weight_scale = random_q(len(counts), n, k)
    shapes = ((n, k), (n, k))
    plan = trtllm_bmm.expert_plan(sum(counts), 1, len(counts), nvfp4, shapes, tile)
    act, act_scale = random_q(plan.rows, k)
    row_scale = torch.rand(plan.rows, device="cuda") if nvfp4 else None
    experts, limits, rows, start = [], [], [], 0
    for expert, count in enumerate(counts):
        for t in range(-(-count // tile)):
            experts.append(expert)
            limits.append(min(start + (t + 1) * tile, start + count))
        rows += [(start + i, expert) for i in range(count)]
        start += -(-count // tile) * tile
    num_tiles = len(experts)
    pad = [0] * (plan.max_ctas - num_tiles)
    tiles = torch.tensor(
        experts + pad + limits + pad + [num_tiles, start],
        dtype=torch.int32,
        device="cuda",
    )
    swizzle = swizzle_block_scales if tile >= 128 else r8c4_block_scales
    out = trtllm_bmm.bmm(
        act,
        swizzle(act_scale),
        row_scale,
        shuffle_rows(weight),
        swizzle_block_scales(shuffle_rows(weight_scale)),
        torch.ones(len(counts), device="cuda"),
        tiles,
        plan,
        0,
    )

    dequant = dequant_nvfp4 if nvfp4 else dequant_mxfp8
    index = torch.tensor([r for r, _ in rows], device="cuda")
    expert = torch.tensor([e for _, e in rows], device="cuda")
    a = dequant(act, act_scale).double()[index]
    w = dequant(weight, weight_scale).double()[expert]
    ref = torch.einsum("rk,rnk->rn", a, w)
    if row_scale is not None:
        ref = ref * row_scale[index, None].double()
    ref = ref.bfloat16().float()
    torch.testing.assert_close(
        out[index].float(), ref, atol=1e-3 * ref.abs().max().item(), rtol=1e-2
    )


def _torch_select_experts(logits, bias, topk, route_scale):
    _, ids = torch.topk(logits.sigmoid() + bias, topk, sorted=True)
    selected = torch.nn.functional.logsigmoid(logits.gather(1, ids))
    return selected.softmax(-1) * route_scale, ids


@pytest.mark.parametrize("tokens", [5, 12, 64, 200, 2100])
def test_routing_matches_torch_and_breaks_ties_low(dist_init, tokens: int):
    """Covers the fused (few tokens), tl.topk and k-pass selections (and the
    TMA router GEMM): each (token, slot) row lands in its expert's tiles with
    its routing weight."""
    moe = _build_moe("nvfp4", _random_moe_weights("nvfp4"))
    gate = moe.gate
    # Equal rows and biases tie experts 1-3, which every token then selects.
    gate.weight.data[[2, 3]] = gate.weight.data[1].clone()
    gate.bias.data[[1, 2, 3]] = 1.0
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    plan = moe.experts.plan(tokens)
    (src2dst, gammas, _, tiles), _ = _route(moe, x, plan)
    ids = tiles[src2dst.long() // plan.tile].view(tokens, -1)
    weights = gammas[src2dst.long()].view(tokens, -1)

    logits = linear_k_groups(x, gate.weight)
    ref_weights, ref_ids = _torch_select_experts(logits, gate.bias, 8, 8.0)
    assert (ids[:, :3] == torch.tensor([1, 2, 3], device="cuda")).all()
    torch.testing.assert_close(ids[:, 3:].long(), ref_ids[:, 3:])
    torch.testing.assert_close(weights, ref_weights)


@pytest.mark.parametrize("tokens", [5, 200, 1100])
def test_routing_is_deterministic(dist_init, tokens: int):
    """Rows are placed within their experts by atomics, so their order may
    change between runs; the selected experts, routing weights and the output
    (combined in slot order) must not. The second run competes with a side
    stream for the GPU to perturb the placement order."""
    moe = _build_moe("nvfp4", _random_moe_weights("nvfp4"))
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    plan = moe.experts.plan(tokens)

    def run():
        (src2dst, gammas, _, tiles), _ = _route(moe, x, plan)
        ids = tiles[src2dst.long() // plan.tile]
        return ids, gammas[src2dst.long()], moe(x)

    expected = run()
    side = torch.Stream("cuda")
    big = torch.randn(4096, 4096, device="cuda")
    with side:
        for _ in range(4):
            big = big @ big
            big /= big.norm()
    actual = run()
    torch.accelerator.synchronize()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [5, 200, 2100])
def test_routing_quantizes_tokens_into_their_rows(dist_init, fmt: str, tokens: int):
    """Covers the fused (few tokens) and the scatter routing (single-warp
    NVFP4 rows for many tokens): each routed
    (token, slot) row holds its token quantized bitwise like `nvfp4_quantize`
    / `mxfp8_quantize`, and (NVFP4) the gate/up output scale of its
    expert."""
    moe = _build_moe(fmt, _random_moe_weights(fmt))
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    plan = moe.experts.plan(tokens)
    (src2dst, _, _, tiles), (values, scales, row_scales) = _route(moe, x, plan)
    routed = src2dst >= 0
    rows = src2dst[routed].long()
    token = torch.arange(src2dst.numel(), device="cuda")[routed] // 8
    if fmt == "nvfp4":
        ref_values, ref_scales, amax = nvfp4_quantize(x)
        experts = moe.experts
        w_amax = experts.w13_weight_amax[tiles[rows // plan.tile].long()]
        denominator = experts.w13_weight_outer_scale_denominator.item()
        ref_row_scales = (amax[token] * w_amax) * (1.0 / (1536.0 * denominator))
        torch.testing.assert_close(row_scales[rows], ref_row_scales, atol=0, rtol=0)
    else:
        ref_values, ref_scales = mxfp8_quantize(x)
    assert torch.equal(
        values[rows].view(torch.uint8), ref_values[token].view(torch.uint8)
    )
    assert torch.equal(
        _unswizzled_scales(scales, rows, plan.tile),
        ref_scales[token].view(torch.uint8),
    )


def _route(
    moe: Inkling2MoE,
    x: torch.Tensor,
    plan: ExpertPlan,
    is_padding: torch.Tensor | None = None,
):
    from vllm.models.inkling2.ops.moe import route

    gate, experts = moe.gate, moe.experts
    return route(
        x,
        gate.weight,
        gate.bias,
        gate.counters,
        experts.counts,
        gate.topk,
        gate.route_scale,
        experts.expert_bounds,
        plan,
        *experts.scales("w13"),
        is_padding,
    )


@pytest.mark.parametrize("tokens", [33, 4000])
def test_linear_k_groups_fp32(tokens: int):
    """Covers both the split-K (few tokens) and the in-program group sum."""
    torch.manual_seed(0)
    x = torch.randn(tokens, 4608, device="cuda").bfloat16()
    w = (torch.randn(384, 4608, device="cuda") * 0.02).bfloat16()
    out = linear_k_groups(x, w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.double() @ w.double().T, check_dtype=False)


def _moe_reference(moe: Inkling2MoE, x: torch.Tensor, cfg) -> torch.Tensor:
    weights, ids = _torch_select_experts(
        linear_k_groups(x, moe.gate.weight),
        moe.gate.bias,
        cfg.num_experts_per_tok,
        cfg.route_scale,
    )
    exp = moe.experts

    def linear(a, name, e):
        w = unshuffle_rows(getattr(exp, f"{name}_weight")[e])
        s = unswizzle_block_scales(getattr(exp, f"{name}_weight_scale")[e])
        s = unshuffle_rows(s)
        if exp.nvfp4:
            av, as_, aa = nvfp4_quantize(a)
            out = dequant_nvfp4(av, as_).double() @ dequant_nvfp4(w, s).double().T
            amax = getattr(exp, f"{name}_weight_amax")[e]
            den = getattr(exp, f"{name}_weight_outer_scale_denominator")
            out = out * (aa * amax / (1536 * den))[:, None]
        else:
            av, as_ = mxfp8_quantize(a)
            out = dequant_mxfp8(av, as_).double() @ dequant_mxfp8(w, s).double().T
        return out.bfloat16()

    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for t in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = ids[t, slot].item()
            gate_up = linear(x[t : t + 1], "w13", e).double()
            gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            act = (
                cfg.mlp_gate_softcap
                * torch.tanh(gate / cfg.mlp_gate_softcap)
                * torch.sigmoid(gate)
                * cfg.mlp_up_softcap
                * torch.tanh(up / cfg.mlp_up_softcap)
                * weights[t, slot]
            )
            out[t] += linear(act.bfloat16(), "w2", e)[0].double()
    return out.bfloat16()


MOE_CONFIG = SimpleNamespace(
    hidden_size=512,
    intermediate_size=256,
    n_routed_experts=16,
    n_shared_experts=0,
    num_experts_per_tok=8,
    route_scale=8.0,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    use_gate_bias=True,
)


def _random_moe_weights(fmt: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    random_q = random_nvfp4 if fmt == "nvfp4" else random_mxfp8
    weights = {
        "gate.weight": (torch.randn(16, 512, device="cuda") * 0.05).bfloat16(),
        "gate.bias": torch.randn(16, device="cuda") * 0.01,
    }
    for name, n, k in (("w13", 512, 512), ("w2", 512, 256)):
        prefix = f"experts.{name}_weight"
        weights[prefix], weights[f"{prefix}_scale"] = random_q(16, n, k)
        if fmt == "nvfp4":
            weights[f"{prefix}_amax"] = torch.rand(16, device="cuda") * 100 + 100
            weights[f"{prefix}_outer_scale_denominator"] = torch.tensor(
                [2688.0], device="cuda"
            )
    return weights


def _build_moe(fmt: str, weights: dict[str, torch.Tensor]) -> Inkling2MoE:
    with torch.device("cuda"):
        moe = Inkling2MoE(MOE_CONFIG, fmt)
    for name, param in moe.named_parameters():
        param.weight_loader(param, weights[name])
    return moe


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [5, 33])
def test_moe_layer_matches_reference(dist_init, fmt: str, tokens: int):
    """Full layer (TP=1) vs. a per-token loop over dequantized experts, with
    fused (few tokens) and separate routing."""
    moe = _build_moe(fmt, _random_moe_weights(fmt))
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    out = moe(x)
    ref = _moe_reference(moe, x, MOE_CONFIG)
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [5, 64])
def test_token_tiles_match(dist_init, monkeypatch, fmt: str, tokens: int):
    """Every token tile (R8c4 and R128c4 scales, one or several tiles per
    expert) gives bitwise the same output."""
    weights = _random_moe_weights(fmt)
    weights["gate.bias"][:6] += 1.0
    moe = _build_moe(fmt, weights)
    x = torch.randn(tokens, 512, device="cuda").bfloat16()

    def run(tile: int) -> torch.Tensor:
        monkeypatch.setattr(trtllm_bmm, "_TILES", ((1 << 31, tile),))
        trtllm_bmm.expert_plan.cache_clear()
        return moe(x)

    expected = run(8)
    for tile in (16, 32, 64, 128, 256):
        torch.testing.assert_close(run(tile), expected, atol=0, rtol=0)


def _nan_filled(empty):
    def wrapper(*args, **kwargs):
        out = empty(*args, **kwargs)
        return out.fill_(float("nan")) if out.is_floating_point() else out

    return wrapper


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [64, 4097])
def test_moe_expert_parallel_matches_single_rank(
    dist_init, monkeypatch, fmt: str, tokens: int
):
    """EP=4, emulated on one GPU, sums to the EP=1 output. Buffers start as
    NaN, so reading rows of experts a rank does not own would show up."""
    weights = _random_moe_weights(fmt)
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    expected = _build_moe(fmt, weights)(x)

    ranks = []
    for rank in range(4):
        monkeypatch.setattr(
            moe_module, "get_tensor_model_parallel_rank", lambda r=rank: r
        )
        monkeypatch.setattr(
            moe_module, "get_tensor_model_parallel_world_size", lambda: 4
        )
        ranks.append(_build_moe(fmt, weights))
    partials: list[torch.Tensor] = []

    def all_reduce(partial: torch.Tensor) -> torch.Tensor:
        partials.append(partial)
        return partial

    monkeypatch.setattr(moe_module, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(torch, "empty", _nan_filled(torch.empty))
    monkeypatch.setattr(torch, "empty_like", _nan_filled(torch.empty_like))
    new_empty = _nan_filled(torch.Tensor.new_empty)
    monkeypatch.setattr(torch.Tensor, "new_empty", new_empty)
    for moe in ranks:
        moe(x)

    assert all(p.dtype == torch.bfloat16 for p in partials) and len(partials) == 4
    actual = sum(p.float() for p in partials).bfloat16()
    assert not actual.isnan().any()
    scale = expected.abs().max().item()
    # Each of the 4 partials is rounded to bf16 (up to 2**-9 of the partial).
    torch.testing.assert_close(actual, expected, atol=4 * 2**-9 * scale, rtol=1e-2)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens, padded", [(5, 8), (33, 48), (200, 256), (2100, 2304)])
def test_padding_tokens_route_to_no_expert(
    dist_init, monkeypatch, fmt: str, tokens: int, padded: int
):
    """CUDA graph padding rows (`is_padding` of the forward context) take no
    expert rows and come out zero, even when they hold NaNs; the real rows
    match the unpadded batch bitwise. Covers the fused and scatter routing."""
    moe = _build_moe(fmt, _random_moe_weights(fmt))
    x = torch.randn(padded, 512, device="cuda").bfloat16()
    x[tokens:] = float("nan")
    expected = moe(x[:tokens])

    is_padding = torch.arange(padded, device="cuda") >= tokens
    context = SimpleNamespace(is_padding=is_padding)
    monkeypatch.setattr(moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe_module, "get_forward_context", lambda: context)
    plan = moe.experts.plan(padded)
    (src2dst, _, _, tiles), _ = _route(moe, x, plan, is_padding)
    assert (src2dst.view(padded, -1)[tokens:] == -1).all()
    assert (src2dst[: tokens * 8] >= 0).all()
    assert tiles[2 * plan.max_ctas + 2].item() == tokens * 8

    out = moe(x)
    assert (out[tokens:] == 0).all()
    torch.testing.assert_close(out[:tokens], expected, atol=0, rtol=0)
