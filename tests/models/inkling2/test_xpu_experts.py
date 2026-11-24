# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU routed-expert backends vs. dequantized torch references, at
the expert shapes of Inkling 2 Mini and Small. The NVFP4 kernel itself
(`nvfp4_emulation_moe.py`) is tested in
`tests/kernels/moe/test_nvfp4_emulation_moe.py`."""

import functools

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform

if not current_platform.is_xpu():
    pytest.skip("Inkling2 XPU expert backends", allow_module_level=True)

from tests.models.inkling2.utils import (  # noqa: E402
    dequant_mxfp8,
    dequant_nvfp4,
)
from vllm.models.inkling2.common.moe import (  # noqa: E402
    Inkling2MoE,
    flat_experts_class,
)
from vllm.models.inkling2.common.ops.moe import select_experts  # noqa: E402
from vllm.models.inkling2.configs import Inkling2ModelConfig  # noqa: E402
from vllm.models.inkling2.xpu.experts import (  # noqa: E402
    XpuMxfp8Experts,
    XpuNvfp4Experts,
    check_fused_moe_support,
    deinterleave_rows,
    quantize_mxfp8_activations,
    split_dots_bound,
    split_dots_config,
    use_split_dots,
)

DEVICE = "xpu"
GATE_CAP, UP_CAP, TOPK = 4.0, 25.0, 8
# hidden, intermediate, experts, local experts (EP rank 1 of 4 for Small).
SHAPES = {"mini": (2304, 1152, 128, 128), "small": (4608, 2304, 384, 96)}
FORMATS = ["mxfp8", "nvfp4"]
# Mini splits below 16 rows per expert (up to 240 tokens), Small below 64
# (up to 3024).
EXPERT_TOKENS = [1, 7, 64, 300, 1600, 3072]
# NVFP4 outer scale denominator of the checkpoints' quantizer (test_moe.py).
DENOMINATOR = 2688.0


def random_nvfp4(*shape: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed values, e4m3 block scales and the weight amax per expert, with
    a GEMM output of about unit size."""
    values = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device=DEVICE)
    scales = torch.rand(*shape[:-1], shape[-1] // 16, device=DEVICE) + 0.5
    amax = torch.rand(shape[0], device=DEVICE) + 0.5
    amax *= DENOMINATOR / (3 * shape[-1] ** 0.5)
    return values.to(torch.uint8), scales.to(torch.float8_e4m3fn), amax


def random_mxfp8(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Values and scales with a GEMM output of about unit size."""
    values = torch.randn(shape, device=DEVICE).to(torch.float8_e4m3fn)
    base = 127 - round(0.5 * torch.log2(torch.tensor(float(shape[-1]))).item())
    scales = torch.randint(base - 1, base + 2, (*shape[:-1], shape[-1] // 32))
    return values, scales.to(torch.uint8).to(DEVICE)


def random_weights(fmt: str, num_experts: int, hidden: int, inter: int):
    """Checkpoint-layout `{w13, w2}: (weight, block scales, global scale)`;
    the NVFP4 global scale is `amax / DENOMINATOR`."""
    shapes = {"w13": (2 * inter, hidden), "w2": (hidden, inter)}
    if fmt == "mxfp8":
        return {
            name: (*random_mxfp8(num_experts, *shape), None)
            for name, shape in shapes.items()
        }
    weights = {}
    for name, shape in shapes.items():
        values, scales, amax = random_nvfp4(num_experts, *shape)
        weights[name] = (values, scales, amax / DENOMINATOR)
    return weights


def dequant(fmt: str, tensors: tuple, expert: int) -> torch.Tensor:
    """fp32 weights `[N, K]` of `expert` from `(weight, scales, global)`."""
    weight, scale, global_scale = tensors
    if fmt == "mxfp8":
        return dequant_mxfp8(weight[expert], scale[expert])
    return dequant_nvfp4(weight[expert], scale[expert]) * global_scale[expert]


def mxfp8_qdq(t: torch.Tensor) -> torch.Tensor:
    """`t` through the MXFP8 activation quantization of the experts."""
    values, scales = quantize_mxfp8_activations(t)
    assert scales is not None
    exp2 = scales.float().repeat_interleave(32, -1)
    return (values.float() * exp2).to(t.dtype)


def reference(fmt, weights, x, ids, gammas, gamma_after_w2: bool, act_qdq=None):
    """Per-expert fp32 loop over the dequantized checkpoint `weights`: the
    bf16 gate/up GEMM output, SiTU of the interleaved (gate, up) columns and
    a bf16 down projection, with the routing weight before the down
    projection (rounded to bf16 with the SiTU) or after it (in fp32).
    `act_qdq`: applied to `x` and to the bf16 SiTU output (quantized
    activations)."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    if act_qdq is not None:
        x = act_qdq(x)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        token, slot = (ids == expert).nonzero(as_tuple=True)
        w13 = dequant(fmt, weights["w13"], expert)
        gate_up = (x[token].float() @ w13.T).bfloat16().float()
        gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
        act = (
            GATE_CAP
            * torch.tanh(gate / GATE_CAP)
            * torch.sigmoid(gate)
            * UP_CAP
            * torch.tanh(up / UP_CAP)
        )
        gamma = gammas.view(ids.shape)[token, slot][:, None]
        act = (act if gamma_after_w2 else act * gamma).bfloat16()
        if act_qdq is not None:
            act = act_qdq(act)
        w2 = dequant(fmt, weights["w2"], expert)
        down = (act.float() @ w2.T).bfloat16().float()
        out.index_add_(0, token, down * gamma if gamma_after_w2 else down)
    return out.bfloat16()


def routing(tokens: int, num_global: int, num_local: int, rank: int):
    """`select_experts`-style local ids `[T, topk]` (-1: not local) with a
    few empty local experts and an all-padding last token, fp32 routing
    weights `[T * topk]` and the identity `src2dst`."""
    start = rank * num_local
    empty = torch.tensor([start, start + 5, start + num_local - 1])
    scores = torch.rand(tokens, num_global)
    # Each token has a local expert (not one of the empty ones).
    scores[torch.arange(tokens), start + 1 + torch.randint(0, 4, (tokens,))] = 2.0
    scores[:, empty] = -1.0
    ids = scores.topk(TOPK, dim=-1).indices
    local = torch.where((ids >= start) & (ids < start + num_local), ids - start, -1)
    if tokens > 1:
        local[-1] = -1
    local = local.to(torch.int32).to(DEVICE)
    gammas = torch.rand(tokens * TOPK, device=DEVICE) + 0.5
    rows = torch.arange(tokens * TOPK, dtype=torch.int32, device=DEVICE)
    src2dst = torch.where(local.flatten() >= 0, rows, -1)
    return local, gammas, src2dst


def gamma_after_w2(fmt: str) -> bool:
    """Where the backend of `fmt` applies the routing weight: MXFP8's fused
    MoE on the down-projection rows in fp32, NVFP4 (Hopper's flow) on the
    SiTU output before the down projection."""
    return fmt == "mxfp8"


def combined(experts, out: torch.Tensor, src2dst: torch.Tensor) -> torch.Tensor:
    """The combined `[T, H]` output of a `FlatExperts` call."""
    if getattr(experts, "combines", False):
        return out
    from vllm.models.inkling2.common.ops.moe import combine

    return combine(out, src2dst, TOPK, torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.uint8, torch.bfloat16])
def test_deinterleave_rows_round_trip(dtype: torch.dtype):
    """Interleaved (gate, up) rows become `[gate | up]` halves, and
    interleaving the halves again gives back the checkpoint rows."""
    torch.manual_seed(0)
    gate, up = torch.randn(2, 6, 64, 96, device=DEVICE).to(dtype)
    checkpoint = torch.stack((gate, up), 2).reshape(6, 128, 96)
    rows = checkpoint.clone()
    deinterleave_rows(rows)
    halves = torch.cat((gate, up), 1)
    assert torch.equal(rows.view(torch.uint8), halves.view(torch.uint8))
    again = torch.stack(rows.chunk(2, 1), 2).reshape(rows.shape)
    assert torch.equal(again.view(torch.uint8), checkpoint.view(torch.uint8))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("model", list(SHAPES))
@pytest.mark.parametrize("tokens", EXPERT_TOKENS)
def test_xpu_experts_match_reference(fmt: str, model: str, tokens: int):
    """The XPU backend of `flat_experts_class` on local experts (one EP
    rank for Small) vs. the dequantized reference, with non-local slots,
    empty experts and a token routed to no local expert (which comes out
    0). NVFP4: below `split_dots_bound` rows per expert the kernel's
    split-dots path runs (Mini up to 240 tokens: 1 / 7 / 64 here; Small up
    to 3024: all but 3072), otherwise its default path."""
    torch.manual_seed(0)
    hidden, inter, num_global, num_local = SHAPES[model]
    weights = random_weights(fmt, num_local, hidden, inter)
    checkpoint = {
        name: tuple(None if t is None else t.clone() for t in tensors)
        for name, tensors in weights.items()
    }
    experts = flat_experts_class(fmt, TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), fmt == "nvfp4"
    )
    assert isinstance(experts, XpuMxfp8Experts if fmt == "mxfp8" else XpuNvfp4Experts)
    rank = 0 if num_local == num_global else 1
    ids, gammas, src2dst = routing(tokens, num_global, num_local, rank)
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    out = experts(x, ids, gammas, src2dst, num_global, GATE_CAP, UP_CAP)
    out = combined(experts, out, src2dst)
    assert getattr(experts, "combines", False) == gamma_after_w2(fmt)
    ref = reference(fmt, checkpoint, x, ids, gammas, gamma_after_w2(fmt))
    if tokens > 1:
        assert (out[-1] == 0).all()
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


def test_expert_tokens_cover_both_nvfp4_paths():
    """`EXPERT_TOKENS` take both the split-dots and the default path at
    each model's shape."""
    for hidden, _, num_global, _ in SHAPES.values():
        paths = {use_split_dots(t, TOPK, num_global, hidden) for t in EXPERT_TOKENS}
        assert paths == {True, False}, hidden


def mini_layer(fmt: str):
    """`Inkling2MoE` (TP=1) at Mini shapes with random checkpoint weights
    loaded through the weight loaders, before `process_weights_after_loading`;
    and those weights."""
    hidden, inter, num_experts, _ = SHAPES["mini"]
    config = Inkling2ModelConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=num_experts,
        num_experts_per_tok=TOPK,
        mlp_gate_softcap=GATE_CAP,
        mlp_up_softcap=UP_CAP,
    )
    with torch.device(DEVICE):
        moe = Inkling2MoE(config, fmt)
    weights = random_weights(fmt, num_experts, hidden, inter)
    loaded = {
        "gate.weight": (torch.randn(num_experts, hidden, device=DEVICE) * 0.02),
        "gate.bias": torch.randn(num_experts, device=DEVICE) * 0.01,
    }
    loaded["gate.weight"] = loaded["gate.weight"].bfloat16()
    for name, (weight, scale, global_scale) in weights.items():
        prefix = f"experts.{name}_weight"
        loaded[prefix], loaded[f"{prefix}_scale"] = weight, scale
        if fmt == "nvfp4":
            loaded[f"{prefix}_amax"] = global_scale * DENOMINATOR
            loaded[f"{prefix}_outer_scale_denominator"] = torch.tensor(
                [DENOMINATOR], device=DEVICE
            )
    for name, param in moe.named_parameters():
        param.weight_loader(param, loaded[name])
    return moe, weights


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("tokens", [5, 64])
def test_xpu_moe_layer_matches_reference(dist_init, fmt: str, tokens: int):
    """`Inkling2MoE` (TP=1) at Mini shapes: `select_experts`, the XPU
    backend and the combine, vs. the reference on the layer's own routing."""
    torch.manual_seed(0)
    moe, weights = mini_layer(fmt)
    moe.experts.process_weights_after_loading()
    x = torch.randn(tokens, SHAPES["mini"][0], device=DEVICE).bfloat16()
    out = moe(x)

    gate = moe.gate
    ids, gammas, _ = select_experts(
        x, gate.weight, gate.bias, TOPK, gate.route_scale, moe.experts.expert_bounds
    )
    ref = reference(fmt, weights, x, ids, gammas, gamma_after_w2(fmt))
    scale = ref.abs().max().item()
    assert scale > 0.1
    torch.testing.assert_close(out, ref, atol=1e-2 * scale, rtol=2e-2)


@pytest.mark.parametrize("fmt", FORMATS)
def test_xpu_experts_keep_kernel_weights_as_parameters(dist_init, fmt: str):
    """After `process_weights_after_loading` the backend computes on the
    layer's Parameters (MXFP8: rewritten into the kernel layout, with the
    library's rebuild mark on the `w13` Parameter; NVFP4: the checkpoint
    layout as loaded), so `named_parameters()` shows the weights in use.
    A second weight load (a reload) is refused by the weight loaders before
    they write anything, and a second processing call fails loudly."""
    torch.manual_seed(0)
    moe, weights = mini_layer(fmt)
    experts = moe.experts
    experts.process_weights_after_loading()
    params = dict(experts.named_parameters())
    backend = experts.flat_experts
    if fmt == "mxfp8":
        assert isinstance(backend, XpuMxfp8Experts)
        assert backend.moe.w13 is params["w13_weight"]
        assert backend.moe.w2 is params["w2_weight"]
        assert getattr(params["w13_weight"], "xpu_fused_moe", False)
        # xe_2 layout [E, K, N] of the de-interleaved checkpoint rows.
        hidden, inter, _, _ = SHAPES["mini"]
        assert params["w2_weight"].shape == (weights["w2"][0].shape[0], inter, hidden)
        ckpt = weights["w2"][0]
        assert torch.equal(
            params["w2_weight"].view(torch.uint8),
            ckpt.transpose(-1, -2).contiguous().view(torch.uint8),
        )
    else:
        assert isinstance(backend, XpuNvfp4Experts)
        for name in ("w13", "w2"):
            weight, scale, _ = backend.weights[name]
            assert weight is params[f"{name}_weight"]
            assert scale is params[f"{name}_weight_scale"]
            assert torch.equal(weight, weights[name][0])
    kernel_bytes = {
        name: param.detach().clone() for name, param in experts.named_parameters()
    }
    for name, param in experts.named_parameters():
        projection = name.split("_", 1)[0]
        weight, scale, global_scale = weights[projection]
        if name.endswith("_weight"):
            loaded = weight
        elif name.endswith("_weight_scale"):
            loaded = scale
        elif name.endswith("_amax"):
            loaded = global_scale * DENOMINATOR
        else:
            loaded = torch.tensor([DENOMINATOR], device=DEVICE)
        with pytest.raises(RuntimeError, match="reloading"):
            param.weight_loader(param, loaded)
    for name, param in experts.named_parameters():
        assert torch.equal(
            param.view(torch.uint8), kernel_bytes[name].view(torch.uint8)
        )
    with pytest.raises(RuntimeError, match="already called"):
        experts.process_weights_after_loading()


def test_mxfp8_build_memory_peak_is_documented(dist_init):
    """The load-time extra memory of `XpuMxfp8Experts` is what its docstring
    states: the de-interleave is one expert at a time, but the relayout
    holds a copy of the whole local `w13` and `w2` at once."""
    torch.manual_seed(0)
    hidden, inter, _, num_local = SHAPES["mini"]
    weights = random_weights("mxfp8", num_local, hidden, inter)
    layer = sum(t.numel() * t.element_size() for w in weights.values() for t in w[:2])
    w13_bytes = weights["w13"][0].numel()
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    before = torch.xpu.memory_allocated()
    experts = flat_experts_class("mxfp8", TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), False
    )
    torch.xpu.synchronize()
    extra = torch.xpu.max_memory_allocated() - before
    assert w13_bytes < extra <= 1.05 * layer + 2**20, (extra, layer)
    del experts


@pytest.mark.parametrize("force_xe3p", [None, "1"])
def test_mxfp8_activation_decision_follows_the_fused_moe_kernel(
    monkeypatch, force_xe3p
):
    """MXFP8 activations are quantized exactly when the built `XpuFusedMoe`
    runs its xe_3 grouped GEMM (`not _uses_xe2`), which `FORCE_XE3P_KERNEL`
    also selects; only the build runs (no GEMM)."""
    if force_xe3p is None:
        monkeypatch.delenv("FORCE_XE3P_KERNEL", raising=False)
    else:
        monkeypatch.setenv("FORCE_XE3P_KERNEL", force_xe3p)
    hidden, inter, _, _ = SHAPES["mini"]
    weights = random_weights("mxfp8", 8, hidden, inter)
    experts = flat_experts_class("mxfp8", TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), False
    )
    assert isinstance(experts, XpuMxfp8Experts)
    assert experts.quantize_activations == (not experts.moe._uses_xe2)
    if force_xe3p:
        assert experts.quantize_activations
    # Both formats build on every device.
    for fmt, cls in (("mxfp8", XpuMxfp8Experts), ("nvfp4", XpuNvfp4Experts)):
        factory = flat_experts_class(fmt, TOPK, GATE_CAP, UP_CAP)
        assert isinstance(factory, functools.partial) and factory.func is cls


@pytest.mark.parametrize("model", list(SHAPES))
@pytest.mark.parametrize("tokens", [1, 64, 300])
def test_xpu_mxfp8_quantized_activations_match_reference(model: str, tokens: int):
    """The MXFP8 experts with quantized activations, on devices whose
    `XpuFusedMoe` runs the xe_3 grouped GEMM (neither Xe2 nor Xe3), vs. the
    reference with the same MXFP8 QDQ of the w13 and w2 inputs."""
    torch.manual_seed(0)
    hidden, inter, num_global, num_local = SHAPES[model]
    weights = random_weights("mxfp8", num_local, hidden, inter)
    checkpoint = {
        name: tuple(None if t is None else t.clone() for t in tensors)
        for name, tensors in weights.items()
    }
    experts = flat_experts_class("mxfp8", TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), False
    )
    assert isinstance(experts, XpuMxfp8Experts)
    if not experts.quantize_activations:
        pytest.skip(
            "this device's XpuFusedMoe runs the xe_2 grouped GEMM (Xe2 / Xe3), "
            "which takes bf16 activations; the quantized path runs on devices "
            "that are neither"
        )
    rank = 0 if num_local == num_global else 1
    ids, gammas, src2dst = routing(tokens, num_global, num_local, rank)
    x = torch.randn(tokens, hidden, device=DEVICE).bfloat16()
    out = experts(x, ids, gammas, src2dst, num_global, GATE_CAP, UP_CAP)
    ref = reference("mxfp8", checkpoint, x, ids, gammas, True, act_qdq=mxfp8_qdq)
    if tokens > 1:
        assert (out[-1] == 0).all()
    scale = ref.abs().max().item()
    assert scale > 0.1
    # A SiTU output that rounds differently can land on a neighbouring e4m3
    # code (2^-3 relative) in the w2 input.
    torch.testing.assert_close(out, ref, atol=3e-2 * scale, rtol=5e-2)
    assert (out.float() - ref.float()).abs().mean() < 2e-3 * scale


@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden,block_m",
    [
        (1, 8, 128, 2304, 16),
        (64, 8, 128, 2304, 16),
        (40, 8, 16, 4608, 32),
        (31, 8, 4, 4608, 64),
    ],
)
def test_split_dots_config_row_block(tokens, top_k, experts, hidden, block_m):
    """The split-dots config's row block is about one expert's rows, 16 to
    64, with tiles that hold whole scale groups (batches that take the
    split-dots path)."""
    assert use_split_dots(tokens, top_k, experts, hidden)
    config = split_dots_config(tokens, top_k, experts)
    assert config["BLOCK_SIZE_M"] == block_m
    assert config["BLOCK_SIZE_K"] % 16 == 0 and config["BLOCK_SIZE_K"] >= 32


@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden,td,split",
    [
        (1, 8, 128, 2304, False, True),
        (240, 8, 128, 2304, False, True),
        (256, 8, 128, 2304, False, False),
        (300, 8, 128, 2304, True, True),
        (512, 8, 128, 2304, True, False),
        (1536, 8, 384, 4608, False, True),
        (3072, 8, 384, 4608, True, False),
    ],
)
def test_split_dots_below_the_bound(
    monkeypatch, tokens, top_k, experts, hidden, td, split
):
    """The split-dots path serves batches below the measured bound of their
    shape and load path (Mini 16 rows per expert, 32 with descriptors; Small
    64), and none under `VLLM_BATCH_INVARIANT`."""
    # Read the variable live (an engine earlier in the session caches envs).
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    assert split_dots_bound(2304, False) == 16
    assert split_dots_bound(2304, True) == 32
    assert split_dots_bound(4608, False) == split_dots_bound(4608, True) == 64
    assert use_split_dots(tokens, top_k, experts, hidden, td) == split
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert not use_split_dots(tokens, top_k, experts, hidden, td)


@pytest.mark.parametrize("model", list(SHAPES))
def test_nvfp4_experts_batch_invariant(monkeypatch, model: str):
    """Under `VLLM_BATCH_INVARIANT` a token's NVFP4 expert rows are
    bitwise the same alone (1 token, the split-dots path otherwise) and in a
    batch of 600 tokens (the default path)."""
    # Read the variable live (an engine earlier in the session caches envs).
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(0)
    hidden, inter, num_global, num_local = SHAPES[model]
    weights = random_weights("nvfp4", num_local, hidden, inter)
    experts = flat_experts_class("nvfp4", TOPK, GATE_CAP, UP_CAP)(
        weights, ((2 * inter, hidden), (hidden, inter)), True
    )
    rank = 0 if num_local == num_global else 1
    ids, gammas, src2dst = routing(600, num_global, num_local, rank)
    x = torch.randn(600, hidden, device=DEVICE).bfloat16()
    batch = experts(x, ids, gammas, src2dst, num_global, GATE_CAP, UP_CAP)
    alone = experts(
        x[:1], ids[:1], gammas[:TOPK], src2dst[:TOPK], num_global, GATE_CAP, UP_CAP
    )
    local = ids[0] >= 0
    assert local.any()
    assert torch.equal(alone[:TOPK][local], batch[:TOPK][local])


REQUIREMENT_ERROR = r"vllm-xpu-kernels >= 0\.1\.17 \(XpuFusedMoe\._uses_xe2\); found"


class _FusedMoeWithSitu:
    def __init__(
        self, w13, *, activation_situ_beta=None, activation_situ_linear_beta=None
    ):
        pass


class _FusedMoeWithoutSitu:
    def __init__(self, w13, *, activation=None):
        pass


def _uses_xe2_grouped_gemm(weight):
    return True


@pytest.mark.parametrize(
    "fused_moe,has_xe2_query,supported",
    [
        (_FusedMoeWithSitu, True, True),
        # 0.1.13-0.1.16: SiTU, but no `_uses_xe2`.
        (_FusedMoeWithSitu, False, False),
        (_FusedMoeWithoutSitu, True, False),
    ],
)
def test_fused_moe_support_check(fused_moe, has_xe2_query, supported):
    """A vllm-xpu-kernels fused-MoE interface without `_uses_xe2` (from
    `_uses_xe2_grouped_gemm`, added in 0.1.17) or without the SiTU betas
    (since 0.1.13) fails with an error naming the requirement; a complete
    one passes (stub modules)."""
    from types import SimpleNamespace

    module = SimpleNamespace(XpuFusedMoe=fused_moe)
    if has_xe2_query:
        module._uses_xe2_grouped_gemm = _uses_xe2_grouped_gemm
    if supported:
        check_fused_moe_support(module)
        return
    with pytest.raises(NotImplementedError, match=REQUIREMENT_ERROR):
        check_fused_moe_support(module)


def test_fused_moe_support_check_on_the_installed_kernels():
    """The installed vllm-xpu-kernels passes the check from release 0.1.17
    on and an older release gets the error naming the requirement; a
    development build without a full release number (`0.1.devN`) may do
    either, but fails with that error only."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version
    from vllm_xpu_kernels import fused_moe_interface

    try:
        installed = Version(version("vllm-xpu-kernels"))
    except PackageNotFoundError:
        pytest.skip("vllm-xpu-kernels has no package metadata")
    if installed.release >= (0, 1, 17):
        check_fused_moe_support(fused_moe_interface)
    elif len(installed.release) >= 3:
        with pytest.raises(NotImplementedError, match=REQUIREMENT_ERROR):
            check_fused_moe_support(fused_moe_interface)
    else:
        try:
            check_fused_moe_support(fused_moe_interface)
        except NotImplementedError as error:
            assert "vllm-xpu-kernels >= 0.1.17" in str(error), error
