# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the gfx950 split-K BF16 router GEMM and where GateLinear's tier 6
sends each batch size of the DeepSeek-V4.1-Flash gates.

Main gate: hidden 5120 x 384 experts (top-6). DSpark draft gate: hidden
5120 x 128 experts, which has no split-K tier and keeps main's paths.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.router import (
    rocm_bf16_router_gemm as rocm_bf16,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

pytestmark = pytest.mark.skipif(
    not rocm_bf16.platform_supported(), reason="requires ROCm gfx950"
)

HIDDEN = 5120
MAIN_EXPERTS = 384
DRAFT_EXPERTS = 128
TOPK = 6
# Gate weights at this scale give logits up to ~7, as on the checkpoint.
WEIGHT_STD = 0.02
# Every token count the tiers serve, from either side of each boundary.
NUM_TOKENS = [1, 21, 22, 48, 64, 128, 224, 256]
_U = 2.0**-24


def _activations(num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", generator=g)
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True))).to(torch.bfloat16)


def _weight(num_experts: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(num_experts, HIDDEN, device="cuda", generator=g)
    return (w * WEIGHT_STD).to(torch.bfloat16)


def _batched(fn, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return torch.cat([fn(x[i : i + num_tokens]) for i in range(0, len(x), num_tokens)])


def _pool(num_tokens: int, min_rows: int, seed: int) -> torch.Tensor:
    """Whole batches of num_tokens rows, at least min_rows in total."""
    return _activations(-(-min_rows // num_tokens) * num_tokens, seed)


def _hipblaslt(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, w.T, out_dtype=torch.float32)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_matches_fp64_within_hipblaslt_error(num_tokens: int) -> None:
    """Both GEMMs sum exact bf16 products in fp32, in different K orders."""
    w = _weight(MAIN_EXPERTS)
    x = _pool(num_tokens, 1024, seed=num_tokens)
    out = _batched(lambda b: rocm_bf16.rocm_bf16_router_gemm(b, w), x, num_tokens)
    blas = _batched(lambda b: _hipblaslt(b, w), x, num_tokens)
    ref = x.double() @ w.double().T

    assert out.dtype == torch.float32
    assert out.shape == (len(x), MAIN_EXPERTS)
    # Holds for any fp32 summation order, so a miss is a dropped or doubled
    # split or a misplaced tile, never rounding.
    gamma = HIDDEN * _U / (1 - HIDDEN * _U)
    bound = gamma * (x.double().abs() @ w.double().abs().T)
    assert ((out.double() - ref).abs() <= bound).all()
    # Measured below hipBLASLt's error on the checkpoint's gate weights
    # (1.6e-6 vs 2.0e-6); the factor 2 allows for hipBLASLt choosing a
    # different kernel at another M.
    ours_err = (out.double() - ref).abs().max().item()
    blas_err = (blas.double() - ref).abs().max().item()
    assert ours_err <= 2 * blas_err, (ours_err, blas_err)


@torch.inference_mode()
def test_declines_what_it_has_no_tier_or_layout_for() -> None:
    """None makes the dispatch fall through to torch.mm."""
    w = _weight(MAIN_EXPERTS)
    x = _activations(257, seed=0)
    wide = _activations(64, seed=1).repeat(1, 2)

    assert rocm_bf16.rocm_bf16_router_gemm(x, w) is None
    assert rocm_bf16.rocm_bf16_router_gemm(x[:64], _weight(DRAFT_EXPERTS)) is None
    assert rocm_bf16.rocm_bf16_router_gemm(x[:64].float(), w) is None
    assert rocm_bf16.rocm_bf16_router_gemm(x[:64], w.float()) is None
    assert rocm_bf16.rocm_bf16_router_gemm(wide[:, ::2], w) is None


@pytest.mark.parametrize("num_tokens", [22, 96, 224])
@torch.inference_mode()
def test_bitwise_across_calls_and_graph_replay(num_tokens: int) -> None:
    """Splits are summed in split order, whatever finishes first."""
    w = _weight(MAIN_EXPERTS)
    x = _activations(num_tokens, seed=0)
    stream = torch.cuda.Stream()
    # The split counters are per stream and must exist before capture.
    with torch.cuda.stream(stream):
        rocm_bf16.rocm_bf16_router_gemm(x, w)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = rocm_bf16.rocm_bf16_router_gemm(x, w)

    new_x = _activations(num_tokens, seed=1)
    x.copy_(new_x)
    graph.replay()
    torch.accelerator.synchronize()
    eager = rocm_bf16.rocm_bf16_router_gemm(new_x, w)

    assert torch.equal(eager, rocm_bf16.rocm_bf16_router_gemm(new_x, w))
    assert torch.equal(captured, eager)


def _top_sets(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    scores = torch.sqrt(F.softplus(logits)) + bias
    return scores.topk(TOPK, dim=-1).indices.sort(dim=-1).values


@pytest.mark.parametrize("num_tokens", [22, 64, 128, 224, 256])
@torch.inference_mode()
def test_expert_selection_matches_hipblaslt(num_tokens: int) -> None:
    """DeepSeek-V4's sqrtsoftplus + correction-bias top-6, against torch.mm.

    sqrt(softplus(z)) has slope below 1/3, so a token whose exact 6th and 7th
    scores differ by more than either GEMM's largest logit error cannot change
    its expert set under either GEMM; only those near-ties are excluded.
    """
    w = _weight(MAIN_EXPERTS)
    g = torch.Generator(device="cuda").manual_seed(1)
    bias = 0.1 * torch.randn(MAIN_EXPERTS, device="cuda", generator=g)
    x = _pool(num_tokens, 4096, seed=100 + num_tokens)
    out = _batched(lambda b: rocm_bf16.rocm_bf16_router_gemm(b, w), x, num_tokens)
    blas = _batched(lambda b: _hipblaslt(b, w), x, num_tokens)
    ref = x.double() @ w.double().T

    logit_err = max((t.double() - ref).abs().max().item() for t in (out, blas))
    exact = (torch.sqrt(F.softplus(ref)) + bias.double()).topk(TOPK + 1, dim=-1)
    clear = exact.values[:, TOPK - 1] - exact.values[:, TOPK] > logit_err + 1e-6
    assert clear.float().mean() > 0.99
    assert torch.equal(_top_sets(out, bias)[clear], _top_sets(blas, bias)[clear])


def _make_gate(monkeypatch: pytest.MonkeyPatch, num_experts: int) -> GateLinear:
    for module in ("layers.linear", "parameter"):
        monkeypatch.setattr(
            f"vllm.model_executor.{module}.get_tensor_model_parallel_rank", lambda: 0
        )
        monkeypatch.setattr(
            f"vllm.model_executor.{module}.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
    gate = GateLinear(
        input_size=HIDDEN,
        output_size=num_experts,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        gate.weight.copy_(_weight(num_experts))
    gate.quant_method.process_weights_after_loading(gate)
    return gate


def _record_kernels(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Which GEMM served each call: main's low-M kernel, split-K, or torch.mm
    once split-K declines."""
    from vllm.model_executor.layers.fused_moe.router import (
        rocm_fp32_router_gemm as rocm_low_m,
    )

    calls: list[str] = []
    low_m = rocm_low_m.rocm_fp32_router_gemm
    split_k = rocm_bf16.rocm_bf16_router_gemm

    def spy_low_m(x, w):
        calls.append("low_m")
        return low_m(x, w)

    def spy_split_k(x, w):
        out = split_k(x, w)
        calls.append("hipblaslt" if out is None else "split_k")
        return out

    monkeypatch.setattr(rocm_low_m, "rocm_fp32_router_gemm", spy_low_m)
    monkeypatch.setattr(rocm_bf16, "rocm_bf16_router_gemm", spy_split_k)
    return calls


@pytest.mark.parametrize(
    ("num_experts", "num_tokens", "expected"),
    [
        # Main's low-M kernel takes M * N <= 8192, i.e. M <= 21 at N = 384.
        *((MAIN_EXPERTS, m, "low_m") for m in (1, 21)),
        *((MAIN_EXPERTS, m, "split_k") for m in (22, 48, 64, 128, 224, 256)),
        # Above 256 the shared expert runs inline, and hipBLASLt alone wins.
        (MAIN_EXPERTS, 257, "hipblaslt"),
        # The draft gate has no split-K tier.
        *((DRAFT_EXPERTS, m, "low_m") for m in (1, 32)),
        *((DRAFT_EXPERTS, m, "hipblaslt") for m in (33, 256)),
    ],
)
@torch.inference_mode()
def test_gate_linear_dispatch_boundaries(
    monkeypatch: pytest.MonkeyPatch, num_experts: int, num_tokens: int, expected: str
) -> None:
    gate = _make_gate(monkeypatch, num_experts)
    assert gate.allow_cublas_router_gemm
    calls = _record_kernels(monkeypatch)
    x = _activations(num_tokens, seed=num_tokens)

    output, _ = gate(x)

    assert calls == [expected]
    assert output.dtype == torch.float32
    if expected == "hipblaslt":
        assert torch.equal(output, _hipblaslt(x, gate.weight))


@torch.inference_mode()
def test_dispatch_keeps_hipblaslt_off_gfx950(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiers are gfx950-tuned; another arch reaching this dispatch must
    keep torch.mm."""
    monkeypatch.setattr(rocm_bf16, "platform_supported", lambda: False)
    calls = _record_kernels(monkeypatch)
    w = _weight(MAIN_EXPERTS)
    x = _activations(64, seed=0)

    output = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)

    assert calls == ["hipblaslt"]
    assert torch.equal(output, _hipblaslt(x, w))
