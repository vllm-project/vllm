# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused Markov bias in DSpark sequential draft sampling.

``fused_markov_bias_add`` (vllm/models/deepseek_v41/amd/
dspark_markov_fused_gemv.py) replaces the stock per-step chain (embed
gather + bf16 GEMV + add) with ONE Triton kernel. The sampler is
untouched (identical RNG stream); only the GEMM fp32 accumulation
order can differ, so the result is NOT bit-exact with rocBLAS. These
tests bound the deviation (<= 2 bf16 ULP vs the stock chain, both
within 1 ULP of an fp64-grounded reference), prove determinism, cover
the strided base-logits slice and int32/int64 token ids the speculator
passes, check the eligibility fallbacks (kill switch, draft-batch
gate, dtype/shape/contiguity misses), and prove CUDA-graph capture +
replay stability (the chain is captured inside the DSpark FULL draft
graph).
"""

import os

import pytest
import torch

from tests.kernels.utils import bf16_ulp_distance
from vllm.models.deepseek_v41.amd.dspark_markov_fused_gemv import (
    fused_markov_bias_add,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernel runs on GPU"
)

_KILL_VAR = "VLLM_DISABLE_MARKOV_FUSED_GEMV"
_GATE_VAR = "VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS"

# The serving shape (DSV4.1-Flash): markov_w2 is [129280, 256] bf16
# replicated on every rank; B = num_reqs, n_spec = 5.
_SERVING_V, SERVING_R = 129280, 256


@pytest.fixture(autouse=True)
def _clean_env():
    os.environ.pop(_KILL_VAR, None)
    # The numerics/eligibility tests exercise the kernel itself: lift the
    # draft-batch gate so every B engages. The gate policy (the measured
    # crossover between the fused chain and the stock GEMM) has its own
    # dedicated test below.
    os.environ[_GATE_VAR] = "1024"
    yield
    os.environ.pop(_KILL_VAR, None)
    os.environ.pop(_GATE_VAR, None)


def _stock_chain(prev, w1, w2, base_slice):
    """Exactly the speculator's stock ops (gather + bf16 GEMM + add)."""
    markov_embed = torch.nn.functional.embedding(prev, w1)
    bias = markov_embed @ w2.t()
    return base_slice + bias


def _make_case(V, R, B, n_spec, dtype, seed, id_dtype):
    torch.manual_seed(seed)
    device = "cuda"
    w1 = (torch.randn(V, R, device=device) * 0.02).to(dtype)
    w2 = (torch.randn(V, R, device=device) * 0.02).to(dtype)
    base = (torch.randn(B, n_spec, V, device=device) * 0.5).to(dtype)
    prev = torch.randint(0, V, (B,), dtype=id_dtype, device=device)
    return w1, w2, base, prev


def _compare_fused_to_stock(prev, w1, w2, base_slice, tag):
    stock = _stock_chain(prev, w1, w2, base_slice)
    fused = fused_markov_bias_add(prev, w1, w2, base_slice)
    assert fused is not None, f"{tag}: unexpectedly fell back"
    torch.accelerator.synchronize()
    assert fused.shape == stock.shape and fused.dtype == stock.dtype
    max_ulp = int(bf16_ulp_distance(fused, stock).max().item())
    assert max_ulp <= 2, f"{tag}: fused deviates from stock by {max_ulp} ULP"
    # Selection stability: identical argmax except where the top-2 logits
    # are within 8 ULP (a rounding-level tie).
    top2 = stock.topk(2, dim=-1)
    gap_ulp = bf16_ulp_distance(top2.values[..., 0:1], top2.values[..., 1:2]).squeeze(
        -1
    )
    mism = (fused.argmax(-1) != stock.argmax(-1)) & ~(gap_ulp <= 8)
    assert not mism.any(), (
        f"{tag}: {int(mism.sum())} rows flipped argmax beyond the 8-ULP tie band"
    )
    return fused, stock


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_matches_stock_chain_serving_shape(B, id_dtype):
    """Serving shape (V=129280, R=256, n_spec=5), base logits as the
    strided [B, n_spec, V][:, 2] view the speculator passes."""
    w1, w2, base, prev = _make_case(
        _SERVING_V, SERVING_R, B, 5, torch.bfloat16, 11, id_dtype
    )
    _compare_fused_to_stock(prev, w1, w2, base[:, 2], "serving")


@pytest.mark.parametrize("V,R", [(257, 130), (64, 8)])
def test_matches_stock_chain_masked_shapes(V, R):
    """Non-power-of-2 / sub-block V and R exercise the load masks."""
    w1, w2, base, prev = _make_case(V, R, 4, 5, torch.bfloat16, 12, torch.int64)
    _compare_fused_to_stock(prev, w1, w2, base[:, 0], f"V={V},R={R}")


def test_fp64_reference_bound():
    """Both paths implement "round the dot to bf16, add to the base in fp32,
    round once" -- the stock chain's semantics, which the fused kernel
    copies verbatim. Against the EXACT fp64 dot, both must agree with
    round_bf16(base + round_bf16(dot_fp64)) within 1 bf16 ULP. (A plain
    single-rounded fp64 reference is the WRONG bound: the intermediate bias
    rounding shifts near-cancellation outputs by many ULPs in BOTH paths --
    that is the stock numerics the fused kernel must preserve.)"""
    w1, w2, base, prev = _make_case(4096, 256, 4, 5, torch.bfloat16, 13, torch.int64)
    base_slice = base[:, 3]
    dot_ref = torch.nn.functional.embedding(prev, w1).double() @ w2.t().double()
    expected = (base_slice.float() + dot_ref.to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    stock = _stock_chain(prev, w1, w2, base_slice)
    fused = fused_markov_bias_add(prev, w1, w2, base_slice)
    torch.accelerator.synchronize()
    for name, got in (("stock", stock), ("fused", fused)):
        max_ulp = int(bf16_ulp_distance(got, expected).max().item())
        assert max_ulp <= 1, (
            f"{name} deviates from the fp64-grounded chain semantics by {max_ulp} ULP"
        )


def test_deterministic():
    """Fixed-order fp32 accumulation, no atomics: repeated runs bitwise
    identical."""
    w1, w2, base, prev = _make_case(
        _SERVING_V, SERVING_R, 2, 5, torch.bfloat16, 14, torch.int64
    )
    a = fused_markov_bias_add(prev, w1, w2, base[:, 1])
    b = fused_markov_bias_add(prev, w1, w2, base[:, 1])
    torch.accelerator.synchronize()
    assert torch.equal(a, b)


def test_env_kill_switch_falls_back():
    os.environ[_KILL_VAR] = "1"
    w1, w2, base, prev = _make_case(512, 64, 2, 5, torch.bfloat16, 15, torch.int64)
    assert fused_markov_bias_add(prev, w1, w2, base[:, 0]) is None


def test_draft_batch_gate_falls_back():
    """The fusion engages at num draft rows <=
    VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS (default 2 -- the measured
    crossover between launch-elimination wins and the stock GEMM's tiled
    weight reuse); larger batches keep the stock chain, a raised gate
    re-engages."""
    os.environ.pop(_GATE_VAR)  # restore the default (2)
    w1, w2, base, prev = _make_case(512, 64, 3, 5, torch.bfloat16, 16, torch.int64)
    # Batch 3 > default gate 2: stock chain.
    assert fused_markov_bias_add(prev, w1, w2, base[:, 0]) is None
    # Batch 2 <= gate: fused.
    out = fused_markov_bias_add(prev[:2], w1, w2, base[:2, 0])
    assert out is not None and out.shape == (2, 512)
    # Raised gate: batch 3 fuses again.
    os.environ[_GATE_VAR] = "8"
    out = fused_markov_bias_add(prev, w1, w2, base[:, 0])
    assert out is not None and out.shape == (3, 512)


@pytest.mark.parametrize(
    "mutate",
    [
        "fp32_weights",
        "noncontig_w2",
        "vocab_mismatch",
    ],
)
def test_ineligible_shapes_fall_back(mutate):
    w1, w2, base, prev = _make_case(512, 64, 2, 5, torch.bfloat16, 16, torch.int64)
    base_slice = base[:, 0]
    if mutate == "fp32_weights":
        w1 = w1.float()
        w2 = w2.float()
    elif mutate == "noncontig_w2":
        # [V, R] view with column-major layout: stride(-1) != 1.
        w2 = w2.t().contiguous().t()
    elif mutate == "vocab_mismatch":
        base_slice = base_slice[:, :500]
    assert fused_markov_bias_add(prev, w1, w2, base_slice) is None, (
        f"{mutate} should fall back to the stock chain"
    )


def test_graph_capture_replay():
    """The chain is captured inside the DSpark FULL draft CUDA graph: the
    fused call must capture and replay correctly on changed inputs."""
    V, R, B, n_spec = 8192, 256, 2, 5
    w1, w2, base, prev = _make_case(V, R, B, n_spec, torch.bfloat16, 17, torch.int64)
    base_s = base.clone()
    prev_s = prev.clone()
    base_slice_s = base_s[:, 4]

    out = fused_markov_bias_add(prev_s, w1, w2, base_slice_s)  # warmup/JIT
    torch.accelerator.synchronize()
    assert out is not None

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = fused_markov_bias_add(prev_s, w1, w2, base_slice_s)
    assert out_g is not None

    def eager():
        return fused_markov_bias_add(prev_s, w1, w2, base_s[:, 4])

    # Replay 1: unchanged inputs.
    g.replay()
    torch.accelerator.synchronize()
    assert torch.equal(out_g, eager())

    # Replay 2: changed inputs must flow through (static-buffer semantics).
    torch.manual_seed(18)
    prev_s.copy_(torch.randint(0, V, (B,), device="cuda", dtype=torch.int64))
    base_s.copy_(torch.randn_like(base_s))
    g.replay()
    torch.accelerator.synchronize()
    assert torch.equal(out_g, eager())
