# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rowwise-fp8 DSpark draft lm_head (VLLM_DSPARK_FP8_DRAFT_HEAD).

CPU tests cover the quantize helper (roundtrip error bound) and draft argmax
agreement between the bf16 reference path and the fp8 path, using a float32
emulation of torch._scaled_mm. The GPU test additionally checks the real
_scaled_mm kernel against the emulation and the bf16 reference.
"""

import pytest
import torch

from vllm.model_executor.layers.fp8_draft_head import (
    _FP8_MAX,
    Fp8DraftHead,
    fp8_draft_head_logits,
    fp8_draft_head_supported,
    quantize_draft_head,
)

VOCAB = 2048
HIDDEN = 256
NUM_TOKENS = 64

# float8_e4m3fn has a 3-bit mantissa: worst-case relative rounding error of a
# representable-range value is 2^-4 of its binade, i.e. <= (1/16) * value and
# <= (32/448) * rowmax absolute (half ulp of the top binade after rowwise
# scaling to [-448, 448]).
_ROUNDTRIP_REL_BOUND = 32.0 / 448.0 / 2.0  # half ulp at the top binade


def _dequant(head: Fp8DraftHead) -> torch.Tensor:
    # row_scale is [1, num_rows] laid out for the GEMM epilogue; transpose to
    # dequantize the [num_rows, hidden] weight.
    return head.weight_fp8.float() * head.row_scale.float().t()


def _emulated_fp8_logits(x: torch.Tensor, head: Fp8DraftHead) -> torch.Tensor:
    """Float32 emulation of fp8_draft_head_logits (no _scaled_mm needed)."""
    act_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    act_fp8 = (x * (_FP8_MAX / act_max)).to(torch.float8_e4m3fn)
    out = act_fp8.float() @ head.weight_fp8.float().t()
    out = out.to(x.dtype)
    out = out * head.row_scale
    out = out * (act_max / _FP8_MAX).to(x.dtype)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rowwise_quant_roundtrip_error_bound(dtype: torch.dtype):
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, HIDDEN, dtype=dtype) * 0.05
    # Give rows wildly different magnitudes to exercise the rowwise scales.
    weight *= torch.logspace(-3, 1, VOCAB, dtype=dtype).unsqueeze(1)

    head = quantize_draft_head(weight)
    assert head.weight_fp8.dtype == torch.float8_e4m3fn
    assert head.weight_fp8.shape == weight.shape
    assert head.row_scale.shape == (1, VOCAB)

    err = (weight.float() - _dequant(head)).abs()
    row_max = weight.float().abs().amax(dim=1, keepdim=True)
    # Small slack over the analytic half-ulp bound for the two roundings
    # (scale multiply in fp32, then fp8 cast).
    bound = row_max * (_ROUNDTRIP_REL_BOUND * 1.25) + 1e-8
    assert (err <= bound).all(), (
        f"rowwise fp8 roundtrip error {err.max().item()} exceeds bound"
    )


def test_draft_argmax_agreement_bf16_vs_fp8_emulated():
    """The fp8 draft head must (almost) always agree with bf16 on argmax.

    Uses the float32 emulation of _scaled_mm; disagreements are only
    acceptable when the bf16 top-2 margin is within the fp8 error scale.
    """
    torch.manual_seed(1234)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)

    ref_logits = hidden.float() @ weight.float().t()
    ref_argmax = ref_logits.argmax(dim=-1)

    head = quantize_draft_head(weight)
    fp8_logits = _emulated_fp8_logits(hidden, head)
    fp8_argmax = fp8_logits.argmax(dim=-1)

    # Primary correctness property: any disagreement must be a near-tie in
    # the reference logits, i.e. the kind of flip that costs at most one
    # rejected draft token during verification, never a wrong output.
    top2 = ref_logits.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).abs()
    err_scale = (
        ref_logits.abs().amax(dim=-1) * 2.0 * _ROUNDTRIP_REL_BOUND * 2.0
    )
    disagree = ref_argmax != fp8_argmax
    assert (margin[disagree] <= err_scale[disagree]).all(), (
        "fp8 draft head flipped an argmax with a large top-2 margin"
    )

    # I.i.d. Gaussian logits are the worst case for argmax stability (the
    # top-2 gap of 2048 i.i.d. samples is tiny); real LM logits have much
    # larger top-1 margins, where the fp8 head measured argmax-identical.
    # Even in this adversarial regime agreement must stay high.
    agree = (ref_argmax == fp8_argmax).float().mean().item()
    assert agree >= 0.90, f"draft argmax agreement too low: {agree:.4f}"


def test_emulated_logits_close_to_reference():
    torch.manual_seed(7)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)

    ref = (hidden.float() @ weight.float().t()).float()
    head = quantize_draft_head(weight)
    fp8 = _emulated_fp8_logits(hidden, head).float()

    scale = ref.abs().amax()
    assert (fp8 - ref).abs().max() <= 0.05 * scale


@pytest.mark.skipif(
    not fp8_draft_head_supported(),
    reason="requires a CUDA device with fp8 support (SM89+)",
)
def test_fp8_draft_head_logits_cuda_matches_emulation():
    torch.manual_seed(42)
    device = torch.device("cuda")
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=device)
    weight *= 0.02
    hidden = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device
    )

    head = quantize_draft_head(weight)
    real = fp8_draft_head_logits(hidden, head).float()
    emulated = _emulated_fp8_logits(hidden, head).float()

    # _scaled_mm accumulates in fp32 like the emulation; only the final
    # bf16 rounding differs.
    scale = emulated.abs().amax()
    assert (real - emulated).abs().max() <= 0.01 * scale

    # And the real kernel's argmax agrees with the bf16 reference head
    # (i.i.d. Gaussian logits are the worst case for argmax stability; see
    # test_draft_argmax_agreement_bf16_vs_fp8_emulated).
    ref_argmax = (hidden.float() @ weight.float().t()).argmax(dim=-1)
    agree = (real.argmax(dim=-1) == ref_argmax).float().mean().item()
    assert agree >= 0.90
