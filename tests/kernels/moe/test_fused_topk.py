# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE fused topk kernel.

Run `pytest tests/kernels/moe/test_fused_topk.py`.
"""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform


def torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.float(), dim=-1)
    else:
        assert scoring_func == "sigmoid"
        scores = torch.sigmoid(gating_output.float())

    if e_score_correction_bias is not None:
        num_experts = gating_output.shape[-1]
        scores_for_choice = scores.view(
            -1, num_experts
        ) + e_score_correction_bias.unsqueeze(0)
        _, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16, 352])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_bias(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_experts", [6, 8, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_nan_inf_clamp(
    num_experts: int,
    topk: int,
    scoring_func: str,
    bad_value: float,
    dtype: torch.dtype,
):
    """Regression test for the NaN/Inf clamp in topk_softmax_kernels.cu.

    Degenerate hidden states (e.g., from CUDA graph padding) can produce
    NaN/Inf gating logits. Without the clamp, softmax/sigmoid outputs are
    NaN and the argmax loop picks expert 0 for every top-k slot (since
    "NaN > NaN" is false per IEEE 754), yielding duplicate expert IDs that
    crash downstream MoE sort kernels. The fix clamps NaN/Inf to 0 before
    argmax so index tie-breaking selects unique experts [0, 1, ..., k-1].
    """
    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")

    # Row 0: all normal. Rows 1-3: fully poisoned with NaN or Inf.
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    gating_output[1:, :] = bad_value

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=False,
        scoring_func=scoring_func,
    )

    # Normal row must still match the torch reference.
    ref_weights, ref_ids = torch_topk(
        gating_output=gating_output[:1],
        topk=topk,
        renormalize=False,
        scoring_func=scoring_func,
    )
    torch.testing.assert_close(
        ref_weights.to(torch.float32), topk_weights[:1], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(ref_ids.to(torch.int32), topk_ids[:1], atol=0, rtol=0)

    # Poisoned rows: IDs must be unique (no duplicates) and weights must be
    # finite (no NaN/Inf propagation into downstream MoE kernels).
    for row in range(1, num_tokens):
        row_ids = topk_ids[row]
        assert row_ids.unique().numel() == topk, (
            f"Row {row} has duplicate expert IDs {row_ids.tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )
        assert torch.isfinite(topk_weights[row]).all(), (
            f"Row {row} has non-finite weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_experts", [6, 8, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_bias_nan_inf_clamp(
    num_experts: int,
    topk: int,
    scoring_func: str,
    bad_value: float,
    dtype: torch.dtype,
):
    """Regression test: NaN/Inf in gating logits must not produce duplicate
    expert IDs or non-finite weights when e_score_correction_bias is present.

    Same scenario as test_fused_topk_nan_inf_clamp but exercising the bias
    path (fused_topk_bias) so the fix in topk_softmax_kernels.cu is covered
    for that entry point as well.
    """
    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    gating_output[1:, :] = bad_value

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=False,
        scoring_func=scoring_func,
    )

    # Normal row must still match the torch reference.
    ref_weights, ref_ids = torch_topk(
        gating_output=gating_output[:1],
        topk=topk,
        renormalize=False,
        e_score_correction_bias=e_score_correction_bias,
        scoring_func=scoring_func,
    )
    torch.testing.assert_close(
        ref_weights.to(torch.float32), topk_weights[:1], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(ref_ids.to(torch.int32), topk_ids[:1], atol=0, rtol=0)

    # Poisoned rows: IDs must be unique (no duplicates) and weights must be
    # finite (no NaN/Inf propagation into downstream MoE kernels).
    for row in range(1, num_tokens):
        row_ids = topk_ids[row]
        assert row_ids.unique().numel() == topk, (
            f"Row {row} has duplicate expert IDs {row_ids.tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )
        assert torch.isfinite(topk_weights[row]).all(), (
            f"Row {row} has non-finite weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value}, scoring_func={scoring_func})"
        )


def _tie_heavy_logits(
    num_tokens: int, num_experts: int, dtype: torch.dtype
) -> torch.Tensor:
    """Small integer logits, so most scores tie exactly with other experts.

    Row 0 is constant (every expert ties) and, with more than one row,
    row 1 is NaN (every score is clamped to 0 and ties).
    """
    logits = torch.randint(-2, 3, (num_tokens, num_experts), device="cuda").to(dtype)
    logits[0] = 1.0
    if num_tokens > 1:
        logits[1] = float("nan")
    return logits


def _run_topk_kernel(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    device = gating_output.device
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=device)
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    token_expert_indices = torch.empty_like(topk_ids)
    kernel = ops.topk_softmax if scoring_func == "softmax" else ops.topk_sigmoid
    kernel(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias=bias,
    )
    return topk_weights, topk_ids, token_expert_indices


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 2048])
@pytest.mark.parametrize("num_experts", [128, 256, 512])
@pytest.mark.parametrize("topk", [8, 10])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("logits_kind", ["normal", "ties"])
def test_fused_topk_without_bias_matches_bias_path(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
    logits_kind: str,
):
    """Rows that one full warp owns select without a bias by redux.sync on
    SM80+. A zero bias gives the same choice scores but keeps the shuffle
    butterfly, so the two launches must agree bit for bit, including the
    lowest-index tie rule and the renormalized weights.
    """
    torch.manual_seed(0)
    if logits_kind == "ties":
        gating_output = _tie_heavy_logits(num_tokens, num_experts, dtype)
    else:
        gating_output = torch.randn(
            (num_tokens, num_experts), dtype=dtype, device="cuda"
        )
    zero_bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")

    weights, ids, token_expert_indices = _run_topk_kernel(
        gating_output, topk, renormalize, scoring_func, bias=None
    )
    ref_weights, ref_ids, ref_token_expert_indices = _run_topk_kernel(
        gating_output, topk, renormalize, scoring_func, bias=zero_bias
    )

    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
    assert torch.equal(token_expert_indices, ref_token_expert_indices)

    if logits_kind == "ties":
        # Equal logits give equal scores, so a stable sort fixes the tie order.
        scores = (
            torch.softmax(gating_output.float(), dim=-1)
            if scoring_func == "softmax"
            else torch.sigmoid(gating_output.float())
        )
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0)
        expected_ids = torch.sort(scores, dim=-1, descending=True, stable=True)[1]
        assert torch.equal(ids, expected_ids[:, :topk].to(torch.int32))
