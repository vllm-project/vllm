# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE fused topk kernel

Run `pytest tests/kernels/moe/test_fused_topk.py`.
"""

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    _use_a100_small_topk,
    fused_topk,
)
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


def current_topk_reference(
    gating_output: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    weights = torch.empty((num_tokens, 4), dtype=torch.float32, device="cuda")
    ids = torch.empty((num_tokens, 4), dtype=torch.int32, device="cuda")
    source_rows = torch.empty_like(ids)
    ops.topk_softmax(
        weights,
        ids,
        source_rows,
        gating_output,
        False,
        is_padding=is_padding,
    )
    return weights, ids, source_rows


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0)),
    reason="The specialized kernel is restricted to SM80.",
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("seed", [0, 17, 2026])
def test_a100_small_topk_bitwise(num_tokens: int, seed: int):
    torch.manual_seed(seed)
    hidden_states = torch.empty((num_tokens, 1), dtype=torch.bfloat16, device="cuda")
    gating_output = torch.randn((num_tokens, 60), dtype=torch.bfloat16, device="cuda")
    reference = current_topk_reference(gating_output)
    actual = fused_topk(hidden_states, gating_output, 4, False)
    assert all(torch.equal(ref, out) for ref, out in zip(reference, actual))


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0)),
    reason="The specialized kernel is restricted to SM80.",
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64])
def test_a100_small_topk_ties(num_tokens: int):
    hidden_states = torch.empty((num_tokens, 1), dtype=torch.bfloat16, device="cuda")
    gating_output = torch.zeros((num_tokens, 60), dtype=torch.bfloat16, device="cuda")
    reference = current_topk_reference(gating_output)
    actual = fused_topk(hidden_states, gating_output, 4, False)
    assert all(torch.equal(ref, out) for ref, out in zip(reference, actual))


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0)),
    reason="The specialized kernel is restricted to SM80.",
)
def test_a100_small_topk_padding_and_opcheck():
    gating_output = torch.randn((8, 60), dtype=torch.bfloat16, device="cuda")
    is_padding = torch.tensor(
        [False, True, False, False, True, False, False, True], device="cuda"
    )
    reference = current_topk_reference(gating_output, is_padding)
    actual = tuple(torch.empty_like(tensor) for tensor in reference)
    ops.topk_softmax_a100(*actual, gating_output, is_padding)
    assert all(torch.equal(ref, out) for ref, out in zip(reference, actual))
    opcheck(
        torch.ops._moe_C.topk_softmax_a100,
        (*actual, gating_output, is_padding),
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0)),
    reason="The specialized kernel is restricted to SM80.",
)
def test_a100_small_topk_dispatch_scope():
    gating_output = torch.empty((8, 60), dtype=torch.bfloat16, device="cuda")
    ids = torch.empty((8, 4), dtype=torch.int32, device="cuda")
    assert _use_a100_small_topk(gating_output, ids, False)
    assert not _use_a100_small_topk(gating_output.half(), ids, False)
    assert not _use_a100_small_topk(gating_output.T.contiguous().T, ids, False)
    assert not _use_a100_small_topk(gating_output[:, :59], ids, False)
    assert not _use_a100_small_topk(gating_output.repeat(9, 1), ids.repeat(9, 1), False)
    assert not _use_a100_small_topk(gating_output, ids[:, :3], False)
    assert not _use_a100_small_topk(gating_output, ids, True)


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0)),
    reason="The specialized kernel is restricted to SM80.",
)
def test_a100_small_topk_cuda_graph():
    hidden_states = torch.empty((16, 1), dtype=torch.bfloat16, device="cuda")
    gating_output = torch.randn((16, 60), dtype=torch.bfloat16, device="cuda")
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            fused_topk(hidden_states, gating_output, 4, False)
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_topk(hidden_states, gating_output, 4, False)
    reference = current_topk_reference(gating_output)
    graph.replay()
    torch.accelerator.synchronize()
    assert all(torch.equal(ref, out) for ref, out in zip(reference, captured))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
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
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16])
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
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
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
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
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
