# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE grouped topk kernel

Run `pytest tests/kernels/moe/test_grouped_topk.py`.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
    fused_grouped_topk,
    grouped_topk,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _run_single_group_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_grouped_topk(
        hidden_states=torch.empty(
            (logits.shape[0], 0), dtype=logits.dtype, device=logits.device
        ),
        gating_output=logits,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=bias,
        num_expert_group=1,
        topk_group=1,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
    )


def _single_group_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "sigmoid":
        scores = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    else:
        scores = torch.softmax(logits, dim=-1).float()
    indices = torch.argsort(
        scores + bias.float(), dim=-1, descending=True, stable=True
    )[:, :topk]
    values = scores.gather(1, indices)
    if renormalize:
        values /= values.sum(dim=-1, keepdim=True) + 1e-20
    values *= routed_scaling_factor
    return values, indices.to(torch.int32)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024, 2048])
@pytest.mark.parametrize(
    "n_expert,topk,num_expert_group,topk_group",
    [
        (16, 2, 8, 2),
        (128, 2, 8, 2),
        (256, 8, 8, 4),
        (384, 8, 1, 1),
        (512, 22, 1, 1),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float32])
def test_grouped_topk(
    monkeypatch: pytest.MonkeyPatch,
    n_token: int,
    n_hidden: int,
    n_expert: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+grouped_topk"])
    )
    get_cached_compilation_config.cache_clear()

    set_random_seed(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=input_dtype, device="cuda")
    gating_output = torch.randn((n_token, n_expert), dtype=input_dtype, device="cuda")
    e_score_correction_bias = torch.randn((n_expert,), dtype=bias_dtype, device="cuda")

    with set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
        m.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        grouped_topk = GroupedTopk(
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )
        assert grouped_topk._forward_method.__name__ == "forward_cuda"
        baseline_topk_weights, baseline_topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            e_score_correction_bias=e_score_correction_bias,
        )

        test_topk_weights, test_topk_ids = fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.testing.assert_close(
            baseline_topk_weights, test_topk_weights, atol=2e-2, rtol=0
        )
        torch.testing.assert_close(baseline_topk_ids, test_topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_grouped_topk_single_group_large_batch():
    set_random_seed(0)
    logits = torch.randn((1536, 896), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((896,), dtype=torch.float32, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,input_dtype,bias_dtype",
    [
        (512, 9, torch.bfloat16, torch.float32),
        (512, 16, torch.float16, torch.float16),
        (513, 9, torch.float32, torch.bfloat16),
        (513, 16, torch.bfloat16, torch.float32),
        (895, 9, torch.float16, torch.bfloat16),
        (896, 16, torch.float32, torch.float16),
        (897, 9, torch.bfloat16, torch.bfloat16),
        (897, 16, torch.float16, torch.float32),
        (1024, 9, torch.float32, torch.bfloat16),
        (1024, 16, torch.bfloat16, torch.float16),
    ],
)
@pytest.mark.parametrize(
    "scoring_func,renormalize,routed_scaling_factor",
    [
        ("sigmoid", True, 1.0),
        ("sigmoid", False, 2.5),
        ("softmax", True, 2.5),
        ("softmax", False, 1.0),
    ],
)
def test_grouped_topk_single_group_tiers(
    num_experts: int,
    topk: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
):
    set_random_seed(7)
    logits = torch.randn((17, num_experts), dtype=input_dtype, device="cuda")
    bias = torch.randn((num_experts,), dtype=bias_dtype, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,scoring_func",
    [
        (128, 8, "sigmoid"),
        (129, 8, "sigmoid"),
        (257, 8, "sigmoid"),
        (385, 8, "sigmoid"),
        (512, 9, "sigmoid"),
        (513, 9, "sigmoid"),
        (769, 9, "sigmoid"),
        (897, 16, "sigmoid"),
        (1024, 16, "sigmoid"),
        (128, 4, "softmax"),
        (128, 5, "softmax"),
        (129, 8, "softmax"),
        (161, 8, "softmax"),
        (256, 9, "softmax"),
        (257, 8, "softmax"),
        (512, 9, "softmax"),
        (512, 17, "softmax"),
        (512, 23, "softmax"),
        (513, 8, "softmax"),
        (577, 9, "softmax"),
        (769, 9, "softmax"),
        (897, 9, "softmax"),
        (1024, 16, "softmax"),
    ],
)
def test_grouped_topk_single_group_capacity_tiers(
    num_experts: int,
    topk: int,
    scoring_func: str,
):
    set_random_seed(11)
    logits = torch.randn((3, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
def test_grouped_topk_single_group_stable_ties(num_experts: int):
    logits = torch.zeros((1, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
    expected_values = torch.full((1, 16), 2.5 / 16, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
@pytest.mark.parametrize("num_finite", [0, 15])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_single_group_nonfinite_scores(
    num_experts: int, num_finite: int, renormalize: bool
):
    logits = torch.full(
        (1, num_experts), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    if num_finite:
        logits[0, :num_finite] = torch.arange(
            num_finite, dtype=torch.bfloat16, device="cuda"
        )
    logits[0, num_finite] = torch.inf
    logits[0, num_finite + 1] = -torch.inf
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
    )

    if num_finite == 0:
        expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
        if renormalize:
            expected_values = torch.full(
                (1, 16), 1 / 16, dtype=torch.float32, device="cuda"
            )
        else:
            expected_values = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    else:
        expected_ids = torch.cat(
            (
                torch.arange(num_finite - 1, -1, -1, dtype=torch.int32, device="cuda"),
                torch.tensor([num_finite], dtype=torch.int32, device="cuda"),
            )
        )[None]
        finite_values = logits[0, :num_finite].float().sigmoid().flip(0)
        if renormalize:
            finite_values /= finite_values.sum()
        finite_values *= 2.5
        expected_values = torch.cat(
            (finite_values, torch.zeros(1, dtype=torch.float32, device="cuda"))
        )[None]

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


# ---------------------------------------------------------------------------
# Determinism of the pure-PyTorch fallback (`grouped_topk`)
#
# Everything above exercises the fused CUDA kernel and is therefore skipped on
# non-CUDA platforms. The tests below deliberately are NOT skipped: the code
# under test is the Python fallback, which is the live path on ROCm (whenever
# AITER MoE is off) and is also reached on CUDA whenever the fused kernel is
# not taken -- `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0`, `e_score_correction_bias
# is None`, or a shape outside the kernel's tier table.
#
# The fallback used to select with `torch.topk(..., sorted=False)`, and
# `sorted=False` licenses the backend to return the k results in any order.
# Above 256 columns `torch.topk` takes a multi-pass path whose order is not
# merely unsorted but differs between calls on identical input. Measured on
# gfx1151 / torch 2.11+rocm10.0, 20 identical `torch.topk(x, k=8,
# sorted=False)` calls on a 64xE tensor gave 1 distinct result for E <= 256
# and 20 distinct for E >= 257, with the same split for any k >= 4.
#
# Hence _DET_NUM_EXPERTS = 288 below: at E <= 256 -- which includes
# DeepSeek-V3/R1's 256, sitting exactly on the safe side of the boundary --
# the stock implementation happens to return sorted, stable output and these
# tests pass against unfixed code.
# ---------------------------------------------------------------------------

# The fallback is device-agnostic; run on whatever the runner has. (On ROCm
# builds torch.cuda.is_available() is True and "cuda" is the right device
# string -- HIP presents itself through the CUDA API.)
_DET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DET_NUM_REPEAT = 20
_DET_NUM_EXPERTS = 288
_DET_TOPK = 8
_DET_NUM_TOKENS = 64


def _run_python_grouped_topk(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    topk: int,
    *,
    num_expert_group: int = 1,
    topk_group: int = 1,
    scoring_func: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the Python fallback exactly as the router does when the fused
    kernel is not taken."""
    return grouped_topk(
        hidden_states=torch.empty(
            (logits.shape[0], 0), dtype=logits.dtype, device=logits.device
        ),
        gating_output=logits,
        topk=topk,
        renormalize=False,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=bias,
    )


def _det_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    return torch.softmax(logits, dim=-1)


def _det_biased_scores(
    logits: torch.Tensor, bias: torch.Tensor | None, scoring_func: str
) -> torch.Tensor:
    scores = _det_scores(logits, scoring_func)
    if bias is not None:
        scores = scores + bias.unsqueeze(0)
    return scores


def _force_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the Python path on every platform (including CUDA), and make it
    # explicit that determinism must not require VLLM_BATCH_INVARIANT, which
    # is documented as NVIDIA SM90+ only and so cannot be enabled on every
    # stack that reaches this code.
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")


def _det_random_inputs(
    bias_is_none: bool, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor | None]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(
        _DET_NUM_TOKENS, _DET_NUM_EXPERTS, generator=gen, dtype=torch.float32
    ).to(_DET_DEVICE)
    bias = (
        None
        if bias_is_none
        else torch.randn(_DET_NUM_EXPERTS, generator=gen, dtype=torch.float32).to(
            _DET_DEVICE
        )
    )
    return logits, bias


def _assert_all_identical(
    results: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    first_w, first_i = results[0]
    for n, (w, i) in enumerate(results[1:], start=1):
        assert torch.equal(i, first_i), (
            f"expert ids differ between call 0 and call {n}: "
            f"{(i != first_i).sum().item()} of {i.numel()} positions"
        )
        # Bitwise, not merely value-wise: this also catches a -0.0/0.0 flip
        # and any reordering among equal weights.
        assert w.view(torch.int32).equal(first_w.view(torch.int32)), (
            f"routing weights differ between call 0 and call {n}"
        )
    return first_w, first_i


def _make_tie_logits(
    num_experts: int, k: int, tie_lo: int, tie_hi: int
) -> torch.Tensor:
    """One row with a deliberate, bitwise-exact tie at the k-boundary.

    Experts ``0 .. k-2`` get well-separated descending logits, experts
    ``tie_lo`` (= k-1) and ``tie_hi`` get the *same* logit value, and every
    other expert sits strictly below the pair. Because the tied pair has
    identical input bits and the scoring activation is elementwise, the two
    biased scores are bitwise-equal on every backend, eager or compiled --
    the tie is constructed, not a rounding accident. The pair occupies
    positions k-1 and k of the value-descending order, i.e. the last selected
    slot and the first dropped one.
    """
    assert tie_lo == k - 1 and tie_hi > tie_lo and tie_hi < num_experts
    logits = torch.zeros(num_experts, dtype=torch.float32)
    logits[: k - 1] = torch.linspace(8.0, 4.0, k - 1)
    logits[tie_lo] = 3.0
    logits[tie_hi] = 3.0
    rest = torch.ones(num_experts, dtype=torch.bool)
    rest[: k - 1] = False
    rest[tie_lo] = False
    rest[tie_hi] = False
    logits[rest] = torch.linspace(2.9, 0.1, int(rest.sum()))
    return logits


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("bias_is_none", [False, True])
@pytest.mark.parametrize(
    ("num_expert_group", "topk_group"), [(1, 1), (8, 4)], ids=["1grp", "8grp"]
)
def test_grouped_topk_repeat_determinism(
    monkeypatch: pytest.MonkeyPatch,
    scoring_func: str,
    bias_is_none: bool,
    num_expert_group: int,
    topk_group: int,
):
    """Identical inputs must give bitwise-identical routing.

    No tie is needed: with ``sorted=False`` the fallback returned the same k
    experts in a *different order* on every call once E > 256, which permutes
    the routing weights and so changes the order the expert outputs are summed
    in downstream.
    """
    _force_python_fallback(monkeypatch)
    logits, bias = _det_random_inputs(bias_is_none)

    # A fresh clone per call so the input buffer address varies, as it does in
    # a real forward.
    results = [
        _run_python_grouped_topk(
            logits.clone(),
            bias,
            _DET_TOPK,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
        )
        for _ in range(_DET_NUM_REPEAT)
    ]
    _assert_all_identical(results)


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("bias_is_none", [False, True])
def test_grouped_topk_returns_value_descending_order(
    monkeypatch: pytest.MonkeyPatch, scoring_func: str, bias_is_none: bool
):
    """The selected experts must come back in descending score order.

    This is the order the fused kernel already produces -- ``moeTopKFuncs.cuh``
    packs ``65535 - idx`` into the comparison key and the multi-group path uses
    ``WarpSelect<..., is_stable=true>`` -- and the order
    ``_single_group_reference`` above already assumes, via a stable descending
    ``argsort``. It is a single-call assertion, so it cannot be flaky.
    """
    _force_python_fallback(monkeypatch)
    logits, bias = _det_random_inputs(bias_is_none)
    biased = _det_biased_scores(logits, bias, scoring_func)

    _, topk_ids = _run_python_grouped_topk(
        logits, bias, _DET_TOPK, scoring_func=scoring_func
    )

    selected = biased.gather(1, topk_ids.to(torch.long))
    bad = (selected[:, :-1] < selected[:, 1:]).any(dim=1)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {bad.numel()} rows are not in descending score "
        f"order; first offending row {int(bad.nonzero()[0])}: "
        f"{selected[int(bad.nonzero()[0])].tolist()}"
    )

    # Same experts as the reference: on tie-free input this order is exactly
    # what topk(..., sorted=True) already returns, i.e. pinning the order does
    # not change the selection.
    ref_ids = biased.topk(_DET_TOPK, dim=-1, sorted=True)[1].to(torch.int32)
    torch.testing.assert_close(topk_ids, ref_ids)


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("bias_is_none", [False, True])
def test_grouped_topk_tie_broken_by_lower_expert_index(
    monkeypatch: pytest.MonkeyPatch, scoring_func: str, bias_is_none: bool
):
    """An exact tie at the k-boundary is resolved by the lower expert index.

    This is the same contract as ``test_grouped_topk_single_group_stable_ties``
    asserts for the fused kernel, but for the fallback. Such ties do occur in
    practice: a live 740x288 router score tensor from a GLM-5.3-Flash prefill
    had k-boundary ties on 2 of 740 rows, between experts whose logits *and*
    biases both differ but whose fp32 sums round to the same value.
    """
    _force_python_fallback(monkeypatch)

    tie_lo, tie_hi = _DET_TOPK - 1, 200
    logits = _make_tie_logits(_DET_NUM_EXPERTS, _DET_TOPK, tie_lo, tie_hi)[None].to(
        _DET_DEVICE
    )
    bias = None if bias_is_none else torch.zeros(_DET_NUM_EXPERTS, device=_DET_DEVICE)

    biased = _det_biased_scores(logits, bias, scoring_func)
    # Self-validate the construction before asserting anything about the op.
    tie_val = biased[0, tie_lo]
    assert biased[0, tie_lo].item() == biased[0, tie_hi].item()
    assert int((biased[0] > tie_val).sum()) == _DET_TOPK - 1
    assert int((biased[0] == tie_val).sum()) == 2

    results = [
        _run_python_grouped_topk(
            logits.clone(), bias, _DET_TOPK, scoring_func=scoring_func
        )
        for _ in range(_DET_NUM_REPEAT)
    ]
    first_w, first_i = _assert_all_identical(results)

    assert int(first_i[0, _DET_TOPK - 1]) == tie_lo
    assert tie_hi not in first_i[0].tolist()

    # The full selection is the value-descending one: experts 0..k-2 by
    # construction, then the tie winner tie_lo = k-1.
    expected_ids = torch.arange(_DET_TOPK, dtype=torch.int32, device=_DET_DEVICE)[None]
    torch.testing.assert_close(first_i, expected_ids)

    # Weights are the unbiased scores of the selected experts. A small
    # tolerance vs the eager reference: the compiled elementwise activation
    # may differ from eager by an ULP on some backends.
    expected_w = _det_scores(logits, scoring_func).gather(1, first_i.to(torch.long))
    torch.testing.assert_close(first_w, expected_w, atol=2e-6, rtol=0)


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
def test_grouped_topk_group_tie_broken_by_lower_group_index(
    monkeypatch: pytest.MonkeyPatch, scoring_func: str
):
    """A bitwise-exact tie at the *group* boundary is broken by ascending
    group index, covering the group-selection site.

    This one stays small on purpose: the group top-k is over
    ``num_expert_group`` values (8 for real DeepSeek/GLM configs), always far
    below topk's 256-column threshold, so a genuine tie is the only way to
    make the group-selection site observable.
    """
    _force_python_fallback(monkeypatch)

    # 4 groups of 8 experts. Group 0 is clearly best (its top-2 dominate),
    # groups 1 and 2 are bitwise-identical -- a deliberate exact tie for the
    # second of the two selected groups -- and group 3 is clearly worst. The
    # non-top logits inside each group are kept low so that the top-4
    # individuals of the union {group 0, group 1} are exactly experts
    # 0, 1, 8, 9; had group 2 won the tie they would be 0, 1, 16, 17.
    g0 = [8.0, 7.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4]
    g12 = [4.0, 3.5, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9]
    g3 = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    logits = torch.tensor(g0 + g12 + g12 + g3, dtype=torch.float32)[None].to(
        _DET_DEVICE
    )
    num_experts = logits.shape[1]
    bias = torch.zeros(num_experts, device=_DET_DEVICE)

    # The group scores (top-2 sum within each group) of groups 1 and 2 must be
    # bitwise-equal: identical inputs, elementwise activation, values-only
    # reduction. (Ties inside that top-2-by-value reduction are between equal
    # values, so the sum is unaffected -- which is why that reduction needs no
    # tie-break of its own.)
    def group_scores(row: torch.Tensor) -> torch.Tensor:
        return row.view(1, 4, -1).topk(2, dim=-1)[0].sum(dim=-1)

    gs = group_scores(_det_biased_scores(logits, bias, scoring_func))
    assert gs[0, 1].item() == gs[0, 2].item()
    assert gs[0, 0] > gs[0, 1] and gs[0, 3] < gs[0, 1]

    results = [
        _run_python_grouped_topk(
            logits.clone(),
            bias,
            4,
            num_expert_group=4,
            topk_group=2,
            scoring_func=scoring_func,
        )
        for _ in range(_DET_NUM_REPEAT)
    ]
    _, first_i = _assert_all_identical(results)

    # Group 1 (the lower index) wins the tie, so the experts are drawn from
    # groups 0 and 1 only.
    expected_ids = torch.tensor([[0, 1, 8, 9]], dtype=torch.int32, device=_DET_DEVICE)
    torch.testing.assert_close(first_i, expected_ids)

    order = _det_biased_scores(logits, bias, scoring_func).gather(
        1, first_i.to(torch.long)
    )
    assert torch.all(order[:, :-1] >= order[:, 1:])
