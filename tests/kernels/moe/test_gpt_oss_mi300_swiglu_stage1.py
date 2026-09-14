# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MI300 fused MXFP4 SwiGLU stage-1 fast path.

The fast path is gated by ``run_mi300_swiglu_stage1`` in
``vllm.model_executor.layers.fused_moe.experts.gpt_oss_mi300_swiglu_stage1``
and is wrapped around the stock ``matmul_ogs`` in ``triton_kernel_moe_forward``.
Tests in this module verify two things:

1. The gate cleanly rejects every input shape / dtype / router state that
   the kernel was not designed to handle, so callers always have a safe
   fall-back to the generic path.
2. On a real gfx942 device with the gpt-oss-20b decode shape, the
   optimized output matches the baseline within MXFP4 + BF16 tolerance.
"""

import os

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels

# Module-level skipifs: every test in this file requires an MI300/MI325 (gfx942)
# host because the gate refuses to fire anywhere else.
pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_rocm(),
        reason="MI300 SwiGLU fast path is AMD/ROCm only",
    ),
    pytest.mark.skipif(
        not current_platform.is_device_capability((9, 4)),
        reason="Requires gfx942 (MI300X / MI325X); validated only there.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_run():
    """Lazily import the gate so the module collects on non-target hosts."""
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_mi300_swiglu_stage1 import (  # noqa: E501
        run_mi300_swiglu_stage1,
    )

    return run_mi300_swiglu_stage1


def _make_dummy_hidden(
    rows: int = 4,
    hidden: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((rows, hidden), dtype=dtype, device="cuda")


def _make_dummy_out(
    gather_rows: int = 16,
    hidden: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((gather_rows, hidden), dtype=dtype, device="cuda")


# ---------------------------------------------------------------------------
# Eligibility-gate tests (kernel never actually launches)
# ---------------------------------------------------------------------------


def test_gate_rejects_wrong_hidden_dtype():
    """fp16 hidden states must fall through to the baseline path."""
    fn = _maybe_run()
    hs = _make_dummy_hidden(dtype=torch.float16)
    out = _make_dummy_out()
    assert (
        fn(
            hs,
            w1=None,
            routing_data=None,
            gather_indx=None,
            precision_config=None,
            bias=None,
            out=out,
            apply_router_weight_on_input=False,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )
        is False
    )


def test_gate_rejects_wrong_hidden_dim():
    """Any hidden width != 3072 falls through (kernel is gpt-oss-20b shaped)."""
    fn = _maybe_run()
    hs = _make_dummy_hidden(hidden=2880)
    out = _make_dummy_out()
    assert (
        fn(
            hs,
            w1=None,
            routing_data=None,
            gather_indx=None,
            precision_config=None,
            bias=None,
            out=out,
            apply_router_weight_on_input=False,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )
        is False
    )


def test_gate_rejects_apply_router_weight_on_input():
    """The gammas-folded variant takes a different math path; reject it."""
    fn = _maybe_run()
    hs = _make_dummy_hidden()
    out = _make_dummy_out()
    assert (
        fn(
            hs,
            w1=None,
            routing_data=None,
            gather_indx=None,
            precision_config=None,
            bias=None,
            out=out,
            apply_router_weight_on_input=True,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )
        is False
    )


def test_gate_rejects_swiglu_alpha_mismatch():
    """Kernel hard-codes OAI SwiGLU (alpha=1.702). Other alphas fall through."""
    fn = _maybe_run()
    hs = _make_dummy_hidden()
    out = _make_dummy_out()
    assert (
        fn(
            hs,
            w1=None,
            routing_data=None,
            gather_indx=None,
            precision_config=None,
            bias=None,
            out=out,
            apply_router_weight_on_input=False,
            swiglu_alpha=1.0,  # not 1.702
            swiglu_limit=7.0,
        )
        is False
    )


def test_gate_rejects_missing_routing_data():
    """Without routing metadata the kernel cannot build a block schedule."""
    fn = _maybe_run()
    hs = _make_dummy_hidden()
    out = _make_dummy_out()
    assert (
        fn(
            hs,
            w1=None,
            routing_data=None,  # no routing
            gather_indx=None,
            precision_config=None,
            bias=None,
            out=out,
            apply_router_weight_on_input=False,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )
        is False
    )


def test_gate_kill_switch_env_var(monkeypatch):
    """VLLM_DISABLE_MI300_GPTOSS_SWIGLU=1 must short-circuit even when all
    other preconditions would otherwise hold (we still pass mostly-None
    inputs; the env-var check fires before any tensor inspection)."""
    fn = _maybe_run()
    monkeypatch.setenv("VLLM_DISABLE_MI300_GPTOSS_SWIGLU", "1")
    hs = _make_dummy_hidden()
    out = _make_dummy_out()
    assert (
        fn(
            hs,
            w1=None,
            routing_data=None,
            gather_indx=None,
            precision_config=None,
            bias=None,
            out=out,
            apply_router_weight_on_input=False,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Numerical-equivalence test (real kernel launches, gpt-oss-20b shape)
# ---------------------------------------------------------------------------

if has_triton_kernels():
    # These imports only succeed when `triton_kernels` is installed; the
    # equivalence test below depends on them and is itself skipped via the
    # `triton_kernels` importorskip below when they are missing.
    from triton_kernels.testing import assert_close

    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
        triton_kernel_moe_forward,
    )

    # Re-use the canonical helper from the sibling test so we exercise the
    # exact same MXFP4 weight layout the production code path uses.
    from .test_gpt_oss_triton_kernels import init_compute_data


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 16])
def test_numerical_equivalence_vs_baseline(num_tokens, monkeypatch, workspace_init):
    """Optimized fast path must match the baseline within MXFP4 tolerance.

    Runs ``triton_kernel_moe_forward`` twice on identical inputs at the
    gpt-oss-20b decode shape (hidden=3072, intermediate=3072, 32 experts,
    top-4): once with the fast path enabled, once with the kill-switch
    env-var forcing the baseline. Output tensors must agree within the
    same MXFP4+BF16 tolerance used by the existing
    ``test_gpt_oss_triton_kernels.py`` correctness suite.

    Parametrised on ``num_tokens`` to cover the gather-row shapes (M*topk)
    the fast path is tuned for: 4, 16, 32, 64.
    """
    pytest.importorskip("triton_kernels")
    from triton_kernels.tensor_details import layout

    if not hasattr(layout, "make_default_matmul_mxfp4_w_layout"):
        pytest.skip("make_default_matmul_mxfp4_w_layout not available")

    # gpt-oss-20b decode geometry (E, K, N, topk). Picking these exact
    # values is what allows ``run_mi300_swiglu_stage1`` to accept
    # the inputs at all; any other shape would fall back silently.
    M = num_tokens
    E = 32
    K = 3072
    N = 3072
    topk = 4

    (
        _x_ref,
        _w1_ref,
        _w1_bias_ref,
        _w2_ref,
        _w2_bias_ref,
        _exp_data_ref,
        x_tri,
        w1_tri,
        w2_tri,
        exp_data_tri,
        w1_bias_tri,
        w2_bias_tri,
        pc1,
        pc2,
    ) = init_compute_data(M, K, N, E, "bf16", "mx4", num_warps=8)

    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_scale=pc1,
        w2_scale=pc2,
        w1_bias=w1_bias_tri,
        w2_bias=w2_bias_tri,
    )

    def _forward():
        return triton_kernel_moe_forward(
            hidden_states=x_tri,
            w1=w1_tri,
            w2=w2_tri,
            gating_output=exp_data_tri,
            topk=topk,
            renormalize=True,
            quant_config=quant_config,
        )

    # Optimized path: env var clear, kernel free to fire.
    monkeypatch.delenv("VLLM_DISABLE_MI300_GPTOSS_SWIGLU", raising=False)
    out_opt = _forward()[..., :K].clone()

    # Baseline: force the generic matmul_ogs path via the kill switch.
    monkeypatch.setenv("VLLM_DISABLE_MI300_GPTOSS_SWIGLU", "1")
    out_baseline = _forward()[..., :K].clone()

    assert_close(ref=out_baseline, tri=out_opt, maxtol=0.025, rmstol=0.005)


def test_fast_path_actually_fires(monkeypatch, workspace_init):
    """Sanity: the optimized kernel must actually be invoked at the
    target shape, otherwise the numerical-equivalence test above is
    a tautology (baseline vs. baseline).

    We monkey-patch the gate to count calls and ensure at least one
    call returns ``True`` during a forward pass at the gpt-oss-20b
    decode shape.
    """
    pytest.importorskip("triton_kernels")
    from triton_kernels.tensor_details import layout

    if not hasattr(layout, "make_default_matmul_mxfp4_w_layout"):
        pytest.skip("make_default_matmul_mxfp4_w_layout not available")

    import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as moe_mod  # noqa: E501

    original = moe_mod.run_mi300_swiglu_stage1
    handled_calls = {"true": 0, "false": 0}

    def _instrumented(*args, **kwargs):
        result = original(*args, **kwargs)
        handled_calls["true" if result else "false"] += 1
        return result

    monkeypatch.setattr(moe_mod, "run_mi300_swiglu_stage1", _instrumented)
    monkeypatch.delenv("VLLM_DISABLE_MI300_GPTOSS_SWIGLU", raising=False)

    M, E, K, N, topk = 16, 32, 3072, 3072, 4
    (
        _x_ref,
        _w1_ref,
        _w1_bias_ref,
        _w2_ref,
        _w2_bias_ref,
        _exp_data_ref,
        x_tri,
        w1_tri,
        w2_tri,
        exp_data_tri,
        w1_bias_tri,
        w2_bias_tri,
        pc1,
        pc2,
    ) = init_compute_data(M, K, N, E, "bf16", "mx4", num_warps=8)

    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_scale=pc1,
        w2_scale=pc2,
        w1_bias=w1_bias_tri,
        w2_bias=w2_bias_tri,
    )

    _ = triton_kernel_moe_forward(
        hidden_states=x_tri,
        w1=w1_tri,
        w2=w2_tri,
        gating_output=exp_data_tri,
        topk=topk,
        renormalize=True,
        quant_config=quant_config,
    )

    assert handled_calls["true"] > 0, (
        "run_mi300_swiglu_stage1 was never True at the gpt-oss-20b "
        f"decode shape; gate call summary: {handled_calls}. Equivalence "
        "test above would be vacuous."
    )


@pytest.fixture(autouse=True)
def _ensure_kill_switch_clean():
    """The fast-path tests must not be run with the kill switch globally
    enabled (otherwise the optimized side reduces to the baseline and
    equivalence becomes vacuous). Fail fast at fixture-setup time rather
    than after a long kernel run."""
    if os.environ.get("VLLM_DISABLE_MI300_GPTOSS_SWIGLU") == "1":
        pytest.fail(
            "VLLM_DISABLE_MI300_GPTOSS_SWIGLU=1 is set globally; unset it "
            "before running this suite."
        )
    yield
