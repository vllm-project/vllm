# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the routed_scaling_factor weight-prefold optimization
(GitHub issue #59640).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


pytestmark = pytest.mark.cpu_test


def _runner_for_scale_test(
    routed_scaling_factor: float, routed_scale_prefolded: bool
) -> MoERunner:
    """Bypass __init__ (which needs a full FusedMoEConfig/router/etc.) and
    set only the attributes _maybe_apply_routed_scale_to_output reads."""
    runner = MoERunner.__new__(MoERunner)
    runner.routed_scaling_factor = routed_scaling_factor
    runner.routed_scale_prefolded = routed_scale_prefolded
    return runner


@pytest.mark.parametrize(
    "dtype,routed_scale_prefolded,expect_multiply",
    [
        (torch.bfloat16, True, False),
        (torch.bfloat16, False, True),
        (torch.float32, True, False),
        (torch.float16, True, True),
        (torch.float16, False, True),
    ],
)
def test_maybe_apply_routed_scale_respects_prefolded_flag(
    dtype, routed_scale_prefolded, expect_multiply
):
    runner = _runner_for_scale_test(
        routed_scaling_factor=5.0, routed_scale_prefolded=routed_scale_prefolded
    )
    fused_output = torch.ones(4, 8, dtype=dtype)
    original = fused_output.clone()

    _, result = runner._maybe_apply_routed_scale_to_output(None, fused_output)

    if expect_multiply:
        torch.testing.assert_close(result, original * 5.0)
    else:
        torch.testing.assert_close(result, original)


def test_maybe_apply_routed_scale_fp16_scales_shared_output_inversely_when_prefolded():
    runner = _runner_for_scale_test(routed_scaling_factor=5.0, routed_scale_prefolded=True)
    fused_output = torch.ones(4, 8, dtype=torch.float16)
    shared_output = torch.ones(4, 8, dtype=torch.float16) * 10.0
    original_shared = shared_output.clone()
    original_fused = fused_output.clone()

    new_shared, new_fused = runner._maybe_apply_routed_scale_to_output(
        shared_output, fused_output
    )

    torch.testing.assert_close(new_fused, original_fused)  # unscaled
    torch.testing.assert_close(new_shared, original_shared * (1.0 / 5.0))


def test_maybe_apply_routed_scale_noop_when_scaling_factor_is_one():
    """routed_scaling_factor == 1.0 is a no-op regardless of the prefolded
    flag -- there is nothing to skip or apply."""
    for prefolded in (True, False):
        runner = _runner_for_scale_test(
            routed_scaling_factor=1.0, routed_scale_prefolded=prefolded
        )
        fused_output = torch.ones(4, 8, dtype=torch.bfloat16)
        original = fused_output.clone()
        _, result = runner._maybe_apply_routed_scale_to_output(None, fused_output)
        torch.testing.assert_close(result, original)


# ---------------------------------------------------------------------------
# NemotronH-side: _routed_scale_prefoldable wiring and the weight-mutation
# hook (_maybe_prefold_routed_scale).
# ---------------------------------------------------------------------------


def _fc2_stub(*, bias: bool = False) -> torch.nn.Module:
    """A minimal stand-in for fc2_latent_proj: just needs .weight (and
    optionally .bias) as mutable tensors, since _maybe_prefold_routed_scale
    only ever calls .mul_ on them."""
    stub = torch.nn.Module()
    stub.weight = torch.nn.Parameter(torch.ones(4, 4))
    if bias:
        stub.bias = torch.nn.Parameter(torch.ones(4))
    else:
        stub.bias = None
    return stub


def _model_for_prefold_test(
    *, prefoldable: bool, routed_scaling_factor: float = 5.0, bias: bool = False
):
    from vllm.model_executor.models.nemotron_h import (
        NemotronHForCausalLM,
        NemotronHMoE,
        NemotronHMoEDecoderLayer,
    )

    fc2 = _fc2_stub(bias=bias)

    mixer = NemotronHMoE.__new__(NemotronHMoE)
    torch.nn.Module.__init__(mixer)
    mixer.fc2_latent_proj = fc2
    mixer._routed_scale_prefoldable = prefoldable
    mixer.routed_scaling_factor = routed_scaling_factor

    layer = NemotronHMoEDecoderLayer.__new__(NemotronHMoEDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.mixer = mixer

    model = type("_StubModel", (), {})()
    model.has_moe = True
    model.layers = [layer]

    causal_lm = NemotronHForCausalLM.__new__(NemotronHForCausalLM)
    causal_lm.model = model
    return causal_lm, fc2


def test_maybe_prefold_routed_scale_folds_weight_when_prefoldable():
    causal_lm, fc2 = _model_for_prefold_test(prefoldable=True, routed_scaling_factor=5.0)
    original_weight = fc2.weight.clone()

    causal_lm._maybe_prefold_routed_scale()

    torch.testing.assert_close(fc2.weight, original_weight * 5.0)
    assert fc2._routed_scale_applied is True


def test_maybe_prefold_routed_scale_also_folds_bias_when_present():
    causal_lm, fc2 = _model_for_prefold_test(
        prefoldable=True, routed_scaling_factor=5.0, bias=True
    )
    original_bias = fc2.bias.clone()

    causal_lm._maybe_prefold_routed_scale()

    torch.testing.assert_close(fc2.bias, original_bias * 5.0)


def test_maybe_prefold_routed_scale_skips_when_not_prefoldable():
    causal_lm, fc2 = _model_for_prefold_test(prefoldable=False, routed_scaling_factor=5.0)
    original_weight = fc2.weight.clone()

    causal_lm._maybe_prefold_routed_scale()

    torch.testing.assert_close(fc2.weight, original_weight)
    assert not getattr(fc2, "_routed_scale_applied", False)


def test_maybe_prefold_routed_scale_is_idempotent():
    causal_lm, fc2 = _model_for_prefold_test(prefoldable=True, routed_scaling_factor=5.0)
    original_weight = fc2.weight.clone()

    causal_lm._maybe_prefold_routed_scale()
    causal_lm._maybe_prefold_routed_scale()  # second call must be a no-op

    torch.testing.assert_close(fc2.weight, original_weight * 5.0)


def test_maybe_prefold_routed_scale_noop_without_moe():
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    causal_lm = NemotronHForCausalLM.__new__(NemotronHForCausalLM)
    model = type("_StubModel", (), {})()
    model.has_moe = False

    def _boom():
        raise AssertionError("must not access .layers when has_moe is False")

    type(model).layers = property(lambda self: _boom())
    causal_lm.model = model

    causal_lm._maybe_prefold_routed_scale()  # should return early, no error
