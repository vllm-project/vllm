# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 MoE W4A16/W4A4 expert weight layout agreement.

Picking the scheme per forward is only sound if one set of expert weights
serves both. These tests pin why that does not hold today, so the constraint
is not rediscovered per backend, and so the opt-in flag fails loudly instead
of being silently ignored.
"""

import pytest

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.nvfp4_a16_dispatch import (
    can_dispatch_a16_per_forward,
    nvfp4_moe_weight_layout_key,
)


def test_b12x_prepares_a_different_w13_for_each_scheme():
    """The concrete blocker: reorder_w13 is driven by use_a16."""
    a16 = nvfp4_moe_weight_layout_key("B12X", True)
    a4 = nvfp4_moe_weight_layout_key("B12X", False)
    assert a16 != a4
    assert a16[1] == "w13_reordered" and a4[1] == "w13_checkpoint"


def test_b12x_is_refused_with_the_responsible_step_named():
    ok, reason = can_dispatch_a16_per_forward("B12X")
    assert not ok
    assert "reorder_w13" in reason


def test_unaudited_backend_is_refused_rather_than_assumed_compatible():
    """Failing open here would mean silently wrong numerics."""
    ok, reason = can_dispatch_a16_per_forward("SOME_FUTURE_BACKEND")
    assert not ok
    assert "not been audited" in reason


@pytest.mark.parametrize("backend", list(NvFp4MoeBackend))
def test_no_backend_can_dispatch_per_forward_yet(backend):
    """Guard: if one becomes sharable, the pairing must be re-reviewed."""
    ok, _ = can_dispatch_a16_per_forward(backend.name)
    assert not ok, (
        f"{backend.name} now reports a shared W4A16/W4A4 expert weight layout; "
        "verify the claim and update the dispatch path"
    )


def test_flag_is_off_by_default():
    import vllm.envs as envs

    assert envs.VLLM_NVFP4_MOE_A16_MAX_M == 0


def test_setting_the_flag_fails_loudly_at_weight_conversion(monkeypatch):
    """The flag must not be silently ignored when no backend can honour it."""
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        convert_to_nvfp4_moe_kernel_format,
    )

    monkeypatch.setattr(envs, "VLLM_NVFP4_MOE_A16_MAX_M", 8)
    with pytest.raises(NotImplementedError, match="reorder_w13"):
        convert_to_nvfp4_moe_kernel_format(
            NvFp4MoeBackend.B12X,
            layer=None,
            w13=None,
            w13_scale=None,
            w13_scale_2=None,
            a13_scale=None,
            w2=None,
            w2_scale=None,
            w2_scale_2=None,
            a2_scale=None,
            is_act_and_mul=True,
        )
