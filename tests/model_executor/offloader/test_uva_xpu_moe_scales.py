# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On XPU, fused-MoE weight scales stay on the device under UVA offload while
the installed vllm-xpu-kernels reads past the end of them
(vllm-project/vllm-xpu-kernels#605)."""

import pytest

from vllm.model_executor.offloader import uva
from vllm.model_executor.offloader.uva import (
    UVAOffloader,
    is_moe_scale_param,
    xpu_moe_scales_stay_on_device,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, False),
        ("0.1.16.3", True),
        ("0.1.17", True),
        ("0.1.17.1", True),
        ("0.1.17.1+local", True),
        ("0.1.17.dev20261030084222+g470c19f.r41027826413.a1", True),
        # Published 0.1.18 nightlies: 1105 predates #605, 1106 is the first
        # with it.
        ("0.1.18.dev20261105", True),
        ("0.1.18.dev20261106", False),
        ("0.1.18.dev20261107", False),
        # Source build counted from the last tag: not known to carry #605.
        ("0.1.18.dev12+g0c9d66b", True),
        ("0.1.18rc1", True),
        ("0.1.18", False),
        ("0.1.18.post1", False),
        ("0.1.18.1", False),
        ("0.1.19.dev3", False),
        ("0.2.0", False),
        ("not-a-version", True),
    ],
)
def test_kernels_version_gate(version, expected):
    assert xpu_moe_scales_stay_on_device(version) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("0.mlp.experts.w13_scales", True),
        ("0.mlp.experts.routed_experts.w2_scales", True),
        ("0.mlp.experts.w13_weight_scale", True),
        ("0.mlp.experts.w2_weight_scale_inv", True),
        ("0.mlp.experts.w13_qweight", False),
        ("0.mlp.experts.w2_weight", False),
        ("0.mlp.shared_expert.down_proj.weight_scale", False),
        ("0.self_attn.qkv_proj.weight_scale", False),
        ("experts_scale", False),
    ],
)
def test_moe_scale_names(name, expected):
    assert is_moe_scale_param(name) is expected


@pytest.mark.parametrize(
    ("is_xpu", "kernels_version", "expected"),
    [
        (True, "0.1.17.1", True),
        (True, "0.1.18", False),
        (True, None, False),
        (False, "0.1.17.1", False),
    ],
)
def test_offloader_gate_wiring(monkeypatch, is_xpu, kernels_version, expected):
    monkeypatch.setattr(uva, "is_uva_available", lambda: True)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: is_xpu)
    monkeypatch.setattr(uva, "_installed_xpu_kernels_version", lambda: kernels_version)
    offloader = UVAOffloader(
        cpu_offload_max_bytes=1 << 30, cpu_offload_params={"experts"}
    )
    assert offloader.keep_moe_scales_on_device is expected
