# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arch gating of the FA3 -> FA4 fallback in get_flash_attn_version (vllm#45847).

FA4 is the CuTe SM100 kernel and only supports SM100-SM110; on SM120 it asserts
at init. The Blackwell FA3->FA4 fallback must therefore not select FA4 on SM120.
"""

from types import SimpleNamespace

import pytest

import vllm.v1.attention.backends.fa_utils as fa_utils
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only dispatch path")
@pytest.mark.parametrize(
    "major,minor,expected",
    [
        (10, 0, 4),  # SM100: CuTe FA4 kernel is supported
        (11, 0, 4),  # SM110: still within the FA4 supported range
        (12, 0, 2),  # SM120: FA4 unsupported -> must fall back to FA2 (#45847)
    ],
)
def test_fa3_to_fa4_fallback_arch_gated(monkeypatch, major, minor, expected):
    # Force fa_version=3 via config so the Blackwell FA3->FA4 fallback runs.
    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=3),
        model_config=None,
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=major, minor=minor),
    )
    # Simulate a build where FA4 is compiled in, so only the arch gate decides.
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface.is_fa_version_supported",
        lambda v: True,
    )
    assert fa_utils.get_flash_attn_version() == expected
