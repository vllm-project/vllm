# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.utils.deep_gemm as deep_gemm


@pytest.mark.parametrize(
    ("env_enabled", "has_module", "device_families", "expected"),
    [
        (True, True, {90}, True),
        (True, True, {100}, True),
        (True, True, {120}, False),
        (False, True, {90}, False),
        (True, False, {90}, False),
    ],
)
def test_deep_gemm_paged_mqa_support_arch_gate(
    monkeypatch,
    env_enabled: bool,
    has_module: bool,
    device_families: set[int],
    expected: bool,
) -> None:
    assert hasattr(deep_gemm, "is_deep_gemm_paged_mqa_supported")

    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", env_enabled)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: has_module)
    monkeypatch.setattr(
        deep_gemm.current_platform,
        "is_device_capability_family",
        lambda capability: capability in device_families,
    )

    deep_gemm.is_deep_gemm_paged_mqa_supported.cache_clear()

    assert deep_gemm.is_deep_gemm_paged_mqa_supported() is expected
