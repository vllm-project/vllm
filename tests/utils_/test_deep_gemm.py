# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.utils.deep_gemm as deep_gemm


def _clear_deep_gemm_support_cache():
    deep_gemm.is_deep_gemm_supported.cache_clear()


def test_deep_gemm_support_does_not_probe_package_when_disabled(monkeypatch):
    _clear_deep_gemm_support_cache()
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", False)

    def fail_if_called():
        raise AssertionError("platform support should not be checked")

    monkeypatch.setattr(deep_gemm.current_platform, "support_deep_gemm", fail_if_called)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", fail_if_called)

    assert deep_gemm.is_deep_gemm_supported() is False


def test_deep_gemm_support_does_not_probe_package_on_unsupported_platform(
    monkeypatch,
):
    _clear_deep_gemm_support_cache()
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(deep_gemm.current_platform, "support_deep_gemm", lambda: False)

    def fail_if_called():
        raise AssertionError("DeepGEMM package should not be probed")

    monkeypatch.setattr(deep_gemm, "has_deep_gemm", fail_if_called)

    assert deep_gemm.is_deep_gemm_supported() is False


def test_deep_gemm_support_probes_package_on_supported_platform(monkeypatch):
    _clear_deep_gemm_support_cache()
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(deep_gemm.current_platform, "support_deep_gemm", lambda: True)

    calls = 0

    def has_deep_gemm():
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(deep_gemm, "has_deep_gemm", has_deep_gemm)

    assert deep_gemm.is_deep_gemm_supported() is True
    assert calls == 1


def test_deep_gemm_support_reports_false_when_supported_platform_lacks_package(
    monkeypatch,
):
    _clear_deep_gemm_support_cache()
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(deep_gemm.current_platform, "support_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: False)

    assert deep_gemm.is_deep_gemm_supported() is False
