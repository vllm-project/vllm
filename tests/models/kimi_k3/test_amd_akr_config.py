# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels route config: parsing, hashing and on() gating (CPU only)."""

import pytest

from vllm.config.kernel import AmdKernelsConfig, KernelConfig
from vllm.models.kimi_k3.amd import akr_config


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("VLLM_KIMI_AKR", raising=False)
    monkeypatch.setattr(akr_config, "_routes", None)
    monkeypatch.setattr(akr_config, "_warned", set())


def _cfg(**kw):
    class _Cfg:
        kernel_config = KernelConfig(amd_kernels=AmdKernelsConfig(**kw))

    return _Cfg()


def _patch(monkeypatch, cfg, available=True):
    monkeypatch.setattr(akr_config, "_available", lambda: available)
    import vllm.config as vc

    monkeypatch.setattr(vc, "get_current_vllm_config_or_none", lambda: cfg)


def test_parse_kernel_config_dict():
    kc = KernelConfig(**{"amd_kernels": {"disabled_routes": ["moe", "gemm"]}})
    assert kc.amd_kernels.enabled
    assert kc.amd_kernels.disabled_routes == ["moe", "gemm"]
    assert KernelConfig().amd_kernels.disabled_routes == []


def test_invalid_route_rejected():
    with pytest.raises(ValueError):
        AmdKernelsConfig(disabled_routes=["bogus"])  # type: ignore[list-item]


def test_hash_changes():
    base = KernelConfig().compute_hash()
    assert KernelConfig(amd_kernels={"enabled": False}).compute_hash() != base
    a = KernelConfig(amd_kernels={"disabled_routes": ["moe", "gemm"]})
    b = KernelConfig(amd_kernels={"disabled_routes": ["gemm", "moe"]})
    assert a.compute_hash() == b.compute_hash() != base


def test_on_all_when_available(monkeypatch):
    _patch(monkeypatch, _cfg())
    assert akr_config.routes() == frozenset(akr_config.ROUTES)
    assert akr_config.on("gemm")


def test_on_false_when_unavailable(monkeypatch):
    _patch(monkeypatch, _cfg(), available=False)
    assert not any(akr_config.on(r) for r in akr_config.ROUTES)


def test_disabled_and_prereqs(monkeypatch):
    _patch(monkeypatch, _cfg(disabled_routes=["moe", "attention", "draft"]))
    assert not akr_config.on("moe")
    assert not akr_config.on("indexer")  # needs attention
    assert not akr_config.on("sampling")  # needs draft
    assert akr_config.on("gemm")


def test_enabled_false(monkeypatch):
    _patch(monkeypatch, _cfg(enabled=False))
    assert akr_config.routes() == frozenset()


def test_env_override(monkeypatch):
    _patch(monkeypatch, _cfg())
    monkeypatch.setenv("VLLM_KIMI_AKR", "0")
    assert akr_config.routes() == frozenset()
    monkeypatch.setattr(akr_config, "_routes", None)
    monkeypatch.setenv("VLLM_KIMI_AKR", "gemm,moe")
    assert akr_config.routes() == frozenset({"gemm", "moe"})


def test_available_requires_rocm_gfx950_and_package(monkeypatch):
    # on this CPU/non-ROCm test host the real probe must be False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        assert not akr_config._available()


def test_unsupported_reason_warns_once(monkeypatch):
    msgs = []
    monkeypatch.setattr(akr_config.logger, "warning", lambda *a: msgs.append(a))
    assert not akr_config.unsupported_reason("moe", None)
    assert akr_config.unsupported_reason("moe", "tp!=8")
    assert akr_config.unsupported_reason("moe", "tp!=8")
    assert len(msgs) == 1
