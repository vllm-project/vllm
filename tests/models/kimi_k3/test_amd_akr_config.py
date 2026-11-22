# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels route config: parsing, hashing and on() gating (CPU only), and the
stock path with every route off."""

import os
import subprocess
import sys

import pytest
import torch

from vllm.config.kernel import AmdKernelsConfig, KernelConfig
from vllm.models.kimi_k3.amd import akr_config
from vllm.platforms import current_platform


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


def test_hash_follows_env_switches(monkeypatch):
    """The VLLM_KIMI_AKR* switches change which code the model traces: a cached
    compile of one setting must not be reused for another."""
    for k in [k for k in os.environ if k.startswith("VLLM_KIMI_AKR")]:
        monkeypatch.delenv(k)
    base = KernelConfig().compute_hash()
    monkeypatch.setenv("VLLM_KIMI_AKR", "0")
    off = KernelConfig().compute_hash()
    monkeypatch.setenv("VLLM_KIMI_AKR", "1")
    monkeypatch.setenv("VLLM_KIMI_AKR_EXPERTS", "0")
    experts_off = KernelConfig().compute_hash()
    assert len({base, off, experts_off}) == 3


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


_rocm_only = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the ROCm Kimi-K3.1 modules"
)


@_rocm_only
def test_dsa_post_load_without_amd_kernels(monkeypatch):
    """Every route off (here: the package is missing): the hook that
    MLAAttention.process_weights_after_loading calls for every Kimi-K3.1 DSA layer
    leaves the layer alone and imports no amd_kernels module."""
    from vllm.models.kimi_k3.amd import dsa

    monkeypatch.setitem(sys.modules, "amd_kernels", None)  # import fails
    _patch(monkeypatch, _cfg(), available=False)
    assert akr_config.routes() == frozenset()
    wrapper = torch.nn.Module()
    dsa._akr_post_load(wrapper)
    assert not hasattr(wrapper, "akr_w")


@_rocm_only
def test_sparse_mla_backend_does_not_import_kimi():
    """The generic ROCm sparse MLA backend reaches the Kimi-K3.1 hooks only through
    the attribute its DSA layers set: importing it loads no Kimi-K3 module."""
    code = (
        "import sys\n"
        "import vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse\n"
        "print(sorted(m for m in sys.modules if m.startswith('vllm.models.kimi_k3')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip().splitlines()[-1] == "[]"
