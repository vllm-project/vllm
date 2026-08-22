# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the lazy NIXL loader and its guard for AVX-compiled UCX.

The UCX library bundled with the NIXL wheels terminates the process from
its native initializer on x86 CPUs without AVX
(https://github.com/vllm-project/vllm/issues/52885), so on such hosts
NIXL must be reported as unavailable instead of being loaded.
"""

import sys

import pytest

import vllm.distributed.nixl_utils as nixl_utils

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def no_avx(monkeypatch):
    """Simulate an x86 CPU without AVX, where loading UCX kills the process."""
    monkeypatch.setattr(nixl_utils, "cpu_supports_avx", lambda: False)


@pytest.fixture
def spy_nixl_warnings(monkeypatch):
    """Collect warning_once calls — vllm's logger does not propagate to
    root, so caplog can't see them."""
    warnings: list[str] = []
    monkeypatch.setattr(
        nixl_utils.logger, "warning_once", lambda msg: warnings.append(msg)
    )
    return warnings


@pytest.fixture
def restore_lazy_globals():
    """Undo the globals() caching done by nixl_utils.__getattr__."""
    saved = {k: v for k, v in nixl_utils.__dict__.items() if k in nixl_utils.__all__}
    yield
    for name in nixl_utils.__all__:
        if name in nixl_utils.__dict__ and name not in saved:
            del nixl_utils.__dict__[name]


def test_load_nixl_attr_is_disabled_without_avx(
    no_avx, restore_lazy_globals, spy_nixl_warnings
):
    assert nixl_utils._load_nixl_attr("NixlWrapper") is None
    assert nixl_utils.NixlWrapper is None
    assert nixl_utils._load_nixl_attr("nixl_agent_config") is None
    assert nixl_utils.nixl_agent_config is None
    # The skip must be visible to users instead of a silent downgrade.
    assert any("Disabling NIXL" in msg for msg in spy_nixl_warnings)


def test_nixl_attr_access_does_not_import_nixl_without_avx(
    no_avx, restore_lazy_globals
):
    assert nixl_utils.nixlXferTelemetry is None
    assert "nixl" not in sys.modules


def test_is_nixl_available_false_without_avx(no_avx):
    assert nixl_utils.is_nixl_available() is False


def test_is_nixl_available_false_when_package_missing(monkeypatch):
    monkeypatch.setattr(nixl_utils, "cpu_supports_avx", lambda: True)
    monkeypatch.setattr(nixl_utils, "_get_nixl_package_name", lambda: "nixl")
    monkeypatch.setattr(
        "vllm.distributed.nixl_utils.importlib.util.find_spec",
        lambda name: None,
    )
    modules_without_nixl = {k: v for k, v in sys.modules.items() if k != "nixl"}
    monkeypatch.setattr(sys, "modules", modules_without_nixl)
    assert nixl_utils.is_nixl_available() is False


def test_is_nixl_available_true_when_package_present(monkeypatch):
    monkeypatch.setattr(nixl_utils, "cpu_supports_avx", lambda: True)
    monkeypatch.setattr(nixl_utils, "_get_nixl_package_name", lambda: "nixl")
    monkeypatch.setattr(
        "vllm.distributed.nixl_utils.importlib.util.find_spec",
        lambda name: object(),
    )
    modules_without_nixl = {k: v for k, v in sys.modules.items() if k != "nixl"}
    monkeypatch.setattr(sys, "modules", modules_without_nixl)
    assert nixl_utils.is_nixl_available() is True
