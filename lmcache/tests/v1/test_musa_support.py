# SPDX-License-Identifier: Apache-2.0
"""
MUSA support unit tests that do not require MUSA hardware.

These tests cover the design contract documented in
``docs/source/developer_guide/musa_support_design.rst``:

- Device detection via the registry-driven
  :func:`lmcache.v1.platform._detect_device`.
- Factory dispatch in :func:`lmcache.v1.gpu_connector.CreateGPUConnector`,
  including fail-fast validation when device-scoped features are requested on
  accelerators without connector support.

The tests stub ``torch_device_type`` / ``torch_dev`` (rather than mutating
the global PyTorch namespace) so they run on any platform.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import importlib

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.metadata import LMCacheMetadata
import lmcache as lmc
import lmcache.v1.gpu_connector as gpu_connector_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata() -> LMCacheMetadata:
    """Minimal metadata accepted by ``CreateGPUConnector``."""
    return LMCacheMetadata(
        model_name="musa_support_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, 16, 8, 64),
    )


def _make_config(**overrides: Any) -> LMCacheEngineConfig:
    """Default config plus the requested overrides."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _FakeTorchDev:
    """Stand-in for ``torch.musa`` / ``torch.xpu`` / ``torch.cuda``."""

    def __init__(self, device_count: int = 1) -> None:
        self._device_count = device_count

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._device_count

    def current_device(self) -> int:
        return 0

    def set_device(self, _idx: int) -> None:
        return


def _patch_device(monkeypatch: pytest.MonkeyPatch, device_type: str) -> None:
    """Pretend the current accelerator is ``device_type`` in the factory."""
    monkeypatch.setattr(gpu_connector_module, "torch_device_type", device_type)
    monkeypatch.setattr(gpu_connector_module, "torch_dev", _FakeTorchDev())


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


class _StubTorch:
    """Minimal stand-in for ``torch`` exposing only what ``_detect_device`` reads."""

    def __init__(
        self,
        *,
        has_musa: bool = False,
        has_xpu: bool = False,
        has_hpu: bool = False,
        musa_available: bool = False,
        xpu_available: bool = False,
        hpu_available: bool = False,
    ) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: True)
        if has_musa:
            self.musa = SimpleNamespace(is_available=lambda: musa_available)
        if has_xpu:
            self.xpu = SimpleNamespace(is_available=lambda: xpu_available)
        if has_hpu:
            self.hpu = SimpleNamespace(is_available=lambda: hpu_available)


def _detect_with_stub(stub: _StubTorch) -> tuple[Any, str]:
    """Run ``_detect_device`` with ``torch`` swapped for the stub."""
    # First Party
    from lmcache.v1.platform._device_detect import _detect_device

    with patch.dict("sys.modules", {"torch": stub}):
        return _detect_device()


def test_detect_device_prefers_musa_when_available() -> None:
    """``_detect_device`` returns MUSA whenever ``torch.musa.is_available()``."""
    # Standard
    import os

    os.environ["DEVICE_TYPE"] = "musa"
    stub = _StubTorch(
        has_musa=True,
        has_xpu=True,
        has_hpu=True,
        musa_available=True,
        xpu_available=True,
        hpu_available=True,
    )
    dev, name = _detect_with_stub(stub)
    assert name == "musa"
    assert dev is stub.musa
    del os.environ["DEVICE_TYPE"]


def test_detect_device_falls_back_past_unavailable_musa() -> None:
    """Falls through MUSA when ``torch.musa.is_available()`` is False."""
    # Standard
    import os

    os.environ["DEVICE_TYPE"] = "musa"
    stub = _StubTorch(
        has_musa=True,
        musa_available=False,
    )
    _, name = _detect_with_stub(stub)
    assert name == "cuda"
    del os.environ["DEVICE_TYPE"]


def test_detect_device_cuda_fallback_when_no_alt_accelerator() -> None:
    """Default fallback is CUDA so existing CUDA tests/paths keep working."""
    stub = _StubTorch()
    dev, name = _detect_with_stub(stub)
    assert name == "cuda"
    assert dev is stub.cuda


# ---------------------------------------------------------------------------
# CreateGPUConnector: MUSA branch + device-scoped feature guards
# ---------------------------------------------------------------------------


def test_create_gpu_connector_blending_rejected_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_blending`` must fail fast on MUSA."""
    _patch_device(monkeypatch, "musa")
    config = _make_config(enable_blending=True, use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="enable_blending"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


@pytest.mark.parametrize("device_type", ["musa", "hpu"])
def test_create_gpu_connector_blending_rejected_on_unsupported_devices(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    """The blending guard rejects devices without a blending connector."""
    _patch_device(monkeypatch, device_type)
    config = _make_config(enable_blending=True, use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="enable_blending"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


def test_create_gpu_connector_v3_rejected_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_gpu_connector_v3`` must fail fast on MUSA."""
    _patch_device(monkeypatch, "musa")
    config = _make_config(use_gpu_connector_v3=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="use_gpu_connector_v3"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


def test_create_gpu_connector_layerwise_rejected_on_hpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HPU ships only ``VLLMPagedMemHPUConnectorV2`` (no layerwise variant).

    Previously, ``use_layerwise=True`` on HPU silently fell through into
    ``VLLMPagedMemLayerwiseGPUConnector`` — the CUDA layerwise connector —
    which then crashed on HPU tensors when constructing
    ``torch.cuda.Stream()``. The guard must reject this combination with a
    clear error before any device-specific construction.
    """
    _patch_device(monkeypatch, "hpu")
    config = _make_config(use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError) as exc:
        CreateGPUConnector(config, metadata, EngineType.VLLM)
    message = str(exc.value)
    assert "use_layerwise" in message
    assert "hpu" in message.lower()


def test_create_gpu_connector_musa_dispatches_to_musa_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch_device_type == 'musa'`` selects a MUSA connector class.

    We do not need real MUSA hardware: the factory's ``torch.device`` call
    is patched (PyTorch builds without ``torch_musa`` reject the 'musa'
    string), and ``from_metadata`` is replaced on the MUSA connectors so
    the assertion checks only the dispatch.
    """
    _patch_device(monkeypatch, "musa")
    # First Party
    from lmcache.v1.gpu_connector import musa_connectors as musa_mod

    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "musa:0"
    )

    sentinel_v2 = object()
    sentinel_layer = object()
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemMUSAConnectorV2,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_v2),
    )
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemLayerwiseMUSAConnector,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_layer),
    )

    metadata = _make_metadata()
    assert (
        CreateGPUConnector(_make_config(use_layerwise=False), metadata, EngineType.VLLM)
        is sentinel_v2
    )
    assert (
        CreateGPUConnector(_make_config(use_layerwise=True), metadata, EngineType.VLLM)
        is sentinel_layer
    )


def test_create_gpu_connector_rejects_sglang_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage2 does not expose SGLang MUSA connector dispatch."""
    _patch_device(monkeypatch, "musa")
    with pytest.raises(ValueError, match="SGLang on MUSA"):
        CreateGPUConnector(_make_config(), _make_metadata(), EngineType.SGLANG)


# ---------------------------------------------------------------------------
# Module import sanity
# ---------------------------------------------------------------------------


def test_lmcache_exports_torch_dev_and_torch_device_type() -> None:
    """The contract used by every accelerator-aware module is the
    ``torch_dev`` / ``torch_device_type`` pair exported from ``lmcache``.

    Re-import to defeat any per-test monkeypatching above.
    """
    importlib.reload(lmc)
    assert hasattr(lmc, "torch_dev")
    assert isinstance(lmc.torch_device_type, str)
    assert lmc.torch_device_type in {"cuda", "musa", "xpu", "hpu", "cpu"}
