# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified ``DeviceOps`` abstraction and spec-based resolution.

These tests are the acceptance gate for the DeviceOps hierarchy.  They stay
platform-agnostic by exercising the torch baseline, the ``DeviceSpec``
dispatch, the ``lmcache.c_ops`` shim, and instance-level ``bind_native``
without requiring any compiled accelerator module.
"""

# Standard
from typing import Any
import inspect

# Third Party
import pytest

# First Party
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.cpu.device_ops import CpuDeviceOps
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)
import lmcache.v1.platform as platform_pkg

# Derive op names from the class body: regular instance-method functions,
# excluding infrastructure helpers.
_OP_NAMES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, member in vars(DeviceOps).items()
        if not name.startswith("_")
        and name not in ("ensure_native", "bind_native")
        and inspect.isfunction(member)
    )
)


@pytest.fixture
def isolated_registry() -> Any:
    """Snapshot the device-spec table so tests can install fakes safely."""
    saved = dict(platform_pkg._DEVICE_REGISTRY)
    try:
        yield saved
    finally:
        platform_pkg._DEVICE_REGISTRY.clear()
        platform_pkg._DEVICE_REGISTRY.update(saved)


# -- Contract --------------------------------------------------------------


def test_base_class_declares_every_op_as_instance_method() -> None:
    """DeviceOps declares every op as a real instance method."""
    ops = DeviceOps()
    for name in _OP_NAMES:
        bound = getattr(ops, name)
        assert callable(bound), name
        # Must be declared on the class body
        assert name in vars(DeviceOps), (
            f"{name} is not declared on the DeviceOps class body"
        )


def test_base_class_has_all_types() -> None:
    """DeviceOps exposes shared types as class attributes."""
    assert DeviceOps.TransferDirection is TransferDirection
    assert DeviceOps.EngineKVFormat is EngineKVFormat
    assert DeviceOps.GPUKVFormat is EngineKVFormat
    assert DeviceOps.PageBufferShapeDesc is PageBufferShapeDesc
    assert callable(DeviceOps.set_shape_desc_dtype)


def test_every_registered_device_has_all_ops(isolated_registry: Any) -> None:
    """Each discovered DeviceSpec resolves an ops instance with all ops."""
    for device_type, spec in isolated_registry.items():
        ops = spec.ops_cls()
        for name in _OP_NAMES:
            assert callable(getattr(ops, name)), (device_type, name)


# -- Dispatch (MRO) --------------------------------------------------------


def test_cpu_inherits_baseline_verbatim() -> None:
    """CpuDeviceOps adds no overrides: every method resolves to the base."""
    for name in _OP_NAMES:
        assert getattr(CpuDeviceOps, name) is getattr(DeviceOps, name), name


def test_musa_overrides_only_one_op() -> None:
    """MusaDeviceOps overrides exactly one hot op; the rest inherit base."""
    musa_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.device_ops",
        reason="musa platform package unavailable",
    )
    overridden = [
        name
        for name in _OP_NAMES
        if getattr(musa_mod.MusaDeviceOps, name) is not getattr(DeviceOps, name)
    ]
    assert overridden == ["multi_layer_block_kv_transfer"]


def test_musa_override_dispatches_native_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling multi_layer_block_kv_transfer on a MusaDeviceOps instance
    dispatches via the native MUSA path when inputs are tensor-backed."""
    # Third Party
    import torch

    musa_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.device_ops",
        reason="musa platform package unavailable",
    )
    native_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.native_kv_transfer",
        reason="musa native_kv_transfer unavailable",
    )

    captured: dict[str, Any] = {}

    def _fake_native(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        native_mod,
        "try_native_multi_layer_block_kv_transfer",
        _fake_native,
    )

    paged = [torch.zeros(4, 4, 16) for _ in range(2)]
    objects = [torch.zeros(2, 8, 16)]

    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = 2
    shape_desc.nb = 1
    shape_desc.bs = 4
    shape_desc.nh = 4
    shape_desc.hs = 16
    shape_desc.kv_size = 2
    shape_desc.element_size = torch.empty((), dtype=torch.float32).element_size()

    # Call through a fresh instance (regular OO polymorphism).
    musa_mod.MusaDeviceOps().multi_layer_block_kv_transfer(
        paged,
        objects,
        [0, 1],
        "musa",
        TransferDirection.D2H,
        shape_desc,
        8,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )

    assert captured, "native path was not invoked"
    assert captured["direction"] == TransferDirection.D2H


# -- bind_native -----------------------------------------------------------


class _FakeNativeModule:
    """Stand-in compiled module: a couple of real ops + a non-OPS symbol."""

    @staticmethod
    def multi_layer_kv_transfer(*a: Any, **k: Any) -> str:
        return "native-mlt"

    @staticmethod
    def calculate_cdf(*a: Any, **k: Any) -> str:
        return "native-cdf"

    @staticmethod
    def not_in_ops(*a: Any, **k: Any) -> str:
        return "ignored"


def test_bind_native_shadows_baseline_for_present_ops() -> None:
    """bind_native rebinds ops found in the module on that instance only."""
    ops = DeviceOps()
    ops.bind_native(_FakeNativeModule())

    assert ops.multi_layer_kv_transfer() == "native-mlt"  # type: ignore[call-arg]
    assert ops.calculate_cdf(None, 0) == "native-cdf"  # type: ignore[call-arg]

    # Other instances still see the torch baseline.
    other = DeviceOps()
    assert other.multi_layer_kv_transfer is not ops.multi_layer_kv_transfer  # type: ignore[attr-defined]


def test_bind_native_discovers_all_public_symbols() -> None:
    """bind_native auto-discovers all public symbols from the native module."""
    ops = DeviceOps()
    ops.bind_native(_FakeNativeModule())
    # Extra symbols on the module are now bound (auto-discovery).
    assert "not_in_ops" in vars(ops)


def test_bind_native_rebinds_types() -> None:
    """bind_native updates type instance attributes from the module."""

    class FakeTypes:
        class TransferDirection:
            pass

        class EngineKVFormat:
            pass

    ops = DeviceOps()
    ops.bind_native(FakeTypes())

    assert ops.TransferDirection is FakeTypes.TransferDirection
    assert ops.EngineKVFormat is FakeTypes.EngineKVFormat
    assert ops.GPUKVFormat is FakeTypes.EngineKVFormat  # alias


# -- Shim ------------------------------------------------------------------


def test_c_ops_shim_has_all_ops_and_types() -> None:
    """The lmcache.c_ops shim module exposes everything from _OP_NAMES + types."""
    # First Party
    import lmcache.c_ops as c_ops

    for name in _OP_NAMES:
        assert hasattr(c_ops, name), f"c_ops missing {name}"
        assert callable(getattr(c_ops, name))
    assert hasattr(c_ops, "TransferDirection")
    assert hasattr(c_ops, "EngineKVFormat")
    assert hasattr(c_ops, "GPUKVFormat")
    assert hasattr(c_ops, "PageBufferShapeDesc")


def test_c_ops_shim_dir() -> None:
    """The shim supports dir() for discoverability."""
    # First Party
    import lmcache.c_ops as c_ops

    names = dir(c_ops)
    assert "multi_layer_kv_transfer" in names
    assert "TransferDirection" in names


# -- DeviceSpec resolution -------------------------------------------------


def test_cpu_and_empty_string_resolve_to_expected_ops_classes() -> None:
    """``cpu`` resolves via CpuDeviceSpec; ``""`` uses the bare fallback."""
    assert type(resolve_device_ops("cpu")) is CpuDeviceOps
    assert type(resolve_device_ops("")) is DeviceOps


def test_resolve_device_ops_returns_cached_singleton() -> None:
    """Two lookups for the same device_type return the same instance."""
    a = resolve_device_ops("cpu")
    b = resolve_device_ops("cpu")
    assert a is b
    assert isinstance(a, CpuDeviceOps)


def test_cpu_without_registered_spec_falls_back_to_base_device_ops(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = {k: v for k, v in isolated_registry.items() if k != "cpu"}
    monkeypatch.setattr(platform_pkg, "_DEVICE_REGISTRY", table)
    assert type(resolve_device_ops("cpu")) is DeviceOps


def test_unregistered_accelerator_fails_fast(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested accelerator with no registered class is a hard error."""
    table = {k: v for k, v in isolated_registry.items() if k != "cuda"}
    monkeypatch.setattr(platform_pkg, "_DEVICE_REGISTRY", table)
    with pytest.raises(
        RuntimeError,
        match="refusing to silently fall back to the torch baseline",
    ):
        resolve_device_ops("cuda")


def test_unknown_accelerator_fails_fast_without_registry_edits() -> None:
    with pytest.raises(
        RuntimeError,
        match="refusing to silently fall back to the torch baseline",
    ):
        resolve_device_ops("definitely-not-a-real-device")


def test_new_device_needs_zero_resolver_edits(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scalability: a fresh DeviceSpec resolves with no resolver change."""

    class DummyDeviceOps(DeviceOps):
        device_type = "dummy"

    class DummyDeviceSpec(DeviceSpec):
        @property
        def device_type(self) -> str:
            return "dummy"

        @property
        def ops_cls(self) -> "type[DeviceOps]":
            return DummyDeviceOps

    monkeypatch.setattr(
        platform_pkg,
        "_DEVICE_REGISTRY",
        {**isolated_registry, "dummy": DummyDeviceSpec()},
    )
    assert type(resolve_device_ops("dummy")) is DummyDeviceOps
