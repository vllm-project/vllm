# SPDX-License-Identifier: Apache-2.0
"""
Tests for MooncakeStoreL2AdapterConfig and factory registration.

Integration tests require the C++ Mooncake extension and a running
Mooncake Store service.  They are skipped automatically when the
extension is not available.
"""

# Standard
from typing import Any, cast
import os
import select
import sys
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters import (
    mooncake_store_l2_adapter as mooncake_store_module,
)
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter import (
    MooncakeStoreL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# =============================================================================
# Helpers
# =============================================================================


def _native_mooncake_available() -> bool:
    """Check if the C++ Mooncake extension can be imported."""
    try:
        # First Party
        from lmcache.lmcache_mooncake import LMCacheMooncakeClient  # noqa: F401

        return True
    except ImportError:
        return False


requires_mooncake = pytest.mark.skipif(
    not _native_mooncake_available(),
    reason="C++ Mooncake extension (lmcache_mooncake) not available",
)


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def create_buffer_memory_obj(
    buffer: torch.Tensor,
    offset_bytes: int,
    size_bytes: int,
    fill_value: float = 1.0,
) -> TensorMemoryObj:
    """Create a tensor-backed memory object from a slice of a buffer.

    Args:
        buffer: Backing tensor containing the bytes to expose.
        offset_bytes: Byte offset where the memory object starts in ``buffer``.
        size_bytes: Size of the memory object in bytes.
        fill_value: Value used to initialize the exposed float32 view.

    Returns:
        A ``TensorMemoryObj`` describing the requested buffer slice.
    """
    raw_data = buffer[offset_bytes : offset_bytes + size_bytes].view(torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size_bytes // 4]),
        dtype=torch.float32,
        address=offset_bytes,
        phy_size=size_bytes,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 10.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


# =============================================================================
# Config Unit Tests (no C++ extension needed)
# =============================================================================


class TestMooncakeStoreL2AdapterConfig:
    """Unit tests for MooncakeStoreL2AdapterConfig."""

    def test_from_dict_minimal(self):
        """Minimal dict with only mooncake keys should work."""
        d = {
            "type": "mooncake_store",
            "local_hostname": "192.168.1.1",
            "metadata_server": "etcd://localhost:2379",
            "global_segment_size": "3221225472",
            "local_buffer_size": "1073741824",
            "protocol": "tcp",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        # LMCache-only keys should be stripped
        assert "type" not in config.setup_config

        # Mooncake keys should be forwarded as strings
        assert config.setup_config["local_hostname"] == "192.168.1.1"
        assert config.setup_config["metadata_server"] == "etcd://localhost:2379"
        assert config.setup_config["protocol"] == "tcp"

        # Default num_workers
        assert config.num_workers == 4

    def test_from_dict_with_num_workers(self):
        """num_workers should be parsed and excluded from setup_config."""
        d = {
            "type": "mooncake_store",
            "num_workers": 8,
            "local_hostname": "10.0.0.1",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert config.num_workers == 8
        assert "num_workers" not in config.setup_config
        assert config.setup_config["local_hostname"] == "10.0.0.1"

    def test_from_dict_with_per_op_workers(self):
        """Per-operation worker counts should be parsed as a dict
        and excluded from setup_config."""
        d = {
            "type": "mooncake_store",
            "num_workers": 16,
            "per_op_workers": {
                "lookup": 4,
                "retrieve": 16,
                "store": 4,
            },
            "local_hostname": "10.0.0.1",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert config.per_op_workers == {"lookup": 4, "retrieve": 16, "store": 4}
        assert "per_op_workers" not in config.setup_config
        assert config.setup_config["local_hostname"] == "10.0.0.1"

    def test_from_dict_forwards_boolean_mooncake_keys_as_strings(self):
        """Non-LMCache boolean keys should be forwarded as strings."""
        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": "127.0.0.1",
                "mooncake_prefer_local_alloc": True,
            }
        )

        assert config.setup_config["mooncake_prefer_local_alloc"] == "True"

    def test_from_dict_forwards_unknown_keys(self):
        """Unknown keys should be forwarded to mooncake unchanged."""
        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "experimental_key": "enabled",
            }
        )

        assert config.setup_config["experimental_key"] == "enabled"

    def test_from_dict_strips_lmcache_only_keys(self):
        """LMCache-only keys should
        not appear in setup_config."""
        d = {
            "type": "mooncake_store",
            "num_workers": 2,
            "eviction": "lru",
            "per_op_workers": {
                "lookup": 3,
                "retrieve": 5,
                "store": 2,
            },
            "local_hostname": "host1",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert "type" not in config.setup_config
        assert "num_workers" not in config.setup_config
        assert "eviction" not in config.setup_config
        assert "per_op_workers" not in config.setup_config
        assert config.setup_config["local_hostname"] == "host1"

    def test_from_dict_converts_values_to_str(self):
        """Non-string values should be converted to strings."""
        d = {
            "type": "mooncake_store",
            "global_segment_size": 3221225472,
            "local_buffer_size": 1073741824,
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert config.setup_config["global_segment_size"] == "3221225472"
        assert config.setup_config["local_buffer_size"] == "1073741824"

    def test_from_dict_skips_none_values(self):
        """Keys with None values should be excluded from setup_config."""
        d = {
            "type": "mooncake_store",
            "local_hostname": "host1",
            "optional_key": None,
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert "optional_key" not in config.setup_config
        assert config.setup_config["local_hostname"] == "host1"

    def test_from_dict_invalid_num_workers_zero(self):
        """num_workers=0 should raise ValueError."""
        d = {"type": "mooncake_store", "num_workers": 0}
        with pytest.raises(ValueError, match="num_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    def test_from_dict_invalid_num_workers_negative(self):
        """Negative num_workers should raise ValueError."""
        d = {"type": "mooncake_store", "num_workers": -1}
        with pytest.raises(ValueError, match="num_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    def test_from_dict_invalid_num_workers_string(self):
        """Non-integer num_workers should raise ValueError."""
        d = {"type": "mooncake_store", "num_workers": "four"}
        with pytest.raises(ValueError, match="num_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    @pytest.mark.parametrize("value", [0, -1, "four"])
    def test_from_dict_invalid_per_op_workers(self, value: Any):
        """Invalid per_op_workers values should raise ValueError."""
        d: dict[str, object] = {
            "type": "mooncake_store",
            "per_op_workers": {
                "lookup": 2,
                "retrieve": value,
                "store": 2,
            },
        }
        with pytest.raises(ValueError, match="per_op_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    def test_from_dict_partial_per_op_workers(self):
        """Partial per_op_workers is valid — unmentioned keys use shared pool."""
        d = {
            "type": "mooncake_store",
            "per_op_workers": {
                "lookup": 4,
                "retrieve": 16,
            },
            "local_hostname": "10.0.0.1",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)
        assert config.per_op_workers == {"lookup": 4, "retrieve": 16}
        assert "store" not in config.per_op_workers

    def test_constructor_copies_setup_config(self):
        """Constructor should copy the setup_config dict."""
        original = {"key": "value"}
        config = MooncakeStoreL2AdapterConfig(setup_config=original)

        # Mutating the original should not affect the config
        original["key"] = "changed"
        assert config.setup_config["key"] == "value"

    def test_help_returns_string(self):
        """help() should return a non-empty string."""
        h = MooncakeStoreL2AdapterConfig.help()
        assert isinstance(h, str)
        assert len(h) > 0


# =============================================================================
# Factory Registration Tests (no C++ extension needed)
# =============================================================================


class TestMooncakeStoreRegistration:
    """Tests for factory and config type registration."""

    def test_mooncake_store_type_registered(self):
        """'mooncake_store' should be in the registered adapter types."""
        assert "mooncake_store" in get_registered_l2_adapter_types()

    def test_config_type_name(self):
        """get_type_name_for_config should return 'mooncake_store'."""
        config = MooncakeStoreL2AdapterConfig(setup_config={})
        name = get_type_name_for_config(config)
        assert name == "mooncake_store"

    def test_factory_raises_without_extension(self):
        """Factory should raise RuntimeError when C++ extension
        is not available."""
        if _native_mooncake_available():
            pytest.skip("C++ Mooncake extension is available")

        config = MooncakeStoreL2AdapterConfig(
            setup_config={"local_hostname": "localhost"},
            num_workers=2,
        )
        with pytest.raises(RuntimeError, match="Mooncake"):
            create_l2_adapter_from_registry(config)


def _install_fake_mooncake_extension(
    monkeypatch: pytest.MonkeyPatch,
    client_cls: type,
) -> None:
    """Install a fake lmcache_mooncake module for factory unit tests."""

    class FakeL1RegistrationConfig:
        def __init__(self):
            self.enabled = False
            self.base = 0
            self.size = 0

    fake_module = types.ModuleType("lmcache.lmcache_mooncake")
    fake_module_any = cast(Any, fake_module)
    fake_module_any.L1RegistrationConfig = FakeL1RegistrationConfig
    fake_module_any.LMCacheMooncakeClient = client_cls
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_mooncake", fake_module)


class TestMooncakeStoreL1RegistrationFactory:
    """Tests for Mooncake TCP/RDMA L1 registration factory behavior.

    RDMA creation must receive a valid L1 memory descriptor; TCP creation
    must not enable preregistration even if a descriptor is provided.
    Uses a fake native extension so these tests do not require Mooncake.
    """

    def test_factory_passes_disabled_l1_registration_for_tcp(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # First Party
        from lmcache.v1.distributed.l2_adapters import native_connector_l2_adapter

        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(
                self,
                config: dict[str, str],
                num_workers: int,
                l1_registration,
                per_op_workers=None,
            ):
                captured["config"] = config
                captured["num_workers"] = num_workers
                captured["l1_registration"] = l1_registration
                captured["per_op_workers"] = per_op_workers

        _install_fake_mooncake_extension(monkeypatch, FakeClient)
        monkeypatch.setattr(
            native_connector_l2_adapter,
            "NativeConnectorL2Adapter",
            lambda client: ("wrapped", client),
        )

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": "127.0.0.1",
                "metadata_server": "P2PHANDSHAKE",
                "num_workers": 3,
                "protocol": "tcp",
            }
        )
        l1_desc = L1MemoryDesc(ptr=123456, size=65536, align_bytes=4096)

        adapter = mooncake_store_module._create_mooncake_store_l2_adapter(
            config,
            l1_memory_desc=l1_desc,
        )
        wrapped_adapter: Any = adapter

        assert wrapped_adapter[0] == "wrapped"
        assert captured["config"] == config.setup_config
        assert captured["num_workers"] == 3
        registration = captured["l1_registration"]
        assert registration.enabled is False
        assert registration.base == 0
        assert registration.size == 0

    def test_factory_passes_enabled_l1_registration_for_rdma(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # First Party
        from lmcache.v1.distributed.l2_adapters import native_connector_l2_adapter

        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(
                self,
                config: dict[str, str],
                num_workers: int,
                l1_registration,
                per_op_workers=None,
            ):
                captured["config"] = config
                captured["num_workers"] = num_workers
                captured["l1_registration"] = l1_registration
                captured["per_op_workers"] = per_op_workers

        _install_fake_mooncake_extension(monkeypatch, FakeClient)
        monkeypatch.setattr(
            native_connector_l2_adapter,
            "NativeConnectorL2Adapter",
            lambda client: ("wrapped", client),
        )

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": "127.0.0.1",
                "metadata_server": "P2PHANDSHAKE",
                "num_workers": 2,
                "protocol": "rdma",
            }
        )
        l1_desc = L1MemoryDesc(ptr=123456, size=65536, align_bytes=4096)

        adapter = mooncake_store_module._create_mooncake_store_l2_adapter(
            config,
            l1_memory_desc=l1_desc,
        )
        wrapped_adapter: Any = adapter

        assert wrapped_adapter[0] == "wrapped"
        assert captured["config"] == config.setup_config
        assert captured["num_workers"] == 2
        registration = captured["l1_registration"]
        assert registration.enabled is True
        assert registration.base == l1_desc.ptr
        assert registration.size == l1_desc.size

    def test_factory_passes_per_op_worker_counts(self, monkeypatch: pytest.MonkeyPatch):
        # First Party
        from lmcache.v1.distributed.l2_adapters import native_connector_l2_adapter

        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(
                self,
                config: dict[str, str],
                num_workers: int,
                l1_registration,
                per_op_workers=None,
            ):
                captured["config"] = config
                captured["num_workers"] = num_workers
                captured["l1_registration"] = l1_registration
                captured["per_op_workers"] = per_op_workers

        _install_fake_mooncake_extension(monkeypatch, FakeClient)
        monkeypatch.setattr(
            native_connector_l2_adapter,
            "NativeConnectorL2Adapter",
            lambda client: ("wrapped", client),
        )

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": "127.0.0.1",
                "num_workers": 16,
                "per_op_workers": {
                    "lookup": 4,
                    "retrieve": 16,
                    "store": 4,
                },
                "protocol": "tcp",
            }
        )

        adapter = mooncake_store_module._create_mooncake_store_l2_adapter(config)
        wrapped_adapter: Any = adapter

        assert wrapped_adapter[0] == "wrapped"
        assert captured["config"] == config.setup_config
        assert captured["num_workers"] == 16
        assert captured["per_op_workers"] == {"lookup": 4, "retrieve": 16, "store": 4}

    def test_factory_requires_l1_memory_descriptor_for_rdma(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        class FakeClient:
            pass

        _install_fake_mooncake_extension(monkeypatch, FakeClient)
        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": "127.0.0.1",
                "protocol": "rdma",
            }
        )

        with pytest.raises(ValueError, match="no L1 memory descriptor"):
            mooncake_store_module._create_mooncake_store_l2_adapter(config)

    def test_factory_rejects_invalid_l1_memory_descriptor_for_rdma(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        class FakeClient:
            pass

        _install_fake_mooncake_extension(monkeypatch, FakeClient)
        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": "127.0.0.1",
                "protocol": "rdma",
            }
        )
        invalid_desc = L1MemoryDesc(ptr=0, size=0, align_bytes=4096)

        with pytest.raises(ValueError, match="invalid"):
            mooncake_store_module._create_mooncake_store_l2_adapter(
                config,
                l1_memory_desc=invalid_desc,
            )


# =============================================================================
# Integration Tests (require C++ Mooncake extension + running service)
# =============================================================================

# Mooncake service connection params from environment
MOONCAKE_LOCAL_HOSTNAME = os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "")
MOONCAKE_METADATA_SERVER = os.environ.get(
    "MOONCAKE_METADATA_SERVER", "etcd://localhost:2379"
)
MOONCAKE_MASTER_SERVER_ADDRESS = os.environ.get(
    "MOONCAKE_MASTER_SERVER_ADDRESS", "localhost:50051"
)
MOONCAKE_DEVICE_NAME = os.environ.get("MOONCAKE_DEVICE_NAME", "")
MOONCAKE_RUN_RDMA_TESTS = os.environ.get("MOONCAKE_RUN_RDMA_TESTS") == "1"

requires_mooncake_service = pytest.mark.skipif(
    not _native_mooncake_available() or not MOONCAKE_LOCAL_HOSTNAME,
    reason=("C++ Mooncake extension not available or MOONCAKE_LOCAL_HOSTNAME not set"),
)
requires_mooncake_rdma = pytest.mark.skipif(
    not MOONCAKE_RUN_RDMA_TESTS,
    reason="RDMA-specific Mooncake test requires MOONCAKE_RUN_RDMA_TESTS=1",
)


@requires_mooncake_service
class TestMooncakeStoreIntegration:
    """Integration tests using real Mooncake Store service.

    These tests require:
    1. The C++ Mooncake extension (lmcache_mooncake) to be built
    2. A running Mooncake Store service
    3. MOONCAKE_LOCAL_HOSTNAME environment variable set

    Set environment variables before running:
        export MOONCAKE_LOCAL_HOSTNAME=<your-ip>
        export MOONCAKE_METADATA_SERVER=etcd://<etcd-host>:2379
    """

    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": MOONCAKE_LOCAL_HOSTNAME,
                "metadata_server": MOONCAKE_METADATA_SERVER,
                "master_server_addr": MOONCAKE_MASTER_SERVER_ADDRESS,
                "num_workers": 2,
            }
        )
        self.adapter = create_l2_adapter(config)
        yield
        adapter = self.adapter
        self.adapter = None
        if adapter is not None:
            adapter.close()

    def test_event_fds_are_distinct(self):
        """Each operation should have a distinct event fd."""
        fds = {
            self.adapter.get_store_event_fd(),
            self.adapter.get_lookup_and_lock_event_fd(),
            self.adapter.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_store_and_lookup(self):
        """Store objects, then verify lookup finds them."""
        keys = [create_object_key(i) for i in range(5)]
        objs = [create_memory_obj(size=64, fill_value=float(i)) for i in range(5)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd)
        completed = self.adapter.pop_completed_store_tasks()
        assert completed[store_tid].is_successful()

        # Lookup all — should find everything
        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap is not None
        for i in range(5):
            assert bitmap.test(i) is True, f"Key {i} not found in lookup"

        # Unlock
        self.adapter.submit_unlock(keys)

    def test_lookup_nonexistent_keys(self):
        """Lookup for keys not stored should return all zeros."""
        keys = [create_object_key(i + 10000) for i in range(3)]
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap is not None
        for i in range(3):
            assert bitmap.test(i) is False

    def test_full_store_lookup_load_workflow(self):
        """End-to-end: store -> lookup -> load, verify data integrity."""
        key = create_object_key(42)
        store_obj = create_memory_obj(size=512, fill_value=3.14)
        load_obj = create_memory_obj(size=512, fill_value=0.0)

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()
        load_fd = self.adapter.get_load_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Lookup
        lookup_tid = self.adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is True

        # Load
        load_tid = self.adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd)
        bitmap = self.adapter.query_load_result(load_tid)
        assert bitmap.test(0) is True

        # Verify data integrity
        assert torch.allclose(load_obj.tensor, store_obj.tensor), (
            "Loaded data does not match stored data"
        )

        # Unlock
        self.adapter.submit_unlock([key])

    def test_batch_store_lookup_load(self):
        """Batch workflow with multiple objects."""
        n = 10
        keys = [create_object_key(i + 100) for i in range(n)]
        store_objs = [
            create_memory_obj(size=128, fill_value=float(i * 7)) for i in range(n)
        ]
        load_objs = [create_memory_obj(size=128, fill_value=0.0) for _ in range(n)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()
        load_fd = self.adapter.get_load_event_fd()

        # Store all
        store_tid = self.adapter.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Lookup all
        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(n):
            assert bitmap.test(i) is True

        # Load all
        load_tid = self.adapter.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd)
        bitmap = self.adapter.query_load_result(load_tid)
        for i in range(n):
            assert bitmap.test(i) is True
            assert torch.allclose(load_objs[i].tensor, store_objs[i].tensor), (
                f"Data mismatch for key {i}"
            )

        self.adapter.submit_unlock(keys)

    def test_mixed_lookup_existing_and_missing(self):
        """Lookup a mix of stored and non-stored keys."""
        stored_keys = [create_object_key(i + 200) for i in range(3)]
        stored_objs = [create_memory_obj(fill_value=float(i)) for i in range(3)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store first 3
        self.adapter.submit_store_task(stored_keys, stored_objs)
        assert wait_for_event_fd(store_fd)
        self.adapter.pop_completed_store_tasks()

        # Lookup 5 keys (3 stored + 2 missing)
        all_keys = stored_keys + [
            create_object_key(10100),
            create_object_key(10101),
        ]
        lookup_tid = self.adapter.submit_lookup_and_lock_task(all_keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)

        for i in range(3):
            assert bitmap.test(i) is True, f"Stored key {i} should be found"
        assert bitmap.test(3) is False, "Missing key should not be found"
        assert bitmap.test(4) is False, "Missing key should not be found"

        self.adapter.submit_unlock(stored_keys)

    # -----------------------------------------------------------------
    # Delete tests
    # -----------------------------------------------------------------

    def test_delete_stored_key(self):
        """Delete a stored key, then lookup confirms it is gone."""
        key = create_object_key(555)
        store_obj = create_memory_obj(size=128, fill_value=7.0)

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Confirm stored
        lookup_tid = self.adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is True
        self.adapter.submit_unlock([key])

        # Delete
        self.adapter.delete([key])

        # Lookup again — should be gone
        lookup_tid = self.adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is False, "Key should not exist after delete"

    def test_delete_nonexistent_keys(self):
        """Deleting keys that don't exist should not raise."""
        keys = [create_object_key(i + 80000) for i in range(3)]
        # Should complete without error
        self.adapter.delete(keys)

    def test_batch_delete_mixed_existing_and_missing(self):
        """Batch delete a mix of stored and non-stored keys."""
        stored_keys = [create_object_key(i + 600) for i in range(3)]
        stored_objs = [
            create_memory_obj(size=64, fill_value=float(i + 1)) for i in range(3)
        ]
        missing_keys = [
            create_object_key(90001),
            create_object_key(90002),
        ]
        all_keys = stored_keys + missing_keys

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store the first 3
        store_tid = self.adapter.submit_store_task(stored_keys, stored_objs)
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Confirm they exist
        lookup_tid = self.adapter.submit_lookup_and_lock_task(
            stored_keys, _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(3):
            assert bitmap.test(i) is True
        self.adapter.submit_unlock(stored_keys)

        # Batch delete mixed keys
        self.adapter.delete(all_keys)

        # Stored keys should now be gone
        lookup_tid = self.adapter.submit_lookup_and_lock_task(
            stored_keys, _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(3):
            assert bitmap.test(i) is False, f"Key {i} should be gone after delete"

    def test_delete_updates_usage_tracking(self):
        """Deleting a stored key reduces tracked byte usage."""
        key = create_object_key(777)
        store_obj = create_memory_obj(size=128, fill_value=9.0)

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Record usage before store
        usage_before = self.adapter.get_usage().total_bytes_used

        # Store
        store_tid = self.adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid].is_successful()

        usage_after_store = self.adapter.get_usage().total_bytes_used
        assert usage_after_store > usage_before, "Usage should increase after store"

        # Confirm stored in lookup so _key_sizes is populated
        _ = self.adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd)
        self.adapter.submit_unlock([key])

        # Delete
        self.adapter.delete([key])

        usage_after_delete = self.adapter.get_usage().total_bytes_used
        assert usage_after_delete == usage_before, (
            f"Usage should return to baseline after delete: "
            f"before={usage_before}, after_delete={usage_after_delete}"
        )

    def test_factory_creates_adapter(self):
        """Verify the factory can create a Mooncake Store L2 adapter."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": MOONCAKE_LOCAL_HOSTNAME,
                "metadata_server": MOONCAKE_METADATA_SERVER,
                "master_server_addr": MOONCAKE_MASTER_SERVER_ADDRESS,
                "num_workers": 2,
            }
        )
        adapter = create_l2_adapter(config)
        try:
            # Should have valid event fds
            assert adapter.get_store_event_fd() >= 0
            assert adapter.get_lookup_and_lock_event_fd() >= 0
            assert adapter.get_load_event_fd() >= 0
        finally:
            adapter.close()

    @requires_mooncake_rdma
    def test_buffer_backed_store_lookup_load(self):
        """Store and load RDMA-preregistered objects backed by an explicit L1 buffer."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        # The class-level fixture creates a default TCP adapter. Close it before
        # creating the RDMA adapter, otherwise Mooncake master may allocate this
        # test object's replica on the TCP segment and the RDMA-only client will
        # fail with NotSupportedTransport.
        self.adapter.close()
        self.adapter = None

        page_size = 4096
        obj_size_bytes = page_size * 16
        l1_buffer = torch.empty(page_size * 256, dtype=torch.uint8, device="cpu")
        l1_desc = L1MemoryDesc(
            ptr=l1_buffer.data_ptr(),
            size=l1_buffer.numel(),
            align_bytes=page_size,
        )

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": MOONCAKE_LOCAL_HOSTNAME,
                "metadata_server": MOONCAKE_METADATA_SERVER,
                "master_server_addr": MOONCAKE_MASTER_SERVER_ADDRESS,
                "num_workers": 2,
                "protocol": "rdma",
                "rdma_devices": MOONCAKE_DEVICE_NAME,
            }
        )
        adapter = create_l2_adapter(config, l1_memory_desc=l1_desc)
        try:
            key = create_object_key(9001, model_name="preregister_model")
            store_obj = create_buffer_memory_obj(
                l1_buffer,
                offset_bytes=0,
                size_bytes=obj_size_bytes,
                fill_value=6.25,
            )
            load_obj = create_buffer_memory_obj(
                l1_buffer,
                offset_bytes=obj_size_bytes * 2,
                size_bytes=obj_size_bytes,
                fill_value=0.0,
            )

            store_tid = adapter.submit_store_task([key], [store_obj])
            assert wait_for_event_fd(adapter.get_store_event_fd())
            assert adapter.pop_completed_store_tasks()[store_tid].is_successful()

            lookup_tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
            assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
            lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
            assert lookup_bitmap.test(0) is True

            load_tid = adapter.submit_load_task([key], [load_obj])
            assert wait_for_event_fd(adapter.get_load_event_fd())
            load_bitmap = adapter.query_load_result(load_tid)
            assert load_bitmap.test(0) is True
            assert torch.equal(load_obj.tensor, store_obj.tensor)

            adapter.submit_unlock([key])
        finally:
            adapter.close()
