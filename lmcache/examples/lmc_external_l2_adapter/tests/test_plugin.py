# SPDX-License-Identifier: Apache-2.0
"""
Tests for the lmc_external_l2_adapter plugin.

Validates that:
1. The adapter class is importable and recognised by the
   LMCache external adapter factory.
2. PluginL2AdapterConfig can parse the plugin's JSON.
3. On Linux the adapter can be fully instantiated and
   performs store / lookup / load round-trips with
   realistic tensor sizes (~4 KB+ per object).
4. FIFO eviction works when the cache is full.
"""

# Standard
from typing import TYPE_CHECKING
import json
import platform
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.config import (
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.plugin_l2_adapter import PluginL2AdapterConfig
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

if TYPE_CHECKING:
    pass

try:
    # Third Party
    import lmc_external_l2_adapter  # noqa: F401

    _HAS_PLUGIN = True
except ModuleNotFoundError:
    _HAS_PLUGIN = False

_skip_unless_plugin = pytest.mark.skipif(
    not _HAS_PLUGIN,
    reason="lmc_external_l2_adapter is not installed",
)

# -- helpers --------------------------------------------------

# Default object size: 1024 float32 = 4 KB
OBJ_SIZE = 1024


def _make_key(tag: int) -> ObjectKey:
    """Create an ObjectKey compatible with both old
    and new API signatures."""
    # Standard
    import inspect

    sig = inspect.signature(ObjectKey)
    params = list(sig.parameters)
    if "chunk_hash" in params and "kv_rank" in params:
        # New API: (chunk_hash: bytes, model_name, kv_rank)
        return ObjectKey(
            chunk_hash=tag.to_bytes(8, "big"),
            model_name="test",
            kv_rank=0,
        )
    # Old API: (model_name, world_size, worker_id, chunk_hash)
    return ObjectKey(  # type: ignore[call-arg]
        model_name="test",
        world_size=1,
        worker_id=0,
        chunk_hash=tag,  # type: ignore[arg-type]
    )


def _make_tensor_obj(
    tensor: torch.Tensor,
) -> TensorMemoryObj:
    """Create a TensorMemoryObj compatible with both
    old and new API."""
    # Standard
    import inspect

    sig = inspect.signature(TensorMemoryObj)
    params = list(sig.parameters)

    # New API requires MemoryObjMetadata
    if "metadata" in params:
        try:
            # First Party
            from lmcache.v1.memory_management import (
                MemoryObjMetadata,
            )

            meta = MemoryObjMetadata(
                shape=tensor.shape,
                dtype=tensor.dtype,
                address=tensor.data_ptr(),
                phy_size=(tensor.nelement() * tensor.element_size()),
                ref_count=1,
            )
            return TensorMemoryObj(
                raw_data=tensor,
                metadata=meta,
                parent_allocator=None,
            )
        except Exception:
            pass

    # Old API: metadata=None is fine
    return TensorMemoryObj(
        raw_data=tensor,
        metadata=None,  # type: ignore[arg-type]
        parent_allocator=None,
    )


def _create_obj(
    size: int = OBJ_SIZE,
    fill: float = 1.0,
) -> TensorMemoryObj:
    """Create a test TensorMemoryObj filled with *fill*."""
    t = torch.empty(size, dtype=torch.float32)
    t.fill_(fill)
    return _make_tensor_obj(t)


def _wait_event_fd(fd: int, timeout: float = 5.0) -> bool:
    """Wait for an eventfd to be signaled."""
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        return True
    return False


# -- constants ------------------------------------------------

ADAPTER_JSON = json.dumps(
    {
        "type": "plugin",
        "module_path": "lmc_external_l2_adapter",
        "class_name": "InMemoryL2Adapter",
        "adapter_params": {
            # ~10 KB capacity: fits 2x 4KB objects
            "max_size_gb": 10240 / (1024**3),
            "mock_bandwidth_gb": 10.0,
        },
    }
)

# Config-class mode: adapter receives a real config object
ADAPTER_JSON_CFG = json.dumps(
    {
        "type": "plugin",
        "module_path": "lmc_external_l2_adapter",
        "class_name": "InMemoryL2Adapter",
        "config_class_name": "InMemoryL2AdapterConfig",
        "adapter_params": {
            "max_size_gb": 10240 / (1024**3),
            "mock_bandwidth_gb": 10.0,
        },
    }
)

_LINUX = platform.system() == "Linux"
_skip_unless_linux = pytest.mark.skipif(
    not _LINUX,
    reason="eventfd requires Linux",
)


# -- config parsing tests ------------------------------------


class TestPluginConfigParsing:
    """Config-only tests (no instantiation)."""

    def test_from_dict(self):
        cfg = PluginL2AdapterConfig.from_dict(json.loads(ADAPTER_JSON))
        assert cfg.module_path == "lmc_external_l2_adapter"
        assert cfg.class_name == "InMemoryL2Adapter"
        assert cfg.config_class_name is None

    def test_from_dict_with_config_class(self):
        cfg = PluginL2AdapterConfig.from_dict(json.loads(ADAPTER_JSON_CFG))
        assert cfg.module_path == "lmc_external_l2_adapter"
        assert cfg.class_name == "InMemoryL2Adapter"
        assert cfg.config_class_name == "InMemoryL2AdapterConfig"

    def test_parse_args(self):
        """Simulate --l2-adapter CLI argument."""
        # Standard
        import argparse

        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            add_l2_adapters_args,
        )

        parser = argparse.ArgumentParser()
        add_l2_adapters_args(parser)
        args = parser.parse_args(["--l2-adapter", ADAPTER_JSON])
        config = parse_args_to_l2_adapters_config(args)
        assert len(config.adapters) == 1
        cfg = config.adapters[0]
        assert isinstance(cfg, PluginL2AdapterConfig)
        assert cfg.class_name == "InMemoryL2Adapter"


# -- import / subclass tests ---------------------------------


@_skip_unless_plugin
class TestPluginImport:
    """Verify the class is importable and correct."""

    def test_import(self):
        # Third Party
        from lmc_external_l2_adapter import (
            InMemoryL2Adapter,
        )

        # First Party
        from lmcache.v1.distributed.l2_adapters.base import (
            L2AdapterInterface,
        )

        assert issubclass(InMemoryL2Adapter, L2AdapterInterface)


# -- full round-trip tests (Linux only) ----------------------


@_skip_unless_plugin
@_skip_unless_linux
class TestPluginRoundTrip:
    """End-to-end tests that create the adapter via the
    external factory and exercise store/lookup/load with
    realistic 4 KB objects."""

    @pytest.fixture(params=["dict", "config_class"])
    def adapter(self, request):
        spec = ADAPTER_JSON if request.param == "dict" else ADAPTER_JSON_CFG
        cfg = PluginL2AdapterConfig.from_dict(json.loads(spec))
        inst = create_l2_adapter_from_registry(cfg)
        yield inst
        inst.close()

    # -- basic interface tests --

    def test_create_via_registry(self, adapter):
        # Third Party
        from lmc_external_l2_adapter import (
            InMemoryL2Adapter,
        )

        assert isinstance(adapter, InMemoryL2Adapter)

    def test_event_fds_are_distinct(self, adapter):
        fds = {
            adapter.get_store_event_fd(),
            adapter.get_lookup_and_lock_event_fd(),
            adapter.get_load_event_fd(),
        }
        assert len(fds) == 3

    # -- store & lookup with 4 KB objects --

    def test_store_and_lookup(self, adapter):
        """Store a 4 KB object then look it up."""
        key = _make_key(42)
        obj = _create_obj(OBJ_SIZE, fill=3.14)
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        tid = adapter.submit_store_task([key], [obj])
        assert _wait_event_fd(store_fd)
        done = adapter.pop_completed_store_tasks()
        assert done.get(tid) is not None
        assert done[tid].is_successful()

        ltid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert _wait_event_fd(lookup_fd)
        bm = adapter.query_lookup_and_lock_result(ltid)
        assert bm is not None
        assert bm.test(0) is True
        adapter.submit_unlock([key])

    # -- store & load with 4 KB objects --

    def test_store_and_load(self, adapter):
        """Store 4 KB then load back and compare."""
        key = _make_key(99)
        src = torch.arange(OBJ_SIZE, dtype=torch.float32)
        obj = _make_tensor_obj(src.clone())
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        tid = adapter.submit_store_task([key], [obj])
        assert _wait_event_fd(store_fd)
        done = adapter.pop_completed_store_tasks()
        assert done.get(tid) is not None
        assert done[tid].is_successful()

        dst = torch.zeros_like(src)
        load_obj = _make_tensor_obj(dst)
        ltid = adapter.submit_load_task([key], [load_obj])
        assert _wait_event_fd(load_fd)
        bm = adapter.query_load_result(ltid)
        assert bm is not None
        assert bm.test(0) is True
        assert torch.equal(dst, src)

    # -- batch store --

    def test_batch_store_and_lookup(self, adapter):
        """Store a batch of 2 objects then lookup both."""
        keys = [_make_key(i) for i in range(2)]
        objs = [_create_obj(OBJ_SIZE, fill=float(i)) for i in range(2)]
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        tid = adapter.submit_store_task(keys, objs)
        assert _wait_event_fd(store_fd)
        done = adapter.pop_completed_store_tasks()
        assert done.get(tid) is not None
        assert done[tid].is_successful()

        ltid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert _wait_event_fd(lookup_fd)
        bm = adapter.query_lookup_and_lock_result(ltid)
        assert bm is not None
        for i in range(2):
            assert bm.test(i) is True
        adapter.submit_unlock(keys)

    # -- mixed lookup (existing + non-existing) --

    def test_lookup_mixed_keys(self, adapter):
        """Lookup of existing + non-existing keys."""
        stored_key = _make_key(200)
        missing_key = _make_key(999)
        obj = _create_obj(OBJ_SIZE)
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_store_task([stored_key], [obj])
        assert _wait_event_fd(store_fd)
        adapter.pop_completed_store_tasks()

        ltid = adapter.submit_lookup_and_lock_task(
            [stored_key, missing_key], _EMPTY_LAYOUT
        )
        assert _wait_event_fd(lookup_fd)
        bm = adapter.query_lookup_and_lock_result(ltid)
        assert bm is not None
        assert bm.test(0) is True  # stored
        assert bm.test(1) is False  # missing
        adapter.submit_unlock([stored_key])

    # -- eviction test --

    def test_fifo_eviction(self, adapter):
        """When the cache is full, FIFO eviction should
        remove the oldest entry.

        Config gives ~10 KB capacity; each object is
        4 KB (1024 * 4 bytes).  Storing 3 objects
        should evict the first one.
        """
        k1 = _make_key(301)
        k2 = _make_key(302)
        k3 = _make_key(303)
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store k1, k2 (fits within 10 KB)
        for k in [k1, k2]:
            tid = adapter.submit_store_task([k], [_create_obj(OBJ_SIZE)])
            assert _wait_event_fd(store_fd)
            done = adapter.pop_completed_store_tasks()
            assert done.get(tid) is not None
            assert done[tid].is_successful()

        # Store k3 -- should evict k1
        tid = adapter.submit_store_task([k3], [_create_obj(OBJ_SIZE)])
        assert _wait_event_fd(store_fd)
        done = adapter.pop_completed_store_tasks()
        assert done.get(tid) is not None
        assert done[tid].is_successful()

        # Lookup all three
        ltid = adapter.submit_lookup_and_lock_task([k1, k2, k3], _EMPTY_LAYOUT)
        assert _wait_event_fd(lookup_fd)
        bm = adapter.query_lookup_and_lock_result(ltid)
        assert bm is not None
        assert bm.test(0) is False  # k1 evicted
        assert bm.test(1) is True  # k2 present
        assert bm.test(2) is True  # k3 present
        adapter.submit_unlock([k2, k3])
