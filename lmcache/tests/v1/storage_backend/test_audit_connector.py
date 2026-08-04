# SPDX-License-Identifier: Apache-2.0
# Standard
from io import StringIO
import asyncio
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.audit_connector import AuditConnector
from lmcache.v1.storage_backend.connector.mock_connector import MockConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def create_test_key(key_id: str) -> CacheEngineKey:
    """Helper to create a test CacheEngineKey"""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.uint8,
    )


def create_mock_memory_obj(backend: LocalCPUBackend, data: bytes) -> MemoryObj:
    """Helper to create a mock MemoryObj with proper structure"""
    tensor = torch.tensor(
        [ord(c) for c in data.decode("latin1") if ord(c) < 256], dtype=torch.uint8
    )
    if len(tensor) == 0:
        tensor = torch.tensor([0], dtype=torch.uint8)

    memory_obj = backend.allocate(
        shapes=[tensor.shape],
        dtypes=[tensor.dtype],
        fmt=MemoryFormat.KV_2LTD,
    )

    if memory_obj is not None and isinstance(memory_obj, TensorMemoryObj):
        memory_obj.tensor.copy_(tensor)

    return memory_obj


@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop

    # Simplified cleanup logic
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        if not loop.is_closed():
            loop.call_soon(loop.stop)
            loop.run_forever()
            loop.close()
    except Exception:
        pass
    finally:
        asyncio.set_event_loop(None)


@pytest.fixture
def local_cpu_backend():
    """Fixture for LocalCPUBackend"""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=1,
        remote_url="mock://test",
        extra_config={},
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(64, 2, 1, 8, 128),
    )
    allocator = AdHocMemoryAllocator(device="cpu")
    return LocalCPUBackend(config=config, metadata=metadata, memory_allocator=allocator)


@pytest.fixture
def mock_connector(event_loop, local_cpu_backend):
    """Fixture for mock connector"""
    connector = MockConnector(
        url="mock://test",
        loop=event_loop,
        local_cpu_backend=local_cpu_backend,
        capacity=1,
        peeking_latency=0.0,
        read_throughput=100.0,
        write_throughput=100.0,
    )
    yield connector
    # Clean up connector to ensure AsyncPQExecutor is properly shut down
    try:
        # Close the connector which will shutdown the AsyncPQExecutor
        connector.close()
    except Exception as e:
        # Log but don't fail the test
        print(f"Failed to close mock connector: {e}")


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.stream = StringIO()

    def emit(self, record):
        self.records.append(record)
        msg = self.format(record)
        self.stream.write(msg + "\n")

    def get_records(self):
        return self.records

    def get_logs(self):
        return self.stream.getvalue()

    def clear(self):
        self.records = []
        self.stream = StringIO()


@pytest.fixture
def log_capture():
    """Fixture for capturing logs"""
    handler = LogCaptureHandler()
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

    # Get audit logger and add handler
    audit_logger = logging.getLogger(
        "lmcache.v1.storage_backend.connector.audit_connector"
    )
    original_level = audit_logger.level
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(handler)

    yield handler

    # Cleanup
    audit_logger.removeHandler(handler)
    audit_logger.setLevel(original_level)


class TestAuditConnector:
    """Test AuditConnector functionality"""

    def test_initialization_basic(self, mock_connector, log_capture):
        """Test basic initialization"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )
        audit = AuditConnector(mock_connector, config)

        assert audit.real_connector is mock_connector
        assert audit.verify_checksum is False
        assert audit.calc_checksum is False
        assert len(audit.excluded_cmds) == 0

        print("\nCaptured logs:")
        print(log_capture.get_logs())

        init_logs = [
            r.msg
            for r in log_capture.get_records()
            if "[REMOTE_AUDIT]" in r.msg and "INITIALIZED" in r.msg
        ]

        assert len(init_logs) > 0, (
            f"Expected INITIALIZED log. Got logs: {log_capture.get_logs()}"
        )

    def test_initialization_with_checksum(self, mock_connector, log_capture):
        """Test initialization with checksum enabled"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={
                "audit_calc_checksum": True,
                "audit_verify_checksum": True,
            },
        )
        audit = AuditConnector(mock_connector, config)

        assert audit.verify_checksum is True
        assert audit.calc_checksum is True
        assert audit.registry_lock is not None

        print("\nCaptured logs:")
        print(log_capture.get_logs())

        init_logs = [r for r in log_capture.get_records() if "INITIALIZED" in r.msg]
        assert len(init_logs) > 0, (
            f"Expected INITIALIZED log. Got logs: {log_capture.get_logs()}"
        )
        assert "Calc Checksum:True" in init_logs[0].msg

    def test_initialization_with_excluded_cmds(self, mock_connector, log_capture):
        """Test initialization with excluded commands"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={
                "audit_exclude_cmds": "exists,list",
            },
        )
        audit = AuditConnector(mock_connector, config)

        assert "exists" in audit.excluded_cmds
        assert "list" in audit.excluded_cmds

        print("\nCaptured logs:")
        print(log_capture.get_logs())

        init_logs = [r for r in log_capture.get_records() if "INITIALIZED" in r.msg]
        assert len(init_logs) > 0, (
            f"Expected INITIALIZED log. Got logs: {log_capture.get_logs()}"
        )
        assert "Excluded Cmds:" in init_logs[0].msg

    def test_put_and_get_with_audit_log(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test put and get operations with audit log verification"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key = create_test_key("test_key")
            memory_obj = create_mock_memory_obj(local_cpu_backend, b"test_data")

            log_capture.clear()

            await audit.put(key, memory_obj)

            print("\nAfter PUT:")
            print(log_capture.get_logs())

            put_logs = [
                r
                for r in log_capture.get_records()
                if "PUT" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(put_logs) > 0, (
                f"Expected PUT audit log. Got logs: {log_capture.get_logs()}"
            )
            assert "Cost:" in put_logs[0].msg
            assert "Size:" in put_logs[0].msg

            log_capture.clear()

            result = await audit.get(key)
            assert result is not None
            assert result.metadata.shape == memory_obj.metadata.shape

            print("\nAfter GET:")
            print(log_capture.get_logs())

            get_logs = [
                r
                for r in log_capture.get_records()
                if "GET" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(get_logs) > 0, (
                f"Expected GET audit log. Got logs: {log_capture.get_logs()}"
            )
            assert "Cost:" in get_logs[0].msg

        event_loop.run_until_complete(run_test())

    def test_exists_with_audit_log(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test exists operation with audit log"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key = create_test_key("test_key")

            log_capture.clear()

            result = await audit.exists(key)
            assert result is False

            print("\nAfter EXISTS:")
            print(log_capture.get_logs())

            exists_logs = [
                r
                for r in log_capture.get_records()
                if "EXISTS" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(exists_logs) > 0, (
                f"Expected EXISTS audit log. Got logs: {log_capture.get_logs()}"
            )
            assert "Cost:" in exists_logs[0].msg

        event_loop.run_until_complete(run_test())

    def test_put_with_checksum_audit_log(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test put with checksum calculation and audit log"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={
                "audit_calc_checksum": True,
                "audit_verify_checksum": True,
            },
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key = create_test_key("test_key")
            memory_obj = create_mock_memory_obj(local_cpu_backend, b"test_data")

            log_capture.clear()

            await audit.put(key, memory_obj)

            print("\nAfter PUT with checksum:")
            print(log_capture.get_logs())

            put_logs = [
                r
                for r in log_capture.get_records()
                if "PUT" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(put_logs) > 0, (
                f"Expected PUT audit log. Got logs: {log_capture.get_logs()}"
            )
            assert "Checksum:" in put_logs[0].msg
            assert "Checksum:N/A" not in put_logs[0].msg

        event_loop.run_until_complete(run_test())

    def test_put_and_get_with_kwargs(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test put and get operations with keyword arguments
        to ensure robust argument handling."""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key = create_test_key("test_key_kwargs")
            memory_obj = create_mock_memory_obj(local_cpu_backend, b"test_data_kwargs")

            log_capture.clear()

            # Use keyword arguments to test argument passing
            await audit.put(key=key, memory_obj=memory_obj)

            put_logs = [
                r
                for r in log_capture.get_records()
                if "PUT" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(put_logs) > 0, "Expected PUT audit log with keyword arguments."

            log_capture.clear()

            result = await audit.get(key=key)
            assert result is not None

            get_logs = [
                r
                for r in log_capture.get_records()
                if "GET" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(get_logs) > 0, "Expected GET audit log with keyword arguments."

        event_loop.run_until_complete(run_test())

    def test_excluded_command_no_audit_log(
        self, mock_connector, event_loop, log_capture
    ):
        """Test that excluded commands don't generate audit logs"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={
                "audit_exclude_cmds": "exists",
            },
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key = create_test_key("test_key")

            log_capture.clear()

            await audit.exists(key)

            print("\nAfter excluded EXISTS:")
            print(log_capture.get_logs())

            exists_logs = [
                r
                for r in log_capture.get_records()
                if "EXISTS" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(exists_logs) == 0, (
                f"Excluded command should not log. Got logs: {log_capture.get_logs()}"
            )

        event_loop.run_until_complete(run_test())

    def test_non_excluded_command_has_audit_log(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test that non-excluded commands generate audit logs"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={
                "audit_exclude_cmds": "exists",
            },
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key = create_test_key("test_key")
            memory_obj = create_mock_memory_obj(local_cpu_backend, b"test_data")

            log_capture.clear()

            await audit.put(key, memory_obj)

            print("\nAfter non-excluded PUT:")
            print(log_capture.get_logs())

            put_logs = [
                r
                for r in log_capture.get_records()
                if "PUT" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(put_logs) > 0, (
                f"Non-excluded command should log. Got logs: {log_capture.get_logs()}"
            )

        event_loop.run_until_complete(run_test())

    def test_not_audit_decorator(self, mock_connector, log_capture):
        """Test that @NotAudit methods don't generate operation audit logs"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )
        audit = AuditConnector(mock_connector, config)

        log_capture.clear()

        audit.post_init()

        print("\nAfter @NotAudit methods:")
        print(log_capture.get_logs())

        operation_logs = [r for r in log_capture.get_records() if "POST_INIT" in r.msg]
        operation_logs = [r for r in operation_logs if "SUCCESS" in r.msg]
        assert len(operation_logs) == 0, (
            f"@NotAudit methods should not generate operation audit logs. "
            f"Got logs: {log_capture.get_logs()}"
        )

    def test_batched_get_with_audit_log(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test batched get with audit log"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key1 = create_test_key("key1")
            key2 = create_test_key("key2")

            await audit.put(key1, create_mock_memory_obj(local_cpu_backend, b"data1"))
            await audit.put(key2, create_mock_memory_obj(local_cpu_backend, b"data2"))

            log_capture.clear()

            results = await audit.batched_get([key1, key2])
            assert len(results) == 2
            assert all(r is not None for r in results)

            print("\nAfter BATCHED_GET:")
            print(log_capture.get_logs())

            batched_logs = [
                r
                for r in log_capture.get_records()
                if "BATCHED_GET" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(batched_logs) > 0, (
                f"Expected BATCHED_GET audit log. Got logs: {log_capture.get_logs()}"
            )
            assert "Cost:" in batched_logs[0].msg

        event_loop.run_until_complete(run_test())

    def test_batched_async_contains_with_audit_log(
        self, mock_connector, local_cpu_backend, event_loop, log_capture
    ):
        """Test batched async contains with audit log"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )

        async def run_test():
            audit = AuditConnector(mock_connector, config)
            key1 = create_test_key("key1")
            key2 = create_test_key("key2")
            key3 = create_test_key("key3")

            await audit.put(key1, create_mock_memory_obj(local_cpu_backend, b"data1"))
            await audit.put(key2, create_mock_memory_obj(local_cpu_backend, b"data2"))

            log_capture.clear()

            count = await audit.batched_async_contains("lookup1", [key1, key2, key3])
            assert count == 2

            print("\nAfter BATCHED_ASYNC_CONTAINS:")
            print(log_capture.get_logs())

            batched_logs = [
                r
                for r in log_capture.get_records()
                if "BATCHED_ASYNC_CONTAINS" in r.msg and "SUCCESS" in r.msg
            ]
            assert len(batched_logs) > 0, (
                f"Expected BATCHED_ASYNC_CONTAINS audit log. "
                f"Got logs: {log_capture.get_logs()}"
            )
            assert "Cost:" in batched_logs[0].msg

        event_loop.run_until_complete(run_test())

    def test_support_methods(self, mock_connector):
        """Test support_* methods are properly forwarded"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )
        audit = AuditConnector(mock_connector, config)

        assert audit.support_batched_get() is True
        assert audit.support_batched_async_contains() is True

    def test_exists_sync_with_audit_log(self, mock_connector, log_capture):
        """Test exists_sync synchronous operation with audit log"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={},
        )
        audit = AuditConnector(mock_connector, config)
        key = create_test_key("test_key")

        log_capture.clear()

        result = audit.exists_sync(key)
        assert result is False

        print("\nAfter EXISTS_SYNC:")
        print(log_capture.get_logs())

        exists_sync_logs = [
            r
            for r in log_capture.get_records()
            if "EXISTS_SYNC" in r.msg and "SUCCESS" in r.msg
        ]
        assert len(exists_sync_logs) > 0, (
            f"Expected EXISTS_SYNC audit log. Got logs: {log_capture.get_logs()}"
        )
        assert "Cost:" in exists_sync_logs[0].msg

    def test_exists_sync_excluded_no_audit_log(self, mock_connector, log_capture):
        """Test that excluded exists_sync doesn't generate audit logs"""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=1,
            remote_url="mock://test",
            extra_config={
                "audit_exclude_cmds": "exists_sync",
            },
        )
        audit = AuditConnector(mock_connector, config)
        key = create_test_key("test_key")

        log_capture.clear()

        result = audit.exists_sync(key)
        assert result is False

        print("\nAfter excluded EXISTS_SYNC:")
        print(log_capture.get_logs())

        exists_sync_logs = [
            r
            for r in log_capture.get_records()
            if "EXISTS_SYNC" in r.msg and "SUCCESS" in r.msg
        ]
        assert len(exists_sync_logs) == 0, (
            f"Excluded exists_sync should not log. Got logs: {log_capture.get_logs()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
