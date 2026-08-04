# SPDX-License-Identifier: Apache-2.0
# Standard
from threading import Lock
from typing import Dict, Optional
import abc
import asyncio
import functools
import hashlib
import logging
import time

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = logging.getLogger(__name__)


class AuditConnectorMeta(abc.ABCMeta):
    """Metaclass that dynamically generates wrapper methods for all
    RemoteConnector methods
    """

    def __new__(mcs, name, bases, namespace):
        # Get all methods from RemoteConnector including abstract methods
        # We need to check both the class dict and inherited methods
        all_methods = {}

        # Collect methods from RemoteConnector and its bases
        for base in RemoteConnector.__mro__:
            for method_name, method_obj in base.__dict__.items():
                if method_name not in all_methods and callable(method_obj):
                    all_methods[method_name] = method_obj

        for method_name, method in all_methods.items():
            # Skip private methods and methods already defined in namespace
            if method_name.startswith("_") or method_name in namespace:
                continue

            # Skip class methods, static methods, and properties
            if isinstance(method, (classmethod, staticmethod, property)):
                continue

            # Check if method is marked with @NotAudit
            is_not_audit = getattr(method, "_not_audit", False)

            # Determine if method is async
            is_async = asyncio.iscoroutinefunction(method)

            # Create appropriate wrapper and add to namespace
            if is_not_audit:
                if is_async:
                    namespace[method_name] = mcs._create_passthrough_async_method(
                        method_name, method
                    )
                else:
                    namespace[method_name] = mcs._create_passthrough_sync_method(
                        method_name, method
                    )
            else:
                if is_async:
                    namespace[method_name] = mcs._create_audit_async_method(
                        method_name, method
                    )
                else:
                    namespace[method_name] = mcs._create_audit_sync_method(
                        method_name, method
                    )

        # Create the class with all methods in namespace
        cls = super().__new__(mcs, name, bases, namespace)

        # Clear abstract methods since we've implemented them all via wrappers
        # The wrappers delegate to real_connector which has the actual implementations
        if hasattr(cls, "__abstractmethods__"):
            cls.__abstractmethods__ = frozenset()

        return cls

    @staticmethod
    def _create_passthrough_async_method(method_name: str, original_method):
        """Create a pass-through async method without logging"""

        @functools.wraps(original_method)
        async def wrapper(self, *args, **kwargs):
            real_method = getattr(self.real_connector, method_name)
            return await real_method(*args, **kwargs)

        wrapper.__name__ = method_name
        wrapper.__qualname__ = f"AuditConnector.{method_name}"
        return wrapper

    @staticmethod
    def _create_passthrough_sync_method(method_name: str, original_method):
        """Create a pass-through sync method without logging"""

        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            real_method = getattr(self.real_connector, method_name)
            return real_method(*args, **kwargs)

        wrapper.__name__ = method_name
        wrapper.__qualname__ = f"AuditConnector.{method_name}"
        return wrapper

    @staticmethod
    def _create_audit_async_method(method_name: str, original_method):
        """Create an audit async method with logging"""

        @functools.wraps(original_method)
        async def wrapper(self, *args, **kwargs):
            # Special handling for put/get methods with checksum
            if method_name == "put":
                return await self._audit_put(*args, **kwargs)
            elif method_name == "get":
                return await self._audit_get(*args, **kwargs)

            # Check if method is in excluded commands
            if hasattr(self, "excluded_cmds") and method_name in self.excluded_cmds:
                real_method = getattr(self.real_connector, method_name)
                return await real_method(*args, **kwargs)

            # Generic audit logging
            self.logger.debug(
                f"[REMOTE_AUDIT][{self.real_connector}]:{method_name.upper()}|START"
            )
            t1 = time.perf_counter()
            try:
                real_method = getattr(self.real_connector, method_name)
                result = await real_method(*args, **kwargs)
                t2 = time.perf_counter()
                cost = (t2 - t1) * 1000
                self.logger.info(
                    f"[REMOTE_AUDIT][{self.real_connector}]:{method_name.upper()}|"
                    f"SUCCESS|Cost:{cost:.6f}ms"
                )
                return result
            except Exception as e:
                self.logger.error(
                    f"[REMOTE_AUDIT][{self.real_connector}]:{method_name.upper()}|"
                    f"FAILED|Error: {str(e)}"
                )
                raise

        wrapper.__name__ = method_name
        wrapper.__qualname__ = f"AuditConnector.{method_name}"
        return wrapper

    @staticmethod
    def _create_audit_sync_method(method_name: str, original_method):
        """Create an audit sync method with logging"""

        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            # Check if method is in excluded commands
            if hasattr(self, "excluded_cmds") and method_name in self.excluded_cmds:
                real_method = getattr(self.real_connector, method_name)
                return real_method(*args, **kwargs)

            self.logger.debug(
                f"[REMOTE_AUDIT][{self.real_connector}]:{method_name.upper()}|START"
            )
            t1 = time.perf_counter()
            try:
                real_method = getattr(self.real_connector, method_name)
                result = real_method(*args, **kwargs)
                t2 = time.perf_counter()
                cost = (t2 - t1) * 1000
                self.logger.info(
                    f"[REMOTE_AUDIT][{self.real_connector}]:{method_name.upper()}|"
                    f"SUCCESS|Cost:{cost:.6f}ms"
                )
                return result
            except Exception as e:
                self.logger.error(
                    f"[REMOTE_AUDIT][{self.real_connector}]:{method_name.upper()}|"
                    f"FAILED|Error: {str(e)}"
                )
                raise

        wrapper.__name__ = method_name
        wrapper.__qualname__ = f"AuditConnector.{method_name}"
        return wrapper


class AuditConnector(RemoteConnector, metaclass=AuditConnectorMeta):
    """Audit wrapper for RemoteConnector that dynamically wraps all methods.

    Features:
    - Automatically wraps all RemoteConnector methods
    - Methods marked with @NotAudit are forwarded without logging
    - Configurable checksum verification via URL parameter
    - Logs all operations with timestamps
    - Optional checksum validation for put/get operations
    """

    def __init__(
        self, real_connector: RemoteConnector, lmcache_config: LMCacheEngineConfig
    ):
        self.real_connector = real_connector
        self.verify_checksum = (
            lmcache_config.extra_config is not None
            and "audit_verify_checksum" in lmcache_config.extra_config
            and lmcache_config.extra_config["audit_verify_checksum"]
        )
        self.calc_checksum = (
            lmcache_config.extra_config is not None
            and "audit_calc_checksum" in lmcache_config.extra_config
            and lmcache_config.extra_config["audit_calc_checksum"]
        )
        self.checksum_registry: Dict[CacheEngineKey, str] = {}
        self.registry_lock = Lock() if self.verify_checksum else None

        # Parse audit exclude commands
        self.excluded_cmds = set()
        if (
            lmcache_config.extra_config
            and "audit_exclude_cmds" in lmcache_config.extra_config
        ):
            exclude_cmds = lmcache_config.extra_config["audit_exclude_cmds"]
            if exclude_cmds:
                self.excluded_cmds = {cmd.strip() for cmd in exclude_cmds.split(",")}

        self.logger = logger.getChild("audit")

        logger.info(
            f"[REMOTE_AUDIT][{self.real_connector}]:INITIALIZED|"
            f"Calc Checksum:{self.calc_checksum}｜"
            f"Verify Checksum: {self.verify_checksum}|"
            f"Excluded Cmds: {self.excluded_cmds}"
        )

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for data validation"""
        return hashlib.sha256(data).hexdigest()

    async def _audit_put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data with optional checksum tracking"""
        data = memory_obj.byte_array
        checksum = self._calculate_checksum(data) if self.calc_checksum else "N/A"
        data_size = len(data)
        self.logger.debug(
            f"[REMOTE_AUDIT][{self.real_connector}]:PUT|START|Size:{data_size}|"
            f"Checksum:{checksum[:8]}|Saved:{len(self.checksum_registry)}|Key:{key}"
        )

        try:
            t1 = time.perf_counter()
            await self.real_connector.put(key, memory_obj)
            t2 = time.perf_counter()
            cost = (t2 - t1) * 1000
            if self.registry_lock:
                with self.registry_lock:
                    self.checksum_registry[key] = checksum
            self.logger.info(
                f"[REMOTE_AUDIT][{self.real_connector}]:PUT|SUCCESS|Size:{data_size}|"
                f"Checksum:{checksum[:8]}|Cost:{cost:.6f}ms|Saved:"
                f"{len(self.checksum_registry)}|Key:{key}"
            )

        except Exception as e:
            self.logger.error(
                f"[REMOTE_AUDIT][{self.real_connector}]:PUT|FAILED|Size:{data_size}|"
                f"Key:{key}|Error: {str(e)}"
            )
            raise

    async def _audit_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve data with optional integrity check"""
        self.logger.debug(
            f"[REMOTE_AUDIT][{self.real_connector}]:GET|START|"
            f"Saved:{len(self.checksum_registry)}|Key:{key}"
        )

        try:
            t1 = time.perf_counter()
            result = await self.real_connector.get(key)
            t2 = time.perf_counter()
            if result is None:
                self.logger.info(
                    f"[REMOTE_AUDIT][{self.real_connector}]:GET|MISS|Key:{key}|"
                    f"Saved: {len(self.checksum_registry)}"
                )
                return None

            current_data = result.byte_array
            current_checksum = (
                self._calculate_checksum(current_data) if self.calc_checksum else "N/A"
            )
            data_size = len(current_data)

            if self.registry_lock:
                with self.registry_lock:
                    expected_checksum = self.checksum_registry.get(key)

                if expected_checksum and current_checksum != expected_checksum:
                    self.logger.error(
                        f"[REMOTE_AUDIT][{self.real_connector}]:"
                        f"GET|MISMATCH|Size:{data_size}|"
                        f"Expected:<{expected_checksum[:8]}>|"
                        f"Actual:<{current_checksum[:8]}>|Key:{key}"
                    )
                    return None

            cost = (t2 - t1) * 1000
            self.logger.info(
                f"[REMOTE_AUDIT][{self.real_connector}]:GET|SUCCESS|"
                f"Checksum:{current_checksum[:8]}|"
                f"Cost:{cost:.6f}ms|Saved:{len(self.checksum_registry)}|Key:{key}"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"[REMOTE_AUDIT][{self.real_connector}]:GET|"
                f"FAILED|Key:{key}|Error: {str(e)}"
            )
            raise
