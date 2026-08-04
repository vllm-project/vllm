# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for distributed module tests.
"""

# Future
from __future__ import annotations

# Standard
from importlib.util import find_spec
from typing import Any, cast
import sys
import types

if find_spec("lmcache.native_storage_ops") is None:

    class Bitmap:
        """Small Python Bitmap fallback for source-only distributed tests."""

        def __init__(self, size: int, first_n: int = 0) -> None:
            self._size = int(size)
            self._bits = {i for i in range(min(int(first_n), self._size))}

        def set(self, index: int) -> None:
            """Set one bit by index."""
            index = int(index)
            if index < 0 or index >= self._size:
                raise IndexError(index)
            self._bits.add(index)

        def test(self, index: int) -> bool:
            """Return whether one bit is set."""
            return int(index) in self._bits

        def get_indices_list(self) -> list[int]:
            """Return set bit indices in ascending order."""
            return sorted(self._bits)

        def popcount(self) -> int:
            """Return the number of set bits."""
            return len(self._bits)

        def count_leading_ones(self) -> int:
            """Return the length of the leading contiguous set-bit prefix."""
            count = 0
            while count in self._bits:
                count += 1
            return count

        def gather(self, values):
            """Return values selected by set bit indices."""
            return [values[i] for i in self.get_indices_list()]

        def __and__(self, other: "Bitmap") -> "Bitmap":
            size = min(self._size, other._size)
            result = Bitmap(size)
            result._bits = {i for i in self._bits & other._bits if i < size}
            return result

        def __iand__(self, other: "Bitmap") -> "Bitmap":
            self._bits &= other._bits
            self._bits = {i for i in self._bits if i < self._size}
            return self

        def __or__(self, other: "Bitmap") -> "Bitmap":
            size = max(self._size, other._size)
            result = Bitmap(size)
            result._bits = set(self._bits | other._bits)
            return result

        def __ior__(self, other: "Bitmap") -> "Bitmap":
            self._size = max(self._size, other._size)
            self._bits |= other._bits
            return self

        def __invert__(self) -> "Bitmap":
            result = Bitmap(self._size)
            result._bits = set(range(self._size)) - self._bits
            return result

        def __str__(self) -> str:
            return "".join("1" if i in self._bits else "0" for i in range(self._size))

    class TTLLock:
        """Minimal TTLLock fallback for tests that only import the symbol."""

    fallback_module = types.ModuleType("lmcache.native_storage_ops")
    fallback_module_any = cast(Any, fallback_module)
    fallback_module_any.Bitmap = Bitmap
    fallback_module_any.TTLLock = TTLLock
    sys.modules["lmcache.native_storage_ops"] = fallback_module


# ---------------------------------------------------------------------------
# CLI options for live-server integration tests (e.g. the Valkey adapter).
#
# These let integration tests point at a real Valkey/Redis without
# environment variables, e.g.::
#
#   pytest test_valkey_l2_adapter_integration.py \
#       --valkey-cluster --valkey-nodes=127.0.0.1:7000,127.0.0.1:7001
#
#   pytest test_valkey_l2_adapter_integration.py \
#       --valkey-host=localhost --valkey-port=6390
#
# Defaults point at localhost so a bare run connects to a locally-running
# server; the test still self-skips when nothing is reachable.
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    """Register Valkey integration-test connection options."""
    group = parser.getgroup("valkey-integration")
    group.addoption(
        "--valkey-cluster",
        action="store_true",
        default=False,
        help="Run Valkey integration tests in cluster mode against --valkey-nodes.",
    )
    group.addoption(
        "--valkey-nodes",
        action="store",
        default="127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002",
        help="Comma-separated host:port cluster seed nodes (cluster mode).",
    )
    group.addoption(
        "--valkey-host",
        action="store",
        default="localhost",
        help="Valkey host (standalone mode).",
    )
    group.addoption(
        "--valkey-port",
        action="store",
        type=int,
        default=6379,
        help="Valkey port (standalone mode).",
    )
