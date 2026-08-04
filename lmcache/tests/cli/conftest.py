# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for CLI tests.

The CLI arg-registration code transitively imports
``lmcache.native_storage_ops`` (a compiled C extension). On CI runners
without a CUDA build the module is absent, so we insert a lightweight
stub into ``sys.modules`` for the duration of each CLI test only.
"""

# Standard
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock
import importlib
import importlib.util
import sys
import types

# Third Party
import pytest

_MOD_NAME = "lmcache.native_storage_ops"


@pytest.fixture(autouse=True)
def _stub_native_storage_ops() -> Generator[None, None, None]:
    """Temporarily stub ``native_storage_ops`` if the extension is not built.

    The stub is removed from ``sys.modules`` after the test so it does not
    interfere with other test suites that ``importorskip`` the real module.
    """
    if importlib.util.find_spec(_MOD_NAME) is not None:
        yield
        return

    stub: Any = types.ModuleType(_MOD_NAME)
    stub.TTLLock = MagicMock()
    stub.Bitmap = MagicMock()
    stub.PeriodicEventNotifier = MagicMock()
    stub.ParallelPatternMatcher = MagicMock()
    stub.RangePatternMatcher = MagicMock()

    sys.modules[_MOD_NAME] = stub
    try:
        yield
    finally:
        if sys.modules.get(_MOD_NAME) is stub:
            del sys.modules[_MOD_NAME]
