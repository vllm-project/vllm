# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest plugin that runs scheduler test suites against the Rust scheduler.

Every scheduler built by the test helpers (`create_scheduler` of
`tests/v1/core/utils.py` and `tests/v1/kv_connector/unit/utils.py`) becomes a
`RustScheduler` (`RustAsyncScheduler` with async scheduling):

    .venv/bin/python -m pytest -p tests.v1.core.rust_scheduler_plugin \\
        tests/v1/core/test_scheduler.py

Tests that construct `Scheduler` directly keep the Python scheduler. Tests
that inspect Python-only internals (`scheduler.waiting`, private attributes)
are expected to fail.

The Rust scheduler brings a Python request up to date only when Python code
reads it through the scheduler. Tests read the request objects they added,
so here every request is synced after each scheduler call.
"""

import functools
import importlib

import pytest

from vllm.v1.core.sched.rust_scheduler import RustAsyncScheduler, RustScheduler

_HELPER_MODULES = ("tests.v1.core.utils", "tests.v1.kv_connector.unit.utils")

_MUTATING_METHODS = (
    "add_request",
    "schedule",
    "update_from_output",
    "update_draft_token_ids",
    "update_draft_token_ids_in_output",
    "finish_requests",
    "reset_prefix_cache",
    "has_requests",
)


def _eagerly_synced(cls):
    def wrap(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            self._core.sync_all_requests()
            return result

        return wrapper

    namespace = {name: wrap(getattr(cls, name)) for name in _MUTATING_METHODS}
    return type(cls.__name__, (cls,), namespace)


EagerRustScheduler = _eagerly_synced(RustScheduler)
EagerRustAsyncScheduler = _eagerly_synced(RustAsyncScheduler)


@pytest.fixture(autouse=True)
def _rust_scheduler(monkeypatch):
    for name in _HELPER_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        monkeypatch.setattr(module, "Scheduler", EagerRustScheduler)
        monkeypatch.setattr(module, "AsyncScheduler", EagerRustAsyncScheduler)
