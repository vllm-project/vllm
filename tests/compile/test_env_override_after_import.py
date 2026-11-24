# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the deferred-patch helper used by vllm/env_override.py.

The Inductor backports in env_override.py are applied lazily, right after
the patched torch module is imported, so that `import vllm` does not pull
in the Inductor stack. These tests use a throwaway module instead of torch so
they run regardless of which backports the installed torch version needs.
"""

import importlib.util
import sys

import pytest

from vllm.env_override import _run_after_module_import


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    name = "_vllm_after_import_probe"
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    meta_path = list(sys.meta_path)
    yield name
    sys.meta_path[:] = meta_path
    sys.modules.pop(name, None)


def test_runs_after_module_body(fake_module):
    seen = []
    _run_after_module_import(
        fake_module, lambda: seen.append(sys.modules[fake_module].VALUE)
    )
    assert seen == []
    assert fake_module not in sys.modules

    importlib.import_module(fake_module)

    assert seen == [1]


def test_survives_find_spec_probe(fake_module):
    # A bare find_spec probe consults the finder without importing the
    # module; the callback must still run on the real import that follows.
    seen = []
    _run_after_module_import(fake_module, lambda: seen.append(True))

    assert importlib.util.find_spec(fake_module) is not None
    assert fake_module not in sys.modules
    assert seen == []

    importlib.import_module(fake_module)

    assert seen == [True]


def test_finder_removed_after_callback(fake_module):
    before = len(sys.meta_path)
    _run_after_module_import(fake_module, lambda: None)
    assert len(sys.meta_path) == before + 1

    importlib.import_module(fake_module)

    assert len(sys.meta_path) == before


def test_already_imported_runs_now(fake_module):
    importlib.import_module(fake_module)
    before = list(sys.meta_path)
    seen = []

    _run_after_module_import(fake_module, lambda: seen.append(True))

    assert seen == [True]
    assert sys.meta_path == before
