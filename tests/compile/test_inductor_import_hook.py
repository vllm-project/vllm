# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys

from vllm.env_override import _run_after_module_import


def test_find_spec_probe_preserves_after_import_callback(tmp_path, monkeypatch):
    name = "_vllm_probe_target"
    (tmp_path / f"{name}.py").write_text("value = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    original_finders = sys.meta_path.copy()
    monkeypatch.setattr(sys, "meta_path", original_finders.copy())
    calls: list[int] = []
    _run_after_module_import(name, lambda: calls.append(sys.modules[name].value))

    try:
        assert importlib.util.find_spec(name) is not None
        assert name not in sys.modules
        assert not calls
        importlib.import_module(name)
        assert calls == [42]
        assert sys.meta_path == original_finders
    finally:
        sys.modules.pop(name, None)
