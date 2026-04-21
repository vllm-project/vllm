# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins

from vllm.entrypoints.cli import main as cli_main


def test_bg_preload_torch_does_not_import_transformers(monkeypatch):
    original_import = builtins.__import__
    imported_modules: list[str] = []

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported_modules.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    cli_main._bg_preload_torch()

    assert "torch" in imported_modules
    assert "transformers" not in imported_modules
