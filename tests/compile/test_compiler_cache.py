# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import os
from pathlib import Path
from unittest.mock import patch

from vllm.compilation.backends import CompilerManager
from vllm.config import CompilationConfig


def test_compiler_cache_is_replaced_atomically(tmp_path: Path):
    manager = CompilerManager(CompilationConfig(backend="eager"))
    manager.initialize_cache(str(tmp_path))
    cache_path = tmp_path / "vllm_compile_cache.py"
    cache_path.write_text("{'old': 'cache'}")
    manager.cache = {("new", 1, "eager"): {"graph_handle": ("a", "b")}}
    manager.is_cache_updated = True
    replace = os.replace

    def check_before_replace(source, target):
        assert cache_path.read_text() == "{'old': 'cache'}"
        assert isinstance(ast.literal_eval(Path(source).read_text()), dict)
        replace(source, target)

    with patch(
        "vllm.compilation.backends.os.replace",
        side_effect=check_before_replace,
    ):
        manager.save_to_file()

    assert ast.literal_eval(cache_path.read_text()) == manager.cache
    assert not list(tmp_path.glob(".vllm_compile_cache.*"))
