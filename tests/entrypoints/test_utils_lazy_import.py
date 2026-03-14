# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that get_max_tokens lazily imports current_platform
rather than relying on a module-level import."""

import subprocess
import sys


def test_get_max_tokens_lazy_platform_import():
    """current_platform should not be imported at module level in utils.py."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ast, inspect, textwrap; "
                "from vllm.entrypoints import utils; "
                "src = inspect.getsource(utils.get_max_tokens); "
                "tree = ast.parse(textwrap.dedent(src)); "
                "imports = [n for n in ast.walk(tree) "
                "  if isinstance(n, ast.ImportFrom) "
                "  and n.module == 'vllm.platforms']; "
                "assert imports, 'get_max_tokens should have a local platform import'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
