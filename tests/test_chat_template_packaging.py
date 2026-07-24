# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skip_global_cleanup


def test_example_jinja_templates_are_declared_for_sdist():
    manifest_lines = (ROOT_DIR / "MANIFEST.in").read_text().splitlines()

    assert "recursive-include examples *.jinja" in manifest_lines


def test_example_jinja_templates_are_declared_for_wheel_data():
    setup_py = (ROOT_DIR / "setup.py").read_text()
    example_template_names = {
        path.name for path in (ROOT_DIR / "examples").glob("*.jinja")
    }
    wheel_data_paths = {
        f".data/data/share/vllm/examples/{name}" for name in example_template_names
    }

    assert '(ROOT_DIR / "examples").glob("*.jinja")' in setup_py
    assert '("share/vllm/examples", example_chat_templates)' in setup_py
    assert "tool_chat_template_gemma4.jinja" in example_template_names
    assert "template_chatml.jinja" in example_template_names
    assert (
        ".data/data/share/vllm/examples/tool_chat_template_gemma4.jinja"
        in wheel_data_paths
    )
    assert ".data/data/share/vllm/examples/template_chatml.jinja" in wheel_data_paths
