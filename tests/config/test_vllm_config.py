# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.config.vllm import _normalize_debug_dump_path


def test_normalize_debug_dump_path_expands_user_before_absolute():
    for raw_path in ("~/vllm-debug", Path("~/vllm-debug")):
        path = _normalize_debug_dump_path(raw_path)

        assert path == (Path.home() / "vllm-debug").absolute()


def test_normalize_debug_dump_path_keeps_relative_paths_cwd_based():
    path = _normalize_debug_dump_path("vllm-debug")

    assert path == Path("vllm-debug").absolute()
