# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.file_utils import atomic_writer


def test_atomic_writer_replaces_file_on_success(tmp_path):
    target = tmp_path / "info.json"
    target.write_text("old", encoding="utf-8")

    with atomic_writer(target, encoding="utf-8") as f:
        f.write("new")

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["info.json"]


def test_atomic_writer_keeps_original_on_error(tmp_path):
    target = tmp_path / "info.json"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError), atomic_writer(target, encoding="utf-8") as f:
        f.write("partial")
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["info.json"]
