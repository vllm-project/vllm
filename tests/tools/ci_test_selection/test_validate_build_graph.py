# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from tools.ci_test_selection.validate_build_graph import validate


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_validate_requires_one_structural_file_to_kernel_path(tmp_path: Path):
    graph = tmp_path / "graph.jsonl"
    rows = [
        {
            "source_kind": "file",
            "source": "cmake/external_projects/flashmla.cmake",
            "destination_kind": "target",
            "destination": "_flashmla_C",
            "edge_kind": "configures",
        },
        {
            "source_kind": "target",
            "source": "_flashmla_C",
            "destination_kind": "artifact",
            "destination": "elf-build-id:abc",
            "edge_kind": "produces",
        },
        {
            "source_kind": "artifact",
            "source": "elf-build-id:abc",
            "destination_kind": "kernel",
            "destination": "_Zkernel",
            "edge_kind": "defines_kernel",
        },
    ]
    _write(graph, rows)

    assert validate(graph, "cmake/external_projects/flashmla.cmake", "_flashmla_C") == [
        ("file", "cmake/external_projects/flashmla.cmake"),
        ("target", "_flashmla_C"),
        ("artifact", "elf-build-id:abc"),
        ("kernel", "_Zkernel"),
    ]


def test_validate_rejects_metadata_only_artifact_match(tmp_path: Path):
    graph = tmp_path / "graph.jsonl"
    rows = [
        {
            "source_kind": "file",
            "source": "cmake/external_projects/flashmla.cmake",
            "destination_kind": "target",
            "destination": "_flashmla_C",
            "edge_kind": "configures",
        },
        {
            "source_kind": "target",
            "source": "_flashmla_C",
            "destination_kind": "artifact",
            "destination": "path:_flashmla_C.so",
            "artifact_build_id": "abc",
            "edge_kind": "produces",
        },
        {
            "source_kind": "artifact",
            "source": "elf-build-id:abc",
            "destination_kind": "kernel",
            "destination": "_Zkernel",
            "artifact_build_id": "abc",
            "edge_kind": "defines_kernel",
        },
    ]
    _write(graph, rows)

    with pytest.raises(ValueError, match="no structurally joined"):
        validate(graph, "cmake/external_projects/flashmla.cmake", "_flashmla_C")
