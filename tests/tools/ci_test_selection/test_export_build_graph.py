# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixture tests for the CMake File API build-graph exporter."""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools.ci_test_selection import (
    export_build_graph,
    extract_kernel_symbols,
    extract_object_kernel_symbols,
    write_build_provenance_manifest,
)

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT_DIR = HERE.parents[2] / "tools/ci_test_selection"


def build_reply(build_dir, source_root):
    reply = build_dir / ".cmake/api/v1/reply"
    reply.mkdir(parents=True)
    (source_root / "csrc").mkdir(parents=True)
    target_mla = {
        "name": "_flashmla_C",
        "id": "_flashmla_C::@6890427a1f51a3e7e1df",
        "type": "MODULE_LIBRARY",
        "artifacts": [{"path": "_flashmla_C.abi3.so"}],
        "sources": [
            {"path": "csrc/flashmla/mla.cu"},
            {"path": "csrc/flashmla/mla_api.cpp"},
            # generated out-of-tree source must be counted, not emitted
            {"path": str(build_dir / "generated/version.cpp")},
        ],
        "dependencies": [{"id": "cutlass_lib::@abc"}],
        "backtrace": 1,
        "backtraceGraph": {
            "files": ["CMakeLists.txt", "cmake/external_projects/flashmla.cmake"],
            "nodes": [{"file": 0}, {"file": 1, "parent": 0}],
        },
    }
    target_dep = {
        "name": "cutlass_lib",
        "id": "cutlass_lib::@abc",
        "type": "STATIC_LIBRARY",
        "artifacts": [{"path": "libcutlass_lib.a"}],
        "sources": [{"path": "csrc/cutlass/stub.cpp"}],
    }
    (reply / "target-mla.json").write_text(json.dumps(target_mla))
    (reply / "target-dep.json").write_text(json.dumps(target_dep))
    codemodel = {
        "paths": {"source": str(source_root), "build": str(build_dir)},
        "configurations": [
            {
                "targets": [
                    {"id": target_mla["id"], "jsonFile": "target-mla.json"},
                    {"id": target_dep["id"], "jsonFile": "target-dep.json"},
                ]
            }
        ],
    }
    (reply / "codemodel-v2-x.json").write_text(json.dumps(codemodel))
    (reply / "index-2026.json").write_text(
        json.dumps(
            {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2-x.json"}]}
        )
    )
    # real ELF with a GNU build-id so the exporter attaches the join key
    shutil.copy("/bin/bash", build_dir / "_flashmla_C.abi3.so")


class TestExport(unittest.TestCase):
    def test_edges_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            build_dir, source_root = base / "build", base / "src"
            build_dir.mkdir()
            source_root.mkdir()
            build_reply(build_dir, source_root)
            out = base / "edges.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "export_build_graph.py"),
                    str(build_dir),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            edges = [json.loads(line) for line in out.read_text().splitlines()]
            summary = json.loads(proc.stderr)

        member = [
            (e["source"], e["destination"])
            for e in edges
            if e["edge_kind"] == "member_of"
        ]
        self.assertIn(("csrc/flashmla/mla.cu", "_flashmla_C"), member)
        self.assertIn(("csrc/cutlass/stub.cpp", "cutlass_lib"), member)
        self.assertEqual(summary["out_of_tree_sources"], 1)
        self.assertNotIn("generated/version.cpp", [e["source"] for e in edges])

        configures = [
            (e["source"], e["destination"])
            for e in edges
            if e["edge_kind"] == "configures"
        ]
        self.assertIn(
            ("cmake/external_projects/flashmla.cmake", "_flashmla_C"), configures
        )

        deps = [
            (e["source"], e["destination"])
            for e in edges
            if e["edge_kind"] == "required_by"
        ]
        self.assertEqual(deps, [("cutlass_lib", "_flashmla_C")])

        produces = {e["source"]: e for e in edges if e["edge_kind"] == "produces"}
        self.assertTrue(
            produces["_flashmla_C"]["destination"].startswith("elf-build-id:")
        )
        self.assertEqual(
            produces["_flashmla_C"]["artifact_path"], "_flashmla_C.abi3.so"
        )
        # /bin/bash stand-in exists on disk -> build-id join key attached
        self.assertTrue(produces["_flashmla_C"].get("artifact_build_id"))
        # static lib never built in fixture -> counted, not emitted as a
        # runtime artifact node.
        self.assertNotIn("cutlass_lib", produces)
        self.assertEqual(summary["artifacts_missing_on_disk"], 1)
        self.assertEqual(summary["targets"], 2)

    def test_artifact_identity_matches_kernel_exporter(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            shared_object = base / "fixture.so"
            shutil.copy("/bin/bash", shared_object)
            identity, build_id = export_build_graph.artifact_identity(shared_object)
            out = base / "kernel-map.jsonl"
            with mock.patch.object(
                extract_kernel_symbols,
                "kernel_names",
                return_value={"_ZfixtureKernel"},
            ):
                extract_kernel_symbols.main([str(base), "--out", str(out)])
            row = json.loads(out.read_text().strip())

        self.assertEqual(row["source"], identity)
        self.assertEqual(row["artifact_build_id"], build_id)

    def test_object_kernel_exporter_retains_exact_translation_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            source_root = base / "src"
            build_dir = base / "build"
            source = source_root / "csrc/ops.cu"
            included_cu = source_root / "csrc/included_impl.cu"
            object_path = build_dir / "CMakeFiles/_C.dir/csrc/ops.cu.o"
            source.parent.mkdir(parents=True)
            object_path.parent.mkdir(parents=True)
            source.write_text("// source\n")
            included_cu.write_text("// included implementation\n")
            object_path.write_bytes(b"object fixture")
            rules = {
                str(object_path.relative_to(build_dir)): [
                    str(source),
                    str(included_cu),
                ]
            }

            with mock.patch.object(
                extract_object_kernel_symbols,
                "kernel_names",
                return_value={"_ZexactKernel"},
            ):
                rows, summary = (
                    extract_object_kernel_symbols.export_object_kernel_edges(
                        build_dir,
                        source_root,
                        rules,
                        {("csrc/ops.cu", "_C")},
                    )
                )

        self.assertEqual(
            rows,
            [
                {
                    "source_kind": "file",
                    "source": "csrc/ops.cu",
                    "edge_kind": "compiles_kernel",
                    "destination_kind": "kernel",
                    "destination": "_ZexactKernel",
                    "target": "_C",
                    "object_path": "CMakeFiles/_C.dir/csrc/ops.cu.o",
                }
            ],
        )
        self.assertEqual(summary["kernel_objects"], 1)
        self.assertEqual(summary["translation_unit_kernel_edges"], 1)
        self.assertEqual(summary["unique_translation_units"], 1)

    def test_static_manifest_hashes_the_exact_image_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / "build-graph.jsonl").write_text('{"graph":1}\n')
            (directory / "kernel-map.jsonl").write_text('{"kernel":1}\n')
            document = write_build_provenance_manifest.build_manifest(
                directory,
                repository_sha="a" * 40,
                image_tag="registry/vllm:test",
                image_digest="sha256:" + "c" * 64,
                created_at="2026-08-12T23:00:00Z",
                buildkite_build_id="build-id",
            )

        self.assertEqual(document["publisher_step_key"], "image-build")
        self.assertEqual(document["repository_sha"], "a" * 40)
        self.assertEqual(document["image_tag"], "registry/vllm:test")
        self.assertEqual(document["image_digest"], "sha256:" + "c" * 64)
        self.assertEqual(
            document["files"]["build-graph.jsonl"]["bytes"],
            len('{"graph":1}\n'),
        )
        self.assertRegex(
            document["files"]["kernel-map.jsonl"]["sha256"], r"^[0-9a-f]{64}$"
        )

    def test_static_manifest_writer_runs_without_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / "build-graph.jsonl").write_text('{"graph":1}\n')
            (directory / "kernel-map.jsonl").write_text('{"kernel":1}\n')
            output = directory / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT_DIR / "write_build_provenance_manifest.py"),
                    str(directory),
                    "--repository-sha",
                    "a" * 40,
                    "--image-tag",
                    "registry/vllm:test",
                    "--image-digest",
                    "sha256:" + "b" * 64,
                    "--created-at",
                    "2026-08-13T00:00:00Z",
                    "--buildkite-build-id",
                    "build-id",
                    "--out",
                    str(output),
                ],
                check=True,
            )

            document = json.loads(output.read_text())

        self.assertEqual(document["kind"], "static-build-provenance")
        self.assertEqual(document["buildkite_build_id"], "build-id")


if __name__ == "__main__":
    unittest.main()
