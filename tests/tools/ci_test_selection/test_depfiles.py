# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for header dependency ingestion (ninja -t deps / .d fallback).

The primary format is `ninja -t deps` tool output because CMake+Ninja
consumes compiler .d files into the binary .ninja_deps database and deletes
them (verified on a real build); classic .d parsing remains only for the
make generator. The exporter integration test proves a header reaches its
including targets as an `includes` edge without duplicating `member_of`.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from tools.ci_test_selection.depfiles import (
    collect_file_target_pairs,
    parse_depfile,
    parse_ninja_deps,
    target_from_object_path,
)

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT_DIR = HERE.parents[2] / "tools/ci_test_selection"

NINJA_DEPS = """\
CMakeFiles/_flashmla_C.dir/csrc/flashmla/mla.cu.o: #deps 3, deps mtime 1 (VALID)
    {src}/csrc/flashmla/mla.cu
    {src}/csrc/flashmla/mla.h
    /usr/include/stdc-predef.h

CMakeFiles/cutlass_lib.dir/csrc/cutlass/stub.cpp.o: #deps 2, deps mtime 1 (STALE)
    {src}/csrc/cutlass/stub.cpp
    {src}/csrc/flashmla/mla.h

_flashmla_C.abi3.so: #deps 1, deps mtime 1 (VALID)
    /usr/lib/x86_64-linux-gnu/crti.o
"""


class TestParsing(unittest.TestCase):
    def test_ninja_deps_skips_stale_keeps_valid_and_link_rules(self):
        rules = parse_ninja_deps(NINJA_DEPS.format(src="/repo"))
        self.assertIn("CMakeFiles/_flashmla_C.dir/csrc/flashmla/mla.cu.o", rules)
        self.assertNotIn("CMakeFiles/cutlass_lib.dir/csrc/cutlass/stub.cpp.o", rules)
        self.assertIn("_flashmla_C.abi3.so", rules)

    def test_depfile_continuations_and_escaped_spaces(self):
        text = (
            "CMakeFiles/_C.dir/a.cu.o: ../a.cu \\\n ../inc/a.h ../inc/with\\ space.h\n"
        )
        rules = parse_depfile(text)
        self.assertEqual(
            rules["CMakeFiles/_C.dir/a.cu.o"],
            ["../a.cu", "../inc/a.h", "../inc/with space.h"],
        )

    def test_target_extraction(self):
        self.assertEqual(
            target_from_object_path("CMakeFiles/_flashmla_C.dir/x.cu.o"),
            "_flashmla_C",
        )
        self.assertIsNone(target_from_object_path("_flashmla_C.abi3.so"))

    def test_collect_scoping(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp)
            build = src / "build"
            build.mkdir()
            rules = parse_ninja_deps(NINJA_DEPS.format(src=src))
            rules["CMakeFiles/_flashmla_C.dir/gen.cu.o"] = [
                str(build / "generated/config.h")
            ]
            pairs, stats = collect_file_target_pairs(rules, build, src)
        self.assertEqual(
            pairs,
            [
                ("csrc/flashmla/mla.cu", "_flashmla_C"),
                ("csrc/flashmla/mla.h", "_flashmla_C"),
            ],
        )
        self.assertEqual(stats["rules_unattributed"], 1)
        self.assertEqual(stats["deps_outside_source_root"], 1)
        self.assertEqual(stats["deps_in_build_tree"], 1)


class TestExporterIntegration(unittest.TestCase):
    def test_includes_edges_without_member_duplication(self):
        from test_export_build_graph import build_reply

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            build_dir, source_root = base / "build", base / "src"
            build_dir.mkdir()
            source_root.mkdir()
            build_reply(build_dir, source_root)
            deps_dump = base / "ninja-deps.txt"
            deps_dump.write_text(NINJA_DEPS.format(src=source_root))
            out = base / "edges.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "export_build_graph.py"),
                    str(build_dir),
                    "--ninja-deps",
                    str(deps_dump),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            edges = [json.loads(line) for line in out.read_text().splitlines()]
            summary = json.loads(proc.stderr)

        includes = [
            (e["source"], e["destination"])
            for e in edges
            if e["edge_kind"] == "includes"
        ]
        # the header reaches its including target as an includes edge
        self.assertEqual(includes, [("csrc/flashmla/mla.h", "_flashmla_C")])
        # the compiled source stays member_of only, never duplicated
        member = [
            (e["source"], e["destination"])
            for e in edges
            if e["edge_kind"] == "member_of"
        ]
        self.assertIn(("csrc/flashmla/mla.cu", "_flashmla_C"), member)
        self.assertNotIn(("csrc/flashmla/mla.cu", "_flashmla_C"), includes)
        self.assertEqual(summary["header_dep_source"], "ninja_deps")
        self.assertEqual(summary["include_edges"], 1)


if __name__ == "__main__":
    unittest.main()
