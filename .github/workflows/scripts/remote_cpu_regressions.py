# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run complete remote CPU suites and record the exact collected test cases."""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

NATIVE_SUITES = {
    "test_gpu_runtime.py",
    "test_mtp_contracts.py",
    "test_remote_config.py",
    "test_target_kv_runtime.py",
    "test_token_gpu.py",
}
EXTERNAL_REGRESSIONS = {
    "tests/v1/core/test_async_scheduler.py": (
        "test_worker_invalid_drafts_follow_queued_step_and_grammar_union",
    ),
    "tests/v1/worker/test_gpu_input_batch_v2.py": (
        "test_model_input_padding_preserves_logical_tokens",
        "test_invalid_spec_tokens_use_current_packed_request_boundaries",
    ),
    "tests/v1/test_serial_utils.py": (
        "test_model_runner_invalid_draft_metadata_round_trip",
    ),
}


def regression_nodes(suite: str = "all", root: Path = Path(".")) -> list[str]:
    tests = root / "tests/v1/spec_decode/remote"
    nodes = [
        str(path.relative_to(root))
        for path in sorted(tests.glob("test_*.py"))
        if suite == "all" or (path.name in NATIVE_SUITES) == (suite == "native")
    ]
    if suite != "native":
        nodes.append(".github/workflows/scripts/test_remote_cpu_regressions.py")
    if suite != "control":
        for name, candidates in EXTERNAL_REGRESSIONS.items():
            path = root / name
            if path.is_file():
                definitions = {
                    node.name
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                }
                nodes.extend(
                    f"{name}::{test}" for test in candidates if test in definitions
                )
    return nodes


class CollectionReport:
    def __init__(self, path: Path, nodes: list[str], suite: str):
        self.path = path
        self.report = {
            "suite": suite,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "selected": nodes,
            "collected": [],
            "collected_count": 0,
            "outcomes": {},
        }
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.report, indent=2) + "\n")

    def pytest_collection_finish(self, session):
        self.report["collected"] = [item.nodeid for item in session.items]
        self.report["collected_count"] = len(session.items)
        self.write()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed or report.skipped:
            self.report["outcomes"][report.nodeid] = report.outcome

    def pytest_sessionfinish(self, session, exitstatus):
        self.report["exitstatus"] = int(exitstatus)
        self.write()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="Print selected pytest nodes"
    )
    parser.add_argument("--suite", choices=("all", "control", "native"), default="all")
    parser.add_argument("--artifacts", type=Path, default=Path("remote-cpu-results"))
    args = parser.parse_args()
    nodes = regression_nodes(args.suite)
    if args.list:
        print("\n".join(nodes))
        return 0

    sys.path.insert(0, str(Path.cwd()))
    import pytest

    args.artifacts.mkdir(parents=True, exist_ok=True)
    report = CollectionReport(args.artifacts / "collection.json", nodes, args.suite)
    if not nodes:
        print(f"No {args.suite} suites exist on this branch.")
        return 0
    return int(
        pytest.main(
            [
                "--noconftest",
                "-p",
                "pytest_timeout",
                "--timeout=60",
                "-q",
                f"--junitxml={args.artifacts / 'junit.xml'}",
                *nodes,
            ],
            plugins=[report],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
