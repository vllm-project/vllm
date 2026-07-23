# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/collect_ascend_diagnostic_env.py"
)
SPEC = importlib.util.spec_from_file_location("collect_ascend_diagnostic_env", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_collect_safe_environment_is_exact_allowlist():
    environ = {
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "HARDWARE_CHIP_MODEL": "910B2",
        "VLLM_PLUGINS": "ascend",
        "GITHUB_TOKEN": "synthetic-github-sentinel",
        "HF_TOKEN": "synthetic-hf-sentinel",
        "BENCHMARK_REPO_GH_TOKEN": "synthetic-publication-sentinel",
        "VLLM_FUTURE_CREDENTIAL": "synthetic-prefix-sentinel",
    }

    assert MODULE.collect_safe_environment(environ) == {
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "HARDWARE_CHIP_MODEL": "910B2",
        "VLLM_PLUGINS": "ascend",
    }


def test_write_safe_environment_excludes_keys_and_values(tmp_path: Path):
    output = tmp_path / "environment.json"
    environ = {
        "ASCEND_HOME_PATH": "/opt/ascend",
        "ASCEND_VISIBLE_DEVICES": "3",
        "SSH_PRIVATE_KEY": "synthetic-ssh-sentinel",
        "HTTP_PROXY": "synthetic-proxy-sentinel",
    }

    MODULE.write_safe_environment(output, environ)

    payload_text = output.read_text(encoding="utf-8")
    assert json.loads(payload_text) == {
        "ASCEND_HOME_PATH": "/opt/ascend",
        "ASCEND_VISIBLE_DEVICES": "3",
    }
    for forbidden in (
        "SSH_PRIVATE_KEY",
        "synthetic-ssh-sentinel",
        "HTTP_PROXY",
        "synthetic-proxy-sentinel",
    ):
        assert forbidden not in payload_text


def test_benchmark_collector_never_dumps_the_raw_environment():
    benchmark_script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/run_ascend_benchmark_ci.sh"
    ).read_text(encoding="utf-8")

    collector = benchmark_script[
        benchmark_script.index("collect_ascend_diagnostics() {") :
        benchmark_script.index("enforce_single_runtime_source_environment() {")
    ]
    assert "collect_ascend_diagnostic_env.py" in benchmark_script
    assert 'environment.json"' in collector
    assert "env |" not in collector
    assert "printenv" not in collector
    assert "env.txt" not in collector
    assert "ASCEND_HOME_PATH:-" not in collector
    assert "ASCEND_RT_VISIBLE_DEVICES:-" not in collector
    assert "LD_LIBRARY_PATH:-" not in collector
