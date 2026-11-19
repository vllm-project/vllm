# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD-disagg CI tests that each run one shell script.

Each script starts the prefill and decode servers (and the proxy), runs its
own pytest file against them and stops them. The step's environment
configures it, as when CI ran the script directly. Run as pytest, the CI
steps can use automatic sharding; each test is one script, so it stays one
job.
"""

import subprocess
from pathlib import Path

import pytest

# The servers run in the script's own processes; this process starts no engine.
pytestmark = pytest.mark.skip_global_cleanup

SCRIPT_DIR = Path(__file__).parent


def run_script(name: str) -> None:
    """Run one of this folder's scripts with this process's environment.

    Args:
        name: The script's file name.

    """
    subprocess.run(["bash", str(SCRIPT_DIR / name)], check=True)


def test_accuracy():
    run_script("run_accuracy_test.sh")


def test_edge_cases():
    run_script("run_edge_case_test.sh")


def test_mamba_prefix_cache():
    run_script("run_mamba_prefix_cache_test.sh")


def test_multi_connector_accuracy():
    run_script("run_multi_connector_accuracy_test.sh")


def test_multi_connector_edge_cases():
    run_script("run_multi_connector_edge_case_test.sh")
