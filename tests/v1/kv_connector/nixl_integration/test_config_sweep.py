# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each config of config_sweep_accuracy_test.sh as its own test, so CI can
spread a sweep over parallel jobs. The script's env vars (DP_EP, HYBRID_SSM,
FLASHINFER, CROSS_LAYERS_BLOCKS, ...) pick the configs and backend as before.
"""

import os
import subprocess
from pathlib import Path

import pytest
import regex as re

SWEEP = Path(__file__).with_name("config_sweep_accuracy_test.sh")
# The script's paths are relative to tests/.
TESTS_DIR = Path(__file__).parents[3]


def _configs() -> list[str]:
    listed = subprocess.run(
        ["bash", str(SWEEP), "--list"],
        cwd=TESTS_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return listed.stdout.splitlines()


def _config_id(config: str) -> str:
    """A short ID that stays the same across runs: CI plans shards from each
    test's past timings, looked up by ID."""
    words = [
        word.rsplit("/", 1)[-1]  # MODEL_NAMES=org/name -> name
        for word in config.split()
        if not word.startswith("GPU_MEMORY_UTILIZATION=")
    ]
    return re.sub(r"[^\w.,=-]", "_", "-".join(words)) or "default"


@pytest.mark.parametrize("config", _configs(), ids=_config_id)
def test_config_sweep(config: str):
    subprocess.run(
        ["bash", str(SWEEP)],
        cwd=TESTS_DIR,
        env={**os.environ, "SWEEP_CONFIG": config},
        check=True,
    )
