# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-weight SAVE/LOAD replay and serving benchmark regression."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.utils import large_gpu_mark
from tests.v1.cudagraph import foundry_utils as harness

pytestmark = [pytest.mark.optional, large_gpu_mark(min_gb=16)]


@pytest.mark.timeout(1800)
def test_foundry_native_save_load_round_trip_and_benchmark(tmp_path: Path):
    """Compare native, SAVE, and LOAD on the same fixed batched workload."""
    pytest.importorskip("foundry")

    workspace = tmp_path / "foundry-archive"
    workspace.mkdir()
    save_config = tmp_path / "foundry-save.toml"
    load_config = tmp_path / "foundry-load.toml"
    harness._write_foundry_config(save_config, "save", workspace)
    harness._write_foundry_config(load_config, "load", workspace)

    log_root = Path(os.environ.get("VLLM_FOUNDRY_LOG_DIR", tmp_path))
    log_root.mkdir(parents=True, exist_ok=True)
    native = harness._run_arm("native", None, log_root / "native.log", workspace)
    save = harness._run_arm("save", save_config, log_root / "save.log", workspace)
    load = harness._run_arm("load", load_config, log_root / "load.log", workspace)

    # Preserve metrics and exact outputs even when the parity gate fails.
    (log_root / "results.json").write_text(
        json.dumps(
            [asdict(result) for result in (native, save, load)],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # The full fixed workload is the correctness contract, not only the canary.
    assert save.outputs == native.outputs
    assert load.outputs == native.outputs

    native_rate = native.benchmark_output_tokens / native.benchmark_seconds
    load_rate = load.benchmark_output_tokens / load.benchmark_seconds
    print(
        "Foundry persistence benchmark: "
        f"startup native/save/load={native.startup_seconds:.2f}/"
        f"{save.startup_seconds:.2f}/{load.startup_seconds:.2f}s, "
        f"graph build={load.graph_build_seconds:.3f}s, "
        f"output tok/s native/load={native_rate:.2f}/{load_rate:.2f}"
    )

    # LOAD should preserve steady-state behavior. Keep this relative threshold
    # loose enough for a shared nightly GPU while catching replay regressions.
    assert load_rate >= native_rate * 0.85
