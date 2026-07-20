# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
SELECTOR_PATH = SCRIPT_DIR / "select_ascend_ci_device.py"
GATE_PATH = SCRIPT_DIR / "ascend_e2e_resource_gate.sh"
SMOKE_PATH = SCRIPT_DIR / "run_e2e_serve_smoke.sh"


def load_selector_module():
    spec = importlib.util.spec_from_file_location(
        "select_ascend_ci_device", SELECTOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selector_chooses_only_a_device_that_meets_the_memory_gate():
    selector = load_selector_module()
    mapping = selector.parse_logical_map("0 0 4 Ascend910B2\n1 0 7 Ascend910B2\n")
    devices = selector.parse_device_memory(
        """
| 0 Ascend 910B2 | OK | 98.0 41 0 / 0 |
| 0 | 0000:C1:00.0 | 0 0 / 0 62000 / 65536 |
| 1 Ascend 910B2 | OK | 90.0 35 0 / 0 |
| 0 | 0000:C2:00.0 | 0 0 / 0 4000 / 65536 |
""",
        mapping,
    )

    selected, considered = selector.choose_device(devices, min_free_ratio=0.92)

    assert [device.logical_id for device in considered] == [7, 4]
    assert selected is not None
    assert selected.logical_id == 7
    assert selected.free_memory_mb == 61536


def test_selector_fails_closed_when_no_candidate_has_enough_memory():
    selector = load_selector_module()
    devices = [
        selector.DeviceMemory(0, 4096, 65536, "test"),
        selector.DeviceMemory(1, 8192, 65536, "test"),
    ]

    selected, considered = selector.choose_device(
        devices,
        min_free_ratio=0.92,
        candidate_devices={0, 1},
    )

    assert selected is None
    assert [device.logical_id for device in considered] == [1, 0]


def test_resource_gate_binds_the_selected_physical_device(tmp_path: Path):
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "print('selected\\t3\\tlogical-map\\t62000\\t65536\\t60294')\n",
        encoding="utf-8",
    )
    command = f"""
source {GATE_PATH}
ASCEND_DEVICE_SELECTOR={fake_selector}
PYTHON_BIN={sys.executable}
select_ascend_e2e_device smoke
printf '%s|%s|%s\\n' "$ASCEND_RT_VISIBLE_DEVICES" \
  "${{ASCEND_VISIBLE_DEVICES-unset}}" "$VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE"
"""

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "ASCEND_VISIBLE_DEVICES": "0,1,2,3"},
    )

    assert result.stdout.rstrip().endswith("3|unset|npu:0")


def test_resource_gate_does_not_turn_unavailable_hardware_into_success(
    tmp_path: Path,
):
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "import sys\n"
        "print('unavailable\\t-\\t-\\t-\\t-\\t-\\tno_device')\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.md"
    command = f"""
source {GATE_PATH}
ASCEND_DEVICE_SELECTOR={fake_selector}
PYTHON_BIN={sys.executable}
GITHUB_STEP_SUMMARY={summary}
select_ascend_e2e_device smoke
"""

    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )

    assert result.returncode == 1
    assert "Ascend resource gate failed" in result.stderr
    assert "Status: failed" in summary.read_text(encoding="utf-8")


def test_serve_smoke_selects_before_preflight_and_passes_the_same_ratio():
    text = SMOKE_PATH.read_text(encoding="utf-8")
    selection = text.index('select_ascend_e2e_device "vLLM serve smoke test"')
    sudo_branch = text.index('if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then', selection)
    preflight = text.index("if ensure_runner_npu_ready; then")

    assert selection < sudo_branch < preflight
    assert "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}" in text
    assert '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"' in text
    assert "ASCEND_RT_VISIBLE_DEVICES" in GATE_PATH.read_text(encoding="utf-8")
