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
REGRESSION_PATH = SCRIPT_DIR / "run_e2e_inference_regression.sh"


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


def test_resource_gate_preserves_original_candidates_for_reselection(
    tmp_path: Path,
):
    calls = tmp_path / "calls.txt"
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['SELECTOR_CALLS']).open('a') as handle:\n"
        "    handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "print('selected\\t3\\tlogical-map\\t62000\\t65536\\t60294')\n",
        encoding="utf-8",
    )
    command = f"""
source {GATE_PATH}
ASCEND_DEVICE_SELECTOR={fake_selector}
PYTHON_BIN={sys.executable}
select_ascend_e2e_device smoke
select_ascend_e2e_device smoke
"""

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ASCEND_VISIBLE_DEVICES": "0,1,2,3",
            "SELECTOR_CALLS": str(calls),
        },
    )

    recorded_calls = calls.read_text(encoding="utf-8").splitlines()
    assert len(recorded_calls) == 2
    assert all("--candidate-devices 0,1,2,3" in call for call in recorded_calls)


def test_resource_gate_confirms_only_the_selected_device(tmp_path: Path):
    calls = tmp_path / "calls.txt"
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['SELECTOR_CALLS']).open('a') as handle:\n"
        "    handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "print('selected\\t3\\tlogical-map\\t62000\\t65536\\t60294')\n",
        encoding="utf-8",
    )
    command = f"""
source {GATE_PATH}
ASCEND_DEVICE_SELECTOR={fake_selector}
PYTHON_BIN={sys.executable}
select_ascend_e2e_device smoke
confirm_selected_ascend_e2e_device smoke-launch
"""

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ASCEND_VISIBLE_DEVICES": "0,1,2,3",
            "SELECTOR_CALLS": str(calls),
        },
    )

    recorded_calls = calls.read_text(encoding="utf-8").splitlines()
    assert "--candidate-devices 0,1,2,3" in recorded_calls[0]
    assert "--candidate-devices 3" in recorded_calls[1]


def test_resource_gate_rejects_a_selected_device_that_became_busy(
    tmp_path: Path,
):
    call_count = tmp_path / "call-count.txt"
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['SELECTOR_CALL_COUNT'])\n"
        "count = int(path.read_text()) if path.exists() else 0\n"
        "path.write_text(str(count + 1))\n"
        "if count == 0:\n"
        "    print('selected\\t3\\tlogical-map\\t62000\\t65536\\t60294')\n"
        "else:\n"
        "    print(\n"
        "        'unavailable\\t3\\tlogical-map\\t4000\\t65536\\t60294\\tno_device'\n"
        "    )\n"
        "    raise SystemExit(3)\n",
        encoding="utf-8",
    )
    command = f"""
source {GATE_PATH}
ASCEND_DEVICE_SELECTOR={fake_selector}
PYTHON_BIN={sys.executable}
select_ascend_e2e_device smoke
confirm_selected_ascend_e2e_device smoke-launch
"""

    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "SELECTOR_CALL_COUNT": str(call_count),
        },
    )

    assert result.returncode == 87
    assert "no longer meets" in result.stderr
    assert "ASCEND_NPU_MEMORY_STATUS=insufficient" in result.stderr
    assert "device=npu:3" in result.stderr


def test_resource_gate_does_not_turn_unavailable_hardware_into_success(
    tmp_path: Path,
):
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "import sys\n"
        "print('unavailable\\t3\\tlogical-map\\t4000\\t65536\\t60294\\tno_device')\n"
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

    assert result.returncode == 87
    assert "Ascend resource gate failed" in result.stderr
    assert "free_gib=3.91" in result.stderr
    assert "required_gib=58.88" in result.stderr
    assert "Status: failed" in summary.read_text(encoding="utf-8")


def test_shared_gate_selects_before_preflight_and_confirmation():
    gate_text = GATE_PATH.read_text(encoding="utf-8")
    selection = gate_text.index('select_ascend_e2e_device "$workload_name"')
    sudo_branch = gate_text.index(
        'if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then', selection
    )
    preflight = gate_text.index("elif ensure_runner_npu_ready; then", sudo_branch)
    confirmation = gate_text.index("confirm_selected_ascend_e2e_device", preflight)

    assert selection < sudo_branch < preflight < confirmation

    for path, workload in (
        (SMOKE_PATH, "vLLM serve smoke test"),
        (REGRESSION_PATH, "vLLM inference regression test"),
    ):
        text = path.read_text(encoding="utf-8")
        prepare = text.rindex(
            f'\nif prepare_ascend_device_for_server "{workload}"; then'
        )
        launch = text.rindex("\nstart_server\n")
        assert prepare < launch
        assert "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}" in text
        assert '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"' in text

    assert "ASCEND_RT_VISIBLE_DEVICES" in GATE_PATH.read_text(encoding="utf-8")


def test_resource_gate_does_not_retry_a_deterministic_selection_failure(
    tmp_path: Path,
):
    calls = tmp_path / "calls.txt"
    fake_selector = tmp_path / "selector.py"
    fake_selector.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['SELECTOR_CALLS'])\n"
        "with path.open('a') as handle:\n"
        "    handle.write('called\\n')\n"
        "print('unavailable\\t3\\tlogical-map\\t4000\\t65536\\t60294\\tno_device')\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    command = f"""
source {GATE_PATH}
ASCEND_DEVICE_SELECTOR={fake_selector}
PYTHON_BIN={sys.executable}
ASCEND_DEVICE_SELECTION_ATTEMPTS=3
ASCEND_E2E_USE_SUDO=0
ensure_runner_npu_ready() {{ return 0; }}
prepare_ascend_device_for_server smoke
"""

    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SELECTOR_CALLS": str(calls)},
    )

    assert result.returncode == 87
    assert calls.read_text(encoding="utf-8").splitlines() == ["called"]


def test_inference_logs_are_scoped_to_the_job_runtime_directory():
    for path in (
        SMOKE_PATH,
        SCRIPT_DIR / "run_e2e_inference_regression.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert 'log_dir="$runtime_root/logs"' in text
        assert "/tmp/vllm-e2e-" not in text
