from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github/workflows/scripts/check_ascend_npu_memory.py"
CI_SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
GIB = 1024**3


def _write_fake_torch(root: Path, *, include_mem_get_info: bool = True) -> None:
    torch_dir = root / "torch"
    torch_dir.mkdir()
    mem_get_info = """
    def mem_get_info(self):
        return (
            int(os.environ["FAKE_NPU_FREE_BYTES"]),
            int(os.environ["FAKE_NPU_TOTAL_BYTES"]),
        )
""" if include_mem_get_info else ""
    (torch_dir / "__init__.py").write_text(
        f"""import os

class _NPU:
    def is_available(self):
        return os.environ.get("FAKE_NPU_AVAILABLE", "1") == "1"

    def set_device(self, device):
        self.device = device
{mem_get_info}
npu = _NPU()

def zeros(size, device=None):
    if os.environ.get("FAKE_NPU_ALLOCATION_FAIL") == "1":
        raise RuntimeError("allocation failed")
    return [0] * size
""",
        encoding="utf-8",
    )
    (root / "torch_npu.py").write_text("# fake torch_npu\n", encoding="utf-8")


def _run(
    tmp_path: Path,
    *,
    free_gib: int = 60,
    total_gib: int = 64,
    utilization: str = "0.92",
    include_mem_get_info: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    _write_fake_torch(fake_modules, include_mem_get_info=include_mem_get_info)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--device",
            "npu:3",
            "--utilization",
            utilization,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(fake_modules),
            "FAKE_NPU_FREE_BYTES": str(free_gib * GIB),
            "FAKE_NPU_TOTAL_BYTES": str(total_gib * GIB),
            **(extra_env or {}),
        },
    )


def test_memory_preflight_reports_capacity_and_succeeds(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.returncode == 0
    assert "ASCEND_NPU_MEMORY_STATUS=ok" in result.stdout
    assert "device=npu:3" in result.stdout
    assert "free_gib=60.00" in result.stdout
    assert "total_gib=64.00" in result.stdout
    assert "required_gib=58.88" in result.stdout


def test_memory_preflight_fails_fast_with_dedicated_exit_code(tmp_path: Path) -> None:
    result = _run(tmp_path, free_gib=4)

    assert result.returncode == 87
    assert "ASCEND_NPU_MEMORY_STATUS=insufficient" in result.stderr
    assert "free_gib=4.00" in result.stderr
    assert "required_gib=58.88" in result.stderr


def test_memory_preflight_rejects_invalid_utilization(tmp_path: Path) -> None:
    result = _run(tmp_path, utilization="1.1")

    assert result.returncode == 2
    assert "must be in (0, 1]" in result.stderr


def test_memory_preflight_preserves_allocation_failure(tmp_path: Path) -> None:
    result = _run(tmp_path, extra_env={"FAKE_NPU_ALLOCATION_FAIL": "1"})

    assert result.returncode == 1
    assert "allocation preflight failed" in result.stderr


def test_memory_preflight_uses_controlled_allocation_fallback(tmp_path: Path) -> None:
    result = _run(tmp_path, include_mem_get_info=False)

    assert result.returncode == 0
    assert "ASCEND_NPU_MEMORY_STATUS=unavailable" in result.stdout
    assert "fallback=allocation-only" in result.stdout


def test_ascend_ci_scripts_share_memory_preflight_and_preserve_exit_code() -> None:
    scripts = (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
        "run_ascend_benchmark_ci.sh",
    )

    for name in scripts:
        content = (CI_SCRIPT_DIR / name).read_text(encoding="utf-8")
        assert "check_ascend_npu_memory.py" in content
        assert '--insufficient-exit-code "$NPU_MEMORY_EXIT_CODE"' in content
        assert 'return "$NPU_MEMORY_EXIT_CODE"' in content
        assert 'exit "$NPU_MEMORY_EXIT_CODE"' in content


def test_ascend_ci_scripts_classify_memory_pressure_on_exit_and_timeout() -> None:
    scripts = (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
        "run_ascend_benchmark_ci.sh",
    )

    for name in scripts:
        content = (CI_SCRIPT_DIR / name).read_text(encoding="utf-8")
        assert "Free memory on device" in content
        assert "server exited before becoming ready" in content
        assert "Timed out waiting for vLLM server to become ready" in content
        assert content.count('exit "$NPU_MEMORY_EXIT_CODE"') >= 2
