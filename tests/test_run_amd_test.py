# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".buildkite"
    / "scripts"
    / "hardware_ci"
    / "run-amd-test.py"
)


def load_run_amd_test_module(env=None):
    env = {} if env is None else env
    with mock.patch.dict(os.environ, env, clear=True):
        spec = importlib.util.spec_from_file_location("run_amd_test", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class RunAmdTestPidConfigTests(unittest.TestCase):
    def test_build_single_node_docker_cmd_uses_safer_default_pids_limit(self):
        module = load_run_amd_test_module()
        with mock.patch.object(module, "build_cache_docker_args", return_value=[]):
            cmd = module.build_single_node_docker_cmd(
                image="rocm/vllm-ci:test",
                name="rocm-test",
                commands="pytest -v -s tests/",
                render_gid="109",
                results_dir=Path("/tmp/results"),
                render_devices="",
                rdma=False,
            )
        self.assertIn("--pids-limit=16384", cmd)

    def test_build_single_node_docker_cmd_honors_pids_limit_override(self):
        module = load_run_amd_test_module({"VLLM_CI_DOCKER_PIDS_LIMIT": "8192"})
        with mock.patch.object(module, "build_cache_docker_args", return_value=[]):
            cmd = module.build_single_node_docker_cmd(
                image="rocm/vllm-ci:test",
                name="rocm-test",
                commands="pytest -v -s tests/",
                render_gid="109",
                results_dir=Path("/tmp/results"),
                render_devices="",
                rdma=False,
            )
        self.assertIn("--pids-limit=8192", cmd)

    def test_validate_container_pids_config_rejects_zero(self):
        module = load_run_amd_test_module({"VLLM_CI_DOCKER_PIDS_LIMIT": "0"})
        with self.assertRaises(SystemExit) as exc:
            module.validate_container_pids_config()
        self.assertEqual(exc.exception.code, 1)

    def test_diagnose_container_exit_detects_ray_thread_budget_exhaustion(self):
        module = load_run_amd_test_module()
        log_text = "\n".join(
            [
                (
                    "2026-04-06 00:31:52 INFO worker.py:2004 -- Started a "
                    "local Ray instance."
                ),
                (
                    "(pid=6089) E0406 00:31:57 pthread_create failed: "
                    "Resource temporarily unavailable"
                ),
                "what(): thread: Resource temporarily unavailable [system:11]",
            ]
        )

        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            fh.write(log_text)
            log_path = Path(fh.name)

        def fake_sh(cmd, *, check=False, capture=False, timeout=None):
            self.assertEqual(cmd[:2], ["docker", "inspect"])
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="false 1 \n",
                stderr="",
            )

        try:
            with mock.patch.object(module, "sh", side_effect=fake_sh):
                diag = module.diagnose_container_exit("ray-test", log_file=log_path)
        finally:
            log_path.unlink(missing_ok=True)

        self.assertTrue(diag["pids_exhausted"])
        self.assertIn(
            "pthread_create failed: Resource temporarily unavailable",
            diag["pid_error"],
        )


if __name__ == "__main__":
    unittest.main()
