# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast Start preload prepares graphs for zero-copy engine restarts."""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import regex as re

from tests.utils import large_gpu_mark
from tests.v1.cudagraph import foundry_utils as harness

pytestmark = [pytest.mark.optional, large_gpu_mark(min_gb=16)]


@pytest.mark.timeout(1800)
def test_foundry_ipc_preload_save_load_and_restart(tmp_path: Path):
    """Preload prepares graphs once for successive zero-copy serving engines."""
    pytest.importorskip("foundry")
    from tests.model_executor.model_loader.test_weight_cache import WeightCacheDaemon
    from vllm.compilation.foundry_adapter.config import CUDAGraphExtensionConfig

    workspace = tmp_path / "foundry-archive"
    workspace.mkdir()
    save_config = tmp_path / "foundry-save.toml"
    load_config = tmp_path / "foundry-load.toml"
    harness._write_foundry_config(save_config, "save", workspace)
    harness._write_foundry_config(load_config, "load", workspace)
    log_root = Path(os.environ.get("VLLM_FOUNDRY_LOG_DIR", tmp_path)) / "ipc-preload"
    log_root.mkdir(parents=True, exist_ok=True)
    checkout = str(Path(__file__).resolve().parents[3])
    env = {
        "PYTHONPATH": os.pathsep.join(
            filter(None, [checkout, os.environ.get("PYTHONPATH")])
        ),
    }
    preparation_start = time.perf_counter()
    hook = CUDAGraphExtensionConfig.from_toml(save_config).hook_library_path
    assert hook is not None
    # Native still needs the import hook for VMM-exported weights, but uses
    # ordinary graph capture without the Foundry graph persistence adapter.
    import_env = {
        "LD_PRELOAD": os.pathsep.join(
            filter(None, [hook, os.environ.get("LD_PRELOAD")])
        )
    }
    with WeightCacheDaemon(
        harness.MODEL,
        tp_size=1,
        extra_args=[
            *harness._server_args(save_config),
            "--prepare-cudagraphs",
            "--kv-cache-memory-bytes",
            str(256 << 20),
            "--disable-flashinfer-autotune",
        ],
        env_extra=env,
        log_file=str(log_root / "daemon.log"),
        enforce_eager=False,
    ) as daemon:
        daemon.wait_until_ready()
        preload_ready_seconds = time.perf_counter() - preparation_start
        assert len(list(workspace.rglob("*.cugraph"))) == 2
        preparation = re.search(
            r"CUDA graph preparation completed in ([0-9.]+) s", daemon.logs
        )
        assert preparation, "preload reported ready without completing SAVE"
        assert daemon.logs.count("Foundry preparation engine completed SAVE") == 1
        ipc_args = [
            "--load-format",
            "ipc_cache",
            "--model-loader-extra-config",
            json.dumps(
                {
                    "socket_dir": daemon.socket_dir,
                    "mode": "zero_copy",
                    "fallback": False,
                }
            ),
            # Fix the KV geometry across processes and separate this graph
            # contract from automatic external-weight memory budgeting.
            "--kv-cache-memory-bytes",
            str(256 << 20),
            "--disable-flashinfer-autotune",
        ]

        def run(name: str, config: Path | None) -> harness.RunResult:
            result = harness._run_arm(
                name,
                config,
                log_root / f"{name}.log",
                workspace,
                extra_args=ipc_args,
                env_dict=import_env,
            )
            text = result.log_path.read_text(encoding="utf-8")
            assert re.search(r"Mapped \d+ tensors .*\(zero_copy mode\)", text)
            assert "falling back to disk" not in text
            daemon.check_health()
            return result

        native = run("ipc-native", None)
        load = run("load", load_config)
        restart = run("load-restart", load_config)
        results = [native, load, restart]
        (log_root / "ipc-results.json").write_text(
            json.dumps(
                {
                    "preload_ready_seconds": preload_ready_seconds,
                    "preparation_seconds": float(preparation.group(1)),
                    "runs": [asdict(result) for result in results],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        for result in results[1:]:
            assert result.outputs == native.outputs
        native_rate = native.benchmark_output_tokens / native.benchmark_seconds
        for result in (load, restart):
            rate = result.benchmark_output_tokens / result.benchmark_seconds
            assert rate >= native_rate * 0.85
