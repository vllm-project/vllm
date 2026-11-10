# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared server, request, and benchmark helpers for Foundry GPU tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import regex as re

from tests.utils import (
    RemoteOpenAIServer,
)

MODEL = os.environ.get(
    "VLLM_FOUNDRY_TEST_MODEL",
    os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
)
CAPTURE_SIZES = [1, 2]
PROMPT = "The capital of France is"


@dataclass(frozen=True)
class RunResult:
    name: str
    startup_seconds: float
    graph_build_seconds: float
    outputs: tuple[str, ...]
    benchmark_seconds: float
    benchmark_output_tokens: int
    full_graph_sizes: tuple[int, ...]
    log_path: Path


class _LoggedServer(RemoteOpenAIServer):
    """RemoteOpenAIServer variant that retains the engine log for assertions."""

    def __init__(self, *args, log_path: Path, **kwargs):
        self._log_path = log_path
        self._log_file = None
        super().__init__(*args, **kwargs)

    def _start_server(self, model, vllm_serve_args, env_dict):
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_LOG_STATS_INTERVAL"] = "1"
        if env_dict:
            env.update(env_dict)
        # Pin the server to this checkout.  The venv can contain an older
        # editable vLLM, which would silently bypass the adapter under test.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH")],
            )
        )
        self._log_file = self._log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                model,
                *vllm_serve_args,
            ],
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _shutdown(self):
        try:
            super()._shutdown()
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None


def _write_foundry_config(path: Path, mode: str, workspace: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'mode = "{mode}"',
                f'workspace_root = "{workspace}"',
                'base_addr = "0x600000000000"',
                'region_size = "256GB"',
                'scratch_space_size = "2GB"',
                'weight_cache_size = "2GB"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _compilation_config(foundry_config: Path | None) -> str:
    # Keep this CI workload on the FULL decode path.  PIECEWISE capture creates
    # dozens of backend subgraphs whose restore order is a separate contract;
    # this test targets the Foundry archive/replay lifecycle for the runner
    # graphs and exercises two deterministic decode sizes.
    config = {
        # Foundry restores driver graphs without rebuilding a Torch compiled
        # model. Match that kernel path in the native reference as well, while
        # retaining FULL decode graph capture/replay in every arm.
        "mode": 0,
        "decode_cudagraph_capture_request_sizes": CAPTURE_SIZES,
        "use_breakable_cudagraph": False,
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }
    if foundry_config is not None:
        # Exercise the adapter's automatic SAVE/LOAD compile-mode pin.
        config.pop("mode")
        config["graph_extension_config_path"] = str(foundry_config)
    return json.dumps(config)


def _server_args(foundry_config: Path | None) -> list[str]:
    return [
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--skip-mm-profiling",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.75",
        "--compilation-config",
        _compilation_config(foundry_config),
        "--cudagraph-metrics",
    ]


def _run_requests(server: _LoggedServer) -> tuple[tuple[str, ...], float, int]:
    client = server.get_client(timeout=600)
    outputs: list[str] = []
    output_tokens = 0
    start = time.perf_counter()
    for _ in range(3):
        response = client.completions.create(
            model=MODEL,
            prompt=[PROMPT] * max(CAPTURE_SIZES),
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        outputs.extend(choice.text for choice in response.choices)
        output_tokens += response.usage.completion_tokens
    return tuple(outputs), time.perf_counter() - start, output_tokens


def _run_benchmark(server: _LoggedServer, result_path: Path) -> tuple[float, int]:
    """Use vLLM's serving benchmark for the steady-state performance gate."""
    with result_path.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "serve",
                "--model",
                MODEL,
                "--base-url",
                server.url_root,
                "--backend",
                "openai",
                "--dataset-name",
                "random",
                "--random-input-len",
                "32",
                "--random-output-len",
                "32",
                "--num-prompts",
                "64",
                "--max-concurrency",
                "2",
                "--num-warmups",
                "2",
                "--ignore-eos",
                "--seed",
                "0",
                "--save-result",
                "--result-dir",
                str(result_path.parent),
                "--result-filename",
                result_path.name,
            ],
            check=True,
            timeout=300,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["completed"] == 64 and result["failed"] == 0, result
    assert result["total_output_tokens"] == 64 * 32, result
    return result["duration"], result["total_output_tokens"]


def _full_graph_sizes(log_text: str) -> tuple[int, ...]:
    rows = re.findall(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|",
        log_text,
    )
    return tuple(
        sorted(
            {
                int(actual)
                for actual, padded, _, mode, count in rows
                if actual == padded and "FULL" in mode and int(count) > 0
            }
        )
    )


def _run_arm(
    name: str,
    foundry_config: Path | None,
    log_path: Path,
    workspace: Path,
    *,
    extra_args: list[str] | None = None,
    env_dict: dict[str, str] | None = None,
) -> RunResult:
    startup_start = time.perf_counter()
    try:
        with _LoggedServer(
            MODEL,
            [*_server_args(foundry_config), *(extra_args or [])],
            log_path=log_path,
            max_wait_seconds=900,
            env_dict=env_dict,
        ) as server:
            startup_seconds = time.perf_counter() - startup_start
            client = server.get_client(timeout=600)

            # The canary is a separate first request: LOAD must replay a restored
            # graph before the steady-state benchmark begins.
            canary = client.completions.create(
                model=MODEL,
                prompt=PROMPT,
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                seed=0,
            )
            assert canary.choices, "canary completion returned no choices"
            canary_text = canary.choices[0].text
            if MODEL == "Qwen/Qwen3-0.6B":
                assert "paris" in canary_text.lower(), canary_text

            outputs, _, _ = _run_requests(server)
            outputs = (canary_text, *outputs)
            benchmark_seconds, output_tokens = _run_benchmark(
                server, log_path.with_name(f"{name}-benchmark.json")
            )
            deadline = time.monotonic() + 10
            while _full_graph_sizes(log_path.read_text()) != tuple(CAPTURE_SIZES):
                assert server.proc.poll() is None, "Server exited before FULL dispatch"
                assert time.monotonic() < deadline, (
                    "Timed out waiting for FULL dispatch"
                )
                time.sleep(0.1)
    except Exception as exc:
        log_tail = ""
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8")[-12000:]
        raise RuntimeError(
            f"{name} Foundry server failed; log={log_path}\n{log_tail}"
        ) from exc

    log_text = log_path.read_text(encoding="utf-8")
    full_graph_sizes = _full_graph_sizes(log_text)
    assert full_graph_sizes == tuple(CAPTURE_SIZES), full_graph_sizes
    graph_build_seconds = 0.0
    if foundry_config is not None:
        assert "native memory profiling completed in" in log_text
    if name.startswith("load"):
        # start_graph_builds only schedules background work; its return time
        # is not the time spent reconstructing and instantiating the graphs.
        match = re.search(r"2 graphs built in ([0-9.]+) ms", log_text)
        assert match, "LOAD did not finish building the two archived graphs"
        graph_build_seconds = float(match.group(1)) / 1000
        assert log_text.count("restored graph descriptor=") == 2
        assert "LOAD recapture fallback" not in log_text
        assert "capture_or_load_graph called on LOAD" not in log_text

    if name == "save":
        archives = list(workspace.rglob("*.cugraph"))
        assert archives, "SAVE produced no Foundry graph archives"

    return RunResult(
        name=name,
        startup_seconds=startup_seconds,
        graph_build_seconds=graph_build_seconds,
        outputs=outputs,
        benchmark_seconds=benchmark_seconds,
        benchmark_output_tokens=output_tokens,
        full_graph_sizes=full_graph_sizes,
        log_path=log_path,
    )
