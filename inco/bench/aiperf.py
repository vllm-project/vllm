# SPDX-License-Identifier: Apache-2.0
"""Build and run `aiperf profile` invocations for one concurrency point.

aiperf is young and has renamed several flags (the genai-perf era
``--synthetic-input-tokens-mean`` became ``--prompt-input-tokens-mean``), so the
flag names used here are resolved against the installed binary's ``--help``
output instead of being hard-coded.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bench.config import SweepConfig, Workload

# Preferred flag name first, older aliases after.
_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "isl": ("--prompt-input-tokens-mean", "--synthetic-input-tokens-mean", "--isl"),
    "isl_stddev": (
        "--prompt-input-tokens-stddev",
        "--synthetic-input-tokens-stddev",
        "--isl-stddev",
    ),
    "osl": ("--prompt-output-tokens-mean", "--output-tokens-mean", "--osl"),
    "osl_stddev": (
        "--prompt-output-tokens-stddev",
        "--output-tokens-stddev",
        "--osl-stddev",
    ),
    "request_count": ("--request-count", "--num-requests"),
    "warmup": ("--warmup-request-count", "--num-warmup-requests"),
    "ui": ("--ui-type", "--ui"),
    "artifact_dir": ("--artifact-dir", "--output-artifact-dir"),
}


class AiperfNotInstalled(RuntimeError):
    pass


@lru_cache(maxsize=8)
def aiperf_help(aiperf_bin: str) -> str:
    """Cached ``aiperf profile --help`` text, used for flag resolution."""
    if shutil.which(aiperf_bin) is None:
        raise AiperfNotInstalled(
            f"{aiperf_bin!r} not found on PATH; `pip install -r inco/requirements.txt`"
        )
    proc = subprocess.run(
        [aiperf_bin, "profile", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "COLUMNS": "200", "TERM": "dumb"},
    )
    return proc.stdout + proc.stderr


def resolve_flag(key: str, help_text: str | None) -> str:
    """Pick the first alias for ``key`` that the installed aiperf accepts.

    When ``help_text`` is None (offline command rendering, tests) the preferred
    modern name is returned.
    """
    aliases = _FLAG_ALIASES[key]
    if help_text is None:
        return aliases[0]
    for alias in aliases:
        if alias in help_text:
            return alias
    raise RuntimeError(
        f"installed aiperf accepts none of {aliases} for {key!r}; "
        "check the aiperf version pinned in inco/requirements.txt"
    )


def build_aiperf_command(
    workload: Workload,
    sweep: SweepConfig,
    concurrency: int,
    help_text: str | None = None,
) -> list[str]:
    """Render the exact aiperf invocation for a single concurrency point."""
    flag = lambda key: resolve_flag(key, help_text)  # noqa: E731
    artifact_dir = sweep.artifact_dir(workload, concurrency)

    cmd = [
        sweep.aiperf_bin,
        "profile",
        "--model",
        workload.model,
        "--tokenizer",
        workload.tokenizer_id,
        "--url",
        sweep.url,
        "--endpoint-type",
        workload.endpoint_type,
        flag("isl"),
        str(workload.isl),
        flag("isl_stddev"),
        str(workload.isl_stddev),
        flag("osl"),
        str(workload.osl),
        flag("osl_stddev"),
        str(workload.osl_stddev),
        "--concurrency",
        str(concurrency),
        flag("request_count"),
        str(sweep.request_count(concurrency)),
        "--random-seed",
        str(workload.random_seed),
        flag("artifact_dir"),
        str(artifact_dir),
        flag("ui"),
        sweep.ui,
    ]
    if workload.streaming:
        cmd.append("--streaming")
    if warmup := sweep.warmup_count(concurrency):
        cmd += [flag("warmup"), str(warmup)]
    if sweep.benchmark_duration:
        cmd += ["--benchmark-duration", str(sweep.benchmark_duration)]
    if workload.ignore_eos:
        # Without this the server may stop early and OSL becomes a
        # model-dependent variable rather than a fixed part of the workload.
        cmd += ["--extra-inputs", "ignore_eos:true"]
    if sweep.extra_aiperf_args:
        cmd += shlex.split(sweep.extra_aiperf_args)
    return cmd


_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def streaming_runner(on_line, heartbeat_s: float = 5.0):
    """Build a ``runner`` that echoes aiperf's output as it arrives.

    aiperf can sit quiet for minutes while a concurrency point runs, and under
    Modal its output is otherwise swallowed. The heartbeat thread emits a line
    every ``heartbeat_s`` seconds so a working run never looks stuck.
    """

    def run(cmd, check=False):  # noqa: ARG001 - matches subprocess.run's seam
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "TERM": "dumb"},
        )
        stop = threading.Event()

        def heartbeat():
            started = time.monotonic()
            while not stop.wait(heartbeat_s):
                on_line(f"... aiperf running ({time.monotonic() - started:.0f}s)")

        beat = threading.Thread(target=heartbeat, daemon=True)
        beat.start()
        try:
            for raw in proc.stdout:
                if line := _ANSI.sub("", raw).strip():
                    on_line(line)
            return subprocess.CompletedProcess(cmd, proc.wait())
        finally:
            stop.set()
            beat.join(timeout=1)

    return run


@dataclass
class AiperfResult:
    concurrency: int
    returncode: int
    artifact_dir: Path
    command: list[str]
    export_json: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.export_json is not None


def find_export_json(artifact_dir: Path) -> Path | None:
    """Locate ``profile_export_aiperf.json`` under an aiperf artifact dir.

    aiperf nests results in a ``{model}-{config}`` subdirectory whose exact
    name depends on the run configuration, so glob for it.
    """
    if not artifact_dir.exists():
        return None
    direct = artifact_dir / "profile_export_aiperf.json"
    if direct.is_file():
        return direct
    matches = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    return matches[-1] if matches else None


def run_concurrency_point(
    workload: Workload,
    sweep: SweepConfig,
    concurrency: int,
    help_text: str | None = None,
    runner=subprocess.run,
) -> AiperfResult:
    """Run one aiperf point, streaming its output to the console."""
    cmd = build_aiperf_command(workload, sweep, concurrency, help_text)
    artifact_dir = sweep.artifact_dir(workload, concurrency)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    proc = runner(cmd, check=False)
    return AiperfResult(
        concurrency=concurrency,
        returncode=proc.returncode,
        artifact_dir=artifact_dir,
        command=cmd,
        export_json=find_export_json(artifact_dir),
    )
