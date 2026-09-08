#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving A/B for the EC CPU offload connector.

Runs one workload twice against identical engine configs that differ only in
whether `--ec-transfer-config` is set:

    recompute   no connector; a re-requested image is re-encoded
    connector   ECCPUConnector in ec_both mode; a re-request reloads from CPU

Load is driven by `vllm bench serve` over the `custom_image` dataset that
`gen_workload.py` emits, so the reported latencies are the ones a client sees.
Alongside them it reads the connector's own DEBUG accounting out of the server
log, and re-applies Phase 0's two gates to the real workload: the connector arm
must actually load entries, and must compute fewer encoder inputs than the
recompute arm. If those fail the timings measure nothing, and this says so
instead of printing a delta.

Because both arms share a port and a GPU, every arm verifies from the fresh log
that the server it is about to measure really is the arm it asked for -- the
connector arm by the EC region's creation line, the recompute arm by that line's
absence. Killing the previous server and hoping is not enough: a survivor would
answer /health and be measured under the wrong label.

`--frag` runs the connector arm with a region deliberately smaller than the
working set and descriptor counting enabled, reporting bandwidth per time window
-- the question being whether an entry still collapses to one descriptor once the
region has churned.

Typical use, driving a server in a pod:

    python run_bench.py --pod vllm-omer-2 \
        --workload-dir /vllm-workspace/bench/wl --out-dir results/

Against a server somebody else is running (an EPD proxy, for instance), where
this cannot manage the process and needs to be told where the log is:

    python run_bench.py --base-url http://proxy:8000 \
        --server-log /vllm-workspace/logs/consumer.log --arms connector
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ec_log_stats import (
    decay_report,
    rewritten_items,
    stage_summary,
    summarize,
    window_stats,
)

ARMS = ("recompute", "connector")
LOG_DIR = "/vllm-workspace/logs"
FRAG_FILE = "/tmp/ec_bench_frag.jsonl"
# Instrumentation lives in its own directory so it is not auto-imported by the
# scripts beside this one; only the server gets it on PYTHONPATH.
PATCH_DIR = "/vllm-workspace/bench/patches"
PROXY = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
QUEUE_CSV = "/tmp/ec_bench_queue.csv"
QUEUE_SAMPLER = "/vllm-workspace/bench/queue_sampler.sh"
QUEUE_SAMPLE_INTERVAL_S = 1.0
# Each EPD configuration is (send the grid instead of the pixels, device the
# encoder's image transform runs on) -- the two independent changes in #50390.
# (send the grid instead of the pixels, encoder transform device, connector).
# An empty connector means no --ec-transfer-config at all.
#
# "cpu" is deliberately kept even though it can transfer nothing: the CPU
# connector locates a peer's entry only through `ec_transfer_params`, which the
# proxy relays exclusively when it rewrites, so without hints the consumer
# recomputes and this arm prices the producer's save with no benefit. The
# example connector has no such dependency -- it finds entries by hash on shared
# storage -- which is why its no-hint arm does transfer.
EPD_CONFIGS: dict[str, tuple[bool, str, str]] = {
    "none": (False, "cpu", ""),
    "example": (False, "cpu", "ECExampleConnector"),
    "cpu": (False, "cpu", "ECCPUConnector"),
    "example-grid": (True, "cpu", "ECExampleConnector"),
    "cpu-grid": (True, "cpu", "ECCPUConnector"),
    "baseline": (False, "cpu", "ECExampleConnector"),
    "grid": (True, "cpu", "ECExampleConnector"),
    "gpu": (False, "cuda", "ECExampleConnector"),
    "both": (True, "cuda", "ECExampleConnector"),
}
# Arms where the consumer must end up loading an encoding from the producer. The
# rest must load nothing, and both directions are asserted.
EPD_EXPECT_LOADS = {
    "example",
    "example-grid",
    "cpu-grid",
    "baseline",
    "grid",
    "gpu",
    "both",
}
# Proof each change actually engaged, read from the log of the process that
# would emit it. A configuration that did not engage is not a measurement.
_GPU_PROCESSOR_MARKER = "Running the multi-modal processor on cuda"

_HEALTH_POLL_S = 5.0
# Generous because an orderly exit has to tear down CUDA and, on the connector
# arm, unlink the EC region; escalating early is what leaks multi-GiB
# /dev/shm files.
_STOP_TIMEOUT_S = 180
_DETACH_TIMEOUT_S = 120
_PGID_READ_ATTEMPTS = 10
_PGID_READ_DELAY_S = 1.0
_EC_REGION_MARKER = "Created EC mmap file"
# Either line proves the log belongs to a server that got as far as serving.
# Two of them because the exact wording is version-dependent, while uvicorn's
# is stable.
_STARTUP_MARKERS = ("Application startup complete", "Starting vLLM server on")


class ServerMismatchError(RuntimeError):
    """The running server is not the arm that was requested."""


class Target:
    """Runs shell commands locally or inside a pod."""

    def __init__(self, pod: str | None) -> None:
        self.pod = pod

    def _argv(self, script: str) -> list[str]:
        argv = ["bash", "-lc", script]
        if self.pod:
            return ["oc", "exec", self.pod, "--", *argv]
        return argv

    def sh(
        self, script: str, *, timeout: int = 600, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self._argv(script), capture_output=True, text=True, timeout=timeout
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"command failed ({result.returncode}): {script}\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return result

    def sh_detached(
        self, script: str, *, timeout: int = _DETACH_TIMEOUT_S
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a command that leaves a daemon behind.

        Output must not be captured: a backgrounded server inherits the pipes
        and never closes them, so waiting on them would block until the timeout
        even though the launching shell exited immediately.

        The returned `CompletedProcess` describes the launching shell, which has
        already exited -- it is not a handle on the daemon. The daemon is
        addressed by the pid it records inside the target, since with `--pod` it
        does not even live on this machine.
        """
        return subprocess.run(
            self._argv(script),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )

    def file_size(self, path: str) -> int:
        out = self.sh(f"stat -c %s {shlex.quote(path)} 2>/dev/null || echo 0").stdout
        return int(out.strip() or 0)

    def read_bytes(self, path: str, start: int, end: int) -> str:
        if end <= start:
            return ""
        quoted = shlex.quote(path)
        return self.sh(f"tail -c +{start + 1} {quoted} | head -c {end - start}").stdout

    def read_text(self, path: str) -> str:
        return self.sh(f"cat {shlex.quote(path)} 2>/dev/null || true").stdout


@dataclass
class BenchServer:
    """One process this harness starts and stops: a vLLM server, or the proxy.

    `managed` is False when `--base-url` points at something this process did not
    start: then start/stop are no-ops and the caller supplies the log path,
    because there is no launch command to derive it from.

    The optional fields exist for the EPD topology, which needs several
    processes at once; the single-server topology leaves them at their defaults.
    `command` runs something that is not a vLLM server (the proxy) through the
    same launch and teardown path, rather than duplicating it.
    """

    target: Target
    args: argparse.Namespace
    arm: str
    log_path: str
    managed: bool = True
    name: str = ""
    port: int = 0
    gpu: str = ""
    command: str = ""
    ec_config: dict[str, Any] | None = None
    gpu_util: float = 0.0
    extra_args: tuple[str, ...] = ()
    extra_env: tuple[str, ...] = ()
    match: tuple[str, str] = ()

    def __post_init__(self) -> None:
        self.name = self.name or self.arm
        self.port = self.port or self.args.port
        self.gpu = self.gpu or self.args.gpu

    @property
    def base_url(self) -> str:
        return self.args.base_url or f"http://127.0.0.1:{self.port}"

    @property
    def pid_file(self) -> str:
        return f"/tmp/ec_bench_{self.name}.pid"

    @property
    def pgid_file(self) -> str:
        return f"/tmp/ec_bench_{self.name}.pgid"

    @property
    def batched_tokens(self) -> int:
        """Token budget per engine step for this instance.

        An encode-only instance needs a budget above one image's token count or
        it can never batch two image requests, which pins it to one image per
        step regardless of load -- the artefact that made every rate=inf
        measurement a queue-depth measurement. The example's README asks for
        "a very high value (effectively unlimited)" here for the same reason.
        Decode keeps the ordinary budget, identical across arms.
        """
        if self.name == "encoder":
            return self.args.encoder_max_num_batched_tokens
        return self.args.max_num_batched_tokens

    @property
    def kill_match(self) -> tuple[str, str]:
        """Two substrings that together identify this process and nothing else.

        Both are required: the port alone also appears in the proxy's
        `--encode-servers-urls`, and the command alone would match every vLLM
        server on a shared dev pod.
        """
        return self.match or ("cli.main serve", f"--port {self.port}")

    @property
    def engine_id(self) -> str:
        return f"ec-bench-{self.args.run_id}-{self.name}"

    def ec_transfer_config(self) -> str:
        """Connector config for this instance.

        Defaults to the local-offload config the single-server A/B uses; the EPD
        topology passes producer and consumer configs in `ec_config`.

        `ec_enable_nixl` is a setting of the connector, so it goes in
        `ec_connector_extra_config`, and is sent only when asked for.
        """
        config: dict[str, Any] = dict(self.ec_config) if self.ec_config else {
            "ec_connector": "ECCPUConnector",
            "ec_role": "ec_both",
            "ec_connector_extra_config": {"ec_cpu_bytes": self.args.ec_cpu_bytes},
        }
        config.setdefault("engine_id", self.engine_id)
        if self.args.ec_enable_nixl:
            # Copy before mutating: `config` is a shallow copy of `ec_config`,
            # so the nested dict is still the caller's.
            extra = dict(config.get("ec_connector_extra_config") or {})
            extra.setdefault("ec_enable_nixl", True)
            config["ec_connector_extra_config"] = extra
        return json.dumps(config)

    def launch_script(self, *, instrument: bool) -> str:
        """Build the launch command.

        `cd /tmp` matters: from /vllm-workspace, `import vllm` resolves the repo
        directory as a namespace package and top-level attributes disappear.
        DEBUG matters: the connector's transfer accounting is on debug lines.
        """
        env = [
            f"CUDA_VISIBLE_DEVICES={self.gpu}",
            "VLLM_USE_V2_MODEL_RUNNER=1",
            "VLLM_LOGGING_LEVEL=DEBUG",
            "VLLM_SERVER_DEV_MODE=1",
            "HF_HOME=/vllm-workspace",
        ]
        if instrument:
            env += [f"PYTHONPATH={PATCH_DIR}", f"EC_BENCH_FRAG_FILE={FRAG_FILE}"]
        env += list(self.extra_env)
        serve = [self.command] if self.command else self._serve_args()
        # setsid puts the process in a new session so its children -- for a vLLM
        # server the API server, EngineCore and workers -- share one process
        # group that teardown can signal as a unit.
        #
        # The setsid'd shell records its OWN pid, which is the new session leader
        # and therefore the group id, then execs the command into that same pid.
        # Reading the group back with `ps` instead would race: setsid has not
        # necessarily moved the process by the time the launching shell looks,
        # and the check then silently falls back to parent-only kills.
        #
        # Env assignments precede setsid because setsid execs its first argument,
        # so `setsid VAR=x cmd` would look for a program named "VAR=x".
        inner = f"echo $$ > {self.pgid_file}; exec " + " ".join(serve)
        return (
            f"mkdir -p {LOG_DIR} && cd /tmp && "
            + " ".join(env)
            + f" setsid bash -c {shlex.quote(inner)}"
            + f" < /dev/null > {self.log_path} 2>&1 &"
            + f" echo $! > {self.pid_file}; disown"
        )

    def _serve_args(self) -> list[str]:
        serve = [
            f"{self.args.python} -m vllm.entrypoints.cli.main serve {self.args.model}",
            f"--port {self.port}",
            "--dtype bfloat16",
            f"--max-model-len {self.args.max_model_len}",
            f"--gpu-memory-utilization "
            f"{self.gpu_util or self.args.gpu_memory_utilization}",
            f"--max-num-batched-tokens {self.batched_tokens}",
            f"--max-num-seqs {self.args.max_num_seqs}",
            f"--tensor-parallel-size {self.args.tensor_parallel_size}",
            # Identical across arms, and both are required by the accounting
            # this harness reads: iteration details print encoder inputs, and
            # excluding video drops the encoder budget floor 32768 -> 16384.
            "--enable-logging-iteration-details",
            """--limit-mm-per-prompt '{"video":0}'""",
        ]
        if self.ec_config is not None or (
            self.arm == "connector" and self.arm in ARMS
        ):
            serve.append(
                f"--ec-transfer-config {shlex.quote(self.ec_transfer_config())}"
            )
        serve.extend(self.extra_args)
        return serve

    def start(self, *, instrument: bool) -> None:
        if not self.managed:
            print(f"[bench] using existing server at {self.base_url}")
            return
        self.target.sh(
            f"rm -f {self.log_path} {FRAG_FILE} {self.pgid_file}", check=False
        )
        self.stop()
        print(f"[bench] launching {self.name}")
        self.target.sh_detached(self.launch_script(instrument=instrument))
        # Without a recorded pid, stop() falls back to a pattern match alone;
        # fail here instead of discovering it at teardown.
        pid = self.target.read_text(self.pid_file).strip()
        if not pid.isdigit():
            raise RuntimeError(
                f"launch did not record a pid in {self.pid_file} "
                f"(got {pid!r}); see {self.log_path}"
            )
        # The setsid'd shell writes its group id a moment after launch, so give
        # it a bounded number of tries rather than reading once and giving up.
        pgid = ""
        for _ in range(_PGID_READ_ATTEMPTS):
            pgid = self.target.read_text(self.pgid_file).strip()
            if pgid.isdigit():
                break
            time.sleep(_PGID_READ_DELAY_S)
        if pgid.isdigit():
            print(f"[bench] {self.name} pid {pid}, process group {pgid}")
        else:
            print(
                f"[bench] WARNING: no process group recorded for pid {pid}; "
                "teardown will signal the parent and a port-scoped pattern "
                "only, which can leave EngineCore children running",
                file=sys.stderr,
            )

    def stop(self) -> None:
        """Stop the whole server tree, waiting for it to actually exit.

        SIGTERM to the process group, not to the parent alone: vLLM's EngineCore
        and worker processes are children, and signalling only the parent leaves
        them alive holding the GPU and the port -- which then presents as the
        next arm's server never becoming healthy.

        SIGTERM before SIGKILL so the EC region's cleanup runs and unlinks its
        /dev/shm file; that cleanup is best-effort and does not survive SIGKILL,
        which is how stale multi-GiB mmap files accumulate. The pattern fallback
        is scoped to this port rather than every vLLM server on the host, which
        would kill unrelated work on a shared dev pod.
        """
        if not self.managed:
            return
        p1, p2 = self.kill_match
        pattern = f"[{p1[0]}]{p1[1:]}.*{p2}"
        script = f"""
        pid=$(cat {self.pid_file} 2>/dev/null || true)
        pgid=$(cat {self.pgid_file} 2>/dev/null || true)
        own=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
        # Never signal our own group: that would kill this shell and its parent.
        if [ -n "$pgid" ] && [ "$pgid" = "$own" ]; then pgid=""; fi
        if [ -n "$pgid" ]; then
            kill -TERM -"$pgid" 2>/dev/null || true
        else
            if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
            pkill -f "{pattern}" 2>/dev/null || true
        fi
        # Liveness must ignore zombies: a reaped-late <defunct> child still
        # matches pgrep, which would make every teardown look hung and escalate
        # to SIGKILL even though the tree exited cleanly. Lines in this shell's
        # own group are skipped so the ps/awk pipeline cannot match itself.
        for _ in $(seq {_STOP_TIMEOUT_S}); do
            alive=$(ps -eo pgid=,stat=,args= | awk \
                -v g="$pgid" -v own="$own" \
                -v p1="{p1}" -v p2="{p2}" '
                $1 == own {{ next }}
                $2 ~ /^Z/ {{ next }}
                ($1 == g && g != "") {{ n++; next }}
                (index($0, p1) > 0 && index($0, p2) > 0) {{ n++ }}
                END {{ print n + 0 }}')
            if [ "$alive" = "0" ]; then
                rm -f {self.pid_file} {self.pgid_file}
                echo stopped
                exit 0
            fi
            sleep 1
        done
        echo escalated
        if [ -n "$pgid" ]; then kill -KILL -"$pgid" 2>/dev/null || true; fi
        if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
        pkill -9 -f "{pattern}" 2>/dev/null || true
        rm -f {self.pid_file} {self.pgid_file}
        """
        result = self.target.sh(script, check=False, timeout=_STOP_TIMEOUT_S + 60)
        if "escalated" in result.stdout:
            detail = (
                f"; the EC region's /dev/shm file for {self.engine_id} may have "
                "leaked"
                if self.arm == "connector"
                else ""
            )
            print(
                f"[bench] WARNING: {self.name} needed SIGKILL after "
                f"{_STOP_TIMEOUT_S}s{detail}",
                file=sys.stderr,
            )

    def wait_healthy(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout_s
        probe = (
            f"curl -s -o /dev/null -w '%{{http_code}}' {self.base_url}/health || true"
        )
        while time.monotonic() < deadline:
            if self.target.sh(probe, check=False).stdout.strip() == "200":
                return
            time.sleep(_HEALTH_POLL_S)
        raise RuntimeError(
            f"{self.name} not healthy within {self.args.startup_timeout_s}s "
            f"(see {self.log_path})"
        )

    def verify_arm(self) -> None:
        """Confirm the live server is this arm, from its own log.

        Guards the case that makes an A/B silently meaningless: a survivor from
        the previous arm answers /health, and its numbers get recorded under
        this arm's name.
        """
        if not self.managed:
            print("[bench] skipping arm verification for an unmanaged server")
            return
        log = self.target.read_text(self.log_path)
        if not any(marker in log for marker in _STARTUP_MARKERS):
            raise ServerMismatchError(
                f"{self.log_path} contains none of {_STARTUP_MARKERS}: the "
                "server answering /health is not the one just launched"
            )
        if self.arm not in ARMS:
            print(f"[bench] verified {self.name} started from {self.log_path}")
            return
        has_region = _EC_REGION_MARKER in log and self.engine_id in log
        if self.arm == "connector" and not has_region:
            raise ServerMismatchError(
                f"connector arm: no '{_EC_REGION_MARKER}' for {self.engine_id}. "
                "Either the connector is not active, or the region was reused "
                "from a previous run (which would hand this arm unearned hits)."
            )
        if self.arm == "recompute" and has_region:
            raise ServerMismatchError(
                "recompute arm: an EC region was created, so a connector is "
                "active in the arm that is supposed to have none"
            )
        print(f"[bench] verified {self.arm} arm from {self.log_path}")

    def wait_for_log(self, marker: str, timeout_s: int = 120) -> None:
        """Wait for a line in this process's log.

        For the proxy, which serves no /health endpoint of its own.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if marker in self.target.read_text(self.log_path):
                return
            time.sleep(2.0)
        raise RuntimeError(
            f"{self.name}: {marker!r} never appeared in {self.log_path}"
        )

    def reset_caches(self) -> None:
        for endpoint in ("reset_prefix_cache", "reset_mm_cache"):
            self.target.sh(
                f"curl -s -X POST {self.base_url}/{endpoint} >/dev/null || true",
                check=False,
            )

    def bench_serve_script(
        self, num_prompts: int, rate: str, out: str, concurrency: int = 0
    ) -> str:
        cmd = [
            f"{self.args.python} -m vllm.entrypoints.cli.main bench serve",
            "--backend openai-chat",
            f"--base-url {self.base_url}",
            "--endpoint /v1/chat/completions",
            f"--model {self.args.model}",
            "--dataset-name custom_image",
            f"--dataset-path {self.args.workload_dir}/workload.jsonl",
            # The emitted order IS the workload; shuffling would destroy the
            # reuse distances the manifest predicts.
            "--disable-shuffle",
            "--custom-ensure-client-side-data",
            f"--custom-output-len {self.args.output_len}",
            f"--num-prompts {num_prompts}",
            f"--request-rate {rate}",
            "--percentile-metrics ttft,tpot,itl,e2el",
            "--metric-percentiles 50,95,99",
            f"--seed {self.args.seed}",
            "--save-result",
            f"--result-filename {out}",
        ]
        if concurrency:
            cmd.append(f"--max-concurrency {concurrency}")
        return f"cd /tmp && HF_HOME=/vllm-workspace {' '.join(cmd)}"


def run_arm(
    target: Target, args: argparse.Namespace, arm: str, num_prompts: int
) -> list[dict[str, Any]]:
    """Measure one arm across every requested request rate."""
    log_path = args.server_log or f"{LOG_DIR}/bench_{arm}.log"
    server = BenchServer(
        target=target,
        args=args,
        arm=arm,
        log_path=log_path,
        managed=not args.base_url,
    )
    server.start(instrument=args.frag)
    try:
        server.wait_healthy()
        server.verify_arm()

        print(f"[bench] {arm}: warmup")
        warm = max(4, min(16, num_prompts // 10))
        target.sh(
            server.bench_serve_script(warm, "inf", "/tmp/warmup.json"),
            timeout=args.bench_timeout_s,
            check=False,
        )

        results: list[dict[str, Any]] = []
        for rate, conc in args.load_points:
            server.reset_caches()
            start = target.file_size(log_path)
            out_path = f"/tmp/bench_{arm}_{rate}_c{conc}.json"
            print(
                f"[bench] {arm}: rate={rate} concurrency={conc or 'unbounded'}, "
                f"{num_prompts} prompts"
            )
            target.sh(
                server.bench_serve_script(num_prompts, rate, out_path, conc),
                timeout=args.bench_timeout_s,
            )
            # Completion reports land after the last response.
            time.sleep(args.settle_s)
            log_text = target.read_bytes(log_path, start, target.file_size(log_path))
            raw = target.read_text(out_path)
            entry: dict[str, Any] = {
                "arm": arm,
                "request_rate": rate,
                "concurrency": conc,
                "client": json.loads(raw) if raw.strip() else {},
                "server": summarize(log_text),
            }
            if args.frag:
                entry["windows"] = window_stats(log_text, args.frag_window_s)
                entry["decay"] = decay_report(entry["windows"], "load")
                entry["descriptors"] = parse_frag(target.read_text(FRAG_FILE))
            results.append(entry)
        return results
    finally:
        server.stop()


def start_queue_sampler(target: Target, instances: dict[str, int]) -> None:
    """Begin sampling each instance's queue depth in the background."""
    pairs = " ".join(f"{name}={port}" for name, port in instances.items())
    target.sh(f"rm -f {QUEUE_CSV}", check=False)
    target.sh_detached(
        f"setsid bash {QUEUE_SAMPLER} {QUEUE_SAMPLE_INTERVAL_S} {QUEUE_CSV} "
        f"{pairs} < /dev/null > /dev/null 2>&1 & echo $! > {QUEUE_CSV}.pid; disown"
    )


def stop_queue_sampler(target: Target) -> None:
    target.sh(
        f'pid=$(cat {QUEUE_CSV}.pid 2>/dev/null || true); '
        f'if [ -n "$pid" ]; then kill -TERM -"$pid" 2>/dev/null || '
        f'kill "$pid" 2>/dev/null || true; fi; rm -f {QUEUE_CSV}.pid',
        check=False,
    )


def queue_stats(csv_text: str, t_start: float, t_end: float) -> dict[str, Any]:
    """Peak and mean queue depth per instance over one load point.

    `waiting` is the number of requests admitted but not yet running: if it
    stays near zero the measurement reflects work, and if it tracks the offered
    concurrency the measurement reflects the queue instead.
    """
    series: dict[tuple[str, str], list[float]] = {}
    for line in csv_text.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        # Prometheus label blocks contain commas, so the metric field cannot be
        # assumed comma-free: take the ends and treat the middle as the name.
        stamp, name, value = parts[0], parts[1], parts[-1]
        metric = ",".join(parts[2:-1])
        try:
            when, amount = float(stamp), float(value)
        except ValueError:
            continue
        if not t_start <= when <= t_end:
            continue
        key = (name, metric.split("{")[0].replace("vllm:num_requests_", ""))
        series.setdefault(key, []).append(amount)
    out: dict[str, Any] = {"samples": sum(len(v) for v in series.values())}
    for (name, metric), values in sorted(series.items()):
        out[f"{name}_{metric}_max"] = round(max(values), 1)
        out[f"{name}_{metric}_mean"] = round(sum(values) / len(values), 2)
    return out


def image_refs_in_prefix(target: Target, workload_dir: str, num_prompts: int) -> int:
    """Image references in the first `num_prompts` lines of the workload.

    The manifest counts the whole workload, so comparing a short run's rewrite
    count against it understates coverage purely arithmetically -- 120 of 400
    requests can never cover 370 references. `--disable-shuffle` means the
    client replays the file in order, so the prefix is exactly what was sent.
    """
    text = target.read_text(f"{workload_dir}/workload.jsonl")
    refs = 0
    for index, line in enumerate(text.splitlines()):
        if index >= num_prompts:
            break
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        refs += sum(
            1
            for part in record.get("content", [])
            if part.get("type") in ("image", "image_url", "image_embeds")
        )
    return refs


def _epd_instances(
    target: Target, args: argparse.Namespace, config: str
) -> tuple[list[BenchServer], BenchServer, BenchServer]:
    """Encoder, decode and proxy for one EPD configuration.

    The encoder is an EC producer, the decode instance an EC consumer, and the
    proxy is what turns one client request into an encode call plus a decode
    call. Only two things vary between configurations: whether the proxy rewrites
    the image into a grid reference, and where the encoder's image transform
    runs.
    """
    rewrite, device, connector = EPD_CONFIGS[config]

    def ec_for(role: str) -> dict[str, Any] | None:
        """Connector config for one role, or None for the no-connector arm."""
        if not connector:
            return None
        if connector == "ECCPUConnector":
            # Two separate processes cannot share one instance's mmap region, so
            # the peer-to-peer transport is what carries an entry across; the
            # producer announces its side-channel address through
            # ec_transfer_params and the consumer dials it.
            return {
                "ec_connector": connector,
                "ec_role": role,
                "ec_connector_extra_config": {
                    "ec_enable_nixl": "True",
                    "ec_cpu_bytes": args.ec_cpu_bytes,
                },
            }
        return {
            "ec_connector": connector,
            "ec_role": role,
            "ec_connector_extra_config": {
                "shared_storage_path": args.shared_storage_path
            },
        }

    # The device list's length is the encoder count; a device repeated in it
    # means those encoders share one GPU and must split its memory.
    devices = [d.strip() for d in args.encoder_devices.split(",") if d.strip()]
    if not devices:
        raise SystemExit("[bench] --encoder-devices names no device")
    shared = {dev: devices.count(dev) for dev in set(devices)}

    # Encoder ports grow from --encoder-port, so they can silently swallow the
    # decode or proxy port as the encoder count rises.
    encoder_ports = {args.encoder_port + i for i in range(len(devices))}
    for label, port in (("decode", args.decode_port), ("proxy", args.proxy_port)):
        if port in encoder_ports:
            raise SystemExit(
                f"[bench] the {label} port {port} lies inside the encoder range "
                f"{min(encoder_ports)}-{max(encoder_ports)} for "
                f"{len(devices)} encoder(s); move --{label}-port or "
                "--encoder-port"
            )
    if args.decode_port == args.proxy_port:
        raise SystemExit("[bench] --decode-port and --proxy-port are the same")

    encoders: list[BenchServer] = []
    for index, dev in enumerate(devices):
        util = args.encoder_gpu_memory_utilization or round(
            args.gpu_memory_utilization / shared[dev], 3
        )
        nixl_env: tuple[str, ...] = ()
        if connector == "ECCPUConnector":
            # Each producer binds its own side channel, and each announces its
            # own address through ec_transfer_params, which is how a consumer
            # ends up holding one session per peer.
            nixl_env = (
                "VLLM_EC_SIDE_CHANNEL_HOST=127.0.0.1",
                f"VLLM_EC_SIDE_CHANNEL_PORT={args.side_channel_port + index}",
            )
        encoders.append(
            BenchServer(
                target=target, args=args, arm=config, name=f"encoder{index}",
                log_path=f"{LOG_DIR}/epd_encoder{index}.log",
                port=args.encoder_port + index, gpu=dev, gpu_util=util,
                ec_config=ec_for("ec_producer"),
                extra_env=nixl_env,
                # mm_tensor_ipc is load-bearing for the device choice: any
                # transport but torch_shm copies the result back to host, so
                # `auto` declines the accelerator and logs that it did.
                # mm_processor_cache_type is a different knob despite the
                # similar name: with "shm" the engine keeps no receiver cache,
                # the processed data is refilled in the worker instead, and the
                # connector reports no grid -- so the proxy would have nothing
                # to substitute.
                extra_args=(
                    f"--mm-processor-device {device}",
                    f"--mm-tensor-ipc {args.mm_tensor_ipc}",
                    f"--mm-processor-cache-type {args.mm_processor_cache_type}",
                )
                + (("--mm-encoder-only",) if args.mm_encoder_only else ())
                + (("--enforce-eager",) if args.encoder_enforce_eager else ()),
            )
        )
    decode = BenchServer(
        target=target, args=args, arm=config, name="decode",
        log_path=f"{LOG_DIR}/epd_decode.log",
        port=args.decode_port, gpu=args.decode_gpu,
        ec_config=ec_for("ec_consumer"),
        extra_env=nixl_env,
        # Without this the decode instance rejects any image_embeds part with
        # "You must set `--enable-mm-embeds`", which is what a rewritten request
        # is made of. Set in every configuration, not just the rewriting ones:
        # it also affects encoder-budget accounting (multimodal/encoder_budget.py),
        # and a flag that differs between arms is a confound rather than a switch.
        extra_args=("--enable-mm-embeds",),
    )
    encode_urls = ",".join(f"http://127.0.0.1:{e.port}" for e in encoders)
    proxy_cmd = " ".join(
        [
            f"{args.python} {args.vllm_repo}/{PROXY}",
            f"--host 127.0.0.1 --port {args.proxy_port}",
            f"--encode-servers-urls {encode_urls}",
            "--prefill-servers-urls disable",
            f"--decode-servers-urls http://127.0.0.1:{args.decode_port}",
        ]
        + ([] if rewrite else ["--no-rewrite"])
    )
    proxy = BenchServer(
        target=target, args=args, arm=config, name="proxy",
        log_path=f"{LOG_DIR}/epd_proxy.log",
        port=args.proxy_port, command=proxy_cmd,
        match=("disagg_epd_proxy.py", f"--port {args.proxy_port}"),
    )
    return encoders, decode, proxy


def _check_coverage(
    config: str,
    rewrote: int,
    image_refs: int,
    floor: float,
    when: str,
    *,
    require_coverage: bool = True,
) -> float:
    """Coverage of the rewrite change, raising when it is too low to measure.

    The proxy logs "Rewrote N" for any N >= 1 and falls back per item, so
    presence proves nothing about how much of the workload was covered.

    An item is rewritable when the encoder reported a grid for it, which needs
    the item's processed data to be present on the scheduler side. A processor
    cache hit does not prevent that: the engine refills it before scheduling
    (`engine/core.py:997`). It goes missing under
    `mm_processor_cache_type=shm`, where the engine keeps no receiver cache and
    the refill happens in the worker instead -- too late for the connector to
    see. Hence the floor is a knob: it catches that configuration empirically
    rather than trusting the reasoning above.
    """
    rewrite = EPD_CONFIGS[config][0]
    coverage = rewrote / image_refs if image_refs else 0.0
    if not rewrite:
        if rewrote:
            raise ServerMismatchError(
                f"{config} ({when}): --no-rewrite was passed, but the proxy "
                f"still rewrote {rewrote} item(s)"
            )
        return 0.0
    if not rewrote:
        raise ServerMismatchError(
            f"{config} ({when}): the proxy rewrote nothing, so the change under "
            "test never engaged"
        )
    # The warmup sends a fraction of the workload, so its rewrite count cannot be
    # measured against the whole workload's reference count -- only the rated
    # runs, which replay all of it, have the right denominator.
    if not require_coverage:
        return coverage
    if coverage < floor:
        raise ServerMismatchError(
            f"{config} ({when}): the proxy rewrote {rewrote} of {image_refs} "
            f"image references ({coverage:.0%}), below the {floor:.0%} floor. "
            "An item is only rewritable when the encoder reported a grid for it, "
            "which needs its processed data on the scheduler side; check the "
            "encoder is not running with mm_processor_cache_type=shm, where the "
            "refill happens in the worker and the connector never sees it. "
            "Lower --min-rewrite-coverage only after checking that."
        )
    return coverage


def verify_epd(
    config: str,
    encoders: list[BenchServer],
    proxy: BenchServer,
    image_refs: int,
) -> float:
    """Fail unless both switches for this configuration actually took effect.

    The GPU transform declines silently when its preconditions are unmet, and a
    proxy that rewrote nothing looks identical in the timings to one that was
    asked not to. Measuring either without checking would attribute a
    configuration's numbers to a change that never happened.

    Coverage, not just presence: the proxy falls back per item, and logs
    "Rewrote N" for any N >= 1, so a run that rewrote one item of hundreds would
    otherwise pass as the rewrite configuration while behaving like baseline.
    An item is only rewritable when the encoder actually ran the transform for
    it -- a repeat served from the encoder's processor cache reports no metadata
    -- so on a reuse-heavy workload coverage is legitimately far below 1.0, and
    the floor is a knob rather than a constant.

    Returns the measured coverage.
    """
    rewrite, device, _connector = EPD_CONFIGS[config]
    encoder = encoders[0]
    encoder_log = encoder.target.read_text(encoder.log_path)
    on_gpu = _GPU_PROCESSOR_MARKER in encoder_log
    if (device == "cuda") != on_gpu:
        raise ServerMismatchError(
            f"{config}: asked for the image transform on {device}, but the "
            f"encoder log {'shows' if on_gpu else 'does not show'} "
            f"{_GPU_PROCESSOR_MARKER!r}. With mm_tensor_ipc="
            f"{encoder.args.mm_tensor_ipc} the accelerator may have been "
            "declined; the encoder log states the reason."
        )
    rewrote = rewritten_items(proxy.target.read_text(proxy.log_path))
    _check_coverage(
        config, rewrote, image_refs, encoder.args.min_rewrite_coverage, "warmup",
        require_coverage=False,
    )
    print(
        f"[bench] verified {config}: transform on {device}, proxy rewrote "
        f"{rewrote} item(s) during warmup"
    )
    return 0.0


def run_epd_config(
    target: Target, args: argparse.Namespace, config: str, num_prompts: int
) -> list[dict[str, Any]]:
    """Measure one EPD configuration across every requested request rate."""
    encoders, decode, proxy = _epd_instances(target, args, config)
    try:
        for server in [*encoders, decode]:
            server.start(instrument=False)
            server.wait_healthy()
            server.verify_arm()
        proxy.start(instrument=False)
        proxy.wait_for_log("Uvicorn running")

        print(f"[bench] {config}: warmup")
        warm = max(4, min(16, num_prompts // 10))
        target.sh(
            proxy.bench_serve_script(warm, "inf", "/tmp/warmup.json"),
            timeout=args.bench_timeout_s, check=False,
        )
        verify_epd(config, encoders, proxy, args.image_refs)

        results: list[dict[str, Any]] = []
        start_queue_sampler(
            target,
            {e.name: e.port for e in encoders} | {"decode": args.decode_port},
        )
        for rate, conc in args.load_points:
            decode.reset_caches()
            trio = (*encoders, decode, proxy)
            marks = {s.name: target.file_size(s.log_path) for s in trio}
            out_path = f"/tmp/bench_epd_{config}_{rate}_c{conc}.json"
            print(
                f"[bench] {config}: rate={rate} concurrency={conc or 'unbounded'}, "
                f"{num_prompts} prompts"
            )
            t_start = time.time()
            target.sh(
                proxy.bench_serve_script(num_prompts, rate, out_path, conc),
                timeout=args.bench_timeout_s,
            )
            t_end = time.time()
            time.sleep(args.settle_s)
            slices = {
                s.name: target.read_bytes(
                    s.log_path, marks[s.name], target.file_size(s.log_path)
                )
                for s in trio
            }
            client_stats = json.loads(raw) if raw.strip() else {}
            done = client_stats.get("completed", 0)
            if done < num_prompts:
                raise ServerMismatchError(
                    f"{config} (c={conc}): only {done} of {num_prompts} requests "
                    "completed, so the latencies describe the few that survived. "
                    f"Check {decode.log_path} and {proxy.log_path} for errors"
                )
            rewrote_this_rate = rewritten_items(slices["proxy"])
            rate_coverage = _check_coverage(
                config, rewrote_this_rate, args.image_refs,
                args.min_rewrite_coverage, f"rate={rate}",
            )
            raw = target.read_text(out_path)
            # A connector arm that transferred nothing looks, in the timings,
            # exactly like the no-connector arm. Assert the direction both ways.
            decode_stats = summarize(slices["decode"])
            loads = decode_stats["ec_load_entries"] or decode_stats["ec_example_loads"]
            expects_loads = config in EPD_EXPECT_LOADS
            if expects_loads and not loads:
                raise ServerMismatchError(
                    f"{config} (c={conc}): the consumer loaded no encodings, so "
                    "nothing was transferred and this arm measures local "
                    "recompute rather than the connector"
                )
            if not expects_loads and loads:
                raise ServerMismatchError(
                    f"{config} (c={conc}): the consumer loaded {loads} encodings, "
                    "but this arm is not supposed to transfer anything"
                )
            # Fan-out must actually reach every encoder: if the proxy's
            # round-robin left one idle we are measuring fewer encoders than we
            # think, with the rest burning memory for nothing.
            per_encoder = {
                e.name: summarize(slices[e.name])["encoder_inputs_computed"]
                for e in encoders
            }
            idle = [name for name, done in per_encoder.items() if not done]
            if len(encoders) > 1 and idle:
                raise ServerMismatchError(
                    f"{config} (c={conc}): {idle} computed no encoder inputs, so "
                    f"the fan-out reached only {len(encoders) - len(idle)} of "
                    f"{len(encoders)} encoders"
                )
            results.append(
                {
                    "arm": config,
                    "encoders": len(encoders),
                    "per_encoder_inputs": per_encoder,
                    "request_rate": rate,
                    "concurrency": conc,
                    "queue": queue_stats(
                        target.read_text(QUEUE_CSV), t_start, t_end
                    ),
                    "client": client_stats,
                    "server": decode_stats,
                    "encoder": summarize(slices[encoders[0].name]),
                    # Per-stage attribution the proxy already logs, which is what
                    # says whether a win came from the decode side or elsewhere.
                    "stages": stage_summary(slices["proxy"]),
                    "rewritten": rewrote_this_rate,
                    "rewrite_coverage": round(rate_coverage, 4),
                }
            )
        return results
    finally:
        stop_queue_sampler(target)
        for server in (proxy, decode, *reversed(encoders)):
            server.stop()


def parse_frag(text: str) -> dict[str, Any]:
    """Aggregate the descriptor-count JSONL the instrumentation patch writes."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        return {"rows": 0, "note": "no descriptor data (patch not active?)"}
    by_caller: dict[str, dict[str, Any]] = {}
    for row in rows:
        agg = by_caller.setdefault(
            row["caller"], {"blocks": 0, "descriptors": 0, "calls": 0, "series": []}
        )
        agg["blocks"] += row["blocks"]
        agg["descriptors"] += row["descriptors"]
        agg["calls"] += row["calls"]
        agg["series"].append((row["t_s"], row["blocks_per_descriptor"]))
    for agg in by_caller.values():
        agg["series"].sort()
        agg["blocks_per_descriptor_mean"] = (
            round(agg["blocks"] / agg["descriptors"], 2) if agg["descriptors"] else 0.0
        )
        first, last = agg["series"][0][1], agg["series"][-1][1]
        agg["blocks_per_descriptor_first"] = first
        agg["blocks_per_descriptor_last"] = last
        agg["verdict"] = (
            "runs held up" if last >= 0.9 * first else "runs shortened (fragmenting)"
        )
    return {"rows": len(rows), "by_caller": by_caller}


def _metric(client: dict[str, Any], key: str) -> Any:
    value = client.get(key)
    return round(value, 2) if isinstance(value, (int, float)) else "-"


def print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'arm':<10} {'rate':>5} {'ttft_p50':>9} {'ttft_p99':>9} {'itl_p50':>8} "
        f"{'out_tok/s':>10} {'ec_loads':>9} {'enc_inputs':>11} {'load_GB/s':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for entry in results:
        client, server = entry["client"], entry["server"]
        print(
            f"{entry['arm']:<10} {str(entry['request_rate']):>5} "
            f"{_metric(client, 'median_ttft_ms'):>9} "
            f"{_metric(client, 'p99_ttft_ms'):>9} "
            f"{_metric(client, 'median_itl_ms'):>8} "
            f"{_metric(client, 'output_throughput'):>10} "
            f"{server['ec_load_entries']:>9} "
            f"{server['encoder_inputs_computed']:>11} "
            f"{server['ec_load_gbps']:>10}"
        )


def _max_encoder_queue(queue: dict[str, Any]) -> Any:
    """Worst waiting depth across however many encoders ran."""
    depths = [
        v
        for k, v in queue.items()
        if k.startswith("encoder") and "waiting_max" in k
    ]
    return max(depths) if depths else "-"


def print_epd_table(results: list[dict[str, Any]]) -> None:
    """One row per configuration, with ratios against `baseline` where present.

    `encode` and `decode_ttfb` are the proxy's own stage timings, which is what
    separates "the decode instance stopped redoing the transform" from a change
    somewhere else.
    """
    header = (
        f"{'config':<9} {'conc':>5} {'ttft_p50':>9} {'ttft_p99':>9} "
        f"{'out_tok/s':>10} {'x_base':>7} {'encode':>8} {'dec_ttfb':>9} "
        f"{'encQmax':>8} {'decQmax':>8} {'rewrote':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    base = {
        (r["request_rate"], r.get("concurrency")): r["client"].get("output_throughput")
        for r in results
        if r["arm"] == "baseline"
    }
    for r in results:
        stages, client = r["stages"], r["client"]
        ref = base.get((r["request_rate"], r.get("concurrency")))
        got = client.get("output_throughput")
        ratio = (
            f"{got / ref:.2f}" if isinstance(ref, float) and isinstance(got, float)
            and ref else "-"
        )
        q = r.get("queue", {})
        print(
            f"{r['arm']:<9} {str(r.get('concurrency', 0)):>5} "
            f"{_metric(client, 'median_ttft_ms'):>9} "
            f"{_metric(client, 'p99_ttft_ms'):>9} "
            f"{_metric(client, 'output_throughput'):>10} {ratio:>7} "
            f"{stages.get('encode_ms_median', '-'):>8} "
            f"{stages.get('decode_ttfb_ms_median', '-'):>9} "
            f"{_max_encoder_queue(q):>8} "
            f"{q.get('decode_waiting_max', '-'):>8} "
            f"{r.get('rewritten', 0):>8}"
        )


def check_gates(results: list[dict[str, Any]]) -> bool:
    """Phase 0's gates, re-applied to the real workload.

    Without these a timing delta is unattributable: if the connector never
    loaded anything, the arms differ by noise and configuration rather than by
    the mechanism under test.
    """
    by_arm = {arm: [r for r in results if r["arm"] == arm] for arm in ARMS}
    if not all(by_arm.values()):
        print("\n[bench] single arm only; both arms are needed for the gates")
        return False
    ok = True
    for conn, base in zip(by_arm["connector"], by_arm["recompute"]):
        rate = conn["request_rate"]
        loads = conn["server"]["ec_load_entries"]
        conn_enc = conn["server"]["encoder_inputs_computed"]
        base_enc = base["server"]["encoder_inputs_computed"]
        gate_load = loads > 0
        gate_skip = conn_enc < base_enc
        ok = ok and gate_load and gate_skip
        print(
            f"[bench] rate={rate}: ec_load fired "
            f"{'PASS' if gate_load else 'FAIL'} ({loads} entries); "
            f"encoder compute dropped {'PASS' if gate_skip else 'FAIL'} "
            f"({conn_enc} vs {base_enc})"
        )
        if base_enc:
            avoided = 1 - conn_enc / base_enc
            print(f"[bench] rate={rate}: encoder inputs avoided {avoided * 100:.1f}%")
    return ok


def print_frag(results: list[dict[str, Any]]) -> None:
    for entry in results:
        print(f"\n[bench] bandwidth decay: {entry['decay']}")
        for caller, agg in entry["descriptors"].get("by_caller", {}).items():
            print(
                f"[bench] {caller}: blocks/descriptor "
                f"{agg['blocks_per_descriptor_first']} -> "
                f"{agg['blocks_per_descriptor_last']} "
                f"(mean {agg['blocks_per_descriptor_mean']}) -- {agg['verdict']}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pod", default="vllm-omer-2", help="empty string runs locally")
    p.add_argument("--python", default="/vllm-workspace/venv-vllm/bin/python")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--workload-dir", default="/vllm-workspace/bench/wl")
    p.add_argument("--out-dir", type=Path, default=Path("bench_results"))
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--gpu", default="0")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--request-rates", default="inf")
    p.add_argument(
        "--max-concurrency",
        default="0",
        help="comma-separated in-flight limits to sweep, e.g. 1,2,4,8. 0 means "
        "unbounded, which at --request-rate inf floods the system and makes "
        "every latency a queue measurement",
    )
    p.add_argument("--num-prompts", type=int, default=0, help="0 = whole workload")
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--ec-cpu-bytes", type=int, default=0, help="0 = from manifest")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--encoder-max-num-batched-tokens",
        type=int,
        default=65536,
        help="token budget for an encode-only instance; must exceed one image's "
        "token count several times over or it cannot batch image requests",
    )
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="for the monolithic reference, whether it gets one GPU or matches "
        "the two an EPD pair occupies",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--startup-timeout-s", type=int, default=900)
    p.add_argument("--bench-timeout-s", type=int, default=3600)
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument(
        "--base-url",
        default="",
        help="measure a server this process does not manage (an EPD proxy, "
        "say); requires --server-log and a single --arms value",
    )
    p.add_argument(
        "--server-log",
        default="",
        help="server log to read EC accounting from; required with --base-url",
    )
    p.add_argument(
        "--frag",
        action="store_true",
        help="connector arm only, region under the working set, descriptor "
        "counting on: measures whether entries stay contiguous as it churns",
    )
    p.add_argument(
        "--ec-enable-nixl",
        action="store_true",
        help="set ec_enable_nixl in the connector arm's extra config; "
        "irrelevant to local offload",
    )
    p.add_argument("--frag-window-s", type=float, default=30.0)
    epd = p.add_argument_group("EPD topology (PR #50390)")
    epd.add_argument("--topology", choices=("single", "epd"), default="single")
    epd.add_argument(
        "--epd-configs",
        default=",".join(EPD_CONFIGS),
        help="which of baseline,grid,gpu,both to measure",
    )
    epd.add_argument("--vllm-repo", default="/vllm-workspace/vllm")
    epd.add_argument(
        "--encoder-devices",
        default="0",
        help="comma-separated devices for the encoder instances; the list's "
        "length is the encoder count, and a device repeated in it means those "
        "encoders share one GPU and split its memory. Accepts plain indices or "
        "MIG UUIDs (MIG-...)",
    )
    epd.add_argument(
        "--encoder-gpu-memory-utilization",
        type=float,
        default=0.0,
        help="per-encoder memory share; 0 divides --gpu-memory-utilization by "
        "the number of encoders sharing that device",
    )
    epd.add_argument(
        "--mm-encoder-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip the language model on encoder instances (~16 GB -> ~1.4 GB), "
        "which is what makes several encoders fit one GPU",
    )
    epd.add_argument(
        "--encoder-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the EPD example states encoder instances are only compatible with "
        "eager mode",
    )
    epd.add_argument(
        "--encoder-port",
        type=int,
        default=8101,
        help="first encoder's port; instance i uses this + i",
    )
    epd.add_argument(
        "--decode-port",
        type=int,
        default=8200,
        help="kept clear of the encoder range, which grows from --encoder-port",
    )
    epd.add_argument("--proxy-port", type=int, default=8000)
    epd.add_argument("--encoder-gpu", default="0")
    epd.add_argument("--decode-gpu", default="1")
    epd.add_argument(
        "--ec-connector",
        default="ECExampleConnector",
        help="only used by the legacy baseline/grid/gpu/both configs",
    )
    epd.add_argument("--side-channel-port", type=int, default=5577)
    epd.add_argument("--shared-storage-path", default="/tmp/ec_bench_shared")
    epd.add_argument(
        "--mm-processor-cache-type",
        default="lru",
        choices=("lru", "shm"),
        help="pinned on the encoder because it decides whether the grid is "
        "reported at all; 'shm' moves the refill into the worker, where the "
        "connector cannot see it",
    )
    epd.add_argument(
        "--min-rewrite-coverage",
        type=float,
        default=0.5,
        help="fraction of image references the proxy must rewrite for a rewrite "
        "configuration to count as engaged; repeats served from the encoder's "
        "processor cache legitimately keep their pixels",
    )
    epd.add_argument(
        "--mm-tensor-ipc",
        default="torch_shm",
        help="torch_shm is required for the encoder's transform to run on the "
        "accelerator; any other transport copies the result back to host and "
        "the device choice is declined",
    )
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    args.request_rates = [r.strip() for r in args.request_rates.split(",") if r.strip()]
    concurrencies = [
        int(c) for c in str(args.max_concurrency).split(",") if c.strip()
    ] or [0]
    args.load_points = [(r, c) for r in args.request_rates for c in concurrencies]
    if args.topology == "epd":
        arms = [c.strip() for c in args.epd_configs.split(",") if c.strip()]
        unknown = set(arms) - set(EPD_CONFIGS)
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"[bench] unknown arm(s)/config(s): {sorted(unknown)}")

    if args.base_url:
        if not args.server_log:
            raise SystemExit("[bench] --base-url requires --server-log")
        if len(arms) != 1:
            raise SystemExit(
                "[bench] --base-url measures one existing server, so name "
                "exactly one arm with --arms"
            )
    if args.frag:
        arms = ["connector"]

    target = Target(args.pod or None)
    raw_manifest = target.read_text(f"{args.workload_dir}/manifest.json")
    manifest = json.loads(raw_manifest) if raw_manifest.strip() else {}
    if not manifest:
        raise SystemExit(
            f"[bench] no manifest.json in {args.workload_dir}; "
            "run gen_workload.py first"
        )
    # A manifest can outlive its images (the pool lives on disk, and /tmp does
    # not survive a pod recreate). Check the images are actually there rather
    # than failing deep inside a load generator with a confusing error.
    pool_entries = manifest.get("pool") or []
    if pool_entries:
        probes = {pool_entries[0]["path"], pool_entries[-1]["path"]}
        missing = [
            path
            for path in sorted(probes)
            # sh() runs a command; read_text() cats a path -- not interchangeable.
            if not target.sh(
                f"test -s {shlex.quote(path)} && echo ok", check=False
            ).stdout.strip()
        ]
        if missing:
            raise SystemExit(
                f"[bench] the manifest in {args.workload_dir} references images "
                f"that are not on disk ({missing}); rebuild the pool with "
                "gen_workload.py"
            )
    expected = manifest["expected"]
    num_prompts = args.num_prompts or manifest["sequence"]["requests"]
    args.image_refs = image_refs_in_prefix(target, args.workload_dir, num_prompts)
    if not args.ec_cpu_bytes:
        args.ec_cpu_bytes = (
            expected["fragmentation_arm_ec_cpu_bytes"]
            if args.frag
            else expected["suggested_ec_cpu_bytes"]
        )

    print(
        f"[bench] workload {num_prompts} requests, working set "
        f"{expected['working_set_bytes'] / 1024**3:.2f} GiB, ec_cpu_bytes "
        f"{args.ec_cpu_bytes / 1024**3:.2f} GiB, max hit rate "
        f"{expected['max_hit_rate'] * 100:.1f}%"
    )

    if args.dry_run and args.topology == "epd":
        rate0, conc0 = args.load_points[0]
        for config in arms:
            encoders, decode, proxy = _epd_instances(target, args, config)
            for server in [*encoders, decode, proxy]:
                print(f"\n=== {config}: {server.name} ===")
                print(server.launch_script(instrument=False))
            print(f"\n=== {config}: load ===")
            print(
                proxy.bench_serve_script(
                    num_prompts, rate0, "/tmp/bench_epd.json", conc0
                )
            )
        return 0

    if args.dry_run:
        for arm in arms:
            server = BenchServer(
                target=target,
                args=args,
                arm=arm,
                log_path=args.server_log or f"{LOG_DIR}/bench_{arm}.log",
                managed=not args.base_url,
            )
            print(f"\n=== {arm} launch ===")
            print(server.launch_script(instrument=args.frag))
            print(f"\n=== {arm} load ===")
            rate0, conc0 = args.load_points[0]
            print(
                server.bench_serve_script(
                    num_prompts, rate0, f"/tmp/bench_{arm}.json", conc0
                )
            )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.topology == "epd":
        out_name = "epd.json"
    else:
        out_name = "frag.json" if args.frag else "ab.json"
    out_path = args.out_dir / out_name

    def persist(results: list[dict[str, Any]]) -> None:
        """Write what has been measured so far.

        Called after every arm so a failure in the second arm does not discard
        the first arm's data, which is expensive to reproduce.
        """
        out_path.write_text(
            json.dumps(
                {
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "manifest_expected": expected,
                    "results": results,
                },
                indent=2,
            )
        )

    results: list[dict[str, Any]] = []
    try:
        for arm in arms:
            if args.topology == "epd":
                # A config must not inherit encodings the previous one saved:
                # those would be free hits it never paid for.
                target.sh(
                    f"rm -rf {args.shared_storage_path} && "
                    f"mkdir -p {args.shared_storage_path}",
                    check=False,
                )
                results.extend(run_epd_config(target, args, arm, num_prompts))
            else:
                results.extend(run_arm(target, args, arm, num_prompts))
            persist(results)
    except Exception:
        persist(results)
        print(f"[bench] partial results saved to {out_path}", file=sys.stderr)
        raise

    if args.topology == "epd":
        print_epd_table(results)
        gates_ok = True
        print(f"\n[bench] wrote {out_path}")
        return 0
    print_table(results)
    if args.frag:
        print_frag(results)
        gates_ok = True
    else:
        gates_ok = check_gates(results)
    print(f"\n[bench] wrote {out_path}")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
