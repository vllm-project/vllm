#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import grp
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("amd-test")

HF_MOUNT = "/root/.cache/huggingface"
JUNIT_CONTAINER_PATH = "/tmp/vllm-results.xml"
ARTIFACT_MOUNT = "/tmp/vllm-rocm-install"
DEFAULT_ARTIFACT_GLOB = "artifacts/vllm-rocm-install/vllm-rocm-install.tar.gz"
WORKSPACE = "/vllm-workspace"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_command(
    cmd: list[str], timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _short_output(output: str | None, limit: int = 500) -> str:
    text = (output or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _load_json_stdout(
    result: subprocess.CompletedProcess[str], description: str
) -> Any:
    stdout = result.stdout.strip()
    if not stdout:
        stderr = _short_output(result.stderr)
        message = f"{description} returned empty stdout"
        if stderr:
            message += f"; stderr: {stderr}"
        raise RuntimeError(message)
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        message = f"{description} returned non-JSON stdout: {_short_output(stdout)!r}"
        stderr = _short_output(result.stderr)
        if stderr:
            message += f"; stderr: {stderr!r}"
        raise RuntimeError(message) from exc


def gpu_snapshot() -> list[dict[str, Any]]:
    result = run_command(["amd-smi", "metric", "--mem-usage", "--json"], timeout=30)
    if result.returncode != 0:
        output = _short_output(result.stderr or result.stdout)
        raise RuntimeError(f"amd-smi metric failed: {output}")

    payload = _load_json_stdout(result, "amd-smi metric --mem-usage --json")
    if isinstance(payload, dict):
        entries = payload.get("gpu_data", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise RuntimeError(
            f"Unexpected amd-smi JSON payload type: {type(payload).__name__}"
        )

    snapshot = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Unexpected amd-smi GPU entry type: {type(entry).__name__}"
            )
        mem = entry.get("mem_usage", {})
        snapshot.append(
            {
                "gpu": entry.get("gpu"),
                "used_mib": mem.get("used_vram", {}).get("value"),
                "total_mib": mem.get("total_vram", {}).get("value"),
            }
        )
    return snapshot


def _count_gpu_entries(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return 0

    for key in ("gpu_data", "gpus", "GPUs", "devices"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)

    return sum(1 for value in payload.values() if isinstance(value, dict))


def amd_smi_list_gpu_count() -> int:
    result = run_command(["amd-smi", "list", "--json"], timeout=30)
    if result.returncode == 0:
        try:
            count = _count_gpu_entries(_load_json_stdout(result, "amd-smi list --json"))
            if count:
                return count
        except RuntimeError as exc:
            log.warning("amd-smi list --json was not usable: %s", exc)

    result = run_command(["amd-smi", "list"], timeout=30)
    if result.returncode != 0:
        output = _short_output(result.stderr or result.stdout)
        raise RuntimeError(f"amd-smi list failed: {output}")

    return sum(
        1
        for line in result.stdout.splitlines()
        if line.strip().lower().startswith("gpu")
    )


@dataclass(frozen=True)
class RunRequest:
    commit: str
    image: str
    commands: str
    test_name: str | None
    results_dir: Path
    hf_cache: Path
    timeout_s: int
    watchdog_interval_s: int
    render_devices: tuple[str, ...]
    artifact_mode: bool
    artifact_glob: str
    execution_mode: str
    num_nodes: int
    num_gpus_per_node: int
    node_commands: tuple[str, ...]

    @classmethod
    def from_env(cls, argv: list[str]) -> "RunRequest":
        commit = os.environ.get("BUILDKITE_COMMIT")
        if not commit:
            raise RuntimeError("BUILDKITE_COMMIT is required")

        artifact_mode = env_flag("VLLM_CI_USE_ARTIFACTS")
        execution_mode = os.environ.get("VLLM_CI_EXECUTION_MODE", "single-node")
        configured_image = os.environ.get("DOCKER_IMAGE_NAME")
        if artifact_mode:
            image = os.environ.get("VLLM_CI_BASE_IMAGE", "rocm/vllm-dev:ci_base")
            if configured_image and configured_image != image:
                log.warning(
                    "Ignoring DOCKER_IMAGE_NAME=%s because "
                    "VLLM_CI_USE_ARTIFACTS=1; using VLLM_CI_BASE_IMAGE=%s",
                    configured_image,
                    image,
                )
        else:
            image = configured_image or f"rocm/vllm-ci:{commit}"

        node_commands = cls._resolve_node_commands()
        hf_cache = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        )

        return cls(
            commit=commit,
            image=image,
            commands=cls._resolve_commands(argv),
            test_name=os.environ.get("VLLM_TEST_GROUP_NAME") or None,
            results_dir=Path(tempfile.mkdtemp(prefix="vllm-ci-results-")),
            hf_cache=hf_cache,
            timeout_s=int(os.environ.get("CONTAINER_TIMEOUT_S", "10200")),
            watchdog_interval_s=int(os.environ.get("WATCHDOG_INTERVAL_S", "15")),
            render_devices=tuple(
                token
                for token in os.environ.get(
                    "BUILDKITE_AGENT_META_DATA_RENDER_DEVICES", ""
                ).split()
                if token.startswith("/dev/")
            ),
            artifact_mode=artifact_mode,
            artifact_glob=os.environ.get(
                "VLLM_CI_ARTIFACT_GLOB", DEFAULT_ARTIFACT_GLOB
            ),
            execution_mode=execution_mode,
            num_nodes=int(os.environ.get("NUM_NODES", "1")),
            num_gpus_per_node=int(os.environ.get("VLLM_NUM_GPUS_PER_NODE", "1")),
            node_commands=node_commands,
        )

    @staticmethod
    def _resolve_commands(argv: list[str]) -> str:
        if os.environ.get("VLLM_CI_EXECUTION_MODE") == "multi-node":
            return ""

        commands = os.environ.get("VLLM_TEST_COMMANDS", "")
        if commands:
            log.info("Commands sourced from VLLM_TEST_COMMANDS")
            return commands

        if len(argv) > 1:
            commands = " ".join(argv[1:])
            log.warning(
                "Using positional args is legacy and may lose quoting. "
                "Prefer VLLM_TEST_COMMANDS."
            )
            return commands

        raise RuntimeError("VLLM_TEST_COMMANDS is required")

    @staticmethod
    def _resolve_node_commands() -> tuple[str, ...]:
        count = int(os.environ.get("VLLM_NODE_COMMAND_COUNT", "0"))
        commands = []
        for idx in range(count):
            command = os.environ.get(f"VLLM_NODE_COMMAND_{idx}")
            if command:
                commands.append(command)
        return tuple(commands)

    def bash_command(self) -> str:
        return (
            f'export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} --junitxml='
            f'{JUNIT_CONTAINER_PATH}"'
            f" && {self.commands}"
        )

    def setup_command(self) -> str:
        return f"""
set -euo pipefail
if [ ! -d "{ARTIFACT_MOUNT}" ]; then
  echo "ROCm artifact directory not mounted at {ARTIFACT_MOUNT}" >&2
  exit 1
fi
wheel="$(find "{ARTIFACT_MOUNT}" -maxdepth 1 -name '*.whl' | head -1)"
if [ -z "${{wheel}}" ]; then
  echo "No vLLM wheel found in {ARTIFACT_MOUNT}" >&2
  exit 1
fi
uv pip install --system --no-deps "${{wheel}}"
rm -rf "{WORKSPACE}"
mkdir -p "{WORKSPACE}"
for path in tests examples benchmarks requirements docker .buildkite pyproject.toml; do
  if [ -e "{ARTIFACT_MOUNT}/${{path}}" ]; then
    cp -a "{ARTIFACT_MOUNT}/${{path}}" "{WORKSPACE}/"
  fi
done
python3 - <<'PY'
import pathlib
import shutil
import sysconfig

src = pathlib.Path("{ARTIFACT_MOUNT}") / "vllm_v1"
if src.exists():
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    dst = site / "vllm" / "v1"
    shutil.copytree(src, dst, dirs_exist_ok=True)
PY
"""


@dataclass
class RunResult:
    mode: str
    exit_code: int
    container_name: str | None = None
    junit_path: Path | None = None
    log_path: Path | None = None


class ArtifactPackage:
    def __init__(self, request: RunRequest) -> None:
        self.request = request
        self.download_dir = request.results_dir / "downloaded-artifacts"
        self.install_dir = request.results_dir / "vllm-rocm-install"

    def prepare(self) -> Path | None:
        if not self.request.artifact_mode:
            return None

        archive = self._find_local_archive()
        if archive is None:
            archive = self._download_archive()
        self._extract_archive(archive)
        self._validate()
        return self.install_dir

    def _find_local_archive(self) -> Path | None:
        configured = Path(self.request.artifact_glob)
        if configured.exists() and configured.is_file():
            return configured
        matches = sorted(Path.cwd().glob(self.request.artifact_glob))
        return matches[0] if matches else None

    def _download_archive(self) -> Path:
        if shutil.which("buildkite-agent") is None:
            raise RuntimeError(
                "VLLM_CI_USE_ARTIFACTS=1 requires buildkite-agent or a local "
                f"artifact at {self.request.artifact_glob}"
            )

        self.download_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading ROCm vLLM artifact: %s", self.request.artifact_glob)
        result = run_command(
            [
                "buildkite-agent",
                "artifact",
                "download",
                self.request.artifact_glob,
                str(self.download_dir),
            ],
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "artifact download failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        matches = sorted(self.download_dir.rglob("vllm-rocm-install.tar.gz"))
        if not matches:
            raise RuntimeError(
                f"artifact download did not produce {self.request.artifact_glob}"
            )
        return matches[0]

    def _extract_archive(self, archive: Path) -> None:
        if self.install_dir.exists():
            shutil.rmtree(self.install_dir)
        self.install_dir.mkdir(parents=True)
        log.info("Extracting ROCm vLLM artifact: %s", archive)
        with tarfile.open(archive, "r:gz") as tar:
            dest = self.install_dir.resolve()
            for member in tar.getmembers():
                member_path = (dest / member.name).resolve()
                if dest not in (member_path, *member_path.parents):
                    raise RuntimeError(
                        f"Refusing to extract unsafe artifact path: {member.name}"
                    )
            tar.extractall(self.install_dir)

    def _validate(self) -> None:
        wheels = list(self.install_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"No vLLM wheel found in {self.install_dir}")
        for required in ("tests", "pyproject.toml"):
            if not (self.install_dir / required).exists():
                raise RuntimeError(
                    f"ROCm artifact is missing required entry: {required}"
                )
        log.info("Prepared ROCm artifact install tree: %s", self.install_dir)


class ContainerRunner:
    _ENV_PASSTHROUGH = [
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "BUILDKITE_PARALLEL_JOB",
        "BUILDKITE_PARALLEL_JOB_COUNT",
    ]

    def __init__(self, request: RunRequest, artifact_dir: Path | None) -> None:
        self.request = request
        self.artifact_dir = artifact_dir
        self.container_name = f"rocm_{request.commit}_{os.urandom(4).hex()}"
        self._created = False
        self._log_thread: threading.Thread | None = None

    def run(self) -> RunResult:
        self.request.hf_cache.mkdir(parents=True, exist_ok=True)
        log.info("--- Single-node job")
        log.info("Container name: %s", self.container_name)

        start = run_command(self._build_cmd(), timeout=120)
        if start.returncode != 0:
            raise RuntimeError(f"docker run failed: {start.stderr.strip()}")
        self._created = True

        log_path = self.request.results_dir / "container.log"
        junit_path = self.request.results_dir / "results.xml"
        exit_code = 1

        try:
            self._log_thread = self._stream_logs(log_path)
            wait = subprocess.run(
                ["docker", "wait", self.container_name],
                capture_output=True,
                text=True,
                timeout=self.request.timeout_s,
            )
            if wait.returncode != 0:
                log.error("docker wait failed: %s", wait.stderr.strip())
            else:
                try:
                    exit_code = int(wait.stdout.strip())
                except ValueError:
                    log.error(
                        "docker wait returned non-integer: %r", wait.stdout.strip()
                    )
        except subprocess.TimeoutExpired:
            log.error("Container exceeded %ds timeout; killing", self.request.timeout_s)
            run_command(["docker", "kill", self.container_name], timeout=30)
            exit_code = 124
        finally:
            if self._log_thread is not None:
                self._log_thread.join(timeout=10)
            self._copy_junit(junit_path)
            self._cleanup()

        return RunResult(
            mode="single-node",
            exit_code=exit_code,
            container_name=self.container_name,
            junit_path=junit_path if junit_path.exists() else None,
            log_path=log_path if log_path.exists() else None,
        )

    def _build_cmd(self) -> list[str]:
        try:
            render_gid = str(grp.getgrnam("render").gr_gid)
        except KeyError as err:
            raise RuntimeError("render group not found") from err

        cmd = ["docker", "run", "--detach", "--device", "/dev/kfd"]
        for device in self.request.render_devices:
            cmd.extend(["--device", device])
        if Path("/dev/infiniband").exists():
            cmd.extend(["--device", "/dev/infiniband", "--cap-add=IPC_LOCK"])
        if self.artifact_dir is not None:
            cmd.extend(["-v", f"{self.artifact_dir}:{ARTIFACT_MOUNT}:ro"])

        cmd.extend(["--network=host", "--shm-size=16gb", "--group-add", render_gid])
        for env_name in self._ENV_PASSTHROUGH:
            cmd.extend(["-e", env_name])
        container_command = self.request.bash_command()
        if self.artifact_dir is not None:
            container_command = f"{self.request.setup_command()} && {container_command}"
        cmd.extend(
            [
                "-e",
                f"HF_HOME={HF_MOUNT}",
                "-e",
                "PYTHONPATH=..",
                "-v",
                f"{self.request.hf_cache}:{HF_MOUNT}",
                "--name",
                self.container_name,
                self.request.image,
                "/bin/bash",
                "-euo",
                "pipefail",
                "-c",
                f"unset PYTORCH_ROCM_ARCH && {container_command}",
            ]
        )
        return cmd

    def _stream_logs(self, log_path: Path) -> threading.Thread:
        def pump() -> None:
            try:
                with open(log_path, "w") as log_file:
                    process = subprocess.Popen(
                        ["docker", "logs", "-f", self.container_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_file.write(line)
                    process.wait()
            except Exception as exc:
                log.warning("log streamer failed: %s", exc)

        thread = threading.Thread(target=pump, name="docker-logs", daemon=True)
        thread.start()
        return thread

    def _copy_junit(self, junit_path: Path) -> None:
        if not self._created:
            return

        result = run_command(
            [
                "docker",
                "cp",
                f"{self.container_name}:{JUNIT_CONTAINER_PATH}",
                str(junit_path),
            ],
            timeout=60,
        )
        if result.returncode == 0:
            log.info("Copied junit xml to %s", junit_path)
        else:
            log.warning("Could not retrieve junit xml: %s", result.stderr.strip())

    def _cleanup(self) -> None:
        if self._created:
            run_command(["docker", "rm", "-f", self.container_name], timeout=30)


class MultiNodeRunner:
    _ENV_PASSTHROUGH = ContainerRunner._ENV_PASSTHROUGH
    _SUBNET = "192.168.10.0/24"
    _HEAD_IP = "192.168.10.10"

    def __init__(self, request: RunRequest, artifact_dir: Path | None) -> None:
        self.request = request
        self.artifact_dir = artifact_dir
        self.suffix = f"{request.commit}_{os.urandom(4).hex()}"
        self.network_name = f"rocm-net-{self.suffix}"
        self.container_names = [
            f"rocm_node{node}_{self.suffix}" for node in range(request.num_nodes)
        ]
        self.container_name = self.container_names[0]
        self._created_containers: list[str] = []
        self._network_created = False

    def run(self) -> RunResult:
        if not self.request.node_commands:
            raise RuntimeError(
                "VLLM_CI_EXECUTION_MODE=multi-node requires VLLM_NODE_COMMAND_*"
            )
        if len(self.request.node_commands) != self.request.num_nodes:
            raise RuntimeError(
                "VLLM_NODE_COMMAND_COUNT must match NUM_NODES "
                f"({len(self.request.node_commands)} != {self.request.num_nodes})"
            )

        log.info("--- Multi-node job")
        log.info("Image: %s", self.request.image)
        log.info("Network: %s (%s)", self.network_name, self._SUBNET)
        log_path = self.request.results_dir / "multi-node.log"
        exit_code = 1

        try:
            self._start_network()
            self._start_nodes()
            self._start_ray()
            exit_code = self._run_node_commands(log_path)
        finally:
            self._cleanup()

        return RunResult(
            mode="multi-node",
            exit_code=exit_code,
            container_name=self.container_name,
            log_path=log_path if log_path.exists() else None,
        )

    def _start_network(self) -> None:
        self._run_checked(
            [
                "docker",
                "network",
                "create",
                "--subnet",
                self._SUBNET,
                self.network_name,
            ],
            "create docker network",
        )
        self._network_created = True

    def _start_nodes(self) -> None:
        self.request.hf_cache.mkdir(parents=True, exist_ok=True)
        for node in range(self.request.num_nodes):
            name = self.container_names[node]
            node_ip = f"192.168.10.{10 + node}"
            cmd = self._docker_run_cmd(node, name, node_ip)
            log.info("Starting %s at %s", name, node_ip)
            self._run_checked(cmd, f"start {name}")
            self._created_containers.append(name)

            if self.artifact_dir is not None:
                self._exec_checked(
                    name,
                    self.request.setup_command(),
                    f"install vLLM artifact on {name}",
                    timeout=600,
                )

    def _docker_run_cmd(self, node: int, name: str, node_ip: str) -> list[str]:
        try:
            render_gid = str(grp.getgrnam("render").gr_gid)
        except KeyError as err:
            raise RuntimeError("render group not found") from err

        cmd = [
            "docker",
            "run",
            "--detach",
            "--device",
            "/dev/kfd",
            "--network",
            self.network_name,
            "--ip",
            node_ip,
            "--shm-size=16gb",
            "--group-add",
            render_gid,
        ]

        render_devices, hip_visible_devices = self._node_gpu_devices(node)
        for device in render_devices:
            cmd.extend(["--device", device])
        if Path("/dev/infiniband").exists():
            cmd.extend(["--device", "/dev/infiniband", "--cap-add=IPC_LOCK"])
        if self.artifact_dir is not None:
            cmd.extend(["-v", f"{self.artifact_dir}:{ARTIFACT_MOUNT}:ro"])

        for env_name in self._ENV_PASSTHROUGH:
            cmd.extend(["-e", env_name])
        cmd.extend(
            [
                "-e",
                f"HIP_VISIBLE_DEVICES={hip_visible_devices}",
                "-e",
                f"HF_HOME={HF_MOUNT}",
                "-e",
                "PYTHONPATH=..",
                "-v",
                f"{self.request.hf_cache}:{HF_MOUNT}",
                "--name",
                name,
                self.request.image,
                "/bin/bash",
                "-lc",
                "tail -f /dev/null",
            ]
        )
        return cmd

    def _node_gpu_devices(self, node: int) -> tuple[list[str], str]:
        start = node * self.request.num_gpus_per_node
        stop = start + self.request.num_gpus_per_node
        if self.request.render_devices:
            selected = list(self.request.render_devices[start:stop])
            if len(selected) != self.request.num_gpus_per_node:
                raise RuntimeError(
                    "Not enough render devices for multi-node job: "
                    f"needed {stop}, found {len(self.request.render_devices)}"
                )
            hip_visible_devices = ",".join(str(idx) for idx in range(len(selected)))
            return selected, hip_visible_devices

        # Fallback for local repro where Buildkite render-device metadata is absent.
        # Mounting /dev/dri exposes all render nodes; HIP_VISIBLE_DEVICES then
        # partitions by host GPU index, matching the historical shell runner.
        host_indices = range(start, stop)
        return ["/dev/dri"], ",".join(str(idx) for idx in host_indices)

    def _start_ray(self) -> None:
        self._exec_detached(
            self.container_names[0],
            "ray start --head --port=6379 --block",
            "start ray head",
        )
        time.sleep(10)
        for name in self.container_names[1:]:
            self._exec_detached(
                name,
                f"ray start --address={self._HEAD_IP}:6379 --block",
                f"start ray worker {name}",
            )
        time.sleep(10)
        self._exec_checked(self.container_names[0], "ray status", "ray status")

    def _run_node_commands(self, log_path: Path) -> int:
        processes: dict[int, subprocess.Popen[str]] = {}
        threads: list[threading.Thread] = []
        write_lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as log_file:
            log_file.write(f"=== Multi-node job started {Watchdog._now()} ===\n")
            for node in range(self.request.num_nodes - 1, -1, -1):
                process = self._exec_process(
                    self.container_names[node],
                    f"cd {WORKSPACE}/tests && {self.request.node_commands[node]}",
                )
                processes[node] = process
                threads.append(
                    self._stream_process_output(
                        node=node,
                        process=process,
                        log_file=log_file,
                        write_lock=write_lock,
                    )
                )
                if node != 0:
                    time.sleep(1)

            deadline = time.monotonic() + self.request.timeout_s
            exit_codes: dict[int, int] = {}
            for node, process in processes.items():
                remaining = max(1, int(deadline - time.monotonic()))
                try:
                    exit_codes[node] = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    log.error(
                        "Node %d exceeded %ds timeout", node, self.request.timeout_s
                    )
                    self._kill_processes(processes)
                    exit_codes[node] = 124
                    break

            for thread in threads:
                thread.join(timeout=10)

            log_file.write(f"=== Exit codes: {exit_codes} ===\n")

        for node in sorted(exit_codes):
            if exit_codes[node] != 0:
                return exit_codes[node]
        return 0

    def _exec_process(self, container: str, command: str) -> subprocess.Popen[str]:
        log.info("Running %s: %s", container, command)
        return subprocess.Popen(
            [
                "docker",
                "exec",
                container,
                "/bin/bash",
                "-euo",
                "pipefail",
                "-c",
                command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _stream_process_output(
        self,
        *,
        node: int,
        process: subprocess.Popen[str],
        log_file,
        write_lock: threading.Lock,
    ) -> threading.Thread:
        def pump() -> None:
            assert process.stdout is not None
            prefix = f"[node{node}] "
            for line in process.stdout:
                text = prefix + line
                with write_lock:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    log_file.write(text)
                    log_file.flush()

        thread = threading.Thread(target=pump, name=f"node{node}-logs", daemon=True)
        thread.start()
        return thread

    def _kill_processes(self, processes: dict[int, subprocess.Popen[str]]) -> None:
        for process in processes.values():
            if process.poll() is None:
                process.kill()

    def _exec_detached(self, container: str, command: str, description: str) -> None:
        self._run_checked(
            ["docker", "exec", "--detach", container, "/bin/bash", "-lc", command],
            description,
        )

    def _exec_checked(
        self,
        container: str,
        command: str,
        description: str,
        timeout: int | None = None,
    ) -> None:
        self._run_checked(
            [
                "docker",
                "exec",
                container,
                "/bin/bash",
                "-euo",
                "pipefail",
                "-c",
                command,
            ],
            description,
            timeout=timeout,
        )

    @staticmethod
    def _run_checked(
        cmd: list[str],
        description: str,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = run_command(cmd, timeout=timeout)
        if result.returncode != 0:
            output = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{description} failed: {output}")
        return result

    def _cleanup(self) -> None:
        for container in reversed(self._created_containers):
            run_command(["docker", "rm", "-f", container], timeout=30)
        if self._network_created:
            run_command(["docker", "network", "rm", self.network_name], timeout=30)


class Watchdog:
    def __init__(self, request: RunRequest, container_name: str) -> None:
        self.request = request
        self.container_name = container_name
        self.log_path = request.results_dir / "health_watchdog.log"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._docker_root = self._detect_docker_root()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as log_file:
            log_file.write(f"=== Watchdog started {self._now()} ===\n")
            self._write_snapshot(log_file, "pre")
            log_file.write("=== Samples ===\n")

        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)
        self._thread.start()
        log.info("Watchdog started (interval %ds)", self.request.watchdog_interval_s)

    def stop(self, exit_code: int | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

        with open(self.log_path, "a") as log_file:
            log_file.write("=== Summary ===\n")
            self._write_snapshot(log_file, "post")
            if exit_code is not None:
                log_file.write(f"exit_code={exit_code}\n")
            log_file.write(f"=== Watchdog stopped {self._now()} ===\n")

        log.info("Watchdog stopped")

    def _loop(self) -> None:
        with open(self.log_path, "a") as log_file:
            while not self._stop.is_set():
                try:
                    log_file.write(self._sample() + "\n")
                    log_file.flush()
                except Exception as exc:
                    log.warning("watchdog sample failed: %s", exc)
                self._stop.wait(self.request.watchdog_interval_s)

    def _write_snapshot(self, log_file, label: str) -> None:
        snapshot = self._try(gpu_snapshot)
        log_file.write(f"  mem_{label}: [{self._container_mem_oneliner()}]\n")
        if snapshot:
            log_file.write(f"  gpu_{label}: [{self._fmt_gpus(snapshot)}]\n")
        log_file.write(f"  disk_{label}: [{self._disk_oneliner()}]\n")

    def _sample(self) -> str:
        return (
            f"{self._now()}"
            f" mem=[{self._container_mem_oneliner()}]"
            f" disk=[{self._disk_oneliner()}]"
            f" container={self._container_state()}"
            f" gpu=[{self._gpu_oneliner()}]"
        )

    def _disk_oneliner(self) -> str:
        paths = [("/", "root"), ("/tmp", "tmp")]
        if self._docker_root and os.path.isdir(self._docker_root):
            paths.append((self._docker_root, "docker"))

        parts = []
        seen_devices: set[int] = set()
        for path, label in paths:
            try:
                stat = os.stat(path)
                if stat.st_dev in seen_devices:
                    continue
                seen_devices.add(stat.st_dev)
                usage = shutil.disk_usage(path)
                pct = round(usage.used * 100 / usage.total, 1) if usage.total else 0
                free = round(usage.free / 1024**3, 1)
                parts.append(f"{label}={pct}%/{free}G")
            except OSError:
                pass
        return ",".join(parts) or "?"

    def _gpu_oneliner(self) -> str:
        try:
            return ",".join(
                f"{gpu['used_mib']}/{gpu['total_mib']}" for gpu in gpu_snapshot()
            )
        except Exception:
            return "?"

    def _container_state(self) -> str:
        try:
            result = run_command(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Status}}/exit={{.State.ExitCode}}/oom={{.State.OOMKilled}}",
                    self.container_name,
                ],
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "?"
        except Exception:
            return "?"

    def _container_mem_oneliner(self) -> str:
        try:
            result = run_command(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.MemUsage}}",
                    self.container_name,
                ],
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "?"
        except Exception:
            return "?"

    @staticmethod
    def _detect_docker_root() -> str | None:
        try:
            result = run_command(
                ["docker", "info", "-f", "{{.DockerRootDir}}"], timeout=10
            )
            docker_root = result.stdout.strip()
            return docker_root or None
        except Exception:
            return None

    @staticmethod
    def _try(func):
        try:
            return func()
        except Exception:
            return None

    @staticmethod
    def _fmt_gpus(snapshot: list[dict[str, Any]]) -> str:
        return ",".join(
            f"GPU{gpu['gpu']}={gpu['used_mib']}/{gpu['total_mib']}MiB"
            for gpu in snapshot
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ArtifactCollector:
    def __init__(self, request: RunRequest) -> None:
        self.request = request

    def finalize(
        self,
        result: RunResult,
        *,
        started_at: datetime,
        watchdog_path: Path | None = None,
    ) -> int:
        exit_code = result.exit_code
        if result.mode == "single-node" and exit_code == 0:
            exit_code = self._validate_junit(result.junit_path)

        summary_path = self._write_summary(
            result=result,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._upload_existing(
            summary_path, result.log_path, result.junit_path, watchdog_path
        )
        return exit_code

    def _validate_junit(self, junit_path: Path | None) -> int:
        if junit_path is None or not junit_path.exists():
            if "pytest" not in self.request.commands:
                log.info("No junit xml and command is not pytest; trusting exit 0")
                return 0
            log.error("junit xml not found; overriding exit 0 -> 1")
            return 1

        try:
            root = ET.parse(junit_path).getroot()
        except ET.ParseError as exc:
            log.error("junit xml parse failed: %s; overriding exit 0 -> 1", exc)
            return 1

        suites = [root] if root.tag == "testsuite" else list(root)
        total = failures = errors = 0
        for suite in suites:
            if suite.tag == "testsuite":
                total += int(suite.get("tests", "0"))
                failures += int(suite.get("failures", "0"))
                errors += int(suite.get("errors", "0"))

        if failures + errors > 0:
            log.error(
                "Junit: %d failure(s) + %d error(s); overriding exit 0 -> 1",
                failures,
                errors,
            )
            return 1
        if total == 0:
            log.error("Junit: 0 tests ran; overriding exit 0 -> 1")
            return 1
        return 0

    def _write_summary(
        self,
        *,
        result: RunResult,
        exit_code: int,
        started_at: datetime,
        finished_at: datetime,
    ) -> Path:
        summary_path = self.request.results_dir / "run-summary.json"
        summary = {
            "test_name": self.request.test_name,
            "mode": result.mode,
            "image": self.request.image,
            "container_name": result.container_name,
            "exit_code": exit_code,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary_path

    def _upload_existing(self, *paths: Path | None) -> None:
        for path in paths:
            if path is not None and path.exists():
                self._upload(path)

    @staticmethod
    def _upload(path: Path) -> None:
        if shutil.which("buildkite-agent") is None:
            return

        result = run_command(
            ["buildkite-agent", "artifact", "upload", str(path)], timeout=300
        )
        if result.returncode != 0:
            log.warning("upload failed for %s: %s", path.name, result.stderr.strip())
        else:
            log.info("Uploaded %s", path.name)


class AmdTestRunner:
    def __init__(self, request: RunRequest) -> None:
        self.request = request
        self.artifacts = ArtifactCollector(request)
        self.artifact_package = ArtifactPackage(request)

    def run(self) -> int:
        started_at = datetime.now(timezone.utc)
        artifact_dir: Path | None = None
        result = RunResult(
            mode=self.request.execution_mode,
            exit_code=1,
        )
        watchdog: Watchdog | None = None

        try:
            self._check_gpus()
            self._ensure_image()
            artifact_dir = self.artifact_package.prepare()
            log.info("Test name: %s", self.request.test_name or "<unset>")
            log.info("Image: %s", self.request.image)
            log.info("Execution mode: %s", self.request.execution_mode)
            if self.request.execution_mode == "multi-node":
                runner = MultiNodeRunner(self.request, artifact_dir)
            else:
                runner = ContainerRunner(self.request, artifact_dir)
            result.container_name = runner.container_name
            watchdog = Watchdog(self.request, result.container_name or "node0")
            watchdog.start()
            if self.request.commands:
                log.info("Commands: %s", self.request.commands)
            result = runner.run()
        except Exception as exc:
            log.error("amd test runner failed: %s", exc)
        finally:
            if watchdog is not None:
                watchdog.stop(result.exit_code)

        return self.artifacts.finalize(
            result,
            started_at=started_at,
            watchdog_path=watchdog.log_path if watchdog is not None else None,
        )

    def _ensure_image(self) -> None:
        if self._image_exists_locally(self.request.image):
            log.info("Image available locally: %s", self.request.image)
            return
        if self._docker_pull(self.request.image):
            return
        raise RuntimeError(f"Failed to pull test image: {self.request.image}")

    def _check_gpus(self) -> None:
        try:
            snapshot = gpu_snapshot()
        except RuntimeError as exc:
            fallback_count = len(self.request.render_devices)
            fallback_source = "Buildkite render-device metadata"
            if fallback_count == 0:
                fallback_count = amd_smi_list_gpu_count()
                fallback_source = "amd-smi list"

            if fallback_count == 0:
                raise RuntimeError("No GPUs detected by amd-smi") from exc

            log.warning(
                "amd-smi memory snapshot failed (%s); detected %d GPU(s) via %s",
                exc,
                fallback_count,
                fallback_source,
            )
            return

        if snapshot:
            log.info("Detected %d GPU(s)", len(snapshot))
            return

        fallback_count = len(self.request.render_devices) or amd_smi_list_gpu_count()
        if fallback_count == 0:
            raise RuntimeError("No GPUs detected by amd-smi")
        log.warning(
            "amd-smi memory snapshot was empty; detected %d GPU(s) via fallback",
            fallback_count,
        )

    @staticmethod
    def _image_exists_locally(image: str) -> bool:
        return (
            run_command(["docker", "image", "inspect", image], timeout=30).returncode
            == 0
        )

    @staticmethod
    def _docker_pull(image: str, retries: int = 3, delay: int = 15) -> bool:
        for attempt in range(retries):
            result = run_command(["docker", "pull", image], timeout=600)
            if result.returncode == 0:
                log.info("Pulled %s on attempt %d", image, attempt + 1)
                return True
            log.warning(
                "docker pull attempt %d failed: %s", attempt + 1, result.stderr.strip()
            )
            if attempt < retries - 1:
                time.sleep(delay)
        return False


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("VLLM_CI_DEBUG") == "1" else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    def raise_interrupt(signum, _frame) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, raise_interrupt)
    signal.signal(signal.SIGINT, raise_interrupt)

    try:
        request = RunRequest.from_env(sys.argv)
        return AmdTestRunner(request).run()
    except KeyboardInterrupt as exc:
        log.error("Interrupted: %s", exc)
        return 130
    except Exception as exc:
        if os.environ.get("VLLM_CI_DEBUG") == "1":
            log.exception("run-amd-test.py failed")
        else:
            log.error("run-amd-test.py failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
