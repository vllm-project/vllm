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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

log = logging.getLogger("amd-test")

HF_MOUNT = "/root/.cache/huggingface"
JUNIT_CONTAINER_PATH = "/tmp/vllm-results.xml"
ARTIFACT_MOUNT = "/tmp/vllm-rocm-install"
DEFAULT_ARTIFACT_GLOB = "artifacts/vllm-rocm-install/vllm-rocm-install.tar.gz"
WORKSPACE = "/vllm-workspace"
ARTIFACT_WORKSPACE_PATHS = (
    "tests",
    "examples",
    "benchmarks",
    "requirements",
    "docker",
    ".buildkite",
    "pyproject.toml",
)
ENV_PASSTHROUGH = (
    "HF_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "BUILDKITE_PARALLEL_JOB",
    "BUILDKITE_PARALLEL_JOB_COUNT",
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_text(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip() or default


def run_command(
    cmd: list[str], timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip()


def render_group_id() -> str:
    try:
        return str(grp.getgrnam("render").gr_gid)
    except KeyError as err:
        raise RuntimeError("render group not found") from err


def docker_env_args(*envs: str) -> list[str]:
    return [arg for env in envs for arg in ("-e", env)]


def docker_device_args(devices: list[str] | tuple[str, ...]) -> list[str]:
    args = ["--device", "/dev/kfd"]
    args.extend(arg for device in devices for arg in ("--device", device))
    if Path("/dev/infiniband").exists():
        args.extend(["--device", "/dev/infiniband", "--cap-add=IPC_LOCK"])
    return args


def make_results_dir(artifact_mode: bool) -> Path:
    configured_root = os.environ.get("VLLM_CI_RESULTS_ROOT")
    if configured_root:
        root_path = Path(configured_root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="vllm-ci-results-", dir=root_path))

    if not artifact_mode:
        return Path(tempfile.mkdtemp(prefix="vllm-ci-results-"))

    candidates = [
        Path(os.environ.get("HF_HOME", "")) / "amd-ci-results",
        Path.cwd() / "amd-ci-results",
        Path(os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH", "")) / "amd-ci-results",
        Path(os.environ.get("BUILDKITE_BUILD_PATH", "")) / "amd-ci-results",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if str(candidate) in {"", "."}:
            continue
        root_path = candidate.resolve()
        if root_path in seen:
            continue
        seen.add(root_path)
        try:
            root_path.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="vllm-ci-results-", dir=root_path))
        except OSError as exc:
            log.warning("Could not use ROCm CI results dir %s: %s", root_path, exc)

    raise RuntimeError(
        "Could not create a Docker-visible ROCm CI results directory. "
        "Set VLLM_CI_RESULTS_ROOT to a writable path shared with Docker."
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
    render_devices: tuple[str, ...]
    artifact_mode: bool
    fallback_image: str | None
    artifact_glob: str
    execution_mode: str
    num_nodes: int
    num_gpus_per_node: int
    node_commands: tuple[str, ...]

    @classmethod
    def from_env(cls, argv: list[str]) -> "RunRequest":
        commit = env_text("BUILDKITE_COMMIT")
        if not commit:
            raise RuntimeError("BUILDKITE_COMMIT is required")

        artifact_mode = env_flag("VLLM_CI_USE_ARTIFACTS")
        execution_mode = env_text("VLLM_CI_EXECUTION_MODE", "single-node")
        configured_image = env_text("DOCKER_IMAGE_NAME")
        fallback_image = None
        if artifact_mode:
            image = env_text("VLLM_CI_BASE_IMAGE", "rocm/vllm-dev:ci_base")
            fallback_image = cls._resolve_fallback_image(
                commit=commit,
                base_image=image,
                configured_image=configured_image,
            )
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
            results_dir=make_results_dir(artifact_mode),
            hf_cache=hf_cache,
            timeout_s=int(os.environ.get("CONTAINER_TIMEOUT_S", "10200")),
            render_devices=tuple(
                token
                for token in os.environ.get(
                    "BUILDKITE_AGENT_META_DATA_RENDER_DEVICES", ""
                ).split()
                if token.startswith("/dev/")
            ),
            artifact_mode=artifact_mode,
            fallback_image=fallback_image,
            artifact_glob=os.environ.get(
                "VLLM_CI_ARTIFACT_GLOB", DEFAULT_ARTIFACT_GLOB
            ),
            execution_mode=execution_mode,
            num_nodes=int(os.environ.get("NUM_NODES", "1")),
            num_gpus_per_node=int(os.environ.get("VLLM_NUM_GPUS_PER_NODE", "1")),
            node_commands=node_commands,
        )

    @staticmethod
    def _resolve_fallback_image(
        *,
        commit: str,
        base_image: str,
        configured_image: str | None,
    ) -> str | None:
        if env_flag("VLLM_CI_DISABLE_FALLBACK") or env_flag("ROCM_CI_ARTIFACT_ONLY"):
            return None

        fallback_image = env_text("VLLM_CI_FALLBACK_IMAGE")
        if not fallback_image and configured_image and configured_image != base_image:
            fallback_image = configured_image
        if not fallback_image:
            fallback_image = f"rocm/vllm-ci:{commit}"
        if fallback_image == base_image:
            log.warning(
                "Ignoring ROCm fallback image because it matches the base image: %s",
                base_image,
            )
            return None
        return fallback_image

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
        return tuple(
            command
            for idx in range(count)
            if (command := os.environ.get(f"VLLM_NODE_COMMAND_{idx}"))
        )

    def bash_command(self) -> str:
        return (
            f'export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} --junitxml='
            f'{JUNIT_CONTAINER_PATH}"'
            f" && {self.commands}"
        )

    def setup_command(self) -> str:
        script = f"""
        import pathlib
        import subprocess
        import sys
        import sysconfig

        artifact_mount = pathlib.Path({ARTIFACT_MOUNT!r})
        workspace = pathlib.Path({WORKSPACE!r})
        workspace_paths = {ARTIFACT_WORKSPACE_PATHS!r}

        def run(*cmd: str) -> None:
            subprocess.run(cmd, check=True)

        if not artifact_mount.is_dir():
            sys.exit(f"ROCm artifact directory not mounted at {{artifact_mount}}")

        wheels = sorted(artifact_mount.glob("*.whl"))
        if not wheels:
            sys.exit(f"No vLLM wheel found in {{artifact_mount}}")

        run("uv", "pip", "install", "--system", "--no-deps", str(wheels[0]))
        run("rm", "-rf", str(workspace))
        workspace.mkdir(parents=True)

        for relative_path in workspace_paths:
            src = artifact_mount / relative_path
            if src.exists():
                run("cp", "-a", str(src), str(workspace))

        v1_src = artifact_mount / "vllm_v1"
        if v1_src.exists():
            site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
            v1_dst = site_packages / "vllm" / "v1"
            v1_dst.parent.mkdir(parents=True, exist_ok=True)
            run("cp", "-a", f"{{v1_src}}/.", str(v1_dst))
        """
        return f"python3 - <<'PY'\n{dedent(script).strip()}\nPY\n"


@dataclass
class RunResult:
    mode: str
    exit_code: int
    container_name: str | None = None
    junit_path: Path | None = None
    log_path: Path | None = None


def copy_artifact_to_container(artifact_dir: Path, container_name: str) -> None:
    # docker cp streams from the client filesystem, avoiding bind-mount path
    # mismatches between the Buildkite workspace and the Docker daemon.
    result = run_command(
        [
            "docker",
            "cp",
            str(artifact_dir),
            f"{container_name}:{ARTIFACT_MOUNT}",
        ],
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"copy ROCm artifact into {container_name} failed: {command_output(result)}"
        )


class ArtifactPackage:
    def __init__(self, request: RunRequest) -> None:
        self.request = request
        self.download_dir = request.results_dir / "downloaded-artifacts"
        self.install_dir = request.results_dir / "vllm-rocm-install"

    def prepare(self) -> Path:
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
            raise RuntimeError(f"artifact download failed: {command_output(result)}")

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
            tar.extractall(
                self.install_dir,
                members=self._validated_archive_members(tar),
            )

    def _validated_archive_members(self, tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
        dest = self.install_dir.resolve()
        members = tar.getmembers()
        for member in members:
            member_path = (dest / member.name).resolve()
            if dest not in (member_path, *member_path.parents):
                raise RuntimeError(
                    f"Refusing to extract unsafe artifact path: {member.name}"
                )
            if member.islnk() or member.issym():
                raise RuntimeError(f"Refusing to extract artifact link: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"Refusing to extract unsupported artifact entry: {member.name}"
                )
        return members

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

        try:
            create = run_command(self._build_cmd(), timeout=120)
            if create.returncode != 0:
                raise RuntimeError(f"docker create failed: {command_output(create)}")
            self._created = True
            if self.artifact_dir is not None:
                copy_artifact_to_container(self.artifact_dir, self.container_name)

            start = run_command(["docker", "start", self.container_name], timeout=120)
            if start.returncode != 0:
                raise RuntimeError(f"docker start failed: {command_output(start)}")
        except Exception:
            self._cleanup()
            raise

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
                log.error("docker wait failed: %s", command_output(wait))
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
        return [
            "docker",
            "create",
            *docker_device_args(self.request.render_devices),
            "--network=host",
            "--shm-size=16gb",
            "--group-add",
            render_group_id(),
            *docker_env_args(*ENV_PASSTHROUGH, f"HF_HOME={HF_MOUNT}", "PYTHONPATH=.."),
            "-v",
            f"{self.request.hf_cache}:{HF_MOUNT}",
            "--name",
            self.container_name,
            self.request.image,
            "/bin/bash",
            "-euo",
            "pipefail",
            "-c",
            self._container_shell_command(),
        ]

    def _container_shell_command(self) -> str:
        commands = ["unset PYTORCH_ROCM_ARCH"]
        if self.artifact_dir is not None:
            commands.append(self.request.setup_command().strip())
        commands.append(self.request.bash_command())
        return "\n".join(commands)

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
        if not self._created or "pytest" not in self.request.commands:
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
            log.info("Could not retrieve junit xml: %s", command_output(result))

    def _cleanup(self) -> None:
        if self._created:
            run_command(["docker", "rm", "-f", self.container_name], timeout=30)


class MultiNodeRunner:
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
                copy_artifact_to_container(self.artifact_dir, name)
                self._exec_checked(
                    name,
                    self.request.setup_command(),
                    f"install vLLM artifact on {name}",
                    timeout=600,
                )

    def _docker_run_cmd(self, node: int, name: str, node_ip: str) -> list[str]:
        render_devices, hip_visible_devices = self._node_gpu_devices(node)
        cmd = [
            "docker",
            "run",
            "--detach",
            *docker_device_args(render_devices),
            "--network",
            self.network_name,
            "--ip",
            node_ip,
            "--shm-size=16gb",
            "--group-add",
            render_group_id(),
            *docker_env_args(
                *ENV_PASSTHROUGH,
                f"HIP_VISIBLE_DEVICES={hip_visible_devices}",
                f"HF_HOME={HF_MOUNT}",
                "PYTHONPATH=..",
            ),
            "-v",
            f"{self.request.hf_cache}:{HF_MOUNT}",
            "--name",
            name,
            self.request.image,
            "/bin/bash",
            "-lc",
            "tail -f /dev/null",
        ]
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
            started_at = utc_now().isoformat(timespec="seconds")
            log_file.write(f"=== Multi-node job started {started_at} ===\n")
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
            raise RuntimeError(f"{description} failed: {command_output(result)}")
        return result

    def _cleanup(self) -> None:
        for container in reversed(self._created_containers):
            run_command(["docker", "rm", "-f", container], timeout=30)
        if self._network_created:
            run_command(["docker", "network", "rm", self.network_name], timeout=30)


class ArtifactCollector:
    def __init__(self, request: RunRequest) -> None:
        self.request = request

    def finalize(
        self,
        result: RunResult,
        *,
        started_at: datetime,
    ) -> int:
        exit_code = result.exit_code
        summary_path = self._write_summary(
            result=result,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=utc_now(),
        )
        self._upload_existing(summary_path, result.log_path, result.junit_path)
        return exit_code

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
            log.warning("upload failed for %s: %s", path.name, command_output(result))
        else:
            log.info("Uploaded %s", path.name)


class AmdTestRunner:
    def __init__(self, request: RunRequest) -> None:
        self.request = request

    def run(self) -> int:
        started_at = utc_now()
        active_request = self.request
        artifact_dir: Path | None = None
        result = RunResult(
            mode=self.request.execution_mode,
            exit_code=1,
        )

        try:
            active_request, artifact_dir = self._prepare_execution()
            log.info("Test name: %s", active_request.test_name or "<unset>")
            log.info("Image: %s", active_request.image)
            log.info("Execution mode: %s", active_request.execution_mode)
            runner_cls = (
                MultiNodeRunner
                if active_request.execution_mode == "multi-node"
                else ContainerRunner
            )
            runner = runner_cls(active_request, artifact_dir)
            result.container_name = runner.container_name
            if active_request.commands:
                log.info("Commands: %s", active_request.commands)
            result = runner.run()
        except Exception as exc:
            log.error("amd test runner failed: %s", exc)

        return ArtifactCollector(active_request).finalize(
            result,
            started_at=started_at,
        )

    def _prepare_execution(self) -> tuple[RunRequest, Path | None]:
        if not self.request.artifact_mode:
            self._ensure_image(self.request.image)
            return self.request, None

        try:
            self._ensure_image(self.request.image)
            artifact_dir = ArtifactPackage(self.request).prepare()
            self._check_artifact_container_visibility(artifact_dir)
            return self.request, artifact_dir
        except Exception as exc:
            if not self.request.fallback_image:
                raise

            log.warning(
                "ROCm artifact setup failed before tests; falling back to %s: %s",
                self.request.fallback_image,
                exc,
            )
            fallback_request = replace(
                self.request,
                image=self.request.fallback_image,
                artifact_mode=False,
                fallback_image=None,
            )
            self._ensure_image(fallback_request.image)
            return fallback_request, None

    def _check_artifact_container_visibility(self, artifact_dir: Path) -> None:
        name = f"rocm_artifact_check_{self.request.commit}_{os.urandom(4).hex()}"
        create = run_command(
            [
                "docker",
                "create",
                "--name",
                name,
                self.request.image,
                "/bin/bash",
                "-euo",
                "pipefail",
                "-c",
                (
                    f'test -n "$(find "{ARTIFACT_MOUNT}" -maxdepth 1 '
                    "-name '*.whl' -print -quit)\""
                ),
            ],
            timeout=120,
        )
        if create.returncode != 0:
            raise RuntimeError(
                f"artifact visibility check create failed: {command_output(create)}"
            )

        try:
            copy_artifact_to_container(artifact_dir, name)
            start = run_command(["docker", "start", "--attach", name], timeout=120)
            if start.returncode != 0:
                raise RuntimeError(
                    "artifact is not visible inside Docker container: "
                    f"{command_output(start)}"
                )
        finally:
            run_command(["docker", "rm", "-f", name], timeout=30)

    def _ensure_image(self, image: str) -> None:
        if self._image_exists_locally(image):
            log.info("Image available locally: %s", image)
            return
        if self._docker_pull(image):
            return
        raise RuntimeError(f"Failed to pull test image: {image}")

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
                "docker pull attempt %d failed: %s", attempt + 1, command_output(result)
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
