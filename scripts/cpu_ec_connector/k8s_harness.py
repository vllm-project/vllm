# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenShift harness for the CPU EC connector e2e test.

Manages two vllm Deployments (producer + consumer) in an OpenShift namespace,
drives HTTP via oc port-forward, and streams logs via oc logs -f.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from shared import ServerSpec

_K8S_DIR = Path(__file__).resolve().parent / "k8s"
_SITECUSTOMIZE = Path(__file__).resolve().parent / "sitecustomize.py"


# ---------------------------------------------------------------------------
# oc subprocess helpers
# ---------------------------------------------------------------------------


def _oc(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["oc"] + cmd, check=check, capture_output=True, text=True)


def _oc_stdin(cmd: list[str], stdin_data: str) -> None:
    subprocess.run(
        ["oc"] + cmd, input=stdin_data, text=True, check=True, capture_output=True
    )


# ---------------------------------------------------------------------------
# YAML manifest patching
# ---------------------------------------------------------------------------


def _set_env(container: dict, name: str, value: str) -> None:
    for e in container.get("env", []):
        if e["name"] == name:
            e["value"] = value
            return
    container.setdefault("env", []).append({"name": name, "value": value})


def patch_deployment_yaml(
    template_path: Path,
    *,
    run_id: str,
    namespace: str,
    image: str,
    model: str,
    port: int,
    gpu_memory_utilization: float,
    ec_role: str,
    engine_id: str,
    num_ec_blocks: int,
    side_channel_port: int,
    producer: bool,
    different_nodes: bool,
) -> dict:
    """Load a deployment YAML template and patch dynamic fields.

    Returns the patched document as a dict (ready for yaml.dump + oc apply).
    """
    with template_path.open() as f:
        doc = yaml.safe_load(f)

    role = "producer" if producer else "consumer"
    run_label = {"run-id": run_id}

    doc["metadata"]["name"] = f"vllm-ec-{role}-{run_id}"
    doc["metadata"]["namespace"] = namespace
    doc["spec"]["selector"]["matchLabels"].update(run_label)
    doc["spec"]["template"]["metadata"]["labels"].update(run_label)

    container = doc["spec"]["template"]["spec"]["containers"][0]
    container["image"] = image

    ec_config = json.dumps(
        {
            "ec_connector": "ECCPUConnector",
            "ec_role": ec_role,
            "engine_id": engine_id,
            "ec_connector_extra_config": {"num_ec_blocks": num_ec_blocks},
        }
    )

    _set_env(container, "VLLM_MODEL", model)
    _set_env(container, "VLLM_PORT", str(port))
    _set_env(container, "GPU_MEMORY_UTILIZATION", str(gpu_memory_utilization))
    _set_env(container, "EC_TRANSFER_CONFIG", ec_config)
    _set_env(container, "VLLM_EC_SIDE_CHANNEL_PORT", str(side_channel_port))

    for vol in doc["spec"]["template"]["spec"]["volumes"]:
        if vol["name"] == "sitecustomize":
            vol["configMap"]["name"] = f"ec-test-sitecustomize-{run_id}"

    pod_spec = doc["spec"]["template"]["spec"]
    if different_nodes:
        pod_spec["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchLabels": {
                                "app": "vllm-ec-test",
                                "run-id": run_id,
                            },
                        },
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ],
            },
        }
    else:
        pod_spec.pop("affinity", None)

    return doc


# ---------------------------------------------------------------------------
# K8sHarness
# ---------------------------------------------------------------------------


class K8sHarness:
    """Context manager that manages producer+consumer vllm Deployments in OpenShift.

    Interface mirrors LocalHarness: exposes .producer, .consumer, .model,
    and restart_producer(). HTTP is accessed via oc port-forward; logs are
    streamed via oc logs -f to local files so log_slice/assert_in_log work
    unchanged.
    """

    def __init__(
        self,
        producer: ServerSpec,
        consumer: ServerSpec,
        model: str,
        *,
        namespace: str,
        image: str,
        k8s_dir: Path = _K8S_DIR,
        different_nodes: bool = False,
        log_delay: float = 0.5,
        keep_on_exit: bool = False,
    ):
        self.producer = producer
        self.consumer = consumer
        self.model = model
        self._namespace = namespace
        self._image = image
        self._k8s_dir = k8s_dir
        self._different_nodes = different_nodes
        self._log_delay = log_delay
        self.keep_on_exit = keep_on_exit
        self._run_id = time.strftime("%Y%m%d-%H%M%S")

        self._producer_pf: subprocess.Popen | None = None
        self._consumer_pf: subprocess.Popen | None = None
        self._producer_logs: subprocess.Popen | None = None
        self._consumer_logs: subprocess.Popen | None = None
        self._pf_watchdog_stop = threading.Event()
        self._pf_watchdog_t: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> K8sHarness:
        for spec in (self.producer, self.consumer):
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[k8s-setup] run_id={self._run_id}, namespace={self._namespace}")
        self._create_configmap()
        self._apply_deployment("producer")
        self._apply_deployment("consumer")
        print("[k8s-setup] waiting for deployments to roll out…")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {
                ex.submit(self._wait_rollout, role): role
                for role in ("producer", "consumer")
            }
            for fut in futs:
                fut.result()
                print(f"  ✓ {futs[fut]} rollout complete")

        self._start_log_stream("producer")
        self._start_log_stream("consumer")

        from shared import HEALTH_TIMEOUT_S

        print("[k8s-setup] waiting on /health for both (via oc exec)…")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs2 = {
                ex.submit(self._wait_vllm_ready, spec, HEALTH_TIMEOUT_S): spec.role
                for spec in (self.producer, self.consumer)
            }
            for fut in futs2:
                fut.result()
                role = futs2[fut]
                port = getattr(self, role).http_port
                print(f"  ✓ {role} healthy on port {port}")

        self._start_port_forward("producer")
        self._start_port_forward("consumer")
        self._start_pf_watchdog()
        return self

    def __exit__(self, *_exc) -> None:
        if self.keep_on_exit:
            print("\n[k8s-teardown] --keep-servers set; leaving deployments running.")
            return
        self._pf_watchdog_stop.set()
        if self._pf_watchdog_t is not None:
            self._pf_watchdog_t.join(timeout=6)
        self._stop_background_procs()
        print("\n[k8s-teardown] deleting deployments and configmap…")
        self._delete_deployment("producer")
        self._delete_deployment("consumer")
        self._delete_configmap()

    # ------------------------------------------------------------------
    # Producer restart
    # ------------------------------------------------------------------

    def restart_producer(self) -> None:
        print("\n[k8s-restart] restarting producer deployment…")
        for attr in ("_producer_pf", "_producer_logs"):
            proc = getattr(self, attr)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            fh = getattr(proc, "_log_fh", None)
            if fh:
                fh.close()

        _oc(
            [
                "rollout",
                "restart",
                f"deployment/{self._deployment_name('producer')}",
                "-n",
                self._namespace,
            ]
        )
        self._wait_rollout("producer")

        self._start_log_stream("producer")

        from shared import HEALTH_TIMEOUT_S

        self._wait_vllm_ready(self.producer, HEALTH_TIMEOUT_S)
        self._start_port_forward("producer")
        print(f"  ✓ producer healthy on {self.producer.http_port}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deployment_name(self, role: str) -> str:
        return f"vllm-ec-{role}-{self._run_id}"

    def _create_configmap(self) -> None:
        cm_name = f"ec-test-sitecustomize-{self._run_id}"
        print(f"[k8s-setup] creating ConfigMap {cm_name}")
        _oc(
            [
                "create",
                "configmap",
                cm_name,
                f"--from-file=sitecustomize.py={_SITECUSTOMIZE}",
                "-n",
                self._namespace,
            ]
        )

    def _delete_configmap(self) -> None:
        cm_name = f"ec-test-sitecustomize-{self._run_id}"
        _oc(["delete", "configmap", cm_name, "-n", self._namespace], check=False)

    def _apply_deployment(self, role: str) -> None:
        spec = self.producer if role == "producer" else self.consumer
        template_path = self._k8s_dir / f"{role}-deployment.yaml"
        patched = patch_deployment_yaml(
            template_path,
            run_id=self._run_id,
            namespace=self._namespace,
            image=self._image,
            model=self.model,
            port=spec.http_port,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            ec_role=f"ec_{role}",
            engine_id=spec.engine_id,
            num_ec_blocks=spec.num_ec_blocks,
            side_channel_port=spec.side_channel_port,
            producer=(role == "producer"),
            different_nodes=self._different_nodes,
        )
        print(f"[k8s-setup] applying {self._deployment_name(role)}")
        _oc_stdin(["apply", "-f", "-", "-n", self._namespace], yaml.dump(patched))

    def _delete_deployment(self, role: str) -> None:
        _oc(
            [
                "delete",
                "deployment",
                self._deployment_name(role),
                "-n",
                self._namespace,
                "--ignore-not-found",
            ],
            check=False,
        )

    def _wait_rollout(self, role: str) -> None:
        _oc(
            [
                "rollout",
                "status",
                f"deployment/{self._deployment_name(role)}",
                "-n",
                self._namespace,
                "--timeout=600s",
            ]
        )

    def _start_log_stream(self, role: str) -> None:
        spec = self.producer if role == "producer" else self.consumer
        log_fh = spec.log_path.open("wb", buffering=0)
        proc = subprocess.Popen(
            [
                "oc",
                "logs",
                "-f",
                f"deployment/{self._deployment_name(role)}",
                "-n",
                self._namespace,
            ],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        proc._log_fh = log_fh  # type: ignore[attr-defined]
        if role == "producer":
            self._producer_logs = proc
        else:
            self._consumer_logs = proc

    def _start_port_forward(self, role: str) -> None:
        spec = self.producer if role == "producer" else self.consumer
        proc = subprocess.Popen(
            [
                "oc",
                "port-forward",
                f"deployment/{self._deployment_name(role)}",
                f"{spec.http_port}:{spec.http_port}",
                "-n",
                self._namespace,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if role == "producer":
            self._producer_pf = proc
        else:
            self._consumer_pf = proc

    def _start_pf_watchdog(self) -> None:
        def _watchdog() -> None:
            while not self._pf_watchdog_stop.wait(5.0):
                for role, attr in (
                    ("producer", "_producer_pf"),
                    ("consumer", "_consumer_pf"),
                ):
                    proc = getattr(self, attr)
                    if proc is not None and proc.poll() is not None:
                        print(
                            f"[watchdog] port-forward for {role} died; restarting",
                            file=sys.stderr,
                        )
                        self._start_port_forward(role)

        self._pf_watchdog_t = threading.Thread(
            target=_watchdog, daemon=True, name="pf-watchdog"
        )
        self._pf_watchdog_t.start()

    def _wait_vllm_ready(self, spec: ServerSpec, timeout_s: int) -> None:
        """Poll the health endpoint via `oc exec` — no port-forward needed."""
        deployment = self._deployment_name(spec.role)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    "oc",
                    "exec",
                    "-n",
                    self._namespace,
                    f"deployment/{deployment}",
                    "--",
                    "curl",
                    "-sf",
                    f"http://localhost:{spec.http_port}/health",
                ],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0:
                return
            time.sleep(5)
        raise TimeoutError(f"{spec.role} did not become healthy within {timeout_s}s")

    def _stop_background_procs(self) -> None:
        procs = [
            self._consumer_pf,
            self._producer_pf,
            self._consumer_logs,
            self._producer_logs,
        ]
        for proc in procs:
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh = getattr(proc, "_log_fh", None)
            if fh is not None:
                fh.close()
