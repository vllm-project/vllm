# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenShift harness for the CPU EC connector e2e test.

Manages two vllm Deployments (producer + consumer) in an OpenShift namespace,
each fronted by a Service and a Route. HTTP goes through the Route; assertions
read the structured event file out of the pod with `oc exec`. Pod logs are also
streamed to local files with `oc logs -f`, for humans reading a failure only.
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

# Where sitecustomize.py writes its structured events inside each pod.
EVENT_FILE = "/tmp/ec_events.jsonl"


# ---------------------------------------------------------------------------
# oc subprocess helpers
# ---------------------------------------------------------------------------


def _oc(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["oc"] + cmd, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(
            f"[oc error] cmd={cmd}\nstdout={result.stdout}\nstderr={result.stderr}",
            file=sys.stderr,
        )
        result.check_returncode()
    return result


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


def patch_manifests(
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
    ec_cpu_bytes: int,
    side_channel_port: int,
    producer: bool,
    different_nodes: bool,
) -> list[dict]:
    """Load a role's Deployment/Service/Route template and patch dynamic fields.

    Returns the patched documents (ready for yaml.dump_all + oc apply).
    """
    with template_path.open() as f:
        docs = list(yaml.safe_load_all(f))

    role = "producer" if producer else "consumer"
    name = f"vllm-ec-{role}-{run_id}"
    run_label = {"run-id": run_id}

    for d in docs:
        d["metadata"]["name"] = name
        d["metadata"]["namespace"] = namespace
        # Label the objects themselves, not just the pod template, so a run that
        # dies before teardown can be swept with
        # `oc delete deployment,service,route -l app=vllm-ec-test`.
        d["metadata"].setdefault("labels", {}).update(
            {"app": "vllm-ec-test", "role": role, "run-id": run_id}
        )

    by_kind = {d["kind"]: d for d in docs}
    doc = by_kind["Deployment"]
    doc["spec"]["selector"]["matchLabels"].update(run_label)
    doc["spec"]["template"]["metadata"]["labels"].update(run_label)
    by_kind["Service"]["spec"]["selector"].update(run_label)
    by_kind["Route"]["spec"]["to"]["name"] = name

    container = doc["spec"]["template"]["spec"]["containers"][0]
    container["image"] = image

    ec_config = json.dumps(
        {
            "ec_connector": "ECCPUConnector",
            "ec_role": ec_role,
            "engine_id": engine_id,
            "ec_enable_nixl": True,
            "ec_connector_extra_config": {"ec_cpu_bytes": ec_cpu_bytes},
        }
    )

    _set_env(container, "VLLM_MODEL", model)
    _set_env(container, "VLLM_PORT", str(port))
    _set_env(container, "GPU_MEMORY_UTILIZATION", str(gpu_memory_utilization))
    _set_env(container, "EC_TRANSFER_CONFIG", ec_config)
    _set_env(container, "VLLM_EC_SIDE_CHANNEL_PORT", str(side_channel_port))
    _set_env(container, "EC_TEST_EVENT_FILE", EVENT_FILE)

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

    return docs


# ---------------------------------------------------------------------------
# K8sHarness
# ---------------------------------------------------------------------------


class K8sHarness:
    """Context manager that manages producer+consumer vllm Deployments in OpenShift.

    Interface mirrors LocalHarness: exposes .producer, .consumer, .model, and
    restart_producer(), and fills in each spec's base_url (its Route) and
    events (its in-pod event file) so the shared test functions are agnostic
    about where the servers run.
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
        keep_on_exit: bool = False,
    ):
        self.producer = producer
        self.consumer = consumer
        self.model = model
        self._namespace = namespace
        self._image = image
        self._k8s_dir = k8s_dir
        self._different_nodes = different_nodes
        self.keep_on_exit = keep_on_exit
        self._run_id = time.strftime("%Y%m%d-%H%M%S")

        self._producer_logs: subprocess.Popen | None = None
        self._consumer_logs: subprocess.Popen | None = None
        self._log_watchdog_stop = threading.Event()
        self._log_watchdog_t: threading.Thread | None = None

        from shared import OcExecEventLog, disable_tls_verify

        # Routes are served with the cluster's default wildcard certificate.
        disable_tls_verify()
        for spec in (self.producer, self.consumer):
            spec.events = OcExecEventLog(
                namespace, self._deployment_name(spec.role), EVENT_FILE
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> K8sHarness:
        # Python skips __exit__ when __enter__ raises, so anything already
        # applied has to be torn down here or it leaks — and a leaked pod holds
        # its GPU, which makes the *next* run fail to schedule for an unrelated-
        # looking reason.
        try:
            return self._setup()
        except BaseException:
            print(
                "\n[k8s-setup] setup failed; tearing down partial state",
                file=sys.stderr,
            )
            try:
                self._teardown()
            except Exception:
                print("[k8s-setup] teardown after failure also failed", file=sys.stderr)
            raise

    def _setup(self) -> K8sHarness:
        for spec in (self.producer, self.consumer):
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._preflight()
        print(f"[k8s-setup] run_id={self._run_id}, namespace={self._namespace}")
        self._create_configmap()
        self._apply_role("producer")
        self._apply_role("consumer")
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

        for spec in (self.producer, self.consumer):
            spec.base_url = self._route_url(spec.role)
            print(f"[k8s-setup] {spec.role} route: {spec.base_url}")

        print("[k8s-setup] waiting on /health for both (via oc exec)…")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs2 = {
                ex.submit(self._wait_vllm_ready, spec, HEALTH_TIMEOUT_S): spec.role
                for spec in (self.producer, self.consumer)
            }
            for fut in futs2:
                fut.result()
                role = futs2[fut]
                print(f"  ✓ {role} healthy in-pod")

        # The pods are up, but the router needs a moment to pick up the Routes.
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs3 = {
                ex.submit(self._wait_route_ready, spec): spec.role
                for spec in (self.producer, self.consumer)
            }
            for fut in futs3:
                fut.result()
                print(f"  ✓ {futs3[fut]} reachable via route")

        self._start_log_watchdog()
        return self

    def __exit__(self, *_exc) -> None:
        if self.keep_on_exit:
            print("\n[k8s-teardown] --keep-servers set; leaving deployments running.")
            return
        self._teardown()

    def _teardown(self) -> None:
        self._log_watchdog_stop.set()
        if self._log_watchdog_t is not None:
            self._log_watchdog_t.join(timeout=6)
        self._stop_background_procs()
        print("\n[k8s-teardown] deleting deployments, services, routes, configmap…")
        self._delete_role("producer")
        self._delete_role("consumer")
        self._delete_configmap()

    # ------------------------------------------------------------------
    # Producer restart
    # ------------------------------------------------------------------

    def restart_producer(self) -> None:
        print("\n[k8s-restart] restarting producer deployment…")
        proc = self._producer_logs
        if proc is not None:
            if proc.poll() is None:
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
        # The Route and Service survive a rollout untouched (the Service selects
        # on labels, which the new pod carries), so base_url still holds.
        self._wait_route_ready(self.producer)
        print(f"  ✓ producer healthy at {self.producer.base_url}")

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

    def _apply_role(self, role: str) -> None:
        spec = self.producer if role == "producer" else self.consumer
        template_path = self._k8s_dir / f"{role}-deployment.yaml"
        patched = patch_manifests(
            template_path,
            run_id=self._run_id,
            namespace=self._namespace,
            image=self._image,
            model=self.model,
            port=spec.http_port,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            ec_role=f"ec_{role}",
            engine_id=spec.engine_id,
            ec_cpu_bytes=spec.ec_cpu_bytes,
            side_channel_port=spec.side_channel_port,
            producer=(role == "producer"),
            different_nodes=self._different_nodes,
        )
        print(f"[k8s-setup] applying {self._deployment_name(role)}")
        _oc_stdin(["apply", "-f", "-", "-n", self._namespace], yaml.dump_all(patched))

    def _delete_role(self, role: str) -> None:
        _oc(
            [
                "delete",
                "deployment,service,route",
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

    def _start_log_stream(self, role: str, *, truncate: bool = True) -> None:
        spec = self.producer if role == "producer" else self.consumer
        log_fh = spec.log_path.open("wb" if truncate else "ab", buffering=0)
        cmd = [
            "oc",
            "logs",
            "-f",
            f"deployment/{self._deployment_name(role)}",
            "-n",
            self._namespace,
        ]
        if not truncate:
            # Reattaching to an existing file: without this, oc would replay the
            # pod's whole history and the file would contain the same events
            # twice. Lines emitted while the stream was down are lost instead,
            # which is fine — the event file, not this log, is what tests read.
            cmd.append("--tail=0")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        proc._log_fh = log_fh  # type: ignore[attr-defined]
        if role == "producer":
            self._producer_logs = proc
        else:
            self._consumer_logs = proc

    def _preflight(self) -> None:
        """Fail fast on an expired login, before spending minutes on a rollout.

        An expired token otherwise surfaces much later and much less obviously:
        `oc exec` starts failing while an already-established `oc logs -f`
        stream keeps working, which reads as a hung server rather than an auth
        problem.
        """
        who = _oc(["whoami"], check=False)
        if who.returncode != 0:
            raise RuntimeError(
                "oc is not authenticated (`oc whoami` failed): "
                f"{who.stderr.strip()[:200]}\nRun `oc login` and retry."
            )
        ns = _oc(
            ["auth", "can-i", "create", "pods/exec", "-n", self._namespace], check=False
        )
        if ns.returncode != 0:
            print(
                f"[k8s-setup] warning: cannot create pods/exec in "
                f"{self._namespace} — event reads and health probes will fail "
                f"({ns.stdout.strip() or ns.stderr.strip()})",
                file=sys.stderr,
            )
        print(f"[k8s-setup] authenticated as {who.stdout.strip()}")

        # Deployments left over from an earlier run still hold their GPUs, which
        # shows up here as a pod that never schedules rather than as a leak.
        stale = _oc(
            [
                "get",
                "deployment",
                "-l",
                "app=vllm-ec-test",
                "-n",
                self._namespace,
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ],
            check=False,
        )
        leftovers = stale.stdout.split()
        if leftovers:
            print(
                f"[k8s-setup] warning: {len(leftovers)} leftover EC test "
                f"deployment(s) still running and holding GPUs: "
                f"{', '.join(leftovers)}\n"
                f"  sweep with: oc delete deployment,service,route "
                f"-l app=vllm-ec-test -n {self._namespace}",
                file=sys.stderr,
            )

    def _route_url(self, role: str, timeout_s: int = 60) -> str:
        """Return the Route's external base URL, waiting for the host to be set."""
        name = self._deployment_name(role)
        deadline = time.monotonic() + timeout_s
        while True:
            result = _oc(
                [
                    "get",
                    "route",
                    name,
                    "-n",
                    self._namespace,
                    "-o",
                    "jsonpath={.spec.host}",
                ],
                check=False,
            )
            host = result.stdout.strip()
            if host:
                return f"https://{host}"
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"route {name} had no host assigned within {timeout_s}s"
                )
            time.sleep(1.0)

    def _wait_route_ready(self, spec: ServerSpec, timeout_s: int = 120) -> None:
        """Wait until /health answers through the Route.

        The pod is already known healthy via `oc exec` at this point; this only
        covers the router noticing the Route.
        """
        from shared import wait_for_health

        wait_for_health(spec.base_url, None, timeout_s)

    def _start_log_watchdog(self) -> None:
        def _watchdog() -> None:
            while not self._log_watchdog_stop.wait(1.0):
                for role, attr in (
                    ("producer", "_producer_logs"),
                    ("consumer", "_consumer_logs"),
                ):
                    proc = getattr(self, attr)
                    if proc is not None and proc.poll() is not None:
                        print(
                            f"[watchdog] log stream for {role} died; restarting",
                            file=sys.stderr,
                        )
                        self._start_log_stream(role, truncate=False)

        self._log_watchdog_t = threading.Thread(
            target=_watchdog, daemon=True, name="log-watchdog"
        )
        self._log_watchdog_t.start()

    def _wait_vllm_ready(self, spec: ServerSpec, timeout_s: int) -> None:
        """Poll the health endpoint from inside the pod, before the Route exists."""
        deployment = self._deployment_name(spec.role)
        deadline = time.monotonic() + timeout_s
        argv = [
            "oc",
            "exec",
            "-n",
            self._namespace,
            f"deployment/{deployment}",
            "--",
            "curl",
            "-sf",
            f"http://localhost:{spec.http_port}/health",
        ]
        attempts = 0
        last = "never ran"
        while time.monotonic() < deadline:
            attempts += 1
            try:
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=15
                )
            except subprocess.TimeoutExpired:
                # `oc exec` itself wedged — a stuck attempt is a symptom worth
                # reporting, not an error to abort the whole wait on.
                last = "oc exec timed out after 15s"
            else:
                if result.returncode == 0:
                    return
                last = (
                    f"rc={result.returncode} "
                    f"stderr={result.stderr.strip()[:400]!r} "
                    f"stdout={result.stdout.strip()[:200]!r}"
                )
            time.sleep(5)
        # Report why it never succeeded. Without this the probe discards every
        # failure and a timeout says nothing about whether the server was down,
        # curl was missing, or `oc exec` could not reach the pod at all.
        raise TimeoutError(
            f"{spec.role} did not become healthy within {timeout_s}s "
            f"({attempts} attempts)\n  probe: {' '.join(argv)}\n  last failure: {last}"
        )

    def _stop_background_procs(self) -> None:
        procs = [
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
