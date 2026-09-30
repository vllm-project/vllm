# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Best-effort, container-visible NVIDIA device samples for command timelines."""

from __future__ import annotations

import csv
import ctypes
import importlib.util
import io
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid as uuid_module
from pathlib import Path

from ci_otel import Span, export_spans, record_spans

INTERVAL_SECONDS = 1.0
BATCH_SECONDS = 30
MAX_DEVICES = 16
QUERY_TIMEOUT_SECONDS = 2
QUERY = "index,uuid,name,utilization.gpu,memory.used,memory.total,mig.mode.current"


def _number(value: str, scale: int = 1, maximum: int | None = None) -> int | None:
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return None
        if maximum is not None and number > maximum:
            return None
        return round(number * scale)
    except ValueError:
        return None


def parse_samples(output: str, timestamp_ns: int) -> list[dict]:
    events = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 7 or len(events) >= MAX_DEVICES:
            continue
        index, uuid, name, util, used, total, mig = (value.strip() for value in row)
        if not index.isdigit() or not uuid.startswith("GPU-"):
            continue
        attributes: dict[str, str | int | bool] = {
            "gpu.uuid": uuid,
            "gpu.index": int(index),
            "gpu.name": name,
        }
        # Parent-device memory on a MIG GPU is not the job's allocated memory.
        if mig.lower() == "enabled":
            attributes["gpu.status"] = "unsupported_mig"
        else:
            for key, value in (
                ("gpu.utilization", _number(util, maximum=100)),
                ("gpu.memory.used", _number(used, 1024**2)),
                ("gpu.memory.total", _number(total, 1024**2)),
            ):
                if value is not None:
                    attributes[key] = value
        events.append(
            {"time_ns": timestamp_ns, "name": "ci.gpu.sample", "attributes": attributes}
        )
    return events


def visible_mig_devices() -> list[str]:
    """Resolve explicit MIG assignments without enumerating sibling instances."""
    nvidia = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").split(",")
    nvidia = [value.strip() for value in nvidia]
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = nvidia if cuda is None else [value.strip() for value in cuda.split(",")]
    if devices and all(value.isdigit() for value in devices):
        if not all(value.startswith("MIG-") for value in nvidia):
            return []
        try:
            devices = [nvidia[int(value)] for value in devices]
        except IndexError:
            return []
    if not devices or not all(value.startswith("MIG-") for value in devices):
        return []
    return list(dict.fromkeys(devices))[:MAX_DEVICES]


def load_nvml():
    """Load the standalone bundled binding without importing vLLM or Torch."""
    # Test images install vLLM as a wheel and keep only tests/helpers in the
    # checkout. Finding the top-level package does not execute its __init__.
    package = importlib.util.find_spec("vllm")
    if package is None or package.origin is None:
        raise ImportError("Installed vLLM package unavailable")
    path = Path(package.origin).parent / "third_party/pynvml.py"
    spec = importlib.util.spec_from_file_location("ci_nvml", path)
    if spec is None or spec.loader is None:
        raise ImportError("Bundled NVML binding unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cuda_visible_mig_devices() -> list[str]:
    """Discover CDI assignments in a subprocess without creating CUDA contexts."""
    pynvml = load_nvml()

    driver = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count)):
        return []
    devices = []
    pynvml.nvmlInit()
    try:
        for index in range(min(count.value, MAX_DEVICES)):
            device = ctypes.c_int()
            identifier = (ctypes.c_ubyte * 16)()
            if driver.cuDeviceGet(ctypes.byref(device), index):
                continue
            # The v2 entry point returns the compute instance UUID under MIG.
            if driver.cuDeviceGetUuid_v2(ctypes.byref(identifier), device):
                continue
            uuid = "MIG-" + str(uuid_module.UUID(bytes=bytes(identifier)))
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                    devices.append(uuid)
            except pynvml.NVMLError:
                continue
    finally:
        pynvml.nvmlShutdown()
    return devices


def query_mig_samples(devices: list[str], timestamp_ns: int) -> list[dict]:
    # Use the binding already shipped with vLLM; never initialize CUDA/Torch.
    pynvml = load_nvml()

    events = []
    pynvml.nvmlInit()
    try:
        for index, uuid in enumerate(devices[:MAX_DEVICES]):
            if not uuid.startswith("MIG-"):
                continue
            attributes: dict[str, str | int | bool] = {
                "gpu.uuid": uuid,
                "gpu.index": index,
                "gpu.name": "NVIDIA MIG",
                "gpu.status": "unavailable_mig_memory",
            }
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                if not pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                    continue
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used = _number(str(memory.used))
                total = _number(str(memory.total))
                if used is not None and total and used <= total:
                    attributes.update(
                        {
                            "gpu.memory.used": used,
                            "gpu.memory.total": total,
                            "gpu.status": "mig_memory_only",
                        }
                    )
                # NVML does not support per-MIG utilization on these devices.
            except pynvml.NVMLError:
                pass
            events.append(
                {
                    "time_ns": timestamp_ns,
                    "name": "ci.gpu.sample",
                    "attributes": attributes,
                }
            )
    finally:
        pynvml.nvmlShutdown()
    return events


def sample_span(events: list[dict]) -> Span:
    return Span(
        trace_id=os.environ["CI_INFRA_TRACE_ID"],
        span_id=secrets.token_hex(8),
        parent_span_id=os.environ["CI_INFRA_COMMAND_SPAN_ID"],
        name="ci.gpu.samples",
        start_ns=events[0]["time_ns"],
        end_ns=events[-1]["time_ns"],
        attributes={
            "ci.span.kind": "gpu-samples",
            "gpu.sample.interval_ms": int(INTERVAL_SECONDS * 1000),
            "gpu.scope": "container-visible-devices",
        },
        events=events,
    )


def collect(parent_pid: int, stop: threading.Event) -> None:
    events: list[dict] = []
    batch_started = time.monotonic()
    failures = 0
    discovery_attempted = False
    mig_devices = visible_mig_devices()
    query = (
        [sys.executable, __file__, "--query-mig", *mig_devices]
        if mig_devices
        else [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        while not stop.is_set() and os.getppid() == parent_pid:
            started = time.monotonic()
            try:
                result = subprocess.run(
                    query,
                    capture_output=True,
                    text=True,
                    timeout=QUERY_TIMEOUT_SECONDS,
                    check=True,
                )
                if stop.is_set():
                    break
                samples = (
                    json.loads(result.stdout)
                    if mig_devices
                    else parse_samples(result.stdout, time.time_ns())
                )
                if (
                    not mig_devices
                    and not discovery_attempted
                    and any(
                        sample["attributes"].get("gpu.status") == "unsupported_mig"
                        for sample in samples
                    )
                ):
                    discovery_attempted = True
                    # CDI can assign a MIG device without exporting its UUID.
                    # Only CUDA-visible devices are inspected, never NVML siblings.
                    try:
                        discovered = subprocess.run(
                            [sys.executable, __file__, "--discover-mig"],
                            capture_output=True,
                            text=True,
                            timeout=QUERY_TIMEOUT_SECONDS,
                            check=True,
                        )
                        mig_devices = json.loads(discovered.stdout)
                    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                        mig_devices = []
                    if mig_devices:
                        query = [sys.executable, __file__, "--query-mig", *mig_devices]
                        continue  # Do not create an extra parent-device series.
                if not samples:
                    break
                events.extend(samples)
                failures = 0
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                failures += 1
                if failures >= 3:
                    print(
                        "CI GPU sampling stopped: device query unavailable",
                        file=sys.stderr,
                    )
                    break
            if events and time.monotonic() - batch_started >= BATCH_SECONDS:
                # Bounded memory and upload time even if the receiver is down.
                # Dropped batches appear as gaps, never as zero utilization.
                export_spans([sample_span(events)], timeout_seconds=2)
                events = []
                batch_started = time.monotonic()
            stop.wait(max(0, INTERVAL_SECONDS - (time.monotonic() - started)))
    finally:
        # The shell joins this process before the ordinary job-end spool flush.
        if events:
            record_spans([sample_span(events)])


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        collect(int(sys.argv[1]), stop)
    except Exception:
        print("CI GPU sampling skipped", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--discover-mig":
        print(json.dumps(cuda_visible_mig_devices()))
    elif len(sys.argv) > 1 and sys.argv[1] == "--query-mig":
        print(json.dumps(query_mig_samples(sys.argv[2:], time.time_ns())))
    else:
        main()
