# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU telemetry sampling for benchmark receipts.

Preferred backend: NVML via `pynvml` (higher sampling rate).
Fallback backend: `nvidia-smi` polling.

Hard rule: no fake outputs. If telemetry is unavailable, callers receive `None` fields.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnergySample:
    t_s: float
    power_w: float
    temp_c: float | None = None
    gpu_util_pct: float | None = None
    mem_util_pct: float | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None


def integrate_energy_j(samples: list[EnergySample], duration_s: float) -> float | None:
    """Trapezoid integration of power over [0, duration_s] => Joules."""
    if duration_s <= 0.0 or not samples:
        return None

    xs = sorted(samples, key=lambda s: float(s.t_s))
    if not xs:
        return None

    energy = 0.0

    # Left edge extrapolation (0 -> first sample)
    t_first = float(xs[0].t_s)
    p_first = float(xs[0].power_w)
    if t_first > 0.0:
        energy += p_first * min(t_first, duration_s)

    # Trapezoids
    for a, b in zip(xs, xs[1:]):
        ta = float(a.t_s)
        tb = float(b.t_s)
        if tb <= ta:
            continue
        if ta >= duration_s:
            break
        seg_a = max(0.0, ta)
        seg_b = min(duration_s, tb)
        if seg_b <= seg_a:
            continue
        dt = seg_b - seg_a
        energy += 0.5 * (float(a.power_w) + float(b.power_w)) * dt

    # Right edge extrapolation (last sample -> duration)
    t_last = float(xs[-1].t_s)
    p_last = float(xs[-1].power_w)
    if t_last < duration_s:
        energy += p_last * (duration_s - t_last)

    return float(energy)


def _run_cmd(cmd: list[str], timeout_s: float) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout_s:.1f}s"
    except FileNotFoundError as e:
        return 127, "", str(e)


def _nvidia_smi_sample(gpu_id: int) -> EnergySample | None:
    code, out, _err = _run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=3.0,
    )
    if code != 0 or not out:
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 6:
        return None
    try:
        return EnergySample(
            t_s=0.0,  # filled by sampler
            power_w=float(parts[0]),
            temp_c=float(parts[1]),
            gpu_util_pct=float(parts[2]),
            mem_util_pct=float(parts[3]),
            mem_used_mb=float(parts[4]),
            mem_total_mb=float(parts[5]),
        )
    except ValueError:
        return None


class EnergySampler:
    """Background sampler that collects power/util/temp while a workload runs."""

    def __init__(self, gpu_id: int, sample_interval_s: float):
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        self.gpu_id = int(gpu_id)
        self.sample_interval_s = float(sample_interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[EnergySample] = []
        self.backend: str = "none"
        self.metadata: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("EnergySampler already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            print(
                "Warning: EnergySampler thread did not terminate within 5 seconds.",
                file=sys.stderr,
            )
        self._thread = None

    def _run(self) -> None:
        t0 = time.perf_counter()

        # Prefer NVML for higher-rate sampling.
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self.backend = "nvml"

            with contextlib.suppress(pynvml.NVMLError):
                self.metadata["power_limit_w"] = (
                    float(pynvml.nvmlDeviceGetPowerManagementLimit(h)) / 1000.0
                )
            with contextlib.suppress(pynvml.NVMLError):
                self.metadata["name"] = str(pynvml.nvmlDeviceGetName(h))

            while not self._stop.is_set():
                t = time.perf_counter() - t0
                p_w = float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0

                try:
                    temp_c = float(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    )
                except Exception:
                    temp_c = None

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    gpu_util = float(util.gpu)
                    mem_util = float(util.memory)
                except Exception:
                    gpu_util = None
                    mem_util = None

                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mem_used_mb = float(mem.used) / (1024.0 * 1024.0)
                    mem_total_mb = float(mem.total) / (1024.0 * 1024.0)
                except Exception:
                    mem_used_mb = None
                    mem_total_mb = None

                self.samples.append(
                    EnergySample(
                        t_s=float(t),
                        power_w=float(p_w),
                        temp_c=temp_c,
                        gpu_util_pct=gpu_util,
                        mem_util_pct=mem_util,
                        mem_used_mb=mem_used_mb,
                        mem_total_mb=mem_total_mb,
                    )
                )
                time.sleep(self.sample_interval_s)
            return
        except ImportError:
            self.backend = "nvidia-smi"
        except Exception:
            self.backend = "nvidia-smi"

        # Fall back to nvidia-smi polling.
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            s = _nvidia_smi_sample(self.gpu_id)
            if s is not None:
                self.samples.append(
                    EnergySample(
                        t_s=float(t),
                        power_w=float(s.power_w),
                        temp_c=s.temp_c,
                        gpu_util_pct=s.gpu_util_pct,
                        mem_util_pct=s.mem_util_pct,
                        mem_used_mb=s.mem_used_mb,
                        mem_total_mb=s.mem_total_mb,
                    )
                )
            time.sleep(self.sample_interval_s)
