# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus /metrics scraper for vLLM performance benchmarking.

Polls vLLM's /metrics endpoint and parses histogram, counter, and gauge
metrics to produce latency percentiles (p50, p95, p99), throughput, and
resource utilization summaries.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import regex as re
import requests


@dataclass
class HistogramData:
    """Parsed Prometheus histogram: bucket boundaries, sum, and count."""

    buckets: list[tuple[float, float]] = field(
        default_factory=list
    )  # (le, cumulative_count)
    sum_val: float = 0.0
    count_val: float = 0.0

    def percentile(self, p: float) -> float | None:
        """Estimate the p-th percentile (p in [0, 1]) via linear interpolation."""
        if not self.buckets or self.count_val == 0:
            return None
        target = p * self.count_val
        prev_bound = 0.0
        prev_count = 0.0
        for le, cumulative_count in self.buckets:
            if cumulative_count >= target:
                bucket_count = cumulative_count - prev_count
                if bucket_count == 0:
                    return le
                fraction = (target - prev_count) / bucket_count
                return prev_bound + fraction * (le - prev_bound)
            prev_bound = le
            prev_count = cumulative_count
        return self.buckets[-1][0] if self.buckets else None

    @property
    def mean(self) -> float | None:
        if self.count_val == 0:
            return None
        return self.sum_val / self.count_val


class VLLMMetricsCollector:
    """Polls vLLM's /metrics endpoint on a background thread and aggregates
    Prometheus-format statistics for percentile computation.

    Usage::

        collector = VLLMMetricsCollector("http://localhost:8000")
        collector.start()
        # ... run workload ...
        collector.stop()
        summary = collector.get_summary()
    """

    HISTOGRAM_METRICS: dict[str, str] = {
        "vllm:time_to_first_token_seconds": "ttft",
        "vllm:time_per_output_token_seconds": "tpot",
        "vllm:e2e_request_latency_seconds": "e2e_latency",
    }

    COUNTER_METRICS: dict[str, str] = {
        "vllm:prompt_tokens_total": "prompt_tokens_total",
        "vllm:generation_tokens_total": "generation_tokens_total",
        "vllm:request_success_total": "request_success_total",
    }

    GAUGE_METRICS: dict[str, str] = {
        "vllm:num_requests_running": "running_requests",
        "vllm:num_requests_waiting": "pending_requests",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage",
    }

    def __init__(self, base_url: str, poll_interval: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.snapshots: list[dict] = []
        self._peak_gauges: dict[str, float] = defaultdict(float)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        # Final poll to capture all completed requests
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=5)
            if resp.status_code == 200:
                parsed = self._parse_all(resp.text)
                self.snapshots.append({"timestamp": time.time(), "parsed": parsed})
                for name, value in parsed.get("gauges", {}).items():
                    self._peak_gauges[name] = max(self._peak_gauges[name], value)
        except requests.exceptions.RequestException:
            pass

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                resp = requests.get(
                    f"{self.base_url}/metrics",
                    timeout=5,
                )
                if resp.status_code == 200:
                    parsed = self._parse_all(resp.text)
                    self.snapshots.append({"timestamp": time.time(), "parsed": parsed})
                    for name, value in parsed.get("gauges", {}).items():
                        self._peak_gauges[name] = max(self._peak_gauges[name], value)
            except requests.exceptions.RequestException:
                pass
            self._stop_event.wait(self.poll_interval)

    def _parse_all(self, text: str) -> dict:
        return {
            "histograms": self._parse_histograms(text),
            "counters": self._parse_counters(text),
            "gauges": self._parse_gauges(text),
        }

    def _parse_histograms(self, text: str) -> dict[str, HistogramData]:
        histograms: dict[str, HistogramData] = {}
        for prom_name, friendly_name in self.HISTOGRAM_METRICS.items():
            hist = HistogramData()

            bucket_pat = re.compile(
                rf'^{re.escape(prom_name)}_bucket\{{[^}}]*le="([^"]+)"[^}}]*\}}\s+([\d.e+\-]+)',
                re.MULTILINE,
            )
            buckets: list[tuple[float, float]] = []
            for m in bucket_pat.finditer(text):
                le_str, count_str = m.group(1), m.group(2)
                if le_str == "+Inf":
                    continue
                buckets.append((float(le_str), float(count_str)))
            hist.buckets = sorted(buckets, key=lambda x: x[0])

            sum_m = re.search(
                rf"^{re.escape(prom_name)}_sum(?:\{{[^}}]*\}})?\s+([\d.e+\-]+)",
                text,
                re.MULTILINE,
            )
            count_m = re.search(
                rf"^{re.escape(prom_name)}_count(?:\{{[^}}]*\}})?\s+([\d.e+\-]+)",
                text,
                re.MULTILINE,
            )
            if sum_m:
                hist.sum_val = float(sum_m.group(1))
            if count_m:
                hist.count_val = float(count_m.group(1))

            histograms[friendly_name] = hist
        return histograms

    def _parse_counters(self, text: str) -> dict[str, float]:
        counters: dict[str, float] = {}
        for prom_name, friendly_name in self.COUNTER_METRICS.items():
            matches = re.findall(
                rf"^{re.escape(prom_name)}(?:\{{[^}}]*\}})?\s+([\d.e+\-]+)",
                text,
                re.MULTILINE,
            )
            if matches:
                counters[friendly_name] = sum(float(v) for v in matches)
        return counters

    def _parse_gauges(self, text: str) -> dict[str, float]:
        gauges: dict[str, float] = {}
        for prom_name, friendly_name in self.GAUGE_METRICS.items():
            matches = re.findall(
                rf"^{re.escape(prom_name)}(?:\{{[^}}]*\}})?\s+([\d.e+\-]+)",
                text,
                re.MULTILINE,
            )
            if matches:
                gauges[friendly_name] = sum(float(v) for v in matches)
        return gauges

    def get_summary(self) -> dict:
        """Compute a performance report: latency percentiles, throughput,
        and peak resource usage."""
        if not self.snapshots:
            return {"error": "No metrics collected"}

        first = self.snapshots[0]
        last = self.snapshots[-1]
        duration = last["timestamp"] - first["timestamp"]

        first_counters = first["parsed"]["counters"]
        last_counters = last["parsed"]["counters"]
        last_histograms = last["parsed"]["histograms"]

        summary: dict = {
            "collection_duration_s": round(duration, 2),
            "total_requests": int(last_counters.get("request_success_total", 0)),
            "total_prompt_tokens": int(last_counters.get("prompt_tokens_total", 0)),
            "total_generation_tokens": int(
                last_counters.get("generation_tokens_total", 0)
            ),
        }

        for name, hist in last_histograms.items():
            summary[f"{name}_avg_s"] = _round(hist.mean)
            summary[f"{name}_p50_s"] = _round(hist.percentile(0.50))
            summary[f"{name}_p95_s"] = _round(hist.percentile(0.95))
            summary[f"{name}_p99_s"] = _round(hist.percentile(0.99))

        if duration > 0:
            gen_delta = last_counters.get(
                "generation_tokens_total", 0
            ) - first_counters.get("generation_tokens_total", 0)
            req_delta = last_counters.get(
                "request_success_total", 0
            ) - first_counters.get("request_success_total", 0)
            summary["throughput_tokens_per_s"] = round(gen_delta / duration, 2)
            summary["throughput_requests_per_s"] = round(req_delta / duration, 2)

        summary["peak_gpu_cache_usage"] = _round(
            self._peak_gauges.get("gpu_cache_usage", 0)
        )
        summary["peak_running_requests"] = int(
            self._peak_gauges.get("running_requests", 0)
        )
        summary["peak_pending_requests"] = int(
            self._peak_gauges.get("pending_requests", 0)
        )

        return {k: v for k, v in summary.items() if v is not None}


def _round(val: float | None, digits: int = 6) -> float | None:
    return round(val, digits) if val is not None else None
