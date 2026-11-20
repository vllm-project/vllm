# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 HTTP serving smoke benchmark plugin."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any

from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.result import ExecutionStatus


def _percentile(sorted_vals: list[float], p: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


class HttpSmokeBenchmark(BenchmarkRunner):
    """Minimal HTTP benchmark exercising `/v1/completions` with excluded warmup."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concurrencies = params.get("concurrency", [1])
        if isinstance(concurrencies, int):
            concurrencies = [concurrencies]
        return [
            WorkloadPoint(
                point_id=f"c{int(c)}",
                params={**params, "concurrency": int(c)},
            )
            for c in concurrencies
        ]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext | DirectRunContext,
    ) -> PointResult:
        if not isinstance(ctx, ServerRunContext):
            raise TypeError("HttpSmokeBenchmark requires ServerRunContext.")

        p = point.params
        concurrency = int(p.get("concurrency", 1))
        num_requests = int(p.get("num_requests", 8))
        warmup_requests = int(p.get("warmup_requests", 2))
        max_tokens = int(p.get("max_tokens", 32))
        prompt = str(p.get("prompt", "Write a short sentence about vLLM:"))

        deadline = time.monotonic() + max(1, ctx.timeout_s)

        def _send_one(_idx: int) -> tuple[float | None, int, str | None, str | None]:
            body = json.dumps(
                {
                    "model": ctx.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{ctx.base_url}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            per_req_timeout = min(60.0, max(1.0, deadline - time.monotonic()))
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=per_req_timeout) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                lat = time.monotonic() - t0
                usage = parsed.get("usage") or {}
                out_toks = int(usage.get("completion_tokens", 0) or 0)
                return lat, out_toks, None, None
            except urllib.error.HTTPError as err:
                kind = "model" if 400 <= err.code < 500 else "infra"
                return None, 0, kind, f"HTTP {err.code}: {err.reason}"
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
            ) as err:
                return None, 0, "infra", f"{type(err).__name__}: {err}"
            except (ValueError, KeyError, TypeError) as err:
                return None, 0, "model", f"{type(err).__name__}: {err}"

        # Excluded warmup requests so cold-start latency does not skew metrics.
        for i in range(max(0, warmup_requests)):
            _send_one(i)

        t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, concurrency)
        ) as pool:
            outcomes = list(pool.map(_send_one, range(num_requests)))
        elapsed = max(1e-6, time.monotonic() - t_start)

        latencies: list[float] = []
        total_out_tokens = 0
        model_errors = 0
        infra_errors = 0
        first_err: str | None = None

        for lat, toks, err_kind, err_msg in outcomes:
            if lat is not None:
                latencies.append(lat)
                total_out_tokens += toks
            else:
                if first_err is None:
                    first_err = err_msg
                if err_kind == "model":
                    model_errors += 1
                else:
                    infra_errors += 1

        latencies.sort()
        n = len(latencies)
        metrics: dict[str, float | int | str | None] = {
            "throughput_req_s": n / elapsed,
            "output_tokens_per_s": total_out_tokens / elapsed,
            "mean_latency_s": (sum(latencies) / n) if n else None,
            "p50_latency_s": _percentile(latencies, 0.50) if n else None,
        }
        # Consistent percentile gating: require N >= ceil(10 / (1 - p)).
        if n >= 100:
            metrics["p90_latency_s"] = _percentile(latencies, 0.90)
        if n >= 1000:
            metrics["p99_latency_s"] = _percentile(latencies, 0.99)
        if n >= 10000:
            metrics["p99_9_latency_s"] = _percentile(latencies, 0.999)

        raw_path = ctx.output_dir / f"http_smoke_{point.point_id}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "point_id": point.point_id,
                    "warmup_requests": warmup_requests,
                    "attempted": num_requests,
                    "completed": n,
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        status = ExecutionStatus.COMPLETED if n > 0 else ExecutionStatus.FAILED
        return PointResult(
            status=status,
            metrics=metrics,
            attempted=num_requests,
            completed=n,
            model_errors=model_errors,
            infra_errors=infra_errors,
            error_message=first_err if n == 0 else None,
            artifacts=[raw_path],
        )
