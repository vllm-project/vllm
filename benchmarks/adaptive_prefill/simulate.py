# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic policy experiment. NOT a GPU or vLLM engine benchmark."""

import json
import math
from dataclasses import dataclass, field

from vllm.v1.core.sched.adaptive_prefill import AdaptivePrefillController


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))]


@dataclass
class Request:
    name: str
    arrival: float
    prompt: int
    output: int
    computed: int = 0
    emitted: int = 0
    times: list[float] = field(default_factory=list)


def run(adaptive, scenario="mixed"):
    controller = AdaptivePrefillController() if adaptive else None
    requests = []
    if scenario != "short_only":
        requests += [Request(f"long-{i}", i * 0.05, 16384, 8) for i in range(4)]
    if scenario != "long_only":
        requests += [Request(f"short-{i}", 0.1 + i * 0.02, 16, 64) for i in range(32)]
    now = 0.0
    steps = 0
    while any(r.emitted < r.output for r in requests):
        steps += 1
        if steps > 200000:
            raise RuntimeError("Policy made no progress")
        active = [r for r in requests if r.arrival <= now and r.emitted < r.output]
        if not active:
            now = min(r.arrival for r in requests if r.arrival > now)
            continue
        decode = [r for r in active if r.computed == r.prompt]
        prefill = [r for r in active if r.computed < r.prompt]
        b = None
        if controller:
            b = controller.begin(now, len(decode), 8192, {r.name for r in prefill})
            prefill.sort(key=lambda r: controller.last_progress[r.name])
        actual = {r.name: 1 for r in decode}
        contexts = {}
        tokens_left = 8192 - len(decode)
        units = 0.0
        finishing = []
        for r in prefill:
            n = min(1024, r.prompt - r.computed, tokens_left)
            if b:
                n = b.limit(r.name, n, r.computed)
            if n <= 0:
                continue
            contexts[r.name] = r.computed
            actual[r.name] = n
            units += n * (1 + r.computed / 8192)
            if b:
                b.admit(r.name, n, r.computed)
            tokens_left -= n
            r.computed += n
            if r.computed == r.prompt:
                finishing.append(r)
        # Explicit synthetic device model, intentionally distinct from bootstrap.
        device_ms = 0.5 + 0.18 * len(decode) + 0.006 * units
        if controller:
            controller.commit(steps, b, actual, contexts, len(decode), now)
        now += device_ms / 1000
        for r in decode + finishing:
            r.emitted += 1
            r.times.append(now)
        if controller:
            controller.complete(steps, device_ms)
    gaps = [
        (b - a) * 1000
        for r in requests
        if r.name.startswith("short")
        for a, b in zip(r.times, r.times[1:])
    ]
    return {
        "scenario": scenario,
        "policy": "adaptive" if adaptive else "baseline",
        "evidence": "synthetic_policy_simulation_not_gpu",
        "synthetic_cost": (
            "0.5 + 0.18*decode_reqs + 0.006*context_weighted_prefill_tokens ms"
        ),
        "decode_gap_p99_ms": percentile(gaps, 0.99),
        "short_ttft_p95_ms": percentile(
            [
                (r.times[0] - r.arrival) * 1000
                for r in requests
                if r.name.startswith("short")
            ],
            0.95,
        ),
        "long_ttft_max_ms": max(
            (
                (r.times[0] - r.arrival) * 1000
                for r in requests
                if r.name.startswith("long")
            ),
            default=None,
        ),
        "output_tokens_per_s": sum(r.emitted for r in requests) / now,
        "all_prefills_completed": all(r.computed == r.prompt for r in requests),
        "steps": steps,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            [
                run(policy, scenario)
                for scenario in ("short_only", "long_only", "mixed")
                for policy in (False, True)
            ],
            indent=2,
        )
    )
