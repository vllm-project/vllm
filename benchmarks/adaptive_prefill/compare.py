# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two live runs only when workload and measurements are comparable."""

import argparse
import json
from pathlib import Path


def compare(baseline, adaptive):
    for run in (baseline, adaptive):
        if run["evidence"] != "live_streaming_endpoint" or run["failures"]:
            raise ValueError("Need two successful live runs, not simulation evidence")
        if not run["all_long_requests_completed"]:
            raise ValueError("Long-prefill completion failed")
    ignored = {"url", "label", "output"}
    a = {k: v for k, v in baseline["config"].items() if k not in ignored}
    b = {k: v for k, v in adaptive["config"].items() if k not in ignored}
    if a != b:
        raise ValueError("Workload/model configuration differs")
    metrics = (
        "decode_token_gap_p99_ms",
        "decode_stream_event_gap_p99_ms",
        "short_ttft_p95_ms",
        "long_ttft_p95_ms",
        "output_tokens_per_s",
    )
    result = {}
    for metric in metrics:
        x, y = baseline[metric], adaptive[metric]
        result[metric] = {
            "baseline": x,
            "adaptive": y,
            "change_percent": (y / x - 1) * 100 if x and y is not None else None,
        }
    result["note"] = (
        "Hardware, server commit/config and idle load must also match; verify manually."
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("adaptive")
    args = p.parse_args()
    print(
        json.dumps(
            compare(
                json.loads(Path(args.baseline).read_text()),
                json.loads(Path(args.adaptive).read_text()),
            ),
            indent=2,
        )
    )
