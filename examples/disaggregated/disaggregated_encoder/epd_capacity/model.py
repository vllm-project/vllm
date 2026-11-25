# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit a deployment model from profiles and estimate rates for a workload."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import fmean

from .encoder import fit_encoder, workload_rates


def pd_rate_bounds(workload, model):
    """Estimate PD rate bounds from KV occupancy and the scheduling budget."""
    requests = list(zip(workload["input_lens"], workload["output_lens"], strict=True))
    if (
        workload["failed"]
        or not requests
        or any(type(n) is not int or n <= 0 for pair in requests for n in pair)
    ):
        raise ValueError("PD requires completed requests with positive token counts")

    kv_tokens, chunk_tokens, step_seconds = (
        model["kv_tokens"],
        model["chunk_tokens"],
        model["step_seconds"],
    )
    scheduled_tokens = model.get("max_num_scheduled_tokens", chunk_tokens)
    if (
        type(kv_tokens) is not int
        or kv_tokens <= 0
        or type(chunk_tokens) is not int
        or chunk_tokens <= 0
        or type(scheduled_tokens) is not int
        or scheduled_tokens <= 0
        or not isfinite(step_seconds)
        or step_seconds <= 0
    ):
        raise ValueError(
            "PD KV, chunk, scheduling budget and step time must be positive"
        )
    largest = max(prompt + output for prompt, output in requests)
    if largest >= kv_tokens:
        raise ValueError("A request reaches the calibrated PD KV capacity")

    demands = []
    for prompt, output in requests:
        full_chunks = (prompt - 1) // chunk_tokens
        demands.append(
            chunk_tokens * full_chunks * (full_chunks + 1) // 2
            + prompt * (1 + output)
            + output * (output + 1) // 2
        )
    token_seconds = step_seconds * fmean(demands)
    # Prefill produces the first output token.
    mean_scheduled = fmean(prompt + output - 1 for prompt, output in requests)
    scheduler_rps = scheduled_tokens / (step_seconds * mean_scheduled)
    return (
        min((kv_tokens - largest) / token_seconds, scheduler_rps),
        min(kv_tokens / token_seconds, scheduler_rps),
    )


def predict_rates(calibration, workload):
    """Reuse frozen component fits for paired E and PD request shapes."""
    shapes = workload["encoder"]
    pd = workload["pd"]
    if len(shapes) != len(pd["input_lens"]):
        raise ValueError("E and PD must describe the same completed requests")
    pd_lower, pd_upper = pd_rate_bounds(pd, calibration["pd"])
    return {
        **workload_rates(calibration["encoder"], shapes, pd_lower),
        "pd_model_rps_interval": [pd_lower, pd_upper],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
        calibration = {**profiles, "encoder": fit_encoder(profiles["measurements"])}
        args.output.write_text(
            json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
