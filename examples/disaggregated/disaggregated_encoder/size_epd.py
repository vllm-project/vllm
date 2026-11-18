# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recommend a single-node E+PD GPU split from measured worker rates."""

import argparse
import json
from fractions import Fraction
from math import isfinite


def positive_rate(value: str) -> Fraction:
    try:
        rate = Fraction(value)
        if rate > 0 and isfinite(float(rate)) and float(rate) > 0:
            return rate
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    raise argparse.ArgumentTypeError("expected a positive finite request rate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-gpus", type=int, required=True, help="Total GPU budget, at least two."
    )
    parser.add_argument(
        "--encoder-rps",
        type=positive_rate,
        required=True,
        help="Measured saturation rate per single-GPU encoder in requests/s.",
    )
    parser.add_argument(
        "--pd-rps",
        type=positive_rate,
        required=True,
        help="Measured saturation rate per single-GPU PD worker in requests/s.",
    )
    args = parser.parse_args()
    if args.num_gpus < 2:
        parser.error("--num-gpus must be at least 2")
    try:
        finite_capacity = isfinite(
            float(max(args.encoder_rps, args.pd_rps) * args.num_gpus)
        )
    except OverflowError:
        finite_capacity = False
    if not finite_capacity:
        parser.error("total worker capacity must be finite")

    candidates = []
    for num_encoders in range(1, args.num_gpus):
        num_pd = args.num_gpus - num_encoders
        encoder_capacity = num_encoders * args.encoder_rps
        pd_capacity = num_pd * args.pd_rps
        if encoder_capacity < pd_capacity:
            bottleneck = "encoder"
        elif pd_capacity < encoder_capacity:
            bottleneck = "pd"
        else:
            bottleneck = "both"
        candidates.append(
            (
                min(encoder_capacity, pd_capacity),
                {
                    "encoder_gpus": num_encoders,
                    "pd_gpus": num_pd,
                    "bottleneck": bottleneck,
                },
            )
        )

    capacity, recommendation = max(candidates, key=lambda candidate: candidate[0])
    print(
        json.dumps(
            {
                **recommendation,
                "encoder_rps": float(args.encoder_rps),
                "pd_rps": float(args.pd_rps),
                "predicted_capacity_rps": float(capacity),
                "candidate_allocations": [
                    {**allocation, "predicted_capacity_rps": float(rate)}
                    for rate, allocation in candidates
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
