# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recommend a single-node E+PD GPU split from measured worker rates."""

import argparse
import json
from fractions import Fraction
from math import isfinite
from pathlib import Path


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
        help="Measured saturation rate per single-GPU encoder in requests/s.",
    )
    parser.add_argument(
        "--pd-rps",
        type=positive_rate,
        help="Measured saturation rate per single-GPU PD worker in requests/s.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--measured-rates", type=Path, nargs=2, metavar=("E", "PD"))
    source.add_argument("--frozen-model", type=Path)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument(
        "--workload", type=Path, help="Paired workload shapes in JSON."
    )
    workload.add_argument(
        "--benchmark-result",
        nargs=2,
        metavar=("JSON", "MERGED_TOKENS"),
        help="Detailed benchmark results and one common merged-token count.",
    )
    args = parser.parse_args()
    if args.num_gpus < 2:
        parser.error("--num-gpus must be at least 2")
    if (args.frozen_model is not None) != bool(args.workload or args.benchmark_result):
        parser.error("--frozen-model requires --workload or --benchmark-result")
    if (args.measured_rates or args.frozen_model) and (
        args.encoder_rps is not None or args.pd_rps is not None
    ):
        parser.error("choose one source for stage rates")

    estimate = None
    if args.measured_rates or args.frozen_model:
        try:
            if args.measured_rates:
                rates = []
                for stage, path in zip(
                    ("encoder", "pd"), args.measured_rates, strict=True
                ):
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if result["stage"] != stage:
                        raise ValueError(f"{path} must contain the {stage} measurement")
                    rates.append(positive_rate(str(result["saturation_rps"])))
                args.encoder_rps, args.pd_rps = rates
            else:
                from epd_capacity.model import predict_rates

                if args.workload:
                    data = json.loads(args.workload.read_text(encoding="utf-8"))
                else:
                    path, tokens = args.benchmark_result
                    pd = json.loads(Path(path).read_text(encoding="utf-8"))
                    data = {
                        "encoder": [{"visual_tokens": int(tokens)}]
                        * len(pd["input_lens"]),
                        "pd": pd,
                    }
                frozen = json.loads(args.frozen_model.read_text(encoding="utf-8"))
                estimate = predict_rates(frozen, data)
                args.encoder_rps = positive_rate(
                    str(estimate["e_rate_upper_bound_rps"])
                )
                args.pd_rps = positive_rate(str(estimate["pd_rate_rps"]))
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            argparse.ArgumentTypeError,
        ) as e:
            parser.error(str(e))
    elif args.encoder_rps is None or args.pd_rps is None:
        parser.error("provide --measured-rates, --frozen-model, or both direct rates")
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
                **({"stage_rate_estimates": estimate} if estimate is not None else {}),
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
