# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure host time submitting one layer's EPLB map updates.

Run this script against both revisions, using identical arguments. Preparation
time is reported separately from foreground commit time; neither is serving
throughput. NPU runs require torch_npu in the environment.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logical-experts", type=int, default=256)
    parser.add_argument("--physical-experts", type=int, default=288)
    parser.add_argument("--max-replicas", type=int, default=33)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    from vllm.distributed.eplb import eplb_state

    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type != "cpu":
        torch.accelerator.set_device_index(device.index or 0)

    physical = torch.arange(args.physical_experts) % args.logical_experts
    # Only the three destination maps are accessed by the commit operation.
    buffers = SimpleNamespace(
        physical_to_logical_map=torch.empty(
            1, args.physical_experts, dtype=torch.long, device=device
        ),
        logical_to_physical_map=torch.empty(
            1, args.logical_experts, args.max_replicas, dtype=torch.long, device=device
        ),
        logical_replica_count=torch.empty(
            1, args.logical_experts, dtype=torch.long, device=device
        ),
    )
    prepare = getattr(eplb_state, "_prepare_eplb_maps", None)
    prepared = (
        prepare(physical, args.logical_experts, args.max_replicas)
        if prepare is not None
        else None
    )

    def commit() -> None:
        if prepared is None:
            eplb_state._commit_eplb_maps_for_layer(buffers, physical, 0)
        else:
            eplb_state._commit_eplb_maps_for_layer(
                buffers, prepared[0], 0, prepared[1], prepared[2]
            )

    for _ in range(12):
        commit()
    rounds = []
    preparation = []
    for _ in range(args.repeats):
        samples = []
        for _ in range(args.iterations):
            if device.type != "cpu":
                torch.accelerator.synchronize()
            start = time.perf_counter_ns()
            commit()
            samples.append((time.perf_counter_ns() - start) / 1000)
        if device.type != "cpu":
            torch.accelerator.synchronize()
        rounds.append(statistics.median(samples))
        if prepare is not None:
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                prepare(physical, args.logical_experts, args.max_replicas)
            preparation.append(
                (time.perf_counter_ns() - start) / args.iterations / 1000
            )

    print(
        json.dumps(
            {
                **vars(args),
                "prepared": prepared is not None,
                "foreground_us": rounds,
                "foreground_median_us": statistics.median(rounds),
                "prepare_us": preparation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
