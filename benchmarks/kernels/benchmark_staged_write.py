# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B full apply_write timing, including host staging copy and H2D allocation.

Each sample stages inputs before the timed region, calls apply_write, and waits
for that operation to finish. No CUDA graph: replay would omit the host work
under study and could reuse UVA slots before their GPU readers have retired.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.buffer_utils import GrowableUvaBufferPool, StagedWriteTensor


def make_payloads(writes, total):
    lengths = [total // writes + int(i < total % writes) for i in range(writes)]
    if writes > 1:
        shift = lengths[0] // 2
        lengths[0] -= shift
        lengths[-1] += shift
    return [list(range(n)) for n in lengths]


def stage(target, payloads):
    for row, payload in enumerate(payloads):
        target.stage_write(row, 3, payload)


def reset_contents(targets, previous_payloads):
    # All previous samples have completed. Reset only the benchmark's source pool;
    # target and metadata allocations stay warm. Allocator caches are not flushed.
    targets[1].write_contents = GrowableUvaBufferPool(torch.int32, max_concurrency=2)
    if previous_payloads is not None:
        for target in targets.values():
            for _ in range(2):
                stage(target, previous_payloads)
                target.apply_write()
                torch.accelerator.synchronize()


def next_capacity(target):
    pool = target.write_contents
    slot = (pool._curr + 1) % pool.max_concurrency
    buf = pool._uva_bufs[slot]
    return slot, 0 if buf is None else buf.cpu.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--modes", nargs="+", choices=["warm", "first", "growth"], default=["warm"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to exercise both pool slots")
    torch.accelerator.set_device_index(0)
    results = []
    for writes in (1, 16, 64):
        cases = [
            (mode, total)
            for mode in args.modes
            for total in (
                (1025, 16385, 262145) if mode == "growth" else (64, 1024, 16384, 262144)
            )
        ]
        for mode, total in cases:
            payloads = make_payloads(writes, total)
            lengths = [len(p) for p in payloads]
            previous_payloads = (
                make_payloads(writes, total - 1) if mode == "growth" else None
            )
            width = max(lengths) + 8
            targets = {}
            for enabled in (0, 1):
                os.environ["VLLM_STAGED_WRITE_USE_UVA_CONTENTS"] = str(enabled)
                targets[enabled] = StagedWriteTensor(
                    (writes, width),
                    torch.int32,
                    torch.device("cuda:0"),
                    max_concurrency=2,
                    uva_instead_of_gpu=True,
                )
                assert (targets[enabled].write_contents is not None) == bool(enabled)

            expected = torch.zeros((writes, width), dtype=torch.int32)
            for row, payload in enumerate(payloads):
                expected[row, 3 : 3 + len(payload)] = torch.tensor(payload)
            for target in targets.values():
                for _ in range(10):
                    stage(target, payloads)
                    target.apply_write()
                    torch.accelerator.synchronize()
                torch.testing.assert_close(target.gpu.cpu(), expected, rtol=0, atol=0)

            # Check the exact allocation/growth transition before timing it.
            if mode != "warm":
                reset_contents(targets, previous_payloads)
                for _ in range(2):
                    for target in targets.values():
                        stage(target, payloads)
                        target.apply_write()
                        torch.accelerator.synchronize()
                        torch.testing.assert_close(
                            target.gpu.cpu(), expected, rtol=0, atol=0
                        )

            samples = {0: [], 1: []}
            # Alternate A/B order to distribute clock and thermal drift.
            for iteration in range(args.repeats):
                if mode != "warm" and iteration % 2 == 0:
                    reset_contents(targets, previous_payloads)
                for enabled in (0, 1) if iteration % 2 == 0 else (1, 0):
                    target = targets[enabled]
                    if enabled:
                        slot, capacity_before = next_capacity(target)
                        if mode == "first":
                            assert capacity_before == 0
                        elif mode == "growth":
                            assert capacity_before == total - 1
                        else:
                            assert capacity_before >= total
                    stage(target, payloads)
                    torch.accelerator.synchronize()
                    start = time.perf_counter_ns()
                    target.apply_write()
                    submitted = time.perf_counter_ns()
                    torch.accelerator.synchronize()
                    done = time.perf_counter_ns()
                    sample = {
                        "submit_us": (submitted - start) / 1000,
                        "complete_us": (done - start) / 1000,
                    }
                    if enabled:
                        capacity_after = target.write_contents._uva_bufs[
                            slot
                        ].cpu.numel()
                        assert capacity_after == 1 << (total - 1).bit_length()
                        sample.update(
                            slot=slot,
                            capacity_before=capacity_before,
                            capacity_after=capacity_after,
                        )
                    samples[enabled].append(sample)
            for target in targets.values():
                torch.testing.assert_close(target.gpu.cpu(), expected, rtol=0, atol=0)
            summary = {}
            for enabled, name in ((0, "h2d"), (1, "uva")):
                summary[name] = {
                    metric: statistics.median(s[metric] for s in samples[enabled])
                    for metric in ("submit_us", "complete_us")
                }
                ordered = sorted(s["complete_us"] for s in samples[enabled])
                summary[name]["complete_p95_us"] = ordered[
                    (95 * len(ordered) + 99) // 100 - 1
                ]
            ratio = summary["h2d"]["complete_us"] / summary["uva"]["complete_us"]
            results.append(
                {
                    "mode": mode,
                    "writes": writes,
                    "elements": total,
                    "dtype": "int32",
                    "lengths": lengths,
                    "summary": summary,
                    "speedup": ratio,
                    "samples": samples,
                }
            )
            print(
                f"mode={mode:6} writes={writes:2} elements={total:6} "
                f"H2D={summary['h2d']['complete_us']:.2f}us "
                f"UVA={summary['uva']['complete_us']:.2f}us "
                f"speedup={ratio:.3f}x",
                flush=True,
            )
    report = {
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "source_file": __file__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "buffer_utils_file": buffer_utils.__file__,
        "buffer_utils_sha256": hashlib.sha256(
            Path(buffer_utils.__file__).read_bytes()
        ).hexdigest(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "boundary": (
            "pre-synchronize outside timer; apply_write + post-synchronize inside; "
            "stage_write and target/metadata construction excluded"
        ),
        "modes": args.modes,
        "allocation_note": (
            "first: fresh contents slot; growth: each slot primed at total-1; "
            "warm: reuse. JIT and allocator caches warmed in all modes. "
            "A=original H2D; B=current growable UVA contents with empty. "
            "No isolated empty-vs-zeros comparison."
        ),
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
