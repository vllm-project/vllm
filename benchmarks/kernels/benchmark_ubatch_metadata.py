# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare MRV2's fused DBO metadata preparation with its PyTorch fallback.

Run with ``python benchmarks/kernels/benchmark_ubatch_metadata.py --output out.json``.
Reports host enqueue and completion time for two microbatches, outside graph replay.
This measures metadata preparation, not model throughput or TPOT.
"""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.ubatch_utils import (
    _slice_query_and_seq_lens_kernel,
    _slice_query_start_loc,
    _slice_seq_lens,
    create_ubatch_slices,
)


def make_case(query_lens, padded=None, reqs_padded=None):
    nr, nt = len(query_lens), sum(query_lens)
    rpad = nr if reqs_padded is None else reqs_padded
    npad = nt if padded is None else padded
    buffers = InputBuffers(rpad, npad, torch.device("cuda"))
    batch = InputBatch.make_dummy(nr, nt, buffers)
    query_start_loc = np.zeros(rpad + 1, dtype=np.int32)
    np.cumsum(query_lens, out=query_start_loc[1 : nr + 1])
    query_start_loc[nr + 1 :] = nt
    buffers.query_start_loc.copy_(torch.from_numpy(query_start_loc))
    buffers.seq_lens.zero_()
    buffers.seq_lens[:nr] = torch.tensor(query_lens, device="cuda") + 37
    batch = replace(
        batch,
        num_reqs_after_padding=rpad,
        num_tokens_after_padding=npad,
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        query_start_loc_np=query_start_loc,
    )
    slices = create_ubatch_slices(batch, 2)
    outputs = [
        (
            torch.empty(
                sl.request_slice.stop - sl.request_slice.start + 1,
                dtype=torch.int32,
                device="cuda",
            ),
            torch.empty(
                sl.request_slice.stop - sl.request_slice.start,
                dtype=torch.int32,
                device="cuda",
            ),
        )
        for sl in slices
    ]

    def torch_metadata():
        for sl, (q, s) in zip(slices, outputs):
            _slice_query_start_loc(
                buffers.query_start_loc,
                q,
                sl.request_slice.start,
                sl.request_slice.stop,
                sl.token_slice.start,
                sl.num_tokens,
            )
            _slice_seq_lens(
                buffers.seq_lens,
                s,
                buffers.query_start_loc,
                sl.request_slice.start,
                sl.request_slice.stop,
                sl.token_slice.stop,
            )

    def fused_metadata():
        for sl, (q, s) in zip(slices, outputs):
            nr = sl.request_slice.stop - sl.request_slice.start
            _slice_query_and_seq_lens_kernel[(triton.cdiv(nr + 1, 128),)](
                buffers.query_start_loc,
                buffers.seq_lens,
                q,
                s,
                sl.request_slice.start,
                sl.request_slice.stop,
                sl.token_slice.start,
                sl.token_slice.stop,
                BLOCK_SIZE=128,
            )

    torch_metadata()
    expected = [(q.clone(), s.clone()) for q, s in outputs]
    fused_metadata()
    for actual, reference in zip(outputs, expected):
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b)
    return torch_metadata, fused_metadata


def measure(fn, repeats):
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    queued = time.perf_counter_ns()
    end.record()
    end.synchronize()
    done = time.perf_counter_ns()
    return {
        "host_enqueue_us": (queued - start) / repeats / 1000,
        "wall_us": (done - start) / repeats / 1000,
        "cuda_span_us": begin.elapsed_time(end) * 1000 / repeats,
    }


def gpu_operations(fn):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        fn()
        torch.accelerator.synchronize()
    return dict(
        Counter(
            e.name
            for e in profiler.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {f"decode_{n}": ([1] * n, None, None) for n in (32, 64, 128, 256, 512)}
    cases.update(
        {
            "decode_padded": ([1] * 44, 64, 64),
            "decode_odd": ([1] * 63, 63, None),
            "prefill_straddle": ([300, 500, 224], None, None),
            "prefill_aligned": ([256] * 4, None, None),
            "mixed": ([1] * 30 + [994], 1024, None),
            "eager_empty_tail": ([1] * 32, 128, None),
        }
    )
    source = Path(__file__).parents[2] / "vllm/v1/worker/gpu/ubatch_utils.py"
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": {},
    }
    for name, case in cases.items():
        functions = dict(zip(("torch", "fused"), make_case(*case)))
        samples = {label: [] for label in functions}
        for round_ in range(args.rounds):
            order = list(functions.items())
            if round_ % 2:
                order.reverse()
            for label, fn in order:
                samples[label].append(measure(fn, args.repeats))
        medians = {
            label: {k: statistics.median(x[k] for x in rows) for k in rows[0]}
            for label, rows in samples.items()
        }
        result["cases"][name] = {"medians": medians, "samples": samples}
        if name == "decode_256":
            result["cases"][name]["gpu_operations"] = {
                label: gpu_operations(fn) for label, fn in functions.items()
            }
        print(name, json.dumps(medians), flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
