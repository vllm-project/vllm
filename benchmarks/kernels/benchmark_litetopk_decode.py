# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paged MQA + top-k chains using CUDA graph replay on SM100.

Both paths reuse their per-forward schedules as in the serving layer. The
baseline uses FP32 scores and the auto selector. These are per-layer kernel-chain
timings; they exclude shared metadata construction and are not model latency.
"""

import argparse
import json
import statistics

import torch

from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.litetopk_decode import (
    CANDIDATE_CAPACITY,
    get_litetopk_bf16_metadata,
    has_litetopk_decode,
    litetopk_bf16_scores,
    litetopk_select,
)
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


def measure(fn, repeats):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 160)
    return samples


@torch.inference_mode()
def run_case(fp4, requests, n, width, repeats):
    torch.manual_seed(37)
    rows = requests * n
    page, head_bytes, k = (128, 64, 512) if fp4 else (64, 128, 2048)
    pages = width // page
    if fp4:
        q = torch.randint(
            0, 256, (rows, 1, 32, head_bytes), device="cuda", dtype=torch.uint8
        ).view(torch.int8)
        sf = torch.full((rows, 1, 32), 0x7D7D7D7D, device="cuda", dtype=torch.int32)
        cache = torch.randint(
            0,
            256,
            (requests * pages, page * (head_bytes + 4)),
            device="cuda",
            dtype=torch.uint8,
        )
        cache[:, page * head_bytes :] = 125
    else:
        q = torch.randn((rows, 1, 32, head_bytes), device="cuda").to(
            torch.float8_e4m3fn
        )
        sf = None
        cache = torch.empty(
            (requests * pages, page * (head_bytes + 4)),
            device="cuda",
            dtype=torch.uint8,
        )
        cache[:, : page * head_bytes] = (
            torch.randn((requests * pages, page * head_bytes), device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        cache[:, page * head_bytes :] = torch.ones(
            (requests * pages, page), device="cuda"
        ).view(torch.uint8)
    cache = cache.view(-1, page, 1, head_bytes + 4)
    weights = torch.randn((rows, 32), device="cuda") / 8
    table = torch.randperm(requests * pages, device="cuda").int().view(requests, pages)
    table = table.repeat_interleave(n, 0).contiguous()
    ids = torch.arange(rows, device="cuda", dtype=torch.int32) // n
    lengths = (
        width - n + 1 + torch.arange(rows, device="cuda", dtype=torch.int32) % n
    ).view(-1, 1)
    schedule = get_paged_mqa_logits_metadata(lengths, page, get_num_sms(), indices=ids)
    bf16_schedule = get_litetopk_bf16_metadata(lengths, ids, n) if fp4 else None
    histogram = torch.zeros((rows, 1024), device="cuda", dtype=torch.int32)
    workspace = torch.zeros(
        rows * (16 + CANDIDATE_CAPACITY * 8), device="cuda", dtype=torch.uint8
    )
    output = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    selector = get_indexer_topk("auto")

    def baseline():
        scores = fp8_fp4_paged_mqa_logits(
            (q, sf),
            cache,
            weights,
            lengths,
            table,
            schedule,
            width,
            False,
            indices=ids,
        )
        selector(scores, lengths, 1, output, k, width)

    def lite():
        if fp4:
            scores = litetopk_bf16_scores(
                (q, sf),
                cache,
                weights,
                lengths,
                table,
                ids,
                width,
                n,
                histogram,
                schedule=bf16_schedule,
            )
        else:
            scores = fp8_fp4_paged_mqa_logits(
                (q, sf),
                cache,
                weights,
                lengths,
                table,
                schedule,
                width,
                False,
                indices=ids,
                histogram=histogram,
            )
        litetopk_select(scores, lengths, histogram, output, workspace)

    # Alternate the order between cases to reduce a systematic warmup bias.
    fns = (("baseline", baseline), ("lite", lite))
    if n % 2 == 0:
        fns = fns[::-1]
    times = {name: measure(fn, repeats) for name, fn in fns}
    medians = {name: statistics.median(values) for name, values in times.items()}
    return {
        "dtype": "bf16" if fp4 else "fp32",
        "requests": requests,
        "n": n,
        "width": width,
        "samples_us": times,
        "median_us": medians,
        "speedup": medians["baseline"] / medians["lite"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not has_litetopk_decode():
        raise RuntimeError("requires SM100, the native selector, and patched DeepGEMM")
    init_workspace_manager(torch.device("cuda"))
    try:
        for requests in args.requests:
            for width in args.widths:
                for fp4, ns in ((False, (1, 4)), (True, range(1, 7))):
                    for n in ns:
                        print(
                            json.dumps(run_case(fp4, requests, n, width, args.repeats)),
                            flush=True,
                        )
    finally:
        reset_workspace_manager()


if __name__ == "__main__":
    main()
