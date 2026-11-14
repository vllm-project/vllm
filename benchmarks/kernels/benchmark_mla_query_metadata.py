# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare expanded and per-request metadata for uniform MLA query blocks."""

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Case:
    batch: int = 8
    query_len: int = 8
    heads: int = 64
    latent: int = 512
    rope: int = 64
    window: int = 8192
    seq_len: int = 131072
    fp8: bool = True
    causal: bool = True
    splits: int = 4
    page_size: int = 64
    page_gap: int = 0


def load_module(path: str | None, name: str):
    if path is None:
        from vllm.v1.attention.ops import triton_decode_attention

        return triton_decode_attention
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(case: Case):
    device = "cuda"
    lengths = torch.tensor(
        [max(case.query_len, case.seq_len - (i % 3) * 17) for i in range(case.batch)],
        device=device,
        dtype=torch.int32,
    )
    max_pages = math.ceil(case.seq_len / case.page_size) + 1
    recent_pages = (
        math.ceil(
            (min(case.seq_len, case.window or case.seq_len) + case.query_len)
            / case.page_size
        )
        + 4
    )
    physical_pages = case.batch * recent_pages * 2
    table = torch.full((case.batch, max_pages), -1, dtype=torch.int32, device=device)
    pages = torch.randperm(physical_pages, device=device)
    for i, length in enumerate(lengths.tolist()):
        begin = max(
            0, length - case.query_len + 1 - (case.window or length) - case.page_size
        )
        first = begin // case.page_size
        end = math.ceil((length + case.page_size) / case.page_size)
        assert end - first <= recent_pages
        table[i, first:end] = pages[i * recent_pages : i * recent_pages + end - first]
    width = case.latent + case.rope
    dtype = torch.float8_e4m3fn if case.fp8 else torch.bfloat16
    cache = torch.randn(
        physical_pages,
        case.page_size + case.page_gap,
        1,
        width,
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)[:, : case.page_size]
    rows = case.batch * case.query_len
    q = torch.randn(rows, case.heads, width, dtype=torch.bfloat16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    return q, cache, table, lengths, scale


def materialize_metadata(table, lengths, case):
    if case.query_len == 1:
        return table, lengths
    table = table.repeat_interleave(case.query_len, dim=0)
    if case.causal:
        offsets = torch.arange(
            1 - case.query_len, 1, device=lengths.device, dtype=lengths.dtype
        )
        lengths = (lengths[:, None] + offsets).flatten().clamp(min=0)
    else:
        lengths = lengths.repeat_interleave(case.query_len)
    return table, lengths


def make_call(module, case, inputs, expanded):
    q, cache, table, lengths, scale = inputs
    rows = q.shape[0]
    out = torch.empty(rows, case.heads, case.latent, device=q.device, dtype=q.dtype)
    lse = torch.empty(rows, case.heads, device=q.device, dtype=q.dtype)
    scratch = torch.empty(
        rows,
        case.heads,
        case.splits,
        case.latent + 1,
        device=q.device,
        dtype=torch.float32,
    )

    def call():
        active_table, active_lengths = table, lengths
        kwargs = {}
        if expanded:
            active_table, active_lengths = materialize_metadata(table, lengths, case)
        else:
            kwargs.update(query_len=case.query_len, causal=case.causal)
        module.decode_attention_fwd(
            q,
            cache,
            cache[..., : case.latent],
            out,
            lse,
            active_table,
            active_lengths,
            scratch,
            case.splits,
            0.0625,
            case.page_size,
            k_scale=scale,
            v_scale=scale,
            is_mla=True,
            sliding_window=case.window,
            **kwargs,
        )
        return out

    return call, out, lse


def measure_graph(call, repeats=7, iterations=100, unroll=8):
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            call()
    for _ in range(20):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (iterations * unroll))
    samples.sort()
    return dict(
        median_us=samples[len(samples) // 2], min_us=samples[0], max_us=samples[-1]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--baseline-source")
    parser.add_argument("--candidate-source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        [Case(**c) for c in json.loads(args.cases.read_text())]
        if args.cases
        else [Case()]
    )
    baseline = load_module(args.baseline_source, "mla_metadata_baseline")
    candidate = load_module(args.candidate_source, "mla_metadata_candidate")
    results = []
    for index, case in enumerate(cases):
        torch.manual_seed(0)
        inputs = make_inputs(case)
        old_call, old_out, old_lse = make_call(baseline, case, inputs, True)
        new_call, new_out, new_lse = make_call(candidate, case, inputs, False)
        old_call()
        new_call()
        assert torch.equal(old_out, new_out), "attention output changed"
        assert torch.equal(old_lse, new_lse), "attention LSE changed"
        old, new = measure_graph(old_call), measure_graph(new_call)
        row = dict(
            case=asdict(case),
            baseline=old,
            candidate=new,
            speedup=old["median_us"] / new["median_us"],
            bitwise_equal=True,
        )
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(dict(index=index, **row)), flush=True)
        del old_call, new_call, inputs, old_out, new_out, old_lse, new_lse
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
