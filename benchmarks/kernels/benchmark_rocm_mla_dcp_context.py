# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-GPU DCP gather/dequant/project comparison; includes real collectives."""

import argparse
import functools
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonPrefillMetadata,
    reorg_kvcache,
)
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import (
    context_row_indices,
    expand_context,
    gather_compressed_context,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-tokens", nargs="+", type=int, default=[128, 1024, 4096, 8192, 16384]
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8 and torch.accelerator.device_count() == 8
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    torch.manual_seed(101 + rank)
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    weight = torch.randn(3072, 512, dtype=torch.bfloat16, device="cuda") * 0.02
    for tokens in args.local_tokens:
        rows.append(run_case(tokens, rank, group, weight))
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "rows": rows}, indent=2) + "\n"
    )
    torch.accelerator.synchronize()
    dist.barrier(group=group)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


def run_case(tokens, rank, group, weight):
    # Two requests, uneven per-rank tails and a non-zero local chunk start.
    padded = [tokens, max(16, tokens // 2)]
    starts = [128, 0]
    lengths = [
        [starts[s] + n - (r * 3 % 13) for r in range(8)] for s, n in enumerate(padded)
    ]
    totals = [
        sum(length - starts[s] for length in row) for s, row in enumerate(lengths)
    ]
    total = sum(totals)
    local_n = sum(padded)
    block_size = 1536
    blocks_per_req = (max(padded) + 128 + block_size - 1) // block_size
    cache = torch.randn(
        2 * blocks_per_req, block_size, 576, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([0.03], device="cuda")
    cache = (cache.float() / scale).to(torch.float8_e4m3fn).view(torch.uint8)
    table = torch.arange(2 * blocks_per_req, device="cuda", dtype=torch.int32).view(
        2, -1
    )
    cu = torch.tensor(
        [0, *torch.tensor(padded).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    seq_cu = torch.tensor([0, totals[0], total], device="cuda", dtype=torch.int32)
    token_to_seq = torch.repeat_interleave(
        torch.arange(2, device="cuda", dtype=torch.int32),
        torch.tensor(padded, device="cuda"),
    )
    chunk = MLACommonPrefillMetadata.ContextChunk(
        index=0,
        request_slice=slice(0, 2),
        token_slice=slice(0, 2),
        continuation_token_end=2,
        is_continuation=False,
        num_context_tokens=total,
        query_start_loc=seq_cu,
        max_query_len=1,
        cu_seq_lens=seq_cu,
        starts=torch.tensor(starts, device="cuda", dtype=torch.int32),
        max_seq_len=max(totals),
        seq_lens=torch.tensor(totals),
        token_to_seq=token_to_seq,
        all_rows_active=True,
        padded_local_seq_lens=padded,
        local_context_lens_allranks=lengths,
        padded_local_cu_seq_lens=cu,
        padded_local_token_to_seq=token_to_seq,
        num_local_context_tokens=local_n,
        local_starts=starts,
    )
    indices = context_row_indices(chunk, torch.device("cuda"))
    workspace = torch.empty(9 * local_n, 576, device="cuda", dtype=torch.bfloat16)
    gather = functools.partial(dist.all_gather_into_tensor, group=dist.group.WORLD)

    def baseline():
        ops.gather_and_maybe_dequant_cache(
            cache,
            workspace,
            table,
            cu,
            token_to_seq,
            local_n,
            "fp8",
            scale,
            chunk.starts,
        )
        gathered = workspace[local_n:]
        gather(gathered, workspace[:local_n])
        latent, rope = gathered.unsqueeze(1).split([512, 64], dim=-1)
        latent, rope = reorg_kvcache(
            latent, rope, padded, lengths, starts, total, max(totals), local_n
        )
        kv = F.linear(latent, weight).view(total, 12, 256)
        k, v = kv.split([128, 128], dim=-1)
        k = torch.cat((k, rope.expand(-1, 12, -1)), dim=-1)
        return k, v

    def fused():
        gathered = gather_compressed_context(
            cache, workspace, table, chunk, gather, torch.float8_e4m3fn
        )
        return expand_context(
            gathered, scale, indices, seq_cu, weight, 12, 128, 64, 128
        )

    reference = baseline()
    candidate = fused()
    errors = []
    for ref, out in zip(reference, candidate):
        error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        assert error < 0.004, (rank, tokens, error)
        errors.append(error)
    torch.testing.assert_close(
        candidate[0][..., 128:], reference[0][..., 128:], atol=0, rtol=0
    )
    functions = {"baseline": baseline, "fused": fused}
    graphs = {}
    for name, fn in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(5):
                fn()
        graphs[name + "_graph"] = graph
    samples = {k: [] for k in [*functions, *graphs]}
    for sample in range(9):
        names = list(samples)
        names = names[sample % 4 :] + names[: sample % 4]
        for name in names:
            dist.barrier(group=group)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            if name in graphs:
                graphs[name].replay()
            else:
                for _ in range(5):
                    functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 5)
    del graph, graphs
    row = {
        "local_padded_tokens": local_n,
        "valid_context_tokens": total,
        "errors": errors,
        "samples_us": samples,
        "median_us": {k: statistics.median(v) for k, v in samples.items()},
    }
    print(json.dumps({"rank": rank, **row}), flush=True)
    return row


if __name__ == "__main__":
    main()
