# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 candidate scorer and four-consumer decode costs (CUDA graph + CUPTI)."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    candidate_blocks_to_sparse_indices,
    sparse_mqa_logits_paged_decode,
    sparse_topk_remap,
)
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.utils.deep_gemm import (
    _import_deep_gemm,
    calc_diff,
    get_paged_mqa_logits_metadata,
)
from vllm.v1.worker.workspace import init_workspace_manager


def bench(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    times = bench_gpu_time_with_cupti(
        fn,
        dry_run_iters=3,
        repeat_iters=25,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    return statistics.median(times) * 1e3


def run_case(dg, reference, rows, context, page, backend, queries_per_request=None):
    # Up to six queries share a request's K pages. Requests use disjoint pages.
    queries = queries_per_request or (6 if rows % 6 == 0 else 1)
    assert rows % queries == 0
    requests, blocks = rows // queries, context // page
    pages = requests * blocks
    cache = torch.empty(pages, page, 1, 132, dtype=torch.uint8, device="cuda")
    raw = cache.view(pages, -1)
    raw[:, : page * 128].copy_(
        torch.randn(pages, page * 128, dtype=torch.bfloat16, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    raw[:, page * 128 :].view(torch.float32).fill_(0.125)
    request_table = torch.randperm(pages, device="cuda", dtype=torch.int32).view(
        requests, blocks
    )
    table = request_table.repeat_interleave(queries, dim=0)
    q = torch.randn(4, rows, 1, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(4, rows, 32, device="cuda")
    lens = torch.full((rows,), context, dtype=torch.int32, device="cuda")
    indices = torch.arange(rows, dtype=torch.int32, device="cuda")
    zeros = torch.zeros_like(lens)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32, device="cuda")
    count = min(2048, context // 8)
    for row in range(rows):
        candidates[row, :count] = torch.randperm(
            context // 8, dtype=torch.int32, device="cuda"
        )[:count]
    sparse_indices = torch.empty_like(candidates)
    end = torch.empty_like(lens)
    cols = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    output = torch.empty(4, rows, 512, dtype=torch.int32, device="cuda")
    candidate_blocks_to_sparse_indices(
        candidates, zeros, lens, 8, 8, out=(sparse_indices, end)
    )
    metadata = dg.get_paged_sparse_mqa_logits_metadata(
        lens,
        table,
        indices,
        page,
        sparse_indices,
        q.dtype,
        8,
        valid_sparse_tokens=end,
    )
    dense_next_n = queries if queries in (2, 4) else 1
    dense_lens = lens.view(-1, dense_next_n)
    dense_table = request_table if dense_next_n > 1 else table
    schedule = get_paged_mqa_logits_metadata(dense_lens, page, dg.get_num_sms())

    def dense_score(i=0):
        return dg.fp8_fp4_paged_mqa_logits(
            (q[i].view(-1, dense_next_n, 32, 128), None),
            cache,
            weights[i],
            dense_lens,
            dense_table,
            schedule,
            context,
        )

    def sparse_score(i=0):
        return dg.fp8_fp4_paged_sparse_mqa_logits(
            (q[i], None), cache, weights[i], metadata, 2048, 8
        )

    def sparse_layer(i, plan):
        return sparse_mqa_logits_paged_decode(
            q[i],
            None,
            cache,
            weights[i],
            lens,
            table,
            indices,
            candidates,
            8,
            8,
            512,
            output[i],
            row_ks=zeros,
            sparse_indices=sparse_indices,
            end=end,
            col_indices=cols,
            kernel_metadata=plan,
            topk_backend=backend,
        )

    def sparse_four():
        plan = None
        for i in range(4):
            plan = sparse_layer(i, plan)

    def expand_table():
        table.view(requests, queries, blocks).copy_(request_table[:, None, :])

    def sparse_four_with_table_expansion():
        expand_table()
        sparse_four()

    def dense_four():
        for i in range(4):
            logits = dense_score(i)
            apply_candidate_mask(logits, None, lens, candidates, 8)
            get_indexer_topk(backend)(
                logits, dense_lens, dense_next_n, output[i], 512, context
            )

    dense = dense_score()
    sparse = sparse_score()
    positions = (
        sparse_indices[:, :, None] * 8 + torch.arange(8, device="cuda")
    ).flatten(1)
    valid = torch.arange(16384, device="cuda")[None, :] < end[:, None]
    diff = float(calc_diff(sparse[valid], dense.gather(1, positions.long())[valid]))
    assert diff < 5e-6, diff
    arms = {
        "dense_score": dense_score,
        "deepgemm_score": sparse_score,
        "deepgemm_first": lambda: sparse_layer(0, None),
        "deepgemm_reuse": lambda: sparse_layer(0, metadata),
        "dense_four": dense_four,
        "deepgemm_four": sparse_four,
    }
    if queries in (2, 4):
        arms["native_page_table_expand"] = expand_table
        arms["deepgemm_four_with_table_expansion"] = sparse_four_with_table_expansion
    winning_tile = None
    if reference is not None:
        triton_logits = torch.empty_like(sparse)

        def triton_score(i=0):
            reference.paged_candidate_logits(
                q[i, :, 0],
                cache.squeeze(2),
                weights[i],
                sparse_indices,
                end,
                lens,
                table,
                8,
                128,
                triton_logits,
            )

        # Tune the reference's cheap tile knob before the repeated comparison.
        tile_times = {}
        for tile in (64, 128, 256):
            reference._BLOCK_N = tile
            triton_score()
            assert float(calc_diff(triton_logits[valid], sparse[valid])) < 5e-6
            tile_times[tile] = bench(triton_score)
        winning_tile = min(tile_times, key=tile_times.get)
        reference._BLOCK_N = winning_tile

        def triton_four():
            reference.candidate_paged_prepare(
                lens[:, None],
                candidates,
                8,
                1,
                rows,
                (zeros, lens_scratch, sparse_indices, end),
            )
            for i in range(4):
                triton_score(i)
                sparse_topk_remap(
                    triton_logits,
                    sparse_indices,
                    end,
                    zeros,
                    8,
                    512,
                    output[i],
                    col_indices=cols,
                    topk_backend=backend,
                )

        lens_scratch = torch.empty_like(lens)
        arms.update(triton_score=triton_score, triton_four=triton_four)
    samples = {name: [] for name in arms}
    # Alternate order to share clock/thermal drift between implementations.
    for repeat in range(3):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            samples[name].append(bench(arms[name]))
    result = dict(
        rows=rows,
        context=context,
        page=page,
        queries_per_request=queries,
        dense_next_n=dense_next_n,
        backend=backend,
        diff=diff,
        triton_tile=winning_tile,
        us={name: statistics.median(values) for name, values in samples.items()},
        samples_us=samples,
    )
    # Useful traffic assumes one read of each selected K and scale per query;
    # page sharing and cache reuse mean this is not measured HBM bandwidth.
    result["deepgemm_useful_gbps"] = (
        rows * count * 8 * 132 / (result["us"]["deepgemm_score"] * 1e3)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 24, 48, 96])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[4096, 16384, 131072, 1048576]
    )
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--queries-per-request", type=int, choices=(1, 2, 4, 6))
    parser.add_argument("--triton-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda"))
    dg = _import_deep_gemm()
    reference = None
    if args.triton_reference:
        spec = importlib.util.spec_from_file_location(
            "triton_reference", args.triton_reference
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        deepgemm=dg.__file__,
        timing="CUPTI, CUDA graph, cold L2, 3 repeats",
        triton_reference=str(args.triton_reference),
        cases=[],
    )
    for context in args.contexts:
        for rows in args.rows:
            result = run_case(
                dg,
                reference,
                rows,
                context,
                args.page,
                args.backend,
                args.queries_per_request,
            )
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
