# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Engram lookup tiles with changing indices and an L2 cache flush.

Example for one DeepSeek-V4.1-Flash host table (about 94.4 GiB):
    python benchmarks/kernels/benchmark_engram_lookup.py \
        --table-rows 384006168 --tokens 1 2 4 8 16 256 --background

Each graph contains distinct batches to exercise a wider address range than
replaying one lookup. Results are kernel times, not model throughput.
"""

import argparse
import json
import random

import torch

from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel
from vllm.triton_utils import triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def capture_lookup(weight, scales, ids, out, block_rows, max_grid):
    id_batches, out_batches = ids.unbind(), out.unbind()
    rows = ids.shape[1] * ids.shape[2]
    grid = min(triton.cdiv(rows, block_rows), max_grid)

    def launch(i):
        _engram_lookup_kernel[(grid,)](
            weight,
            scales,
            id_batches[i],
            out_batches[i],
            0,
            weight.shape[0],
            rows,
            ids.stride(1),
            ids.stride(2),
            HEAD_START=0,
            LOCAL_HEADS=ids.shape[2],
            TOTAL_HEADS=ids.shape[2],
            DIM=weight.shape[1],
            QUANT_BLOCK=32,
            BLOCK_R=block_rows,
            GRID=grid,
        )

    launch(0)
    torch.testing.assert_close(out[0], torch.ones_like(out[0]), rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(ids.shape[0]):
            launch(i)
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def main(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.cuda.get_device_properties(0)
    max_grid = device.multi_processor_count // (2 if args.background else 1)
    host = args.placement == "cpu"
    weight_owner = torch.empty(
        args.table_rows,
        args.dim,
        dtype=torch.uint8,
        device=args.placement,
        pin_memory=host,
    ).fill_(56)
    scale_owner = torch.empty(
        args.table_rows,
        args.dim // 32,
        dtype=torch.uint8,
        device=args.placement,
        pin_memory=host,
    ).fill_(127)
    weight = weight_owner.view(torch.float8_e4m3fn)
    scales = scale_owner
    if host:
        weight = get_accelerator_view_from_cpu_tensor(weight)
        scales = get_accelerator_view_from_cpu_tensor(scales)
    print(
        json.dumps(
            dict(
                vars(args),
                device=device.name,
                sms=device.multi_processor_count,
                torch=torch.__version__,
                triton=triton.__version__,
            )
        ),
        flush=True,
    )

    for tokens in args.tokens:
        ids = torch.randint(
            0,
            args.table_rows,
            (args.batches, tokens, args.heads),
            dtype=torch.int32,
            device="cuda",
        )
        out = torch.empty(
            args.batches,
            tokens,
            args.heads,
            args.dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        graphs = {
            block_rows: capture_lookup(weight, scales, ids, out, block_rows, max_grid)
            for block_rows in args.block_rows
        }

        order = list(args.block_rows)
        for repeat in range(args.repeats):
            rng.shuffle(order)
            for block_rows in order:
                times = triton.testing.do_bench(
                    graphs[block_rows].replay,
                    warmup=25,
                    rep=args.duration_ms,
                    return_mode="all",
                )
                print(
                    json.dumps(
                        dict(
                            tokens=tokens,
                            block_rows=block_rows,
                            repeat=repeat,
                            us_per_lookup=[1000 * t / args.batches for t in times],
                        )
                    ),
                    flush=True,
                )
        del graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-rows", type=int, default=2**22)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 256])
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dim", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--placement", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--block-rows", type=int, nargs="+", default=[16, 2])
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--seed", type=int, default=137)
    main(parser.parse_args())
