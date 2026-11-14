# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare one local Engram lookup with the GPUNETIO reader on matching rows."""

import argparse
import hashlib
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.distributed.engram_share.rdma.owner import inspect_checkpoint
from vllm.distributed.engram_share.rdma.provider import VllmRemoteReader
from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def load_local_rows(manifest, layer, count):
    tensors = []
    for kind, width in (("weight", 256), ("scale", 8)):
        spec = next(
            x for x in manifest["specs"] if x["layer"] == layer and x["kind"] == kind
        )
        shard = manifest["shards"][spec["shard"]]
        offset = shard["data_start"] + spec["data_offsets"][0]
        with open(shard["path"], "rb", buffering=0) as source:
            data = os.pread(source.fileno(), count * width, offset)
        if len(data) != count * width:
            raise ValueError("short checkpoint table")
        host = torch.empty((count, width), dtype=torch.uint8, pin_memory=True)
        host.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).view_as(host))
        if kind == "weight":
            host = host.view(torch.float8_e4m3fn)
        tensors.append(host)
    return tensors


def measure(fn):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin = time.perf_counter_ns()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000, (time.perf_counter_ns() - begin) / 1000


def summarize(samples):
    gpu_us, wall_us = zip(*samples)
    return {
        "gpu_event_median_us": statistics.median(gpu_us),
        "host_wall_median_us": statistics.median(wall_us),
        "gpu_event_samples_us": gpu_us,
        "host_wall_samples_us": wall_us,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reader-port-base", type=int, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024]
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--table-rows", type=int, default=65536)
    parser.add_argument("--layer", type=int, choices=[1, 14], default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens) <= 0 or min(args.heads, args.table_rows, args.samples) <= 0:
        parser.error("tokens, heads, table rows and samples must be positive")
    if args.warmup < 1:
        parser.error("warmup must be positive")
    os.environ.update(
        ENGRAM_SHARED_MODEL=str(args.model),
        ENGRAM_SHARED_READY=str(args.ready),
        ENGRAM_SHARED_LIBRARY=str(args.library),
        ENGRAM_SHARED_READER_PORT_BASE=str(args.reader_port_base),
        ENGRAM_SHARED_MAX_QUERIES=str(max(args.tokens) * args.heads),
    )
    manifest = inspect_checkpoint(args.model)
    if args.table_rows > manifest["rows"][(1, 14).index(args.layer)]:
        parser.error("table rows exceed checkpoint size")
    host = load_local_rows(manifest, args.layer, args.table_rows)
    views = tuple(get_accelerator_view_from_cpu_tensor(tensor) for tensor in host)
    table = SimpleNamespace(
        dim=256,
        block_size=32,
        num_embeddings=args.table_rows,
        head_start=0,
        part_n_hash_cols=args.heads,
        n_hash_cols=args.heads,
        vocab_start_idx=0,
        vocab_end_idx=args.table_rows,
        rdma_layer_id=args.layer,
        rdma_full_decode_graph=False,
        rdma_provider=None,
        peer_rows=0,
        _uses_small_host_pages=False,
        _num_sms=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        _storage=lambda: views,
    )
    reader = VllmRemoteReader(args.model)
    results = []
    try:
        for tokens in args.tokens:
            generator = torch.Generator().manual_seed(20261029 + tokens)
            ids = torch.randint(
                args.table_rows,
                (tokens, args.heads),
                generator=generator,
                dtype=torch.int32,
            ).cuda()
            local = torch.empty(
                (tokens, args.heads, 256), dtype=torch.bfloat16, device="cuda"
            )
            remote = torch.empty_like(local)
            local_fn = partial(ParallelEngramEmbedding.lookup, table, ids, local)
            remote_fn = partial(reader.lookup, table, ids, remote)
            for _ in range(args.warmup):
                local_fn()
                remote_fn()
            torch.cuda.synchronize()
            torch.testing.assert_close(local, remote, rtol=0, atol=0)
            timings = {"local": [], "remote": []}
            for index in range(args.samples):
                order = (("local", local_fn), ("remote", remote_fn))
                for name, fn in order if index % 2 == 0 else reversed(order):
                    timings[name].append(measure(fn))
            local_result = summarize(timings["local"])
            remote_result = summarize(timings["remote"])
            torch.testing.assert_close(local, remote, rtol=0, atol=0)
            results.append(
                {
                    "tokens": tokens,
                    "heads": args.heads,
                    "rows": tokens * args.heads,
                    "local": local_result,
                    "remote": remote_result,
                    "correct": True,
                }
            )
    finally:
        reader.close()
    args.output.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "device": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                "table_rows": args.table_rows,
                "warmup": args.warmup,
                "samples": args.samples,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
