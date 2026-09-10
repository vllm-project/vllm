# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded TP2PCP2 TopK component timing; no model or KV-cache changes."""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, output):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    from vllm import _custom_ops as ops
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.model_executor.layers import tp_topk_publication as pub
    from vllm.v1.attention.backends.mla.indexer import (
        balanced_prefill_row_shard,
        split_indexer_prefill_chunks,
    )

    cpu_groups, gpu_groups = [], []
    for start in (0, 2):
        cpu_groups.append(dist.new_group([start, start + 1], backend="gloo"))
        gpu_groups.append(dist.new_group([start, start + 1], backend="nccl"))
    lane, tp_rank = divmod(rank, 2)
    cpu_group, gpu_group = cpu_groups[lane], gpu_groups[lane]
    nccl = PyNcclCommunicator(cpu_group, device=rank)
    assert not nccl.disabled
    pub.get_tp_group = lambda: SimpleNamespace(
        device_group=gpu_group, rank_in_group=tp_rank, world_size=2
    )
    os.environ["VLLM_TP_TOPK_DIRECT"] = "1"
    buffer = pub.allocate_topk_buffer(
        32768, 2048, dtype=torch.int32, device=torch.device("cuda", rank)
    )
    publisher = pub.get_topk_publication(buffer)
    rows, width = buffer.shape
    records = []
    for prefix in (0, 65536):
        # Actual PCP mirrored chunks: lane 0 owns chunks 0,3; lane 1 owns 1,2.
        seq = torch.tensor(
            [prefix + (lane + 1) * 16384, prefix + (4 - lane) * 16384],
            dtype=torch.int32,
        )
        qlens = torch.tensor([16384, 16384], dtype=torch.int32)
        sizes = balanced_prefill_row_shard(seq, qlens, 1, 2)
        start = sum(sizes[:tp_rank])
        stop = start + sizes[tp_rank]
        local = torch.arange(
            start * width, stop * width, dtype=torch.int32, device=buffer.device
        ).reshape(-1, width)
        # Use actual production query chunk planner; bound synthetic logits
        # at 512 MiB and regenerate outside timed intervals.
        specs = split_indexer_prefill_chunks(seq, qlens, 2_000_000, 512 * 1024**2)
        native_chunks = []
        for req, query in specs:
            assert req.stop - req.start == 1
            lo = max(req.start * 16384 + query.start, start)
            hi = min(req.start * 16384 + query.stop, stop)
            if hi > lo:
                native_chunks.append(
                    (
                        lo,
                        hi,
                        int(seq[req.start]),
                        int(seq[req.start]) - 16384 + lo - req.start * 16384,
                    )
                )

        # Time each native chunk separately; no MQA or logits generation in timing.
        native = []
        for lo, hi, keys, first in native_chunks:
            logits = torch.randn((hi - lo, keys), device=buffer.device)
            ks = torch.zeros(hi - lo, device=buffer.device, dtype=torch.int32)
            ke = torch.arange(
                first + 1, first + hi - lo + 1, device=buffer.device, dtype=torch.int32
            )
            dst = buffer[lo:hi]

            produce = partial(
                ops.top_k_per_row_prefill,
                logits,
                ks,
                ke,
                dst,
                hi - lo,
                logits.stride(0),
                1,
                width,
            )

            for _ in range(2):
                produce()
            timings = []
            for _ in range(5):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                produce()
                b.record()
                b.synchronize()
                timings.append(a.elapsed_time(b) * 1000)
            native.append({"rows": hi - lo, "keys": keys, "us": timings})
            del logits, ks, ke

        buffer.fill_(-1)
        buffer[start:stop].copy_(local)
        torch.accelerator.synchronize()
        # Measure grouped NCCL broadcasts exactly as PyNCCL all_gatherv uses.
        # Allocate its output on every call, like CudaCommunicator.
        samples = []
        for repeat in range(16):
            for mode in (
                ("collective", "direct")
                if repeat % 2 == 0
                else ("direct", "collective")
            ):
                dist.barrier()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                events[0].record()
                if mode == "collective":
                    gathered = torch.empty_like(buffer)
                    nccl.all_gatherv(gathered, buffer[start:stop].contiguous(), sizes)
                    events[1].record()
                    buffer.copy_(gathered)
                    events[2].record()
                    labels = ["all_gatherv", "copy_back", "unused"]
                else:
                    publisher.begin()
                    events[1].record()
                    pub._publish_rows[((sizes[tp_rank] * width + 1023) // 1024, 2)](
                        buffer,
                        publisher.peers,
                        start,
                        sizes[tp_rank],
                        width,
                        width,
                        tp_rank,
                        1024,
                    )
                    events[2].record()
                    publisher.finish()
                    labels = ["begin_barrier", "peer_stores", "finish_barrier"]
                events[3].record()
                events[3].synchronize()
                if repeat >= 4:
                    samples.append(
                        {
                            "mode": mode,
                            "repeat": repeat,
                            "total_us": events[0].elapsed_time(events[3]) * 1000,
                            **{
                                label: events[i].elapsed_time(events[i + 1]) * 1000
                                for i, label in enumerate(labels)
                            },
                        }
                    )
                if repeat == 15:
                    expected = torch.arange(
                        rows * width, dtype=torch.int32, device=buffer.device
                    ).reshape(rows, width)
                    torch.testing.assert_close(buffer, expected)
        records.append(
            {
                "prefix": prefix,
                "rank": rank,
                "pcp_lane": lane,
                "sizes": sizes,
                "native_chunks": native,
                "samples": samples,
            }
        )
    results = [None] * 4
    dist.all_gather_object(results, records)
    if rank == 0:
        Path(output).write_text(
            json.dumps({"torch": torch.__version__, "results": results}, indent=2)
        )
    torch.accelerator.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from vllm.utils.network_utils import get_open_port

    mp.spawn(worker, args=(get_open_port(), args.output), nprocs=4, join=True)
