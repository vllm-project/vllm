# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete K1+K2 against a single kernel with group overlap (TP8)."""

import argparse
import json
import os
import statistics
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import torch
import torch.distributed as dist

from benchmarks.kernels.benchmark_kimi_k3_mono import checkpoint_weights
from benchmarks.kernels.benchmark_kimi_k3_mono_k2 import checkpoint_moe_weights
from tests.models.kimi_k3.test_amd_mono import (
    COS_MIN,
    _cos,
    _kda_inputs,
    _moe_args,
    _moe_weights,
    _randn,
    _rel_l2,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.kimi_k3.amd.mono import layer as k2
from vllm.models.kimi_k3.amd.mono import overlap
from vllm.models.kimi_k3.amd.mono.attention import kda as k1
from vllm.models.kimi_k3.amd.mono.attention.parallel import pack_projection_weights
from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer


def measure(args, rows, length, rank, cpu_group):
    device = torch.device(f"cuda:{rank}")
    comm = PyNcclCommunicator(cpu_group, device=device)
    assert not comm.disabled
    sync_input = torch.zeros(1, device=device)
    sync_output = torch.empty_like(sync_input)
    comm.all_reduce(sync_input, sync_output)
    first = _kda_inputs(4, 1, False, rows, qlen=length)
    if args.model:
        if rank == 0:
            print(f"Loading checkpoint layer {args.layer}", flush=True)
        first.update(checkpoint_weights(args.model, args.layer, rank))
        weights, front = checkpoint_moe_weights(args.model, args.layer, rank, device)
    else:
        weights, front = _moe_weights(rank, device), {}
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    core = torch.empty(rows, k1.PROJ, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, k1.HIDDEN, dtype=torch.bfloat16, device=device)
    baseline_pipeline = {
        (6, 6): 35667968,
        (12, 6): 35667968,
        (8, 8): 33562624,
        (16, 8): 33562624,
    }.get((rows, length), 33562624)
    base1 = k1.KdaPreBuild(
        tokens=rows,
        qlen=length,
        nblocks=4,
        delta=True,
        state_len=first["conv_state"].shape[-1],
        state_rows=16,
        pipeline=baseline_pipeline,
    )
    base2 = k2.K2Build(tokens=rows, nblocks=4)
    baseline_tiles = [
        (first, min(16 // length * length, rows - first))
        for first in range(0, rows, 16 // length * length)
    ]
    configs = {}
    for workers in (
        [] if args.reuse_only or args.pipeline or args.tune else args.k1_workers
    ):
        configs[f"overlap_w{workers}"] = overlap.OverlapBuild(
            replace(base1, pipeline=33562624 | k1.TOKEN_PIPELINE | 4),
            replace(base2, parallel_shared_activation=True, late_shared_wait=True),
            requests_per_group=1,
            k1_workers=workers,
        )
    if rows <= 16:
        configs["reuse_workers"] = overlap.OverlapBuild(
            base1, base2, k1_workers=k1.BLOCKS, reuse_workers=True
        )
        configs["tagged_core"] = replace(configs["reuse_workers"], tagged_core=True)
        for workers in args.quant_workers:
            configs[f"tagged_quant_{workers}"] = replace(
                configs["tagged_core"],
                k2=replace(base2, latent_quant_workers=workers),
            )
    if rows <= 16 and (args.producer_quant or args.pipeline or args.tune):
        configs["tagged_producer_quant"] = replace(
            configs["tagged_core"], k2=replace(base2, producer_latent_quant=True)
        )
    if args.tune:
        assert rows <= 16
        for batch in args.down_batches:
            configs[f"prefetch_{batch}"] = replace(
                configs["tagged_producer_quant"],
                k2=replace(
                    base2, producer_latent_quant=True, down_batch=batch, inter_vec=2
                ),
            )
            if args.trim_oproj:
                configs[f"trim_oproj_{batch}"] = replace(
                    configs[f"prefetch_{batch}"],
                    k2=replace(configs[f"prefetch_{batch}"].k2, trim_oproj=True),
                )
            if args.stream_oproj:
                configs[f"stream_oproj_{batch}"] = replace(
                    configs[f"prefetch_{batch}"],
                    k2=replace(configs[f"prefetch_{batch}"].k2, stream_oproj=True),
                )
                if args.trim_oproj:
                    configs[f"trim_stream_oproj_{batch}"] = replace(
                        configs[f"trim_oproj_{batch}"],
                        k2=replace(
                            configs[f"trim_oproj_{batch}"].k2, stream_oproj=True
                        ),
                    )
            if args.balanced_oproj:
                configs[f"balanced_oproj_{batch}"] = replace(
                    configs[f"prefetch_{batch}"],
                    k2=replace(configs[f"prefetch_{batch}"].k2, balanced_oproj=True),
                )
                if args.trim_oproj and args.stream_oproj:
                    configs[f"balanced_trim_stream_{batch}"] = replace(
                        configs[f"trim_stream_oproj_{batch}"],
                        k2=replace(
                            configs[f"trim_stream_oproj_{batch}"].k2,
                            balanced_oproj=True,
                        ),
                    )
    if args.pipeline:
        for workers in args.k1_workers:
            for down in args.down_workers:
                configs[f"pipeline_w{workers}_d{down}_g{args.requests_per_group}"] = (
                    overlap.OverlapBuild(
                        replace(
                            base1,
                            pipeline=33562624
                            | (k1.TOKEN_PIPELINE | 4 if args.token_pipeline else 0),
                        ),
                        replace(
                            base2,
                            producer_latent_quant=True,
                            down_workers=down,
                            parallel_shared_activation=True,
                            late_shared_wait=True,
                            full_shared_pipeline=args.full_shared,
                        ),
                        k1_workers=workers,
                        requests_per_group=args.requests_per_group,
                        ring_buffers=True,
                    )
                )
        if rows <= 16:
            for down in args.down_workers:
                configs[f"stream_d{down}"] = replace(
                    configs["tagged_producer_quant"],
                    k2=replace(
                        base2,
                        producer_latent_quant=True,
                        down_workers=down,
                        shared_down_workers=16,
                        parallel_shared_activation=True,
                        late_shared_wait=True,
                        full_shared_pipeline=args.full_shared,
                    ),
                )
    separate_keys = {}
    if args.separate_tuned:
        separate_keys = {
            "separate_" + name: key.k2
            for name, key in configs.items()
            if name.startswith(
                (
                    "prefetch_",
                    "trim_oproj_",
                    "stream_oproj_",
                    "trim_stream_oproj_",
                    "balanced_oproj_",
                    "balanced_trim_stream_",
                )
            )
        }
    if args.arms:
        configs = {name: key for name, key in configs.items() if name in args.arms}
        separate_keys = {
            name: key for name, key in separate_keys.items() if name in args.arms
        }
        assert configs or separate_keys, args.arms
    peers = PeerBuffer(
        max(
            [
                sum(k2.peer_bytes(n) for _, n in baseline_tiles),
                *(overlap.peer_bytes(key) for key in configs.values()),
            ]
        ),
        cpu_group,
        rank,
        8,
        device,
    )
    peers.bytes.zero_()
    baseline_peers = []
    peer_offset = 0
    for _, count in baseline_tiles:
        baseline_peers.append(peers.addresses + peer_offset)
        peer_offset += k2.peer_bytes(count)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    packed = pack_projection_weights(first["w_in"])
    second = dict(
        core=core,
        w_o=_randn(gr, k1.HIDDEN, k1.PROJ, scale=0.02),
        prefix=first["prefix"],
        blocks=first["blocks"],
        ares_nw=first["ares_nw"],
        ares_qk=first["ares_qk"],
        in_nw=first["in_nw"],
        out=out,
        **_moe_args(weights),
    )
    second.update(front)
    initial = {
        name: first[name].clone()
        for name in ("prefix", "blocks", "conv_state", "rstate")
    }
    first_launch = dict(
        prefix=first["prefix"],
        delta=first["delta"],
        blocks=first["blocks"],
        ares_nw=first["ares_nw"],
        ares_qk=first["ares_qk"],
        in_nw=first["in_nw"],
        w_in=packed if baseline_pipeline & 33554432 else first["w_in"],
        w_fb=first["w_fb"],
        conv_w=first["conv_w"],
        conv_state=first["conv_state"],
        a_log=first["a_log"],
        dt_bias=first["dt_bias"],
        on_w=first["on_w"],
        rstate=first["rstate"],
        st_idx=first["st_idx"],
        num_acc=first["num_acc"],
        core_out=core,
    )
    packed_first = {
        **first_launch,
        "w_in": packed,
        "conv_st": first["conv_state"],
        "blk_sm": first["blocks"].stride(0),
        "blk_sr": first["blocks"].stride(1),
        "cs_seq": first["conv_state"].stride(0),
        "cs_dim": first["conv_state"].stride(1),
        "cs_tok": first["conv_state"].stride(2),
        "rs_seq": first["rstate"].stride(0),
        "st_stride": first["st_idx"].stride(0),
    }
    packed_second = {
        **second,
        "blk_sm": first["blocks"].stride(0),
        "blk_sr": first["blocks"].stride(1),
    }
    arguments = overlap.pack_arguments(packed_first, packed_second, device)
    scratch1 = torch.zeros(k1.scratch_bytes(base1), dtype=torch.uint8, device=device)
    scratch2 = torch.zeros(
        max(
            k2.scratch_bytes(replace(key, tokens=n))
            for key in (base2, *separate_keys.values())
            for _, n in baseline_tiles
        ),
        dtype=torch.uint8,
        device=device,
    )
    baseline_seconds = []
    for first_row, count in baseline_tiles:
        item = dict(second)
        for name in ("core", "prefix", "blocks", "out"):
            item[name] = item[name][first_row : first_row + count]
        baseline_seconds.append(item)
    workspaces = {
        name: torch.zeros(overlap.scratch_bytes(key), dtype=torch.uint8, device=device)
        for name, key in configs.items()
    }
    flush = torch.empty(
        args.flush_l2_mb * 1024 * 1024, dtype=torch.uint8, device=device
    )

    def prepare():
        for name, value in initial.items():
            first[name].copy_(value)
        epoch.add_(1)

    def launch(name):
        if name == "current_two_launches" or name in separate_keys:
            second_key = separate_keys.get(name, base2)
            k1.kda_pre(base1, **first_launch, scratch=scratch1, epoch=epoch, layer=2)
            for i, (_, count) in enumerate(baseline_tiles):
                k2.k2_launch(
                    replace(second_key, tokens=count),
                    **baseline_seconds[i],
                    scratch=scratch2,
                    peers=baseline_peers[i],
                    rank=rank,
                    epoch=epoch,
                    layer=3 + i,
                )
        else:
            overlap.launch(
                configs[name],
                arguments,
                workspaces[name],
                peers.addresses,
                epoch,
                rank,
                layer=4,
            )

    names = ["current_two_launches", *configs, *separate_keys]
    expected = None
    expected_core = None
    expected_state = None
    expected_prefix = None
    expected_conv = None
    checks, graphs = {}, {}
    for name in names:
        if rank == 0:
            print(f"Compile/check M{rows}/L{length} {name}", flush=True)
        prepare()
        launch(name)
        torch.accelerator.synchronize()
        if expected is None:
            expected = out.clone()
            expected_core = core.clone()
            expected_state = first["rstate"].clone()
            expected_prefix = first["prefix"].clone()
            expected_conv = first["conv_state"].clone()
        checks[name] = dict(
            output_cosine=_cos(out, expected),
            output_rel_l2=_rel_l2(out, expected),
            core_cosine=_cos(core, expected_core),
            core_rel_l2=_rel_l2(core, expected_core),
            state_rel_l2=_rel_l2(first["rstate"], expected_state),
        )
        assert checks[name]["output_cosine"] > COS_MIN, (name, rank, checks[name])
        assert checks[name]["core_cosine"] > COS_MIN, (name, rank, checks[name])
        assert checks[name]["state_rel_l2"] < 1e-3, (name, rank, checks[name])
        if args.pipeline or args.tune or name in ("reuse_workers", "tagged_core"):
            assert torch.equal(core, expected_core), (name, rank, checks[name])
            assert torch.equal(out, expected), (name, rank, checks[name])
            assert torch.equal(first["prefix"], expected_prefix), (name, rank, "prefix")
            assert torch.equal(first["rstate"], expected_state), (name, rank, "state")
            assert torch.equal(first["conv_state"], expected_conv), (name, rank, "conv")
        outputs = [torch.empty_like(out) for _ in range(8)]
        dist.all_gather(outputs, out)
        assert all(torch.equal(outputs[0], value) for value in outputs)
        for _ in range(args.warmup):
            prepare()
            launch(name)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            prepare()
            flush.zero_()
            comm.all_reduce(sync_input, sync_output)
            start.record()
            launch(name)
            end.record()
        graphs[name] = graph, start, end
        graph.replay()
        torch.accelerator.synchronize()
        assert _cos(out, expected) > COS_MIN, ("graph", name, rank)

    samples = {name: [] for name in names}
    latency = torch.zeros(1, device=device)
    for repeat in range(args.repeats):
        for name in names if repeat % 2 == 0 else reversed(names):
            graph, start, end = graphs[name]
            round_values = []
            for _ in range(args.iterations):
                graph.replay()
                torch.accelerator.synchronize()
                latency.fill_(start.elapsed_time(end) * 1000)
                dist.all_reduce(latency, op=dist.ReduceOp.MAX)
                round_values.append(latency.item())
            samples[name].append(round_values)
        if rank == 0:
            print(
                "round",
                repeat,
                {name: statistics.median(samples[name][-1]) for name in names},
                flush=True,
            )

    for name, key in [] if args.skip_trace else configs.items():
        traced = replace(
            key, k1=replace(key.k1, timeline=True), k2=replace(key.k2, timeline=True)
        )
        timeline = torch.zeros(
            overlap.timeline_words(traced), dtype=torch.int64, device=device
        )

        def trace_launch(traced=traced, name=name, timeline=timeline):
            overlap.launch(
                traced,
                arguments,
                workspaces[name],
                peers.addresses,
                epoch,
                rank,
                layer=4,
                timeline=timeline,
            )

        prepare()
        trace_launch()
        torch.accelerator.synchronize()
        dist.barrier()
        trace_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(trace_graph):
            prepare()
            timeline.zero_()
            flush.zero_()
            comm.all_reduce(sync_input, sync_output)
            trace_launch()
        trace_graph.replay()
        torch.accelerator.synchronize()
        trace_path = (
            args.output.parent / f"trace-m{rows}-l{length}-{name}-rank{rank}.pt"
        )
        timestamps = timeline.cpu()
        record = {
            "timestamps_100mhz": timestamps,
            "layout": overlap.layouts(traced)[0],
            "k1_points": k1.TL_POINTS,
            "k2_points": overlap.TL_POINTS,
            "config": asdict(traced),
            "method": "warmed HIP graph replay with device barrier and cold L2",
        }
        torch.save(record, trace_path)
        record["timestamps_100mhz"] = timestamps.tolist()
        trace_path.with_suffix(".json").write_text(json.dumps(record) + "\n")

    # Replay the same captured graphs with changed inputs and request-state slots.
    original_slots = first["st_idx"].clone()
    first["st_idx"].copy_(original_slots.roll(1, dims=1))
    initial["prefix"].add_(0.03125)
    changed_expected = None
    for name in names:
        graphs[name][0].replay()
        torch.accelerator.synchronize()
        if changed_expected is None:
            changed_expected = out.clone()
        assert _cos(out, changed_expected) > COS_MIN, ("changed inputs", name, rank)
        if args.pipeline or args.tune or name in ("reuse_workers", "tagged_core"):
            assert torch.equal(out, changed_expected), ("changed inputs", name, rank)
    result = dict(
        rows=rows,
        qlen=length,
        checks=checks,
        changed_input_replay=True,
        configs={name: asdict(key) for name, key in configs.items()},
        separate_configs={name: asdict(key) for name, key in separate_keys.items()},
        median_us={
            name: statistics.median([statistics.median(v) for v in rounds])
            for name, rounds in samples.items()
        },
        samples_max_rank_us=samples,
    )
    dist.barrier()
    graphs.clear()
    torch.accelerator.synchronize()
    peers.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--shapes", nargs="+", default=["12/6", "16/8"])
    parser.add_argument("--k1-workers", nargs="+", type=int, default=[112, 128, 144])
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--quant-workers", nargs="*", type=int, default=[])
    parser.add_argument("--producer-quant", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trim-oproj", action="store_true")
    parser.add_argument("--separate-tuned", action="store_true")
    parser.add_argument("--stream-oproj", action="store_true")
    parser.add_argument("--balanced-oproj", action="store_true")
    parser.add_argument("--down-batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--token-pipeline", action="store_true")
    parser.add_argument("--full-shared", action="store_true")
    parser.add_argument("--down-workers", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--requests-per-group", type=int, choices=[1, 2], default=1)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flush-l2-mb", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    cpu_group = dist.new_group(backend="gloo")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        torch=torch.__version__,
        hip=torch.version.hip,
        gpu=torch.cuda.get_device_name(rank),
        results=[],
        inputs="checkpoint weights, synthetic activations/state; TP8"
        if args.model
        else "seeded synthetic weights, activations, and state; TP8",
        checkpoint=str(args.model) if args.model else None,
        layer=args.layer if args.model else None,
        boundary="complete K1+K2; input reset, epoch, L2 flush, barrier outside events",
        flush_l2_mb=args.flush_l2_mb,
    )
    for shape in args.shapes:
        rows, length = map(int, shape.split("/"))
        assert 0 < rows <= 64 and rows % length == 0
        report["results"].append(measure(args, rows, length, rank, cpu_group))
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
