# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8 K2 scheduling benchmark; run with torch.distributed.run, eight ranks."""

import argparse
import importlib
import json
import os
import statistics
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import torch
import torch.distributed as dist

from tests.models.kimi_k3.test_amd_mono import (
    COS_MIN,
    _cos,
    _moe_args,
    _moe_reference,
    _moe_weights,
    _randn,
    _rel_l2,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.kimi_k3.amd.mono import layer as k2
from vllm.models.kimi_k3.amd.mono.attention.kda import HIDDEN, PROJ
from vllm.models.kimi_k3.amd.mono.common.debug import (
    DIAG_BYTES,
    given_up_waits,
    region_namer,
)
from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer


@lru_cache(maxsize=8)
def checkpoint_moe_weights(path, layer, rank, device):
    """Load checkpoint K2 weights using the production TP8 packing/sharding."""
    from aiter.utility.fp4_utils import e8m0_shuffle
    from safetensors import safe_open

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.mono.stages.moe import LAT, RI, SI, UP_N, E

    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"language_model.model.layers.{layer}."
    with ExitStack() as stack:
        handles = {}

        def read(name, shard=None):
            full = prefix + name
            file = index[full]
            if file not in handles:
                handles[file] = stack.enter_context(
                    safe_open(path / file, framework="pt", device="cpu")
                )
            handle = handles[file]
            return (
                handle.get_tensor(full)
                if shard is None
                else handle.get_slice(full)[shard]
            ).contiguous()

        def moe(name, shard=None):
            return read("block_sparse_moe." + name, shard)

        w13 = torch.empty(E, 2 * RI, LAT // 2, dtype=torch.uint8)
        s13 = torch.empty(E, 2 * RI, LAT // 32, dtype=torch.uint8)
        w2 = torch.empty(E, LAT, RI // 2, dtype=torch.uint8)
        s2 = torch.empty(E, LAT, RI // 32, dtype=torch.uint8)
        rows = slice(rank * RI, (rank + 1) * RI)
        for expert in range(E):
            for name, lo in (("w1", 0), ("w3", RI)):
                w13[expert, lo : lo + RI].copy_(
                    moe(f"experts.{expert}.{name}.weight_packed", rows)
                )
                s13[expert, lo : lo + RI].copy_(
                    moe(f"experts.{expert}.{name}.weight_scale", rows)
                )
            w2[expert].copy_(
                moe(
                    f"experts.{expert}.w2.weight_packed",
                    (slice(None), slice(rank * RI // 2, (rank + 1) * RI // 2)),
                )
            )
            s2[expert].copy_(
                moe(
                    f"experts.{expert}.w2.weight_scale",
                    (slice(None), slice(rank * RI // 32, (rank + 1) * RI // 32)),
                )
            )
        shared_rows = slice(rank * SI, (rank + 1) * SI)
        weights = {
            "w_gate": moe("gate.weight"),
            "bias": moe("gate.e_score_correction_bias"),
            "w_ld": moe("routed_expert_down_proj.weight"),
            "ln_w": moe("routed_expert_norm.weight"),
            "w_up_shard": moe(
                "routed_expert_up_proj.weight", slice(rank * UP_N, (rank + 1) * UP_N)
            ),
            "w_sgu": torch.cat(
                [
                    moe(f"shared_experts.{part}_proj.weight", shared_rows)
                    for part in ("gate", "up")
                ]
            ),
            "w_sd": moe("shared_experts.down_proj.weight", (slice(None), shared_rows)),
        }
        front = {
            "w_o": read(
                "self_attn.o_proj.weight",
                (slice(None), slice(rank * PROJ, (rank + 1) * PROJ)),
            ),
            "ares_nw": read("mlp_res_norm.weight").flatten(),
            "ares_qk": read("mlp_res_proj.weight").flatten(),
            "in_nw": read("post_attention_layernorm.weight").flatten(),
        }
    weights = {name: value.to(device) for name, value in weights.items()}
    front = {name: value.to(device) for name, value in front.items()}
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    weights.update(
        w13=rocm_aiter_ops.shuffle_weight_a16w4(w13.to(device).view(fp4), 16, False),
        w2=rocm_aiter_ops.shuffle_weight_a16w4(w2.to(device).view(fp4), 16, False),
        w13s=rocm_aiter_ops.shuffle_scale_a16w4(
            s13.to(device).view(e8).view(-1, LAT // 32), E, False
        ),
        w2s=e8m0_shuffle(s2.to(device).view(e8).view(-1, RI // 32)),
    )
    return weights, front


def load_reference(root):
    import vllm.models.kimi_k3.amd as amd

    prefix = "vllm.models.kimi_k3.amd.mono"
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == prefix or name.startswith(prefix + ".")
    }
    old_path, old_mono = amd.__path__, amd.mono
    for name in saved:
        del sys.modules[name]
    amd.__path__ = [str(root / "vllm/models/kimi_k3/amd"), *old_path]
    try:
        module = importlib.import_module(prefix + ".layer")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)
        amd.__path__, amd.mono = old_path, old_mono
    return module


def measure(args, rows, rank, cpu_group):
    reference_k2 = (
        load_reference(args.reference_source) if args.reference_source else k2
    )
    device = torch.device(f"cuda:{rank}")
    comm = PyNcclCommunicator(cpu_group, device=device)
    assert not comm.disabled
    sync_input = torch.zeros(1, device=device)
    sync_output = torch.empty_like(sync_input)
    comm.all_reduce(sync_input, sync_output)
    bf = torch.bfloat16
    peers = PeerBuffer(k2.peer_bytes(rows), cpu_group, rank, 8, device)
    peers.bytes.zero_()
    dist.barrier()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    if args.model:
        if rank == 0:
            print(f"Loading checkpoint layer {args.layer}", flush=True)
        weights, front = checkpoint_moe_weights(args.model, args.layer, rank, device)
    else:
        weights, front = _moe_weights(rank, device), {}
    if args.shared_route:
        weights = dict(weights, bias=weights["bias"].clone())
        weights["bias"].fill_(-1000)
        weights["bias"][:16] = 1000
    gc = torch.Generator(device=device).manual_seed(11)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    core = _randn(gr, rows, PROJ)
    w_o = _randn(gr, HIDDEN, PROJ, scale=0.02)
    prefix0 = _randn(gc, rows, HIDDEN, scale=2.0)
    blocks = _randn(gc, rows, 8, HIDDEN, scale=2.0)
    ares_nw = (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf)
    ares_qk = _randn(gc, HIDDEN, scale=0.05)
    in_nw = (1 + _randn(gc, HIDDEN, scale=0.1, dtype=torch.float32)).to(bf)
    if front:
        w_o, ares_nw, ares_qk, in_nw = (
            front[name] for name in ("w_o", "ares_nw", "ares_qk", "in_nw")
        )
    prefix = prefix0.clone()
    out = torch.empty(rows, HIDDEN, dtype=bf, device=device)
    keys = {"old_k2": reference_k2.K2Build(tokens=min(rows, 16), nblocks=4)}
    if args.mode in (
        "wide",
        "wide-pipeline",
        "wide-latq",
        "wide-stream-x",
        "wide-front",
    ):
        if rows <= 16:
            keys["selected_v8"] = k2.K2Build(
                tokens=rows,
                nblocks=4,
                producer_latent_quant=True,
                balanced_oproj=True,
                down_batch=2,
                inter_vec=2,
                diag=args.diag,
            )
            keys["separate_wide_2"] = replace(
                keys["selected_v8"], wide=True, separate_wide_lds=True
            )
        for batch in args.down_batches:
            keys[f"wide_{batch}"] = k2.K2Build(
                tokens=rows,
                nblocks=4,
                wide=True,
                diag=args.diag,
                producer_latent_quant=True,
                balanced_oproj=True,
                down_batch=batch,
                inter_vec=2,
            )
        for batch in args.down_batches:
            keys[f"sparse_{batch}"] = replace(
                keys[f"wide_{batch}"],
                compact_lds=True,
                sparse_routes=True,
                cache_wide_inter=rows <= 32,
                stream_attn=rows > 32 or rows > 16 and args.nblocks >= 8,
            )
            if args.mode == "wide-pipeline":
                for workers in args.down_workers:
                    keys[f"sparse_{batch}_dw{workers}"] = replace(
                        keys[f"sparse_{batch}"],
                        down_workers=workers,
                        cache_wide_inter=False,
                    )
            if args.mode == "wide-latq":
                keys[f"sparse_{batch}_lv2"] = replace(
                    keys[f"sparse_{batch}"], latq_vec=2
                )
            if args.mode == "wide-stream-x":
                keys[f"sparse_{batch}_sx"] = replace(
                    keys[f"sparse_{batch}"], stream_x=True
                )
            if args.mode == "wide-front":
                keys[f"sparse_{batch}_front"] = replace(
                    keys[f"sparse_{batch}"], stream_x=True, front_panels=True
                )
                keys[f"sparse_{batch}_front_unbalanced"] = replace(
                    keys[f"sparse_{batch}_front"], balanced_oproj=False
                )
                keys[f"sparse_{batch}_front_shared_first"] = replace(
                    keys[f"sparse_{batch}_front_unbalanced"], shared_down_first=True
                )
                keys[f"sparse_{batch}_front_h_once"] = replace(
                    keys[f"sparse_{batch}_front_unbalanced"], shared_h_once=True
                )
                keys[f"sparse_{batch}_front_shared_first_h_once"] = replace(
                    keys[f"sparse_{batch}_front_shared_first"], shared_h_once=True
                )
                keys[f"sparse_{batch}_front_tail"] = replace(
                    keys[f"sparse_{batch}_front_unbalanced"], tail_panels=True
                )
                keys[f"sparse_{batch}_front_shared_first_h_once_tail"] = replace(
                    keys[f"sparse_{batch}_front_shared_first_h_once"], tail_panels=True
                )
            keys[f"compact_{batch}"] = replace(
                keys[f"wide_{batch}"], compact_lds=True, cache_wide_inter=True
            )
            if rows > 16:
                keys[f"stream_attn_{batch}"] = replace(
                    keys[f"wide_{batch}"],
                    compact_lds=True,
                    stream_attn=True,
                    stream_latent=rows > 48,
                )
        if rows > 32:
            keys = {
                name: key
                for name, key in keys.items()
                if name == "old_k2" or name.startswith(("stream_attn_", "sparse_"))
            }
    elif args.mode in ("oproj", "oproj-stream", "oproj-balance"):
        keys["producer_quant"] = replace(keys["old_k2"], producer_latent_quant=True)
        keys["prefetch_2"] = replace(keys["producer_quant"], down_batch=2, inter_vec=2)
        keys["trim_oproj"] = replace(keys["prefetch_2"], trim_oproj=True)
        if args.mode in ("oproj-stream", "oproj-balance"):
            keys["stream_oproj"] = replace(keys["prefetch_2"], stream_oproj=True)
            keys["trim_stream_oproj"] = replace(keys["trim_oproj"], stream_oproj=True)
        if args.mode == "oproj-balance":
            keys["balanced_oproj"] = replace(keys["prefetch_2"], balanced_oproj=True)
            keys["balanced_trim_stream"] = replace(
                keys["trim_stream_oproj"], balanced_oproj=True
            )
    elif args.mode == "tune":
        keys["producer_quant"] = replace(keys["old_k2"], producer_latent_quant=True)
        keys["vector_inter"] = replace(keys["producer_quant"], inter_vec=2)
        for batch in args.down_batches:
            if batch != 5:
                keys[f"prefetch_{batch}"] = replace(
                    keys["vector_inter"], down_batch=batch
                )
    elif args.mode == "pipeline":
        keys["producer_quant"] = replace(keys["old_k2"], producer_latent_quant=True)
        for workers in args.down_workers:
            keys[f"stream_down_{workers}"] = replace(
                keys["producer_quant"],
                down_workers=workers,
                shared_down_workers=16,
                parallel_shared_activation=True,
                late_shared_wait=True,
                full_shared_pipeline=args.full_shared,
            )
    elif args.mode == "shared-quant":
        for workers in args.quant_workers:
            keys[f"shared_quant_{workers}"] = replace(
                keys["old_k2"], latent_quant_workers=workers
            )
        if args.producer_quant:
            keys["producer_quant"] = replace(keys["old_k2"], producer_latent_quant=True)
    else:
        keys["activation_only"] = replace(
            keys["old_k2"], parallel_shared_activation=True
        )
        for workers in args.workers:
            keys[f"early_shared_{workers}"] = replace(
                keys["activation_only"], shared_down_workers=workers
            )
        keys["late_shared_wait"] = replace(
            keys[f"early_shared_{args.workers[0]}"], late_shared_wait=True
        )
    keys = {
        name: replace(key, nblocks=args.nblocks, reset=args.reset)
        for name, key in keys.items()
    }
    if args.arms:
        keys = {
            name: key
            for name, key in keys.items()
            if name == "old_k2" or name in args.arms
        }
        assert len(keys) > 1, args.arms
    scratch = {
        name: torch.zeros(
            (reference_k2 if name == "old_k2" else k2).scratch_bytes(
                replace(key, tokens=rows)
            ),
            dtype=torch.uint8,
            device=device,
        )
        for name, key in keys.items()
    }
    flush = torch.empty(
        args.flush_l2_mb * 1024 * 1024, dtype=torch.uint8, device=device
    )

    diag = torch.zeros(DIAG_BYTES, dtype=torch.uint8, device=device)

    def prepare():
        diag.zero_()
        prefix.copy_(prefix0)
        epoch.add_(1)

    control_x = torch.empty_like(prefix)
    timelines = {}

    def launch(name, collect_x=False):
        module = reference_k2 if name == "old_k2" else k2
        width = keys[name].tokens
        for first in range(0, rows, width):
            count = min(width, rows - first)
            key = replace(keys[name], tokens=count)
            part = slice(first, first + count)
            module.k2_launch(
                key,
                core=core[part],
                w_o=w_o,
                prefix=prefix[part],
                blocks=blocks[part],
                ares_nw=ares_nw,
                ares_qk=ares_qk,
                in_nw=in_nw,
                out=out[part],
                scratch=scratch[name],
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=3 + first // width,
                diag=diag,
                tl=timelines.get(name),
                **_moe_args(weights),
            )
            if collect_x:
                offset = module.scratch_layout(key)["xrow"][0]
                x = scratch[name][offset : offset + count * HIDDEN * 2].view(bf)
                control_x[part].copy_(x.view(count, HIDDEN))

    expected = None
    expected_prefix, expected_x = None, None
    errors = {}
    graphs = {}
    for name in keys:
        if rank == 0:
            print(f"Compile/check S{rows} {name}", flush=True)
        prepare()
        launch(name, collect_x=True)
        torch.accelerator.synchronize()
        waits = given_up_waits(diag, region_namer(k2.REGIONS))
        if waits:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(
                f".s{rows}.{name}.rank{rank}.waits.json"
            ).write_text(json.dumps(waits, indent=2))
            module = reference_k2 if name == "old_k2" else k2
            offset, size = module.scratch_layout(keys[name])["inter"]
            tagged = scratch[name][offset : offset + size].view(torch.int32).view(-1, 2)
            expected_tag = (int(epoch.item()) % (1 << 22)) * 256 + 4
            bad = (tagged[:, 1] != expected_tag).reshape(rows * 16, 12, 4)
            detail = dict(
                expected_tag=expected_tag,
                missing=bad.nonzero().cpu().tolist(),
                missing_per_group=bad.sum(dim=(0, 2)).cpu().tolist(),
                missing_per_word=bad.sum(dim=(0, 1)).cpu().tolist(),
                prefix_error=_rel_l2(prefix, expected_prefix)
                if expected_prefix is not None
                else None,
                x_error=_rel_l2(control_x, expected_x)
                if expected_x is not None
                else None,
            )
            args.output.with_suffix(
                f".s{rows}.{name}.rank{rank}.detail.json"
            ).write_text(json.dumps(detail, indent=2))
            raise RuntimeError((name, rank, waits[:12], detail["missing_per_group"]))
        if expected is None:
            expected = out.clone()
            expected_prefix = prefix.clone()
            expected_x = control_x.clone()
            reference = _moe_reference(control_x, weights, rank)
            cosine = _cos(out, reference)
            assert cosine > COS_MIN, (name, rank, cosine)
            errors["old_vs_production_moe_cosine"] = cosine
        errors[name] = {"cosine": _cos(out, expected), "rel_l2": _rel_l2(out, expected)}
        assert torch.equal(control_x, expected_x), (
            "MoE input",
            name,
            rank,
            _rel_l2(control_x, expected_x),
        )
        assert torch.equal(prefix, expected_prefix), ("prefix", name, rank)
        assert torch.equal(out, expected), (name, rank, errors[name])
        outputs = [torch.empty_like(out) for _ in range(8)]
        dist.all_gather(outputs, out)
        assert all(torch.equal(outputs[0], value) for value in outputs)
        if args.check_only:
            print(f"PASS S{rows} {name} rank{rank}", flush=True)
            continue
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
        assert torch.equal(out, expected), ("graph replay", name, rank)

    if args.check_only:
        dist.barrier()
        peers.close()
        return dict(
            rows=rows,
            rank=rank,
            checks=errors,
            configs={name: asdict(key) for name, key in keys.items()},
        )

    original_core = core.clone()
    core.add_(0.25 * (rank + 1))
    prepare()
    launch("old_k2")
    torch.accelerator.synchronize()
    changed_out, changed_prefix = out.clone(), prefix.clone()
    assert not torch.equal(changed_out, expected)
    for name, (graph, _, _) in graphs.items():
        for _ in range(3):
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out, changed_out), ("changed-input replay", name, rank)
            assert torch.equal(prefix, changed_prefix), ("prefix replay", name, rank)
        errors[name]["changed_input_replay"] = True
    core.copy_(original_core)

    samples = {name: [] for name in keys}
    latency = torch.zeros(1, device=device)
    names = list(keys)
    for repeat in range(args.repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            graph, start, end = graphs[name]
            values = []
            for _ in range(args.iterations):
                graph.replay()
                torch.accelerator.synchronize()
                latency.fill_(start.elapsed_time(end) * 1000)
                dist.all_reduce(latency, op=dist.ReduceOp.MAX)
                values.append(latency.item())
            samples[name].append(values)
        if rank == 0:
            print(
                "round",
                repeat,
                {
                    name: round(statistics.median(samples[name][-1]), 3)
                    for name in names
                },
                flush=True,
            )
    timeline_paths = []
    if args.timeline:
        for name in keys:
            if name == "old_k2":
                continue
            original_key = keys[name]
            keys[name] = replace(original_key, timeline=True)
            timelines[name] = torch.zeros(
                k2.BLOCKS * k2.TL_POINTS, dtype=torch.int64, device=device
            )
            prepare()
            launch(name)
            torch.accelerator.synchronize()
            dist.barrier()
            traced_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(traced_graph):
                prepare()
                timelines[name].zero_()
                flush.zero_()
                comm.all_reduce(sync_input, sync_output)
                launch(name)
            traced_graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out, expected), ("timeline output", name, rank)
            path = args.output.with_suffix(f".s{rows}.{name}.rank{rank}.timeline.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "config": asdict(keys[name]),
                        "rank": rank,
                        "clock_hz": 100_000_000,
                        "timestamps": timelines[name]
                        .view(k2.BLOCKS, k2.TL_POINTS)
                        .cpu()
                        .tolist(),
                        "method": "instrumented cold-L2 HIP graph; excluded from perf",
                    }
                )
                + "\n"
            )
            timeline_paths.append(str(path))
            keys[name] = original_key

    result = {
        "rows": rows,
        "rank": rank,
        "checks": errors,
        "configs": {name: asdict(key) for name, key in keys.items()},
        "median_us": {
            name: statistics.median([statistics.median(v) for v in rounds])
            for name, rounds in samples.items()
        },
        "samples_max_rank_us": samples,
        "timelines": timeline_paths,
    }
    dist.barrier()
    graphs.clear()
    torch.accelerator.synchronize()
    peers.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--nblocks", type=int, default=4)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--shared-route", action="store_true")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--rows", nargs="+", type=int, default=[6, 12, 8, 16])
    parser.add_argument("--workers", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument(
        "--mode",
        choices=[
            "wide",
            "wide-pipeline",
            "wide-latq",
            "wide-stream-x",
            "wide-front",
            "scheduling",
            "shared-quant",
            "pipeline",
            "tune",
            "oproj",
            "oproj-stream",
            "oproj-balance",
        ],
        default="scheduling",
    )
    parser.add_argument("--down-workers", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--full-shared", action="store_true")
    parser.add_argument("--down-batches", nargs="+", type=int, default=[2, 3, 4, 6, 8])
    parser.add_argument("--quant-workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--producer-quant", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--flush-l2-mb", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    cpu_group = dist.new_group(backend="gloo")
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(rank),
        "boundary": (
            "K2 kernel only; reset, epoch, L2 flush and device barrier outside events"
        ),
        "flush_l2_mb": args.flush_l2_mb,
        "inputs": "checkpoint weights, synthetic activations/residuals; TP8"
        if args.model
        else "seeded synthetic weights, activations and residuals; TP8",
        "checkpoint": str(args.model) if args.model else None,
        "layer": args.layer if args.model else None,
        "shared_route": args.shared_route,
        "results": [],
    }
    for rows in args.rows:
        report["results"].append(measure(args, rows, rank, cpu_group))
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
