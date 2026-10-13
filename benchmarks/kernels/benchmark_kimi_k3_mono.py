# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time Kimi-K3 K1 against its numerical reference chain on MI355X.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_kimi_k3_mono.py \
        --rows 8 16 32 64 --output /tmp/kimi-k1.json

Uses the numerical suite's random weights. Graph replay excludes allocations,
JIT and state reset. The reference is not the production model's dispatch;
use serving benchmarks to assess end-to-end gains.
Packed projection modes prepare their immutable weights outside the timed region.

"""

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from tests.models.kimi_k3.test_amd_mono import (
    COS_MIN,
    _cos,
    _kda_inputs,
    _kda_reference,
    _rel_l2,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.kimi_k3.amd.mono.attention import kda
from vllm.models.kimi_k3.amd.mono.common.tiles import token_tiles
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated


def checkpoint_weights(path, layer, rank):
    """Read the BF16 K1 parameters with the production TP8 shard layout."""
    from safetensors import safe_open

    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"language_model.model.layers.{layer}."

    def read(name, shard=None):
        name = prefix + name
        with safe_open(path / index[name], framework="pt", device="cpu") as f:
            value = f.get_tensor(name) if shard is None else f.get_slice(name)[shard]
        return value.contiguous()

    rows = slice(rank * kda.PROJ, (rank + 1) * kda.PROJ)
    heads = slice(rank * kda.NH, (rank + 1) * kda.NH)
    projections = [read(f"self_attn.{part}_proj.weight", rows) for part in "qkvg"]
    projections += [
        read("self_attn.f_a_proj.weight"),
        read("self_attn.b_proj.weight", heads),
        torch.zeros(4, kda.HIDDEN, dtype=torch.bfloat16),
    ]
    weights = {
        "w_in": torch.cat(projections),
        "w_fb": read("self_attn.f_b_proj.weight", rows),
        "conv_w": torch.cat(
            [
                read(f"self_attn.{part}_conv1d.weight", rows).view(kda.PROJ, 4)
                for part in "qkv"
            ]
        ).float(),
        "a_log": read("self_attn.A_log").flatten()[heads].float(),
        "dt_bias": read("self_attn.dt_bias").flatten()[rows].float(),
        "on_w": read("self_attn.o_norm.weight").flatten(),
        "ares_nw": read("self_attention_res_norm.weight").flatten(),
        "ares_qk": read("self_attention_res_proj.weight").flatten(),
        "in_nw": read("input_layernorm.weight").flatten(),
    }
    return {
        name: value.to(
            device="cuda",
            dtype=torch.float32
            if name in ("conv_w", "a_log", "dt_bias")
            else torch.bfloat16,
        )
        for name, value in weights.items()
    }


def measure(
    rows,
    length,
    warmup,
    iterations,
    repeats,
    flush_l2_mb,
    nblocks=4,
    write_block=False,
    compare_state_rows=(),
    weights=None,
    check=False,
    compare_local_post=False,
    pipelines=(),
    head_workers=16,
    projection_chunks=2,
    kda_starts=(),
    pipeline_state_rows=0,
):
    inputs = _kda_inputs(nblocks, 1, False, rows, qlen=length)
    if weights is not None:
        for name, value in weights.items():
            assert value.shape == inputs[name].shape, (name, value.shape)
            assert value.dtype == inputs[name].dtype, (name, value.dtype)
        inputs.update(weights)
    packed_weight = None
    if any(pipeline & 33554432 for pipeline in pipelines):
        from vllm.models.kimi_k3.amd.mono.attention.parallel import (
            pack_projection_weights,
        )

        packed_weight = pack_projection_weights(inputs["w_in"])
    key = kda.KdaPreBuild(
        tokens=rows,
        qlen=length,
        nblocks=nblocks,
        delta=True,
        state_len=inputs["conv_state"].size(-1),
        state_rows=rows if length == 8 and rows in (32, 64) else 0,
        write_idx=nblocks if write_block else -1,
    )
    baseline = {
        name: inputs[name].clone()
        for name in ("prefix", "blocks", "conv_state", "rstate")
    }
    scratch = torch.zeros(kda.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    epoch = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, kda.PROJ, dtype=torch.bfloat16, device="cuda")
    eviction = (
        torch.empty(flush_l2_mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if flush_l2_mb
        else None
    )
    with set_current_vllm_config(VllmConfig()):
        norm = FusedRMSNormGated(kda.HD, activation="sigmoid").cuda()
    norm.weight.data.copy_(inputs["on_w"])

    def reset():
        for name, value in baseline.items():
            inputs[name].copy_(value)
        epoch.add_(1)

    def mono(build_key=key, workspace=scratch):
        kda.kda_pre(
            build_key,
            prefix=inputs["prefix"],
            delta=inputs["delta"],
            blocks=inputs["blocks"],
            ares_nw=inputs["ares_nw"],
            ares_qk=inputs["ares_qk"],
            in_nw=inputs["in_nw"],
            w_in=packed_weight if build_key.pipeline & 33554432 else inputs["w_in"],
            w_fb=inputs["w_fb"],
            conv_w=inputs["conv_w"],
            conv_state=inputs["conv_state"],
            a_log=inputs["a_log"],
            dt_bias=inputs["dt_bias"],
            on_w=inputs["on_w"],
            rstate=inputs["rstate"],
            st_idx=inputs["st_idx"],
            num_acc=inputs["num_acc"],
            core_out=out,
            scratch=workspace,
            layer=0,
            epoch=epoch,
        )

    variants = {
        "reference": lambda: _kda_reference(
            inputs, nblocks, rows, key.write_idx, norm=norm, inplace=True
        ),
        "mono": mono,
    }
    if compare_local_post:
        from vllm.models.kimi_k3.amd.mono.attention import local_post as kda_post
        from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

        def local_post():
            x = attn_res(
                inputs["prefix"],
                inputs["delta"],
                inputs["blocks"],
                inputs["ares_nw"],
                inputs["ares_qk"],
                inputs["in_nw"],
                nblocks,
                key.write_idx,
                1e-5,
                1e-5,
            )
            projected = torch.nn.functional.linear(x, inputs["w_in"])
            kda_post.run(key, projected, inputs, out)

        variants["local_post"] = local_post
    for state_rows in compare_state_rows:
        variant_key = replace(key, state_rows=state_rows)
        workspace = torch.zeros(
            kda.scratch_bytes(variant_key), dtype=torch.uint8, device="cuda"
        )
        variants[f"mono_v{state_rows}"] = lambda k=variant_key, w=workspace: mono(k, w)
    for pipeline in pipelines:
        for state_rows in [0, pipeline_state_rows] if pipeline_state_rows else [0]:
            variant_key = replace(
                key,
                pipeline=pipeline,
                head_workers=head_workers,
                projection_chunks=projection_chunks,
                state_rows=state_rows or key.state_rows,
            )
            workspace = torch.zeros(
                kda.scratch_bytes(variant_key), dtype=torch.uint8, device="cuda"
            )
            name = f"parallel_p{pipeline}_hw{head_workers}"
            if state_rows:
                name += f"_v{state_rows}"
            variants[name] = lambda k=variant_key, w=workspace: mono(k, w)
    for offset in kda_starts:
        variant_key = replace(key, pipeline=1024, kda0=offset)
        workspace = torch.zeros(
            kda.scratch_bytes(variant_key), dtype=torch.uint8, device="cuda"
        )
        variants[f"resident_kda{offset}"] = lambda k=variant_key, w=workspace: mono(
            k, w
        )
    errors = {}
    if check:
        reset()
        expected = _kda_reference(inputs, nblocks, rows, key.write_idx, norm=norm)
        for name, fn in variants.items():
            if name == "reference":
                continue
            print(f"Checking {name}: M{rows}/L{length}", flush=True)
            reset()
            fn()
            torch.testing.assert_close(
                inputs["prefix"],
                expected["prefix"],
                rtol=0,
                atol=0,
                msg=lambda detail, variant=name: f"{variant}: prefix\n{detail}",
            )
            torch.testing.assert_close(
                inputs["blocks"], expected["blocks"], rtol=0, atol=0
            )
            errs = {
                field: _rel_l2(inputs[field], expected[field])
                for field in ("conv_state", "rstate")
            }
            errs["core_cos"] = _cos(out, expected["core"])
            assert errs["conv_state"] < 1e-3 and errs["rstate"] < 1e-3, (name, errs)
            assert errs["core_cos"] > COS_MIN, (name, errs)
            errors[name] = errs
    graphs = {}
    for name, fn in variants.items():
        for _ in range(warmup):
            reset()
            fn()
        torch.accelerator.synchronize()
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    samples = {name: [] for name in variants}
    for repeat in range(repeats):
        order = list(variants)
        if repeat % 2:
            order.reverse()
        for name in order:
            events = []
            for _ in range(iterations):
                reset()
                if eviction is not None:
                    eviction.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                events.append((start, end))
            torch.accelerator.synchronize()
            samples[name].append(
                statistics.median(a.elapsed_time(b) * 1000 for a, b in events)
            )
    return {
        "rows": rows,
        "sequence_length": length,
        "nblocks": nblocks,
        "write_block": write_block,
        "projection_chunks": projection_chunks,
        "flush_l2_mb": flush_l2_mb,
        "median_us_per_run": samples,
        "numerics": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument(
        "--sequence-length", type=int, choices=(*range(1, 9), 16), default=8
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flush-l2-mb", type=int, default=0)
    parser.add_argument("--nblocks", type=int, choices=(*range(1, 9), 16), default=4)
    parser.add_argument("--write-block", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--rank", type=int, choices=range(8), default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compare-local-post", action="store_true")
    parser.add_argument(
        "--pipelines",
        type=int,
        nargs="*",
        choices=(
            *range(1, 9),
            16,
            32,
            64,
            96,
            128,
            256,
            512,
            520,
            768,
            1024,
            1536,
            2048,
            2560,
            4096,
            8192,
            16384,
            2113536,
            33562624,
            33570816,
            35667968,
            67108868,
            100671492,
            102776836,
            234889220,
            236994564,
        ),
        default=[],
    )
    parser.add_argument("--head-workers", type=int, choices=range(1, 33), default=16)
    parser.add_argument(
        "--pipeline-state-rows", type=int, choices=(0, 16, 32, 64), default=0
    )
    parser.add_argument(
        "--compare-state-rows", type=int, nargs="*", choices=(16, 32, 64), default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--projection-chunks", type=int, choices=(1, 2, 3, 4), default=2
    )
    parser.add_argument(
        "--kda-starts", type=int, nargs="*", choices=range(256), default=[]
    )
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1 or args.flush_l2_mb < 0:
        parser.error("counts must be positive and flush-l2-mb nonnegative")
    for rows in args.rows:
        token_tiles(rows, args.sequence_length, tile_tokens=16)
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "layer": args.layer if args.checkpoint else None,
        "rank": args.rank,
        "results": [],
    }
    weights = (
        checkpoint_weights(args.checkpoint, args.layer, args.rank)
        if args.checkpoint
        else None
    )
    for rows in args.rows:
        measured = measure(
            rows,
            args.sequence_length,
            args.warmup,
            args.iterations,
            args.repeats,
            args.flush_l2_mb,
            args.nblocks,
            args.write_block,
            args.compare_state_rows,
            weights,
            args.check,
            args.compare_local_post,
            args.pipelines,
            args.head_workers,
            args.projection_chunks,
            args.kda_starts,
            args.pipeline_state_rows,
        )
        result["results"].append(measured)
        print(json.dumps(measured), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
