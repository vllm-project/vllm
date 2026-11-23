#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune FlashInfer `mm_bf16` for every BF16 GEMM shape of a model.

The (N, K) shapes are read from the safetensors headers of the model on the
Hugging Face Hub (BF16 2D weights only, no weights are downloaded). Tensor
parallelism is applied by layer name: q/k/v, gate, up, in_proj* and lm_head
divide N by TP; o_proj, down_proj and out_proj divide K. Fused layers (qkv,
gate_up, in_proj_qkvz, in_proj_ba) are tuned as well as their parts.

Every shape is tuned with `mm_bf16(pdl=True, backend="auto")` for the M buckets
below (the sizes vLLM captures CUDA graphs for) and the winners are written to a
FlashInfer autotune JSON file. Put
the file into `vllm/model_executor/layers/gemmbf16_configs` and start vLLM with
`--linear-backend flashinfer`: matching shapes are then not tuned again at warmup.

Usage:
    python tools/tune_flashinfer_bf16_gemm.py --model M --tp 1 --out F.json
    python tools/tune_flashinfer_bf16_gemm.py --model M --tp 4 --mtp 1 2 3 --out F.json
    python tools/tune_flashinfer_bf16_gemm.py --model M --tp 1 4 --gpus 4 ...

With --gpus N the shapes are split over N worker processes, one per GPU, and the
results are merged into one file.

If the output file already exists, the (N, K, M) points it holds are kept and only
the missing ones are measured, so the script can be restarted after an
interruption or run again with more --mtp values. Results are saved after every
shape (with --gpus, into `<out>.parts/` until the final merge).
"""

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta

import regex as re
import torch

EXTRA_SHAPES = {
    "Qwen/Qwen3.5-397B-A17B": [
        (4608, 4096, (4,)),
    ],
    "Qwen/Qwen3.8-Flash-Next": [
        (320, 2560, (8,)),
        (336, 10240, (1, 2, 4, 8)),
        (1280, 2560, (2,)),
        (4224, 2560, (4,)),
        (12800, 2560, (1, 2, 4, 8)),
        (13952, 2560, (1,)),
    ],
    "deepseek-ai/DeepSeek-V4-Pro": [
        (512, 7168, (1, 8)),
        (1024, 7168, (2, 4)),
        (2048, 7168, (1, 2, 4, 8)),
        (7168, 384, (1, 4)),
        (7168, 512, (1, 4)),
        (7168, 1024, (1, 4)),
        (7168, 2048, (1, 4)),
        (16160, 7168, (8,)),
        (32320, 7168, (4,)),
        (64640, 7168, (2,)),
    ],
    "deepseek-ai/DeepSeek-V4.1-Flash": [
        (1024, 5120, (1, 2, 4, 8)),
        (5120, 384, (1, 4)),
        (5120, 512, (4,)),
        (5120, 1024, (1, 4)),
        (16160, 5120, (8,)),
        (32320, 5120, (4,)),
        (64640, 5120, (2,)),
    ],
    "meta-models/Muse-Spark-Open-816B-A42B": [
        (1024, 8192, (4,)),
        (2304, 8192, (8,)),
        (4608, 8192, (4,)),
        (9216, 8192, (2,)),
        (18432, 8192, (1,)),
    ],
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4": [
        (1024, 8192, (4,)),
        (2304, 8192, (8,)),
        (4608, 8192, (4,)),
        (9216, 8192, (2,)),
        (18432, 8192, (1,)),
    ],
    "moonshotai/Kimi-K3": [
        (24, 128, (4,)),
        (24, 512, (4,)),
        (96, 128, (1,)),
        (96, 512, (1,)),
        (1536, 128, (8,)),
        (2112, 7168, (1, 2, 4, 8)),
        (3072, 128, (4,)),
        (6144, 128, (2,)),
        (7168, 896, (1, 4)),
        (12448, 7168, (4,)),
        (49376, 7168, (1,)),
    ],
    "nvidia/DeepSeek-V4-Pro-NVFP4": [
        (512, 7168, (1, 2, 4)),
        (1024, 7168, (8,)),
        (2048, 7168, (1, 2, 4, 8)),
        (7168, 384, (1, 4)),
        (7168, 512, (1, 4)),
        (7168, 1024, (1, 4)),
        (7168, 2048, (1, 4)),
        (16160, 7168, (8,)),
        (32320, 7168, (4,)),
        (64640, 7168, (2,)),
    ],
    "nvidia/DeepSeek-V4.1-Flash-NVFP4": [
        (1024, 5120, (1, 2, 4, 8)),
        (5120, 384, (1, 4)),
        (5120, 512, (4,)),
        (5120, 1024, (1, 4)),
        (16160, 5120, (8,)),
        (32320, 5120, (4,)),
        (64640, 5120, (2,)),
    ],
    "nvidia/GLM-5.3-NVFP4": [
        (16, 192, (4,)),
        (16, 512, (4,)),
        (64, 192, (1,)),
        (64, 512, (1,)),
        (160, 6144, (1, 2, 4, 8)),
        (2624, 6144, (1, 2, 4, 8)),
        (6144, 256, (1, 4)),
        (8192, 2048, (4,)),
    ],
    "nvidia/Kimi-K3-NVFP4": [
        (24, 128, (4,)),
        (24, 512, (4,)),
        (96, 128, (1,)),
        (96, 512, (1,)),
        (7168, 896, (1, 4)),
        (7168, 1536, (8,)),
        (7168, 3072, (4,)),
        (7168, 6144, (2,)),
    ],
    "nvidia/Qwen3.5-397B-A17B-NVFP4": [
        (4608, 4096, (4,)),
    ],
    "nvidia/Qwen3.8-Flash-Next-NVFP4": [
        (320, 2560, (8,)),
        (336, 10240, (1, 2, 4, 8)),
        (1280, 2560, (2,)),
        (4224, 2560, (4,)),
        (12800, 2560, (1, 2, 4, 8)),
        (13952, 2560, (1,)),
    ],
    "zai-org/GLM-5.3": [
        (16, 192, (4,)),
        (16, 512, (4,)),
        (64, 192, (1,)),
        (64, 512, (1,)),
        (160, 6144, (1, 2, 4, 8)),
        (6144, 256, (1, 4)),
    ],
}


COLUMN_PARALLEL = {
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "w1",
    "w3",
    "fc1",
    "linear_fc1",
    "q_b_proj",
    "kv_b_proj",
    "lm_head",
    "fc",
}
ROW_PARALLEL = {"o_proj", "down_proj", "out_proj", "w2", "fc2", "linear_fc2"}
FUSED_LAYERS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_b", "in_proj_a"),
    ("w1", "w3"),
)
SKIPPED_NAMES = ("visual", "vision", "embed_tokens", "experts.")
VOCAB_PADDING = 64


def hf_linear_weights(model: str) -> dict[str, tuple[int, int]]:
    from huggingface_hub import get_safetensors_metadata

    weights: dict[str, tuple[int, int]] = {}
    meta = get_safetensors_metadata(model)
    for file_meta in meta.files_metadata.values():
        for name, tensor in file_meta.tensors.items():
            if (
                name.endswith(".weight")
                and len(tensor.shape) == 2
                and (tensor.dtype == "BF16")
            ):
                weights[name.removesuffix(".weight")] = (
                    tensor.shape[0],
                    tensor.shape[1],
                )
    embed = [v for k, v in weights.items() if k.endswith("embed_tokens")]
    weights = {
        k: v
        for k, v in weights.items()
        if "shared_experts." in k or not any(x in k for x in SKIPPED_NAMES)
    }
    if not any(k.split(".")[-1] == "lm_head" for k in weights) and embed:
        weights["lm_head"] = embed[0]
    return weights


def split_for_tp(leaf: str, n: int, k: int, tp: int) -> tuple[int, int]:
    if leaf == "lm_head":
        n = (n + VOCAB_PADDING - 1) // VOCAB_PADDING * VOCAB_PADDING
    if leaf in COLUMN_PARALLEL or leaf.startswith("in_proj"):
        return n // tp, k
    if leaf in ROW_PARALLEL:
        return n, k // tp
    return n, k


def shapes_from_hf(model: str, tps: list[int]) -> set[tuple[int, int]]:
    modules: dict[str, dict[str, tuple[int, int]]] = {}
    for name, (n, k) in hf_linear_weights(model).items():
        parent, _, leaf = name.rpartition(".")
        modules.setdefault(re.sub(r"\.\d+(?=\.|$)", ".#", parent), {})[leaf] = (n, k)
    shapes: set[tuple[int, int]] = set()
    for tp in tps:
        for leaves in modules.values():
            split = {leaf: split_for_tp(leaf, *nk, tp) for leaf, nk in leaves.items()}
            shapes.update(split.values())
            for group in FUSED_LAYERS:
                if (
                    all(g in split for g in group)
                    and len({split[g][1] for g in group}) == 1
                ):
                    shapes.add((sum(split[g][0] for g in group), split[group[0]][1]))
    if not shapes:
        raise SystemExit(f"no BF16 linear weights found for {model}")
    return shapes


def extra_shapes(model: str, tps: list[int]) -> set[tuple[int, int]]:
    return {
        (n, k)
        for n, k, shape_tps in EXTRA_SHAPES.get(model, [])
        if set(shape_tps) & set(tps)
    }


def collect_shapes(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.shapes_json:
        with open(args.shapes_json) as f:
            shapes = {(x[0], x[1]) for x in json.load(f)}
    else:
        shapes = shapes_from_hf(args.model, args.tp) | extra_shapes(args.model, args.tp)
    print(f"{len(shapes)} shapes")
    return sorted(shapes, key=lambda nk: -nk[0] * nk[1])


# CUDA graph token sizes of vLLM (`_set_cudagraph_sizes`) plus the default
# max_num_batched_tokens, which vLLM adds as one more bucket.
BASE_BUCKETS = (
    1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64,
    72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256,
    272, 288, 304, 320, 336, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640,
    672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1280, 1536,
    1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608,
    4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 8192,
)  # fmt: skip
MAX_NUM_BATCHED_TOKENS = 16384
# Largest CUDA graph (tokens) on data center Blackwell; uniform decode graphs
# with speculative decoding are only captured up to it.
MAX_DECODE_GRAPH_TOKENS = 1024


def mtp_decode_buckets(num_speculative_tokens: int) -> set[int]:
    """Token counts of uniform decode batches with this many draft tokens.

    A decode step verifies `1 + num_speculative_tokens` tokens per request, so
    vLLM also captures n * (1 + num_speculative_tokens) tokens for the request
    counts n of its decode grid (1, 2, 4, then multiples of 8 up to 256 and of
    16 above), as long as they fit into the largest decode graph.
    """
    query_len = 1 + num_speculative_tokens
    max_requests = MAX_DECODE_GRAPH_TOKENS
    counts = {n for n in (1, 2, 4) if n <= max_requests}
    counts |= set(range(8, min(max_requests + 1, 256), 8))
    counts |= set(range(256, max_requests + 1, 16))
    counts.add(max_requests)
    sizes = {n * query_len for n in counts}
    return {m for m in sizes if m <= MAX_DECODE_GRAPH_TOKENS}


def extra_buckets(num_speculative_tokens: list[int]) -> set[int]:
    """Decode sizes of the given MTP depths that the base buckets do not have."""
    extra: set[int] = set()
    for k in num_speculative_tokens:
        extra |= mtp_decode_buckets(k)
    return extra - set(BASE_BUCKETS) - {MAX_NUM_BATCHED_TOKENS}


def buckets_for(num_speculative_tokens: list[int]) -> tuple[int, ...]:
    """M buckets to tune: the base ones, max_num_batched_tokens, and the decode
    sizes of MTP that are not among them. The draft model decodes one token per
    request, so the plain sizes stay. max_num_batched_tokens is not a decode
    size and is never scaled."""
    buckets = set(BASE_BUCKETS) | {MAX_NUM_BATCHED_TOKENS}
    buckets |= extra_buckets(num_speculative_tokens)
    return tuple(sorted(buckets))


_POINT = re.compile(r"\(\((\d+), (\d+)\), \((\d+), (\d+)\)")


def measured_points(path: str) -> set[tuple[int, int, int]]:
    """(N, K, M) points that a tuning file already holds a winner for."""
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    points = set()
    for key in data:
        if not key.startswith("('bf16_gemm_pdl'"):
            continue
        match = _POINT.search(key)
        if match and match.group(2) == match.group(3):
            m, k, _, n = map(int, match.groups())
            points.add((n, k, m))
    return points


def tune(args: argparse.Namespace, shapes: list[tuple[int, int]]) -> None:
    from flashinfer import autotune, mm_bf16
    from flashinfer.autotuner import AutoTuner

    buckets = buckets_for(args.mtp)
    torch.accelerator.set_device_index(0)
    start = time.perf_counter()
    with (
        torch.inference_mode(),
        autotune(tune_mode=True, cache=args.out, tuning_buckets=buckets, round_up=True),
    ):
        for i, (n, k) in enumerate(shapes):
            t = time.perf_counter()
            a = torch.randn(buckets[-1], k, device="cuda", dtype=torch.bfloat16) * 0.1
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
            mm_bf16(a, w.T, None, pdl=True, out_dtype=torch.bfloat16, backend="auto")
            torch.accelerator.synchronize()
            AutoTuner.get().save_configs(args.out)  # checkpoint, kept if we are killed
            seconds = round(time.perf_counter() - t, 1)
            print(f"[{i + 1}/{len(shapes)}] n={n} k={k} {seconds} s", flush=True)
            del a, w
            torch.accelerator.empty_cache()
    total = time.perf_counter() - start
    elapsed = timedelta(seconds=round(total))
    print(f"tuned {len(shapes)} shapes in {elapsed} ({total:.0f} s) -> {args.out}")
    release_cudnn_graphs()


def split_shapes(
    shapes: list[tuple[int, int]], parts: int
) -> list[list[tuple[int, int]]]:
    """Greedy balance of the shapes over `parts` workers by weight size."""
    groups: list[list[tuple[int, int]]] = [[] for _ in range(parts)]
    load = [0] * parts
    for n, k in sorted(shapes, key=lambda nk: -nk[0] * nk[1]):
        i = load.index(min(load))
        groups[i].append((n, k))
        load[i] += n * k + (1 << 24)
    return [g for g in groups if g]


def merge_tuning_files(paths: list[str], out: str) -> None:
    """Merge FlashInfer autotune JSON files written by the workers."""
    metadata: dict = {}
    records: dict = {}
    configs: dict = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        metadata = metadata or data.get("_metadata", {})
        for namespace, items in data.get("_records", {}).items():
            records.setdefault(namespace, {}).update(items)
        configs.update({k: v for k, v in data.items() if not k.startswith("_")})
    merged: dict = {"_metadata": metadata}
    if records:
        merged["_records"] = records
    for key in sorted(configs):
        merged[key] = configs[key]
    source = json.dumps(merged, sort_keys=True, separators=(",", ":")).encode()
    merged["_generation"] = hashlib.sha256(source).hexdigest()
    with open(f"{out}.tmp", "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(f"{out}.tmp", out)


def tune_on_gpus(args: argparse.Namespace, shapes: list[tuple[int, int]]) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(args.gpus)]
    if len(devices) < args.gpus:
        sys.exit(f"--gpus {args.gpus} but only {len(devices)} GPUs are visible")
    groups = split_shapes(shapes, args.gpus)
    start = time.perf_counter()
    # The parts stay next to the output until the final merge, so an interrupted
    # run is continued from them. Every worker starts from everything known so far.
    parts_dir = f"{args.out}.parts"
    os.makedirs(parts_dir, exist_ok=True)
    known = [p for p in [args.out] if os.path.isfile(p)]
    known += sorted(
        os.path.join(parts_dir, f)
        for f in os.listdir(parts_dir)
        if f.startswith("part-") and f.endswith(".json")
    )
    seed = os.path.join(parts_dir, "seed.json")
    if known:
        merge_tuning_files(known, seed)
    workers = []
    for i, group in enumerate(groups):
        shapes_file = os.path.join(parts_dir, f"shapes-{i}.json")
        out_file = os.path.join(parts_dir, f"part-{i}.json")
        with open(shapes_file, "w") as f:
            json.dump(group, f)
        if known:
            shutil.copyfile(seed, out_file)
        command = [
            sys.executable,
            __file__,
            "--shapes-json",
            shapes_file,
            "--mtp",
            *map(str, args.mtp),
            "--out",
            out_file,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[i]}
        print(f"GPU {devices[i]}: {len(group)} shapes", flush=True)
        workers.append((subprocess.Popen(command, env=env), out_file))
    codes = [worker.wait() for worker, _ in workers]
    if any(codes):
        sys.exit(f"worker exit codes: {codes}; rerun to continue from {parts_dir}")
    merge_tuning_files([out_file for _, out_file in workers], args.out)
    for name in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, name))
    os.rmdir(parts_dir)
    total = time.perf_counter() - start
    elapsed = timedelta(seconds=round(total))
    print(
        f"tuned {len(shapes)} shapes on {len(groups)} GPUs in {elapsed} -> {args.out}"
    )


def release_cudnn_graphs() -> None:
    from flashinfer.gemm.gemm_base import clear_cudnn_graph_cache

    clear_cudnn_graph_cache()
    for name, module in list(sys.modules.items()):
        if name.startswith("flashinfer"):
            for obj in list(vars(module).values()):
                if hasattr(obj, "cache_clear"):
                    obj.cache_clear()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="", help="Hugging Face model name")
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--gpus", type=int, default=1, help="tune on this many GPUs in parallel"
    )
    parser.add_argument("--out", required=True, help="output FlashInfer tuning JSON")
    parser.add_argument(
        "--mtp",
        type=int,
        nargs="*",
        default=[],
        metavar="K",
        help="num_speculative_tokens of MTP, e.g. --mtp 1 2 3: also tune the "
        "decode sizes n * (1 + K) that the base buckets do not have",
    )
    parser.add_argument(
        "--shapes-json", default="", help="JSON list of [n, k] replacing the model"
    )
    args = parser.parse_args()
    shapes = collect_shapes(args)
    buckets = buckets_for(args.mtp)
    extra = sorted(extra_buckets(args.mtp))
    print(f"{len(buckets)} M buckets, {len(extra)} from --mtp {args.mtp}: {extra}")
    measured = measured_points(args.out)
    parts_dir = f"{args.out}.parts"
    if os.path.isdir(parts_dir):
        for name in os.listdir(parts_dir):
            if name.startswith("part-") and name.endswith(".json"):
                measured |= measured_points(os.path.join(parts_dir, name))
    todo = [
        (n, k) for n, k in shapes if any((n, k, m) not in measured for m in buckets)
    ]
    points = len(shapes) * len(buckets)
    done = sum((n, k, m) in measured for n, k in shapes for m in buckets)
    print(
        f"{done} of {points} points already measured, {len(todo)} of "
        f"{len(shapes)} shapes still have something to measure"
    )
    if not todo:
        return
    if args.gpus > 1:
        tune_on_gpus(args, todo)
    else:
        tune(args, todo)


if __name__ == "__main__":
    main()
