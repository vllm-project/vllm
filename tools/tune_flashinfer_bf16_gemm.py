#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune FlashInfer `mm_bf16` for every BF16 GEMM shape of a model.

The (N, K) shapes are read from the safetensors headers of the model on the
Hugging Face Hub (BF16 2D weights only, no weights are downloaded). Tensor
parallelism is applied by layer name: q/k/v, gate, up, in_proj* and lm_head
divide N by TP; o_proj, down_proj and out_proj divide K. Fused layers (qkv,
gate_up, in_proj_qkvz, in_proj_ba) are tuned as well as their parts.

Every shape is tuned with `mm_bf16(pdl=True, backend="auto")` for the given
M buckets and the winners are written to a FlashInfer autotune JSON file. Put
the file into the directory named by `VLLM_FLASHINFER_BF16_TUNING_DIR` and
start vLLM with `--linear-backend flashinfer`: matching shapes are then not
tuned again at warmup.

Usage:
    python tools/tune_flashinfer_bf16_gemm.py --model M --tp 1 --bs 1,2,4 --out F.json
"""

import argparse
import json
import time

import regex as re
import torch

EXTRA_SHAPES = {
    "Qwen/Qwen3.5-397B-A17B": [
        (1, 4096, (4,)),
        (32, 4096, (4,)),
        (512, 4096, (4,)),
        (1024, 8192, (4,)),
        (4096, 256, (4,)),
        (4096, 2048, (4,)),
        (4608, 4096, (4,)),
        (5120, 4096, (4,)),
        (62080, 4096, (4,)),
    ],
    "Qwen/Qwen3.8-Flash-Next": [
        (1, 2560, (1, 2, 4, 8)),
        (12, 2560, (8,)),
        (24, 2560, (4,)),
        (48, 2560, (2,)),
        (96, 2560, (1,)),
        (160, 2560, (8,)),
        (320, 2560, (4, 8)),
        (320, 10240, (1, 2, 4, 8)),
        (336, 10240, (1, 2, 4, 8)),
        (512, 2560, (1, 2, 4, 8)),
        (640, 2560, (2, 4)),
        (1280, 2560, (1, 2)),
        (2048, 2560, (8,)),
        (2560, 80, (8,)),
        (2560, 160, (4,)),
        (2560, 320, (2,)),
        (2560, 640, (1,)),
        (2560, 768, (8,)),
        (2560, 1536, (4,)),
        (2560, 2560, (1,)),
        (2560, 3072, (2,)),
        (2560, 6144, (1,)),
        (4096, 2560, (4,)),
        (4224, 2560, (4,)),
        (8192, 2560, (2,)),
        (10240, 320, (1, 2, 4, 8)),
        (12800, 2560, (1, 2, 4, 8)),
        (13952, 2560, (1,)),
        (16384, 2560, (1,)),
        (31040, 2560, (8,)),
        (62080, 2560, (4,)),
        (124160, 2560, (2,)),
        (248320, 2560, (1,)),
    ],
    "deepseek-ai/DeepSeek-V4-Pro": [
        (64, 7168, (1, 2, 4, 8)),
        (384, 7168, (1, 2, 4, 8)),
        (512, 7168, (1, 2, 4, 8)),
        (1024, 7168, (1, 2, 4, 8)),
        (2048, 7168, (1, 2, 4, 8)),
        (7168, 384, (1, 4)),
        (7168, 512, (1, 4)),
        (7168, 1024, (1, 4)),
        (7168, 2048, (1, 4)),
        (16160, 7168, (8,)),
        (32320, 7168, (4,)),
        (64640, 7168, (2,)),
        (129280, 7168, (1,)),
    ],
    "deepseek-ai/DeepSeek-V4.1-Flash": [
        (32, 5120, (1, 2, 4, 8)),
        (128, 512, (1, 2, 4, 8)),
        (384, 5120, (1, 2, 4, 8)),
        (512, 5120, (4,)),
        (1024, 5120, (1, 2, 4, 8)),
        (5120, 384, (1, 4)),
        (5120, 512, (4,)),
        (5120, 1024, (1, 4)),
        (16160, 5120, (8,)),
        (32320, 5120, (4,)),
        (64640, 5120, (2,)),
        (129280, 256, (4,)),
        (129280, 5120, (1,)),
    ],
    "meta-models/Muse-Spark-Open-816B-A42B": [
        (1024, 8192, (4,)),
        (2048, 4096, (1, 4)),
        (2304, 8192, (8,)),
        (4096, 8192, (1,)),
        (4608, 8192, (4,)),
        (8192, 1024, (8,)),
        (8192, 2048, (2, 4)),
        (8192, 4096, (1, 2)),
        (8192, 8192, (1,)),
        (9216, 8192, (2,)),
        (18432, 8192, (1,)),
        (25256, 8192, (8,)),
        (50512, 8192, (4,)),
        (101024, 8192, (2,)),
        (202048, 8192, (1,)),
    ],
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4": [
        (1024, 8192, (4,)),
        (2048, 4096, (1, 4)),
        (2304, 8192, (8,)),
        (4096, 8192, (1,)),
        (4608, 8192, (4,)),
        (8192, 1024, (8,)),
        (8192, 2048, (2, 4)),
        (8192, 4096, (1, 2)),
        (8192, 8192, (1,)),
        (9216, 8192, (2,)),
        (18432, 8192, (1,)),
        (25256, 8192, (8,)),
        (50512, 8192, (4,)),
        (101024, 8192, (2,)),
        (202048, 8192, (1,)),
    ],
    "moonshotai/Kimi-K3": [
        (24, 128, (4,)),
        (24, 512, (4,)),
        (96, 128, (1,)),
        (96, 512, (1,)),
        (896, 7168, (1, 2, 4, 8)),
        (1536, 128, (8,)),
        (1536, 7168, (8,)),
        (2112, 7168, (1, 2, 4, 8)),
        (2304, 1536, (8,)),
        (3072, 128, (4,)),
        (3072, 512, (8,)),
        (3072, 7168, (4,)),
        (3584, 7168, (1, 2, 4, 8)),
        (4608, 1536, (4,)),
        (6144, 128, (2,)),
        (6144, 512, (4,)),
        (6144, 7168, (2,)),
        (7168, 768, (8,)),
        (7168, 896, (1, 4)),
        (7168, 1536, (4, 8)),
        (7168, 3072, (2, 4)),
        (7168, 3584, (1, 4)),
        (7168, 4224, (8,)),
        (7168, 6144, (1, 2)),
        (7168, 8448, (4,)),
        (7168, 12288, (1,)),
        (7168, 16896, (2,)),
        (7168, 33792, (1,)),
        (8448, 7168, (8,)),
        (9216, 1536, (2,)),
        (12288, 128, (1,)),
        (12288, 512, (2,)),
        (12288, 7168, (1,)),
        (12448, 7168, (4,)),
        (16896, 7168, (4,)),
        (18432, 1536, (1,)),
        (24576, 512, (1,)),
        (33792, 7168, (2,)),
        (49376, 7168, (1,)),
        (67584, 7168, (1,)),
        (163840, 7168, (1,)),
    ],
    "nvidia/DeepSeek-V4-Pro-NVFP4": [
        (64, 7168, (1, 2, 4, 8)),
        (384, 7168, (1, 2, 4, 8)),
        (512, 7168, (1, 2, 4, 8)),
        (1024, 7168, (1, 2, 4, 8)),
        (2048, 7168, (1, 2, 4, 8)),
        (7168, 384, (1, 4)),
        (7168, 512, (1, 4)),
        (7168, 1024, (1, 4)),
        (7168, 2048, (1, 4)),
        (16160, 7168, (8,)),
        (32320, 7168, (4,)),
        (64640, 7168, (2,)),
        (129280, 7168, (1,)),
    ],
    "nvidia/DeepSeek-V4.1-Flash-NVFP4": [
        (32, 5120, (1, 2, 4, 8)),
        (128, 512, (1, 2, 4, 8)),
        (384, 5120, (1, 2, 4, 8)),
        (512, 5120, (4,)),
        (1024, 5120, (1, 2, 4, 8)),
        (5120, 384, (1, 4)),
        (5120, 512, (4,)),
        (5120, 1024, (1, 4)),
        (16160, 5120, (8,)),
        (32320, 5120, (4,)),
        (64640, 5120, (2,)),
        (129280, 256, (4,)),
        (129280, 5120, (1,)),
    ],
    "nvidia/GLM-5.3-NVFP4": [
        (16, 192, (4,)),
        (16, 512, (4,)),
        (64, 192, (1,)),
        (64, 512, (1,)),
        (160, 6144, (1, 2, 4, 8)),
        (256, 6144, (1, 2, 4, 8)),
        (512, 6144, (8,)),
        (1024, 6144, (4,)),
        (2048, 2048, (8,)),
        (2048, 6144, (2,)),
        (2624, 6144, (1, 2, 4, 8)),
        (4096, 2048, (1, 2, 4, 8)),
        (4096, 6144, (1,)),
        (6144, 256, (1, 4, 8)),
        (6144, 512, (4,)),
        (6144, 1024, (2,)),
        (6144, 2048, (1, 8)),
        (6144, 4096, (4,)),
        (6144, 8192, (2,)),
        (6144, 12288, (1, 2, 4, 8)),
        (6144, 16384, (1,)),
        (7168, 512, (4,)),
        (8192, 2048, (2, 4)),
        (16384, 2048, (1,)),
        (19360, 6144, (8,)),
        (38720, 6144, (4,)),
        (77440, 6144, (2,)),
        (154880, 6144, (1,)),
    ],
    "nvidia/Kimi-K3-NVFP4": [
        (24, 128, (4,)),
        (24, 512, (4,)),
        (96, 128, (1,)),
        (96, 512, (1,)),
        (896, 7168, (1, 2, 4, 8)),
        (1536, 7168, (8,)),
        (3072, 7168, (4,)),
        (3584, 7168, (1, 2, 4, 8)),
        (6144, 7168, (2,)),
        (7168, 768, (8,)),
        (7168, 896, (1, 4)),
        (7168, 1536, (4, 8)),
        (7168, 3072, (2, 4)),
        (7168, 3584, (1, 4)),
        (7168, 4224, (8,)),
        (7168, 6144, (1, 2)),
        (7168, 8448, (4,)),
        (7168, 16896, (2,)),
        (7168, 33792, (1,)),
        (8448, 7168, (8,)),
        (12288, 7168, (1,)),
        (16896, 7168, (4,)),
        (20480, 7168, (8,)),
        (33792, 7168, (2,)),
        (40960, 7168, (4,)),
        (67584, 7168, (1,)),
        (81920, 7168, (2,)),
        (163840, 7168, (1,)),
    ],
    "nvidia/Qwen3.5-397B-A17B-NVFP4": [
        (1, 4096, (4,)),
        (32, 4096, (4,)),
        (512, 4096, (4,)),
        (1024, 8192, (4,)),
        (4096, 256, (4,)),
        (4096, 2048, (4,)),
        (4608, 4096, (4,)),
        (5120, 4096, (4,)),
        (62080, 4096, (4,)),
    ],
    "nvidia/Qwen3.8-Flash-Next-NVFP4": [
        (1, 2560, (1, 2, 4, 8)),
        (12, 2560, (8,)),
        (24, 2560, (4,)),
        (48, 2560, (2,)),
        (96, 2560, (1,)),
        (160, 2560, (8,)),
        (320, 2560, (4, 8)),
        (320, 10240, (1, 2, 4, 8)),
        (336, 10240, (1, 2, 4, 8)),
        (512, 2560, (1, 2, 4, 8)),
        (640, 2560, (2, 4)),
        (1280, 2560, (1, 2)),
        (2048, 2560, (8,)),
        (2560, 80, (8,)),
        (2560, 160, (4,)),
        (2560, 320, (2,)),
        (2560, 640, (1,)),
        (2560, 768, (8,)),
        (2560, 1536, (4,)),
        (2560, 2560, (1,)),
        (2560, 3072, (2,)),
        (2560, 6144, (1,)),
        (4096, 2560, (4,)),
        (4224, 2560, (4,)),
        (8192, 2560, (2,)),
        (10240, 320, (1, 2, 4, 8)),
        (12800, 2560, (1, 2, 4, 8)),
        (13952, 2560, (1,)),
        (16384, 2560, (1,)),
        (31040, 2560, (8,)),
        (62080, 2560, (4,)),
        (124160, 2560, (2,)),
        (248320, 2560, (1,)),
    ],
    "zai-org/GLM-5.3": [
        (16, 192, (4,)),
        (16, 512, (4,)),
        (64, 192, (1,)),
        (64, 512, (1,)),
        (160, 6144, (1, 2, 4, 8)),
        (256, 6144, (1, 2, 4, 8)),
        (6144, 256, (1, 4)),
        (6144, 12288, (1, 2, 4, 8)),
        (19360, 6144, (8,)),
        (38720, 6144, (4,)),
        (77440, 6144, (2,)),
        (154880, 6144, (1,)),
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
    usable = [(n, k) for n, k in shapes if n % 8 == 0 and k % 8 == 0]
    skipped = sorted(shapes - set(usable))
    print(f"{len(usable)} shapes, skipped {len(skipped)} (N or K % 8 != 0): {skipped}")
    return sorted(usable, key=lambda nk: -nk[0] * nk[1])


def parse_buckets(values: list[str]) -> tuple[int, ...]:
    return tuple(sorted({int(x) for v in values for x in v.replace(",", " ").split()}))


def tune(args: argparse.Namespace, shapes: list[tuple[int, int]]) -> None:
    from flashinfer import autotune, mm_bf16

    buckets = parse_buckets(args.bs)
    torch.accelerator.set_device_index(0)
    start = time.perf_counter()
    done = []
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
            seconds = round(time.perf_counter() - t, 1)
            done.append({"n": n, "k": k, "seconds": seconds})
            print(f"[{i + 1}/{len(shapes)}] n={n} k={k} {seconds} s", flush=True)
            del a, w
            torch.accelerator.empty_cache()
    total = time.perf_counter() - start
    print(f"tuned {len(shapes)} shapes in {total:.0f} s -> {args.out}")
    summary = {
        "model": args.model,
        "tp": args.tp,
        "sizes": list(buckets),
        "shapes": done,
    }
    with open(args.out + ".shapes.json", "w") as f:
        json.dump(summary, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="", help="Hugging Face model name")
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    parser.add_argument("--out", required=True, help="output FlashInfer tuning JSON")
    parser.add_argument(
        "--bs", nargs="+", required=True, help="M buckets, e.g. --bs 1,2,4,8"
    )
    parser.add_argument(
        "--shapes-json", default="", help="JSON list of [n, k] replacing the model"
    )
    args = parser.parse_args()
    tune(args, collect_shapes(args))


if __name__ == "__main__":
    main()
