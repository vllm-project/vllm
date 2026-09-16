# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MiniMax-M3 sparse prefill implementations on identical inputs.

The optional source arguments make it possible to compare historical source
snapshots without copying benchmark-only kernels into production modules.
Each source must export ``minimax_m3_sparse_attn``. Sources that expose
``_PREFILL_TILE_Q`` are swept over ``--tiles``.

Example::

    python benchmarks/kernels/benchmark_minimax_m3_sparse_prefill.py \
      --original-source /tmp/original.py \
      --separate-source /tmp/separate.py \
      --unified-source vllm/models/minimax_m3/common/ops/sparse_attn.py \
      --output /tmp/results.json
"""

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from vllm import envs
from vllm.triton_utils import triton

TOPK = 16
BLOCK_SIZE = 128
HEAD_DIM = 128
GQA_GROUP_SIZE = 8

# Historical source snapshots are loaded into the installed vLLM package. Add
# the new option's default when that package predates the option; each loaded
# module is subsequently assigned the tile under test by ``invoke``.
envs.environment_variables.setdefault("VLLM_MINIMAX_SPARSE_PREFILL_TILE_Q", lambda: 0)


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    query_len: int
    prefix_len: int


FULL_SHAPES = (
    Shape("q128_ctx8k", 1, 128, 8192),
    Shape("q512_ctx8k", 1, 512, 8192),
    Shape("q2k_ctx60k", 1, 2048, 60000),
    Shape("q8k_ctx58k", 1, 8192, 57600),
    Shape("b8_q1k_ctx8k", 8, 1024, 8192),
    Shape("b32_q256_ctx58k", 32, 256, 57600),
)


def load_source(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_topk(shape: Shape, shared: int) -> torch.Tensor:
    """Build valid unique per-row lists with a controlled shared prefix."""
    rows = shape.batch * shape.query_len
    result = torch.empty((1, rows, TOPK), dtype=torch.int32)
    row = 0
    for _ in range(shape.batch):
        for q_offset in range(shape.query_len):
            valid = min(
                TOPK,
                (shape.prefix_len + q_offset + BLOCK_SIZE) // BLOCK_SIZE,
            )
            common_count = min(shared, valid)
            ids = list(range(common_count))
            unique_count = valid - common_count
            unique_space = max(
                1,
                (shape.prefix_len + q_offset + BLOCK_SIZE) // BLOCK_SIZE - common_count,
            )
            for index in range(unique_count):
                ids.append(
                    common_count
                    + (q_offset * max(1, unique_count) + index) % unique_space
                )
            ids.extend([-1] * (TOPK - valid))
            result[0, row] = torch.tensor(ids, dtype=torch.int32)
            row += 1
    return result


def make_case(shape: Shape, shared: int, kv_dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(7)
    total_q = shape.batch * shape.query_len
    seq_len = shape.prefix_len + shape.query_len
    pages_per_request = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    q = torch.randn(
        total_q,
        GQA_GROUP_SIZE,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache = torch.randn(
        pages_per_request,
        1,
        BLOCK_SIZE,
        2 * HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(kv_dtype)
    topk = make_topk(shape, shared).cuda()
    block_table = torch.arange(
        pages_per_request, device="cuda", dtype=torch.int32
    ).expand(shape.batch, -1)
    cu_seqlens_q = torch.arange(
        0,
        total_q + 1,
        shape.query_len,
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = torch.full((shape.batch,), seq_len, device="cuda", dtype=torch.int32)
    prefix_lens = torch.full(
        (shape.batch,), shape.prefix_len, device="cuda", dtype=torch.int32
    )
    return {
        "q": q,
        "kv_cache": cache,
        "topk_idx": topk,
        "block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seq_lens": seq_lens,
        "prefix_lens": prefix_lens,
        "max_query_len": shape.query_len,
        "num_kv_heads": 1,
        "sm_scale": HEAD_DIM**-0.5,
        "output": torch.empty_like(q),
    }


def union_stats(topk: torch.Tensor, shape: Shape, tile: int) -> tuple[float, int]:
    values = topk.cpu()[0]
    sizes: list[int] = []
    for request in range(shape.batch):
        begin = request * shape.query_len
        end = begin + shape.query_len
        for q_begin in range(begin, end, tile):
            ids = values[q_begin : min(q_begin + tile, end)].flatten()
            sizes.append(int(torch.unique(ids[ids >= 0]).numel()))
    return sum(sizes) / len(sizes), max(sizes)


def invoke(module: ModuleType, tile: int, args: dict[str, object]) -> None:
    if hasattr(module, "_PREFILL_TILE_Q"):
        module._PREFILL_TILE_Q = tile
    module.minimax_m3_sparse_attn(**args)


def bench(
    module: ModuleType,
    tile: int,
    args: dict[str, object],
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float, float]:
    fn = lambda: invoke(module, tile, args)
    median, p20, p80 = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return float(median), float(p20), float(p80)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp8":
        return torch.float8_e4m3fn
    raise ValueError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument("--separate-source", type=Path, required=True)
    parser.add_argument("--unified-source", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--shared", type=int, nargs="+", default=[16, 8, 0])
    parser.add_argument("--kv-dtypes", nargs="+", default=["bf16", "fp8"])
    parser.add_argument("--suite", choices=["smoke", "full"], default="full")
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    modules = {
        "original": load_source("minimax_sparse_original", cli.original_source),
        "separate": load_source("minimax_sparse_separate", cli.separate_source),
        "unified": load_source("minimax_sparse_unified", cli.unified_source),
    }
    shapes = FULL_SHAPES if cli.suite == "full" else (FULL_SHAPES[1],)
    shared_values = cli.shared if cli.suite == "full" else [16, 0]
    dtype_names = cli.kv_dtypes if cli.suite == "full" else ["bf16"]
    results: list[dict[str, object]] = []

    for dtype_name in dtype_names:
        kv_dtype = dtype_from_name(dtype_name)
        for shape in shapes:
            for shared in shared_values:
                args = make_case(shape, shared, kv_dtype)
                original_output = args["output"]
                assert isinstance(original_output, torch.Tensor)
                invoke(modules["original"], 1, args)
                reference = original_output.clone()
                torch.accelerator.synchronize()
                original_ms, p20, p80 = bench(
                    modules["original"], 1, args, cli.warmup_ms, cli.rep_ms
                )
                original_row = {
                    "variant": "original",
                    "tile": 1,
                    "shape": shape.name,
                    "batch": shape.batch,
                    "query_len": shape.query_len,
                    "prefix_len": shape.prefix_len,
                    "total_q": shape.batch * shape.query_len,
                    "shared_topk": shared,
                    "kv_dtype": dtype_name,
                    "median_ms": original_ms,
                    "p20_ms": p20,
                    "p80_ms": p80,
                    "speedup": 1.0,
                    "max_abs_error": 0.0,
                }
                results.append(original_row)
                print(json.dumps(original_row), flush=True)

                for tile in cli.tiles:
                    mean_union, max_union = union_stats(args["topk_idx"], shape, tile)
                    for variant in ("separate", "unified"):
                        invoke(modules[variant], tile, args)
                        output = args["output"]
                        assert isinstance(output, torch.Tensor)
                        max_error = float(
                            (output.float() - reference.float()).abs().max()
                        )
                        if max_error > 0.03:
                            raise AssertionError(
                                f"{variant=} {tile=} {shape.name=} {max_error=}"
                            )
                        median, p20, p80 = bench(
                            modules[variant], tile, args, cli.warmup_ms, cli.rep_ms
                        )
                        row = {
                            "variant": variant,
                            "tile": tile,
                            "shape": shape.name,
                            "batch": shape.batch,
                            "query_len": shape.query_len,
                            "prefix_len": shape.prefix_len,
                            "total_q": shape.batch * shape.query_len,
                            "shared_topk": shared,
                            "mean_union": mean_union,
                            "max_union": max_union,
                            "kv_dtype": dtype_name,
                            "median_ms": median,
                            "p20_ms": p20,
                            "p80_ms": p80,
                            "speedup": original_ms / median,
                            "max_abs_error": max_error,
                        }
                        results.append(row)
                        print(json.dumps(row), flush=True)

                del args, reference
                torch.accelerator.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "sources": {
            "original": str(cli.original_source),
            "separate": str(cli.separate_source),
            "unified": str(cli.unified_source),
        },
        "results": results,
    }
    cli.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(results)} measurements to {cli.output}", flush=True)


if __name__ == "__main__":
    main()
