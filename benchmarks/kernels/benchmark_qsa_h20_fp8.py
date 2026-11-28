# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare QSA operators with a saved baseline on their H20 FP8 layouts."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import warnings
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.qwen4_exp.nvidia.ops import qsa, qsa_indexer


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ops-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-stride", type=int, default=10444800)
    parser.add_argument("--logits-width", type=int, default=8192)
    args = parser.parse_args()
    assert not args.output.exists()
    assert qsa._is_h20(), "this benchmark requires a full 78-SM H20"
    assert args.block_stride >= 1600 * 512
    base = {
        "attention": load(args.baseline_ops_dir / "qsa.py", "qsa_baseline"),
        "indexer": load(args.baseline_ops_dir / "qsa_indexer.py", "indexer_baseline"),
    }
    blocks = 3930
    arena = torch.empty(
        (blocks - 1) * args.block_stride + 1600 * 512,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": "CUPTI, CUDA Graph, cold L2; 2 interleaved rounds x31 per seed",
        "block_stride": args.block_stride,
        "sources": {},
        "rows": [],
        "status": "PARTIAL",
    }
    for label, module in {
        **base,
        "candidate_attention": qsa,
        "candidate_indexer": qsa_indexer,
    }.items():
        path = Path(module.__file__)
        result["sources"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        for family in ("attention", "indexer"):
            for rows in (1, 7, 8) if family == "attention" else (1, 127, 4096):
                for seed in (617, 20261125):
                    torch.manual_seed(seed)
                    if family == "attention":
                        query = torch.randn(
                            rows, 6, 256, device="cuda", dtype=torch.bfloat16
                        )
                        keys = arena.as_strided(
                            (blocks, 1600, 1, 256), (args.block_stride, 512, 512, 1)
                        )
                        values = arena.as_strided(keys.shape, keys.stride(), 256)
                        for tensor in (keys, values):
                            tensor[-2:].copy_(
                                torch.randn_like(tensor[-2:], dtype=torch.float32)
                            )
                        table = torch.tensor(
                            [[blocks - 2, blocks - 1]], device="cuda", dtype=torch.int32
                        )
                        indices = torch.empty(
                            rows, 2049, device="cuda", dtype=torch.int32
                        )
                        indices[:, :2048] = torch.randperm(3200, device="cuda")[:2048]
                        indices[:, -1] = 2048
                        inputs = dict(
                            q=query,
                            k_cache=keys,
                            v_cache=values,
                            indices=indices,
                            block_table=table,
                            token_to_req=torch.zeros(
                                rows, device="cuda", dtype=torch.int32
                            ),
                            use_prefill_config=False,
                            k_scale=0.5,
                            v_scale=2.0,
                            output_gate=torch.randn_like(query),
                        )
                        output = torch.empty_like(query)
                        functions = {
                            "baseline": partial(
                                base[family].qsa_sparse_paged_attention,
                                **inputs,
                                out=output,
                            ),
                            "candidate": partial(
                                qsa.qsa_sparse_paged_attention, **inputs, out=output
                            ),
                        }
                    else:
                        width = args.logits_width
                        pages = math.ceil(width / 400)
                        assert 0 < pages <= blocks
                        keys = arena.as_strided(
                            (blocks, 400, 1, 128), (args.block_stride, 512, 512, 1)
                        )
                        keys[-pages:].copy_(
                            torch.randn_like(keys[-pages:], dtype=torch.float32)
                        )
                        query = torch.randn(rows, 4, 128, device="cuda").to(
                            torch.float8_e4m3fn
                        )
                        inputs = dict(
                            q=query,
                            k_cache=keys,
                            page_table=torch.arange(
                                blocks - pages, blocks, device="cuda", dtype=torch.int32
                            )[None],
                            query_start_loc=torch.tensor(
                                [0, rows], device="cuda", dtype=torch.int32
                            ),
                            visible_blocks=torch.arange(
                                1, rows + 1, device="cuda", dtype=torch.int32
                            )
                            .mul(8)
                            .clamp_max(width),
                            max_query_len=rows,
                            logits_width=width,
                            query_offset=0,
                            num_queries=rows,
                        )
                        output = torch.full((rows * width,), torch.nan, device="cuda")
                        functions = {
                            "baseline": partial(
                                base[family]._prefill_logits,
                                **inputs,
                                logits_workspace=output,
                            ),
                            "candidate": partial(
                                qsa_indexer._prefill_logits,
                                **inputs,
                                logits_workspace=output,
                            ),
                        }
                    reference = functions["baseline"]().clone()
                    actual = functions["candidate"]()
                    torch.testing.assert_close(
                        actual, reference, atol=2e-2, rtol=2e-2, equal_nan=True
                    )
                    samples = {label: [] for label in functions}
                    for repeat in range(2):
                        order = list(functions)
                        random.Random(seed + repeat).shuffle(order)
                        for label in order:
                            with warnings.catch_warnings(record=True) as caught:
                                times = bench_gpu_time_with_cupti(
                                    functions[label],
                                    cold_l2_cache=True,
                                    use_cuda_graph=True,
                                    dry_run_iters=8,
                                    repeat_iters=31,
                                )
                            assert not any(
                                "fall" in str(w.message).lower()
                                or "disabled" in str(w.message).lower()
                                for w in caught
                            ), caught
                            assert len(times) == 31 and all(
                                math.isfinite(float(x)) and x > 0 for x in times
                            )
                            samples[label].extend(float(x) * 1000 for x in times)
                    row = dict(
                        family=family, query_rows=rows, seed=seed, samples_us=samples
                    )
                    for label in functions:
                        row[label + "_us"] = statistics.median(samples[label])
                    row["speedup"] = row["baseline_us"] / row["candidate_us"]
                    result["rows"].append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                        flush=True,
                    )
        result["status"] = "PASSED"
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
