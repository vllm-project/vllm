# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile DSV4.1 metadata builders without loading model weights.

Run before and after a change with distinct --output directories. Timings cover
the complete builder calls, including allocations and host metadata work.
"""

import argparse
import dataclasses
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.config import AttentionConfig
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def make_case(name, batch_size):
    spec = "dspark" in name
    adaptive = "adaptive" in name
    query_lens = ([5 if "draft" in name else 6] if spec else [1]) * batch_size
    if name in ("prefill", "chunked_prefill"):
        query_lens = [2048] * batch_size
    elif name == "mixed":
        query_lens[-1] = 2048
    cpu_lens = list(query_lens)
    if adaptive and "draft" not in name:
        cpu_lens = [3] * batch_size
        # Budget redistribution preserves the total but changes device boundaries.
        query_lens = [1, 5] * (batch_size // 2) + [3] * (batch_size % 2)
    n = sum(query_lens)

    def tensor(x):
        return torch.tensor(x, dtype=torch.int32, device="cuda")

    qsl_cpu = torch.tensor([0] + cpu_lens, dtype=torch.int32).cumsum(0).int()
    qsl = tensor([0] + query_lens).cumsum(0).int()
    prefixes = [
        (0 if name == "prefill" else 4096)
        + (i % 7 if adaptive and "draft" in name else 0)
        for i in range(batch_size)
    ]
    upper_lens = [6] * batch_size if adaptive and "draft" not in name else cpu_lens
    seq_cpu = torch.tensor(
        [p + q for p, q in zip(prefixes, upper_lens)], dtype=torch.int32
    )
    seq = tensor([p + q for p, q in zip(prefixes, query_lens)])
    bt = torch.arange(batch_size * 64, device="cuda", dtype=torch.int32).view(
        batch_size, 64
    )
    positions = torch.cat(
        [torch.arange(p, p + q, device="cuda") for p, q in zip(prefixes, query_lens)]
    )
    cm = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq,
        seq_lens_cpu_upper_bound=seq_cpu,
        num_reqs=batch_size,
        num_actual_tokens=n,
        max_query_len=max(upper_lens),
        max_seq_len=int(seq_cpu.max()),
        block_table_tensor=bt,
        slot_mapping=torch.arange(n, device="cuda"),
        positions=positions,
        causal="draft" not in name,
    )
    config = NS(
        model_config=NS(
            max_model_len=8192,
            hf_config=NS(sliding_window=128, compress_ratios=[0, 1, 2], index_topk=512),
        ),
        scheduler_config=NS(max_num_batched_tokens=max(n, 8192), max_num_seqs=256),
        parallel_config=NS(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        attention_config=AttentionConfig(),
        num_speculative_tokens=5 if spec else 0,
        speculative_config=NS(
            num_speculative_tokens=5,
            parallel_drafting=False,
            enable_adaptive_verification=adaptive,
            use_dspark=lambda: True,
        )
        if spec
        else None,
    )
    device = torch.device("cuda:0")
    swa = DeepseekV41SparseSWAMetadataBuilder(
        SlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        ),
        [],
        config,
        device,
    )
    builders = [("swa", swa)]
    for ratio in (1, 2):
        kv = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
        )
        builders.extend(
            [
                (
                    f"mla{ratio}",
                    DeepseekV4SparseMLAMetadataBuilder(kv, [], config, device),
                ),
                (
                    f"indexer{ratio}",
                    DeepseekV32IndexerMetadataBuilder(
                        kv, [], config, device, block_table_width=64
                    ),
                ),
            ]
        )

    def run():
        outputs = []
        for _, builder in builders:
            # Each cache group receives fresh common metadata in the runner.
            cm._token_to_req_indices_cache = None
            outputs.append(builder.build(0, cm))
        return outputs

    return run, builders, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "prefill",
            "chunked_prefill",
            "mixed",
            "decode",
            "dspark",
            "dspark_adaptive",
            "dspark_draft",
            "dspark_draft_adaptive",
        ],
    )
    args = parser.parse_args()
    if args.baseline_source:

        def load(name):
            spec = importlib.util.spec_from_file_location(
                f"_baseline_{name}", args.baseline_source / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

        original_backend = load("backend")
        original_indexer = load("indexer")
        CommonAttentionMetadata.token_to_req_indices = (
            original_backend.CommonAttentionMetadata.token_to_req_indices
        )
        DeepseekV32IndexerMetadataBuilder.build = (
            original_indexer.DeepseekV32IndexerMetadataBuilder.build
        )
        original_builder = original_indexer.DeepseekV32IndexerMetadataBuilder
        DeepseekV32IndexerMetadataBuilder._build_varlen_decode_indices = (
            original_builder._build_varlen_decode_indices
        )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in args.cases:
        for batch in args.batches:
            run, builders, cm = make_case(case, batch)
            for _ in range(5):
                run()
            torch.accelerator.synchronize()

            def tensors(value, prefix=""):
                result = {}
                if isinstance(value, torch.Tensor):
                    result[prefix] = value.cpu().clone()
                elif dataclasses.is_dataclass(value):
                    for field in dataclasses.fields(value):
                        if not field.name.startswith("tile_sched"):
                            result.update(
                                tensors(
                                    getattr(value, field.name), f"{prefix}.{field.name}"
                                )
                            )
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        result.update(tensors(item, f"{prefix}.{i}"))
                return result

            actual = tensors(run())
            snapshot = f"{case}_b{batch}.pt"
            if args.reference:
                expected = torch.load(args.reference / snapshot, weights_only=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.save(actual, args.output / snapshot)
            if args.validate_only:
                print(f"{case} batch={batch}: metadata validated", flush=True)
                continue
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(3):
                    with torch.profiler.record_function(f"{case}_b{batch}"):
                        for label, builder in builders:
                            cm._token_to_req_indices_cache = None
                            with torch.profiler.record_function(label):
                                builder.build(0, cm)
                torch.accelerator.synchronize()
            prof.export_chrome_trace(str(args.output / f"{case}_b{batch}.json"))
            (args.output / f"{case}_b{batch}.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
            )
            wall = []
            for _ in range(30):
                start = time.perf_counter()
                run()
                torch.accelerator.synchronize()
                wall.append((time.perf_counter() - start) * 1e6)
            gpu = bench_gpu_time_with_cupti(
                run, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=100
            )
            row = dict(
                case=case,
                batch=batch,
                tokens=cm.num_actual_tokens,
                wall_us=statistics.median(wall),
                graph_gpu_us=statistics.median(gpu) * 1000,
            )
            results.append(row)
            print(row, flush=True)
            (args.output / "results.json").write_text(
                json.dumps(
                    dict(
                        gpu=torch.cuda.get_device_name(),
                        torch=torch.__version__,
                        results=results,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
