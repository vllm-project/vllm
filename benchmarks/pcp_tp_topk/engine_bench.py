# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded same-engine A/B: replicated versus TP-row-sharded PCP Indexer."""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def set_mode(worker, mode):
    import importlib

    import vllm.v1.attention.backends.mla.indexer as indexer
    from vllm.model_executor.layers import tp_topk_publication as publication
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    # Benchmark-only: consistently bypass unsupported FI multicast allocation.
    for name in (
        "vllm.models.common.ops.fused_allreduce_rms_norm",
        "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm",
    ):
        importlib.import_module(name)._can_use_flashinfer = lambda *args: (False, 0)
    if not hasattr(publication, "_bench_original_lookup"):
        publication._bench_original_lookup = publication.get_topk_publication

    if not hasattr(indexer, "_bench_original_build"):
        indexer._bench_original_build = DeepseekV32IndexerMetadataBuilder.build
    stats = {"mode": mode, "sharded": 0, "prefill_rows": [], "direct_lookups": 0}
    worker._tp_indexer_stats = stats

    def lookup(buffer):
        result = (
            publication._bench_original_lookup(buffer) if mode == "direct" else None
        )
        if result is not None:
            stats["direct_lookups"] += 1
        return result

    publication.get_topk_publication = lookup

    def build(self, *args, **kwargs):
        result = indexer._bench_original_build(self, *args, **kwargs)
        if result.prefill is not None:
            sizes = result.prefill.row_shard_sizes
            stats["prefill_rows"].append(result.num_prefill_tokens)
            if mode == "baseline":
                result.prefill.row_shard_sizes = None
            elif sizes is not None:
                stats["sharded"] += 1
        return result

    DeepseekV32IndexerMetadataBuilder.build = build
    return stats


def get_stats(worker):
    return worker._tp_indexer_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pcp", type=int, default=2)
    parser.add_argument("--input", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        load_format="fastsafetensors",
        tensor_parallel_size=args.tp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=args.input + 32,
        max_num_seqs=1,
        max_num_batched_tokens=65536,
        enforce_eager=True,
        disable_log_stats=False,
        disable_custom_all_reduce=True,
        moe_backend="flashinfer_cutlass",
        attention_config={"mla_prefill_backend": "FLASH_ATTN"},
        kernel_config={"enable_jit_warmup": False, "enable_flashinfer_autotune": False},
        compilation_config={"mode": "NONE", "cudagraph_mode": "NONE"},
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    seed_tokens = tokenizer.encode(
        "Archive record: the access code is cedar-amber. "
        "The shipment was inspected at the northern depot. ",
        add_special_tokens=False,
    )
    tokens = (seed_tokens * (args.input // len(seed_tokens) + 1))[: args.input]
    sampling = SamplingParams(
        temperature=0, max_tokens=8, min_tokens=8, ignore_eos=True
    )
    records = []
    for i, mode in enumerate(
        ("baseline", "sharded", "direct", "baseline", "sharded", "direct")
    ):
        llm.collective_rpc(set_mode, args=(mode,))
        start = time.perf_counter()
        output = llm.generate([{"prompt_token_ids": tokens}], sampling, use_tqdm=False)[
            0
        ]
        elapsed = time.perf_counter() - start
        metrics = output.metrics
        assert metrics is not None and not metrics.is_corrupted
        stats = llm.collective_rpc(get_stats)
        record = {
            "mode": mode,
            "warmup": i < 3,
            "elapsed_s": elapsed,
            "ttft_s": metrics.first_token_latency,
            "tokens": list(output.outputs[0].token_ids),
            "route": stats,
        }
        records.append(record)
        args.output.write_text(
            json.dumps(
                {
                    "settings": vars(args) | {"output": str(args.output)},
                    "records": records,
                },
                indent=2,
            )
        )
        print(json.dumps(record), flush=True)
    assert records[-1]["tokens"] == records[-2]["tokens"] == records[-3]["tokens"]
    assert all(s["sharded"] > 0 for s in records[-1]["route"])
    assert all(s["direct_lookups"] > 0 for s in records[-1]["route"])


if __name__ == "__main__":
    main()
