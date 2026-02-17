"""Benchmark block table kernel launch overhead.

Measures:
1. apply_staged_writes() time (N kernel launches for N groups)
2. gather_block_tables() time (1 kernel launch)
3. compute_slot_mappings() time (1 kernel launch)
4. gather + compute_slot_mappings combined (2 kernel launches)
5. fused gather+slot_mappings (1 kernel launch)
6. Total scheduling overhead per step
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm.v1.worker.gpu.block_table import BlockTables


def benchmark_block_table_ops(
    num_kv_cache_groups: int = 1,
    block_size: int = 16,
    max_num_reqs: int = 256,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 4096,
    num_reqs: int = 128,
    num_tokens: int = 2048,
    num_warmup: int = 50,
    num_iters: int = 200,
):
    device = torch.device("cuda:0")
    block_sizes = [block_size] * num_kv_cache_groups

    bt = BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        device=device,
    )

    # Setup: populate block tables with realistic data
    max_blocks = max_model_len // block_size
    for req_idx in range(num_reqs):
        num_blocks = min(req_idx + 1, max_blocks)
        block_ids = tuple(
            list(range(req_idx * max_blocks, req_idx * max_blocks + num_blocks))
            for _ in range(num_kv_cache_groups)
        )
        bt.append_block_ids(req_idx, block_ids, overwrite=True)
    bt.apply_staged_writes()

    # Create idx_mapping (identity for simplicity)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

    # Create query_start_loc (uniform distribution)
    tokens_per_req = num_tokens // num_reqs
    query_start_loc = torch.arange(
        0,
        num_reqs * tokens_per_req + 1,
        tokens_per_req,
        dtype=torch.int32,
        device=device,
    )[: num_reqs + 1]
    query_start_loc[-1] = num_tokens

    # Create positions (ensure non-negative)
    positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        n_tok = end - start
        seq_len = max(n_tok, (i + 1) * (max_model_len // num_reqs))
        positions[start:end] = torch.arange(seq_len - n_tok, seq_len)

    results = {}

    # Benchmark apply_staged_writes
    # First stage some writes
    for req_idx in range(min(num_reqs, 32)):
        new_blocks = tuple(
            [req_idx * max_blocks + max_blocks - 1] for _ in range(num_kv_cache_groups)
        )
        bt.append_block_ids(req_idx, new_blocks, overwrite=False)

    # Warmup
    for _ in range(num_warmup):
        bt.apply_staged_writes()
        # Re-stage writes for next iteration
        for req_idx in range(min(num_reqs, 32)):
            new_blocks = tuple(
                [req_idx * max_blocks + max_blocks - 1]
                for _ in range(num_kv_cache_groups)
            )
            bt.append_block_ids(req_idx, new_blocks, overwrite=False)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.apply_staged_writes()
        end_events[i].record()
        # Re-stage
        for req_idx in range(min(num_reqs, 32)):
            new_blocks = tuple(
                [req_idx * max_blocks + max_blocks - 1]
                for _ in range(num_kv_cache_groups)
            )
            bt.append_block_ids(req_idx, new_blocks, overwrite=False)

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["apply_staged_writes_ms"] = sum(times) / len(times)

    # Ensure writes are applied before gather/slot tests
    bt.apply_staged_writes()

    # Benchmark gather_block_tables
    for _ in range(num_warmup):
        bt.gather_block_tables(idx_mapping)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.gather_block_tables(idx_mapping)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["gather_block_tables_ms"] = sum(times) / len(times)

    # Benchmark compute_slot_mappings
    for _ in range(num_warmup):
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["compute_slot_mappings_ms"] = sum(times) / len(times)

    # Benchmark gather + compute_slot_mappings combined (separate kernels)
    for _ in range(num_warmup):
        bt.gather_block_tables(idx_mapping)
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.gather_block_tables(idx_mapping)
        bt.compute_slot_mappings(idx_mapping, query_start_loc, positions)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["gather_plus_slots_ms"] = sum(times) / len(times)

    # Benchmark fused gather+slot_mappings (single kernel)
    for _ in range(num_warmup):
        bt.gather_and_compute_slot_mappings(idx_mapping, query_start_loc, positions)

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        bt.gather_and_compute_slot_mappings(idx_mapping, query_start_loc, positions)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["fused_gather_slots_ms"] = sum(times) / len(times)

    results["savings_ms"] = (
        results["gather_plus_slots_ms"] - results["fused_gather_slots_ms"]
    )
    results["savings_pct"] = (
        results["savings_ms"] / results["gather_plus_slots_ms"] * 100
        if results["gather_plus_slots_ms"] > 0
        else 0
    )

    results["total_separate_ms"] = (
        results["apply_staged_writes_ms"] + results["gather_plus_slots_ms"]
    )
    results["total_fused_ms"] = (
        results["apply_staged_writes_ms"] + results["fused_gather_slots_ms"]
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark block table kernels")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save JSON results"
    )
    args = parser.parse_args()

    print("Block Table Kernel Benchmark")
    print("=" * 60)

    configs = [
        {
            "num_kv_cache_groups": 1,
            "num_reqs": 32,
            "num_tokens": 512,
            "label": "small_batch_1grp",
        },
        {
            "num_kv_cache_groups": 1,
            "num_reqs": 128,
            "num_tokens": 2048,
            "label": "medium_batch_1grp",
        },
        {
            "num_kv_cache_groups": 1,
            "num_reqs": 256,
            "num_tokens": 8192,
            "label": "large_batch_1grp",
        },
        {
            "num_kv_cache_groups": 2,
            "num_reqs": 128,
            "num_tokens": 2048,
            "label": "medium_batch_2grp",
        },
        {
            "num_kv_cache_groups": 4,
            "num_reqs": 128,
            "num_tokens": 2048,
            "label": "medium_batch_4grp",
        },
    ]

    all_results = {}

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n--- {label} ---")
        print(f"  Config: {cfg}")
        results = benchmark_block_table_ops(**cfg)
        all_results[label] = {**results, **cfg}
        for k, v in results.items():
            print(f"  {k}: {v:.4f} ms" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("| Config | Separate (ms) | Fused (ms) | Savings (us) | Savings (%) |")
    print("|--------|--------------|------------|-------------|-------------|")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        sep = r["gather_plus_slots_ms"]
        fused = r["fused_gather_slots_ms"]
        savings_us = r["savings_ms"] * 1000
        savings_pct = r["savings_pct"]
        print(
            f"| {key:25s} | {sep:.4f} | {fused:.4f} "
            f"| {savings_us:>8.1f} | {savings_pct:>8.1f}% |"
        )
