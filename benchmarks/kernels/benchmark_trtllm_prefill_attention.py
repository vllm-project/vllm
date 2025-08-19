# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import os
import random
from datetime import datetime

import flashinfer
import torch

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8

# KV Cache Layout for TRT-LLM
# kv_cache_shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@torch.no_grad()
def benchmark_prefill(
    num_seqs,
    max_seq_len,
    page_size=16,
    dtype=torch.bfloat16,
    kv_layout="HND",
    num_kv_heads=8,
    kv_cache_dtype="auto",
    head_dim=128,
    warmup=10,
    trials=20,
):
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    HEAD_GRP_SIZE = 8
    MAX_SEQ_LEN = max_seq_len

    # large number to reduce kv_cache reuse
    NUM_BLOCKS = int(256000 / page_size)

    workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8)

    num_qo_heads = num_kv_heads * HEAD_GRP_SIZE
    sm_scale = float(1.0 / (head_dim**0.5))

    q_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    q_lens[-1] = MAX_SEQ_LEN
    max_q_len = max(q_lens)
    q_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            torch.cumsum(
                torch.tensor(q_lens, dtype=torch.int32), dim=0, dtype=torch.int32
            ),
        ]
    )
    q = torch.randn(sum(q_lens), num_qo_heads, head_dim, dtype=dtype)

    kv_lens = [random.randint(0, MAX_SEQ_LEN) for _ in range(num_seqs)]
    kv_lens[-1] = MAX_SEQ_LEN

    seq_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, kv_lens)]
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_seq_len + page_size - 1) // page_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=dtype)
    k_scale = v_scale = 1.0

    if kv_cache_dtype.startswith("fp8"):
        kv_cache, _ = to_float8(kv_cache)

    output_trtllm = torch.empty(q.shape, dtype=dtype)

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = seq_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + page_size - 1) // page_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % page_size
        if kv_last_page_len == 0:
            kv_last_page_len = page_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    output_baseline = torch.empty(q.shape, dtype=dtype)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=kv_cache.dtype,
    )

    def time_fn(fn, warmup=10, trials=20):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for i in range(warmup):
            fn()
        for i in range(trials):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
        return sum(times) / len(times), torch.std(torch.tensor(times))

    def baseline_prefill():
        return wrapper.run(
            q, kv_cache, k_scale=k_scale, v_scale=v_scale, out=output_baseline
        )

    def trt_prefill():
        return flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_q_len=max_q_len,
            max_kv_len=max_seq_len,
            bmm1_scale=k_scale * sm_scale,
            bmm2_scale=v_scale,
            batch_size=num_seqs,
            cum_seq_lens_q=q_indptr,
            cum_seq_lens_kv=kv_indptr,
            out=output_trtllm,
        )

    trt_mean, trt_std = time_fn(trt_prefill)
    baseline_mean, baseline_std = time_fn(baseline_prefill)

    # Calculate percentage speedup (positive means TRT is faster)
    speedup_percent = (baseline_mean - trt_mean) / baseline_mean

    print(
        f"\t{num_seqs}\t{max_seq_len}\t{trt_mean:.5f}\t{trt_std.item():.5f}"
        f"\t{baseline_mean:.5f}\t{baseline_std.item():.5f}\t{speedup_percent:.5f}"
    )

    # Return results for CSV writing
    return {
        "num_seqs": num_seqs,
        "trt_mean": trt_mean,
        "trt_std": trt_std.item(),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std.item(),
        "speedup_percent": speedup_percent,
        "q_dtype": str(dtype),
        "kv_cache_dtype": kv_cache_dtype,
        "page_size": page_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
    }


def write_results_to_csv(results, filename=None):
    """Write benchmark results to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashinfer_trtllm_benchmark_{timestamp}.csv"

    fieldnames = [
        "num_seqs",
        "trt_mean",
        "trt_std",
        "baseline_mean",
        "baseline_std",
        "speedup_percent",
        "q_dtype",
        "kv_cache_dtype",
        "page_size",
        "num_kv_heads",
        "head_dim",
        "max_seq_len",
    ]

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"Results written to {filename}")


if __name__ == "__main__":
    num_seqs = [1, 4, 8, 16, 32, 64, 128, 256]
    max_seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    all_results = []

    print(
        "Running benchmark for q_dtype = bfloat16, kv_cache_dtype: bfloat16, "
        "output_dtype: bfloat16"
    )
    print(
        "\tnum_seqs\tmax_seq_len\ttrt_mean\ttrt_std\tbaseline_mean\t"
        "baseline_std\tspeedup_percent"
    )
    for max_seq_len in max_seq_lens:
        for bs in num_seqs:
            result = benchmark_prefill(
                bs,
                max_seq_len,
                dtype=torch.bfloat16,
                kv_cache_dtype="auto",
            )
            all_results.append(result)

    # Write all results to CSV
    write_results_to_csv(all_results)
