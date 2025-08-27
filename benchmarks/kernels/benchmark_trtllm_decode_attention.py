# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import os
from datetime import datetime
from typing import Optional

import flashinfer
import torch

from vllm.utils import round_up

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
FP8_DTYPE = torch.float8_e4m3fn
FP4_DTYPE = torch.uint8


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@torch.no_grad()
def benchmark_decode(
    dtype: torch.dtype,
    quant_dtypes: tuple[
        Optional[torch.dtype], Optional[torch.dtype], Optional[torch.dtype]
    ],
    batch_size: int,
    max_seq_len: int,
    num_heads: tuple[int, int] = (64, 8),
    head_size: int = 128,
    kv_layout: str = "HND",
    block_size: int = 16,
    warmup: int = 10,
    trials: int = 20,
):
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtypes
    q_quant_dtype = q_quant_dtype or dtype
    kv_quant_dtype = kv_quant_dtype or dtype
    o_quant_dtype = o_quant_dtype or dtype

    num_qo_heads, num_kv_heads = num_heads
    assert num_qo_heads % num_kv_heads == 0

    sm_scale = float(1.0 / (head_size**0.5))

    # large number to reduce kv_cache reuse
    NUM_BLOCKS = int(256000 / block_size)

    kv_cache_shape = None
    if kv_layout == "NHD":
        kv_cache_shape = (NUM_BLOCKS, 2, block_size, num_kv_heads, head_size)
    elif kv_layout == "HND":
        kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, block_size, head_size)
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")

    # Always using 1.0 scale to reflect the real perf in benchmarking
    q_scale = 1.0
    ref_query = torch.randn(batch_size, num_qo_heads, head_size, dtype=dtype)
    if q_quant_dtype == FP8_DTYPE:
        query, _ = to_float8(ref_query)
    else:
        query = ref_query

    kv_lens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)
    kv_lens[-1] = max_seq_len

    seq_lens = kv_lens
    max_seq_len = torch.max(seq_lens).item()

    # Always using 1.0 scale to reflect the real perf in benchmarking
    k_scale = v_scale = 1.0
    ref_kv_cache = torch.randn(kv_cache_shape, dtype=dtype)
    if kv_quant_dtype == FP8_DTYPE:
        kv_cache, _ = to_float8(ref_kv_cache)
    else:
        kv_cache = ref_kv_cache

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(batch_size):
        seq_len = seq_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)
    workspace_buffer = torch.zeros(1024 * 1024 * 1024, dtype=torch.int8)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_tensor_cores=True,
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_qo_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
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

    o_scale = 1.0
    o_sf_scale = None
    output_baseline = torch.empty(ref_query.shape, dtype=dtype)
    if o_quant_dtype == FP4_DTYPE:
        o_sf_scale = 500.0
        output_trtllm = flashinfer.utils.FP4Tensor(
            torch.empty(query.shape[:-1] + (query.shape[-1] // 2,), dtype=torch.uint8),
            torch.empty(
                (
                    round_up(query.shape[0], 128),
                    round_up(query.shape[1] * query.shape[2] // 16, 4),
                ),
                dtype=torch.float8_e4m3fn,
            ),
        )
    else:
        output_trtllm = torch.empty(query.shape, dtype=o_quant_dtype)

    def baseline_decode():
        return wrapper.run(
            ref_query,
            ref_kv_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            out=output_baseline,
        )

    def trtllm_decode():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=q_scale * k_scale * sm_scale,
            bmm2_scale=v_scale / o_scale,
            o_sf_scale=o_sf_scale,
            out=output_trtllm,
        )

    baseline_mean, baseline_std = time_fn(baseline_decode)
    trtllm_mean, trtllm_std = time_fn(trtllm_decode)

    # Calculate percentage speedup (positive means TRT is faster)
    speedup_percent = (baseline_mean - trtllm_mean) / baseline_mean

    print(
        f"\t{batch_size}\t{max_seq_len}\t{trtllm_mean:.3f}\t{trtllm_std.item():.3f}"
        f"\t{baseline_mean:.3f}\t{baseline_std.item():.3f}\t{speedup_percent:.3f}"
    )

    # Return results for CSV writing
    return {
        "batch_size": batch_size,
        "trtllm_mean": trtllm_mean,
        "trtllm_std": trtllm_std.item(),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std.item(),
        "speedup_percent": speedup_percent,
        "q_dtype": str(q_quant_dtype),
        "kv_cache_dtype": str(kv_quant_dtype),
        "output_dtype": str(o_quant_dtype),
        "block_size": block_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "max_seq_len": max_seq_len,
    }


def write_results_to_csv(results, filename=None):
    """Write benchmark results to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashinfer_trtllm_benchmark_{timestamp}.csv"

    fieldnames = [
        "batch_size",
        "trtllm_mean",
        "trtllm_std",
        "baseline_mean",
        "baseline_std",
        "speedup_percent",
        "q_dtype",
        "kv_cache_dtype",
        "output_dtype",
        "block_size",
        "num_kv_heads",
        "head_size",
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
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    max_seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    all_results = []

    dtype = torch.bfloat16
    quant_dtypes = [
        # (q_quant_dtype, kv_quant_dtype, o_quant_dtype)
        (None, None, None),
        (None, FP8_DTYPE, None),
        (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
        (FP8_DTYPE, FP8_DTYPE, FP4_DTYPE),
    ]

    for quant_dtype in quant_dtypes:
        q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtype
        q_quant_dtype = q_quant_dtype or dtype
        kv_quant_dtype = kv_quant_dtype or dtype
        o_quant_dtype = o_quant_dtype or dtype

        print(
            f"Running benchmark for q_dtype = {q_quant_dtype}, "
            f"kv_cache_dtype: {kv_quant_dtype}, "
            f"output_dtype: {o_quant_dtype}"
        )
        print(
            "\tbatch_size\tmax_seq_len\ttrtllm_mean\ttrtllm_std\tbaseline_mean\t"
            "baseline_std\tspeedup_percent"
        )
        for max_seq_len in max_seq_lens:
            for bs in batch_sizes:
                result = benchmark_decode(
                    dtype=dtype,
                    quant_dtypes=quant_dtype,
                    batch_size=bs,
                    max_seq_len=max_seq_len,
                )
                all_results.append(result)

    # Write all results to CSV
    write_results_to_csv(all_results)
