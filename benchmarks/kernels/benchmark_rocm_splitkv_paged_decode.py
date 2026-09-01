# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare Triton and native ROCm paged-decode kernels on gfx1x."""

import argparse
import math

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    kernel_paged_attention_2d,
    paged_attention_2d_splitkv_decode,
    paged_attention_rocm_splitkv_decode,
)


def make_inputs(
    batch_size: int,
    context_len: int,
    page_size: int,
    kv_cache_dtype: str,
    num_query_heads: int,
) -> tuple[torch.Tensor, ...]:
    num_kv_heads = 4
    head_size = 256
    query_dtype = torch.bfloat16
    device = "cuda"
    num_pages_per_seq = math.ceil(context_len / page_size)
    num_pages = batch_size * num_pages_per_seq

    query = torch.randn(
        batch_size,
        num_query_heads,
        head_size,
        device=device,
        dtype=query_dtype,
    )
    if kv_cache_dtype == "auto":
        cache_dtype = query_dtype
        k_scale = torch.ones(1, device=device, dtype=torch.float32)
        v_scale = torch.ones(1, device=device, dtype=torch.float32)
    else:
        cache_dtype = current_platform.fp8_dtype()
        k_scale = torch.tensor(0.02, device=device, dtype=torch.float32)
        v_scale = torch.tensor(0.025, device=device, dtype=torch.float32)

    x = 16 // torch.empty((), dtype=cache_dtype).element_size()
    key_shape = (num_pages, num_kv_heads, head_size // x, page_size, x)
    value_shape = (num_pages, num_kv_heads, head_size, page_size)
    if kv_cache_dtype == "auto":
        key_cache = torch.randn(key_shape, device=device, dtype=cache_dtype)
        value_cache = torch.randn(value_shape, device=device, dtype=cache_dtype)
    else:
        key_cache = (torch.randn(key_shape, device=device) / k_scale).to(cache_dtype)
        value_cache = (torch.randn(value_shape, device=device) / v_scale).to(
            cache_dtype
        )

    block_tables = torch.arange(
        num_pages, device=device, dtype=torch.int32
    ).view(batch_size, num_pages_per_seq)
    # Exercise a non-page-aligned final page.
    seq_lens = torch.full(
        (batch_size,), context_len - 1, device=device, dtype=torch.int32
    )
    return query, key_cache, value_cache, block_tables, seq_lens, k_scale, v_scale


def benchmark_case(
    batch_size: int,
    context_len: int,
    page_size: int,
    kv_cache_dtype: str,
    num_query_heads: int,
    num_splits: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = make_inputs(
        batch_size, context_len, page_size, kv_cache_dtype, num_query_heads
    )
    output = torch.empty_like(query)
    triton_split_output = torch.empty_like(query)
    native_output = torch.empty_like(query)
    triton_mid_out = torch.empty(
        batch_size,
        query.shape[1],
        num_splits,
        query.shape[2],
        device=query.device,
        dtype=torch.float32,
    )
    triton_mid_lse = torch.empty(
        batch_size,
        query.shape[1],
        num_splits,
        device=query.device,
        dtype=torch.float32,
    )
    native_partial_out = torch.empty_like(triton_mid_out)
    native_partial_max = torch.empty_like(triton_mid_lse)
    native_partial_sum = torch.empty_like(triton_mid_lse)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    gqa_ratio = num_query_heads // num_kv_heads
    scale = query.shape[2] ** -0.5
    triton_tile = 32

    def triton_nonsplit() -> None:
        kernel_paged_attention_2d[(batch_size, num_kv_heads)](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=None,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=None,
            scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out_scale_inv=1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=gqa_ratio,
            num_queries_per_kv_padded=max(triton.next_power_of_2(gqa_ratio), 16),
            block_table_stride=block_tables.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=triton_tile,
            PHYSICAL_BLOCK_SIZE=page_size,
            HEAD_SIZE=query.shape[2],
            HEAD_SIZE_PADDED=triton.next_power_of_2(query.shape[2]),
            USE_ALIBI_SLOPES=False,
            SLIDING_WINDOW=0,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=False,
            query_start_len_ptr=None,
            USE_SINKS=False,
            USE_FP8=False,
        )

    def triton_splitkv() -> None:
        paged_attention_2d_splitkv_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            output=triton_split_output,
            actual_max_splits=num_splits,
            max_seq_len=context_len,
            mid_out=triton_mid_out,
            mid_lse=triton_mid_lse,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def native_splitkv() -> None:
        paged_attention_rocm_splitkv_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            output=native_output,
            actual_max_splits=num_splits,
            max_seq_len=context_len,
            partial_out=native_partial_out,
            partial_max=native_partial_max,
            partial_sum=native_partial_sum,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # Compile/warm all variants, then guard the measurements with a direct
    # output comparison so a fast but incorrect result is never reported.
    triton_nonsplit()
    triton_splitkv()
    native_splitkv()
    torch.cuda.synchronize()
    torch.testing.assert_close(triton_split_output, output, atol=0.02, rtol=0.02)
    torch.testing.assert_close(native_output, output, atol=0.02, rtol=0.02)

    return {
        "triton_nonsplit_ms": triton.testing.do_bench(
            triton_nonsplit, warmup=warmup_ms, rep=rep_ms
        ),
        "triton_splitkv_ms": triton.testing.do_bench(
            triton_splitkv, warmup=warmup_ms, rep=rep_ms
        ),
        "native_splitkv_ms": triton.testing.do_bench(
            native_splitkv, warmup=warmup_ms, rep=rep_ms
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[1024, 4096, 8192, 16384]
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 24])
    parser.add_argument("--page-size", type=int, default=1584)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument(
        "--num-query-heads", type=int, choices=[16, 24], default=16
    )
    parser.add_argument(
        "--kv-cache-dtype", choices=["auto", "fp8_e4m3"], default="auto"
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()

    print(
        "batch,context,page_size,query_heads,kv_dtype,num_splits,"
        "triton_nonsplit_ms,triton_splitkv_ms,native_splitkv_ms"
    )
    for batch_size in args.batches:
        for context_len in args.contexts:
            result = benchmark_case(
                batch_size,
                context_len,
                args.page_size,
                args.kv_cache_dtype,
                args.num_query_heads,
                args.num_splits,
                args.warmup_ms,
                args.rep_ms,
            )
            print(
                f"{batch_size},{context_len},{args.page_size},"
                f"{args.num_query_heads},{args.kv_cache_dtype},{args.num_splits},"
                f"{result['triton_nonsplit_ms']:.6f},"
                f"{result['triton_splitkv_ms']:.6f},"
                f"{result['native_splitkv_ms']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    if not current_platform.is_rocm():
        raise RuntimeError("This benchmark requires ROCm.")
    main()
