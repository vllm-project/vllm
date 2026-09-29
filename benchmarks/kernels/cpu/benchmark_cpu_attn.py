# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import time

import numpy as np
import torch

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_quantized_kv_cache,
    set_random_seed,
)
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend, _get_attn_isa

# Enable AMX tile data registers so isolated runs don't rely on other operations
# to trigger oneDNN's _init_amx() first (which can cause hangs otherwise).
if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()

KV_CACHE_DTYPE_CHOICES = ["auto", "fp8_e4m3", "fp8_e5m2"]


def parse_query_lens(value: str) -> list[int]:
    try:
        query_lens = [int(entry.strip()) for entry in value.split(",")]
    except ValueError:
        raise ValueError(
            "--query-lens must be comma-separated positive integers"
        ) from None
    if any(query_len <= 0 for query_len in query_lens):
        raise ValueError("--query-lens must contain only positive integers")
    return query_lens


def get_attn_isa(
    block_size: int | None = None,
    dtype: torch.dtype | None = None,
    head_size: int | None = None,
    kv_cache_dtype: str = "auto",
):
    # Delegate to _get_attn_isa so the fallback path applies the same arch
    # gating (e.g. RISC-V RVV is only chosen when the build's hardcoded
    # VLEN=128 kernel is actually present; on VLEN=256 / scalar hosts it
    # correctly falls through to vec/vec16).
    return _get_attn_isa(
        dtype if dtype is not None else torch.bfloat16,
        block_size if block_size else 32,
        head_size=head_size,
        kv_cache_dtype=kv_cache_dtype if kv_cache_dtype != "auto" else None,
    )


# rand number generation takes too much time, cache rand tensors
@functools.lru_cache(maxsize=128, typed=False)
def tensor_cache(
    elem_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.randn(elem_num, dtype=dtype)
    return tensor


@torch.inference_mode()
def main(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int = None,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
    num_blocks: int = 4096,
    use_sink: bool = False,
    enable_kv_split: bool = False,
    isa: str | None = None,
    kv_cache_dtype: str = "auto",
    seed: int = 0,
    iters: int = 20,
    benchmark_scheduler: bool = False,
) -> None:
    set_random_seed(seed)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5
    token_num = sum(query_lens)

    # Resolve kv_cache_dtype: "auto" means same as compute dtype (no quantization)
    effective_kv_cache_dtype = kv_cache_dtype if kv_cache_dtype != "auto" else None
    is_fp8_kv = (
        is_quantized_kv_cache(effective_kv_cache_dtype)
        if effective_kv_cache_dtype
        else False
    )
    # FP8 KV cache is stored as uint8 (byte-level view of fp8 values)
    kv_cache_torch_dtype = torch.uint8 if is_fp8_kv else dtype

    if isa is None:
        isa = get_attn_isa(block_size, dtype, kv_cache_dtype)

    s_aux = (
        15 * torch.rand((num_query_heads,), dtype=torch.bfloat16) if use_sink else None
    )

    query = tensor_cache(
        elem_num=token_num * num_query_heads * head_size,
        dtype=dtype,
    )
    query = query.view(
        token_num,
        num_query_heads,
        head_size,
    )

    key_value = tensor_cache(
        elem_num=2 * num_blocks * num_kv_heads * block_size * head_size,
        dtype=dtype,
    )
    key_value = key_value.view(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
    )
    key_cache, value_cache = key_value.unbind(0)

    # KV cache for CPU attention; FP8 variants store quantized values as uint8
    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=kv_cache_torch_dtype
    )
    packed_value_cache = torch.empty_like(packed_key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # use reshape_and_cache to pack key_cache and value_cache
    slot_mapping = torch.arange(0, num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, num_kv_heads, head_size),
        value=value_cache.view(-1, num_kv_heads, head_size),
        key_cache=packed_key_cache,
        value_cache=packed_value_cache,
        slot_mapping=slot_mapping,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )

    def make_metadata() -> torch.Tensor:
        return cpu_attn_get_scheduler_metadata(
            num_reqs=num_seqs,
            num_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            seq_lens=kv_lens_tensor,
            dtype=dtype,
            query_start_loc=cu_query_lens,
            causal=True,
            sliding_window_size=sliding_window if sliding_window is not None else -1,
            isa=isa,
            enable_kv_split=enable_kv_split,
            kv_cache_dtype=kv_cache_dtype,
        )

    metadata = make_metadata()

    out_with_split = torch.empty_like(query)

    def run_benchmark(iters: int) -> list[float]:
        times = []
        for _ in range(iters):
            start_time = time.perf_counter_ns()
            cpu_attention_with_kv_cache(
                query=query,
                key_cache=packed_key_cache,
                value_cache=packed_value_cache,
                output=out_with_split,
                query_start_loc=cu_query_lens,
                seq_lens=kv_lens_tensor,
                scale=scale,
                causal=True,
                alibi_slopes=None,
                sliding_window=window_size if sliding_window is not None else -1,
                block_table=block_tables,
                softcap=0,
                scheduler_metadata=metadata,
                s_aux=s_aux,
                kv_cache_dtype=kv_cache_dtype,
            )
            end_time = time.perf_counter_ns()
            times.append((end_time - start_time) / 1e6)
        return times

    print("benchmark mode: synthetic CPU attention; kernel-only evidence")

    # Warmup, then benchmark the attention kernel.
    run_benchmark(5)
    times = run_benchmark(iters)

    time_min = min(times)
    time_max = max(times)
    time_mean = np.mean(times)
    time_std = np.std(times)

    print("\tmin (ms) = ", time_min)
    print("\tmax (ms) = ", time_max)
    print("\tmean (ms) = ", time_mean)
    print("\tstd = ", time_std)
    print("\tmedian (ms) = ", np.median(times))
    if benchmark_scheduler:
        for _ in range(5):
            make_metadata()
        scheduler_times = []
        for _ in range(iters):
            start_time = time.perf_counter_ns()
            make_metadata()
            scheduler_times.append((time.perf_counter_ns() - start_time) / 1e6)
        print("\tscheduler metadata median (ms) = ", np.median(scheduler_times))


def generate_seq_lens(
    batch_size: int,
    q_len_min: int,
    q_len_max: int,
    kv_len_min: int,
    kv_len_max: int,
    seed: int = 0,
    query_lens: list[int] | None = None,
) -> list[tuple[int, int]]:
    assert 1 <= kv_len_min <= kv_len_max
    if query_lens is None:
        assert 1 <= q_len_min <= q_len_max
        assert kv_len_max >= q_len_min
    else:
        if len(query_lens) != batch_size:
            raise ValueError("query_lens length must match batch_size")
        if any(query_len <= 0 for query_len in query_lens):
            raise ValueError("query_lens must contain only positive integers")
        if max(query_lens) > kv_len_max:
            raise ValueError("kv_len_max must be at least the largest query length")

    g = torch.Generator(device="cpu").manual_seed(seed)

    def rint(lo: int, hi: int) -> int:
        return torch.randint(lo, hi + 1, (1,), generator=g).item()

    seq_lens: list[tuple[int, int]] = []
    for i in range(batch_size):
        if query_lens is None:
            # ensure q <= kv
            kv = rint(max(kv_len_min, q_len_min), kv_len_max)
            q = rint(q_len_min, min(q_len_max, kv))
        else:
            q = query_lens[i]
            kv = rint(max(kv_len_min, q), kv_len_max)
        seq_lens.append((q, kv))

    return seq_lens


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the paged attention kernel.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--q-len-min", type=int, default=512)
    parser.add_argument("--q-len-max", type=int, default=512)
    parser.add_argument(
        "--query-lens",
        type=parse_query_lens,
        default=None,
        metavar="1,4,...",
        help="Explicit per-request query lengths; overrides --q-len-min/--q-len-max.",
    )
    parser.add_argument("--kv-len-min", type=int, default=512)
    parser.add_argument("--kv-len-max", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4096)

    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=CPUAttentionBackend.get_supported_head_sizes(),
        default=128,
    )
    parser.add_argument("--enable-kv-split", action="store_true")
    parser.add_argument("--block-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--use-sink", action="store_true")
    parser.add_argument(
        "--isa",
        type=str,
        choices=["vec", "neon", "amx", "amx_fp8", "vec16", "rvv"],
        default=None,
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=KV_CACHE_DTYPE_CHOICES,
        default="auto",
        help="KV cache dtype: 'auto' uses compute dtype (bfloat16), "
        "'fp8_e4m3'/'fp8_e5m2' enables FP8 quantized KV cache "
        "(requires AMX-FP8 capable hardware for amx_fp8 ISA).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--benchmark-scheduler",
        action="store_true",
        help="Also time native scheduler metadata creation.",
    )

    args = parser.parse_args()
    if args.query_lens is not None:
        if len(args.query_lens) != args.batch_size:
            parser.error(
                "--query-lens must contain one value per request "
                f"(expected {args.batch_size}, got {len(args.query_lens)})"
            )
        if max(args.query_lens) > args.kv_len_max:
            parser.error(
                "--kv-len-max must be at least the largest value in --query-lens"
            )
    print(args)

    seq_lens = generate_seq_lens(
        args.batch_size,
        args.q_len_min,
        args.q_len_max,
        args.kv_len_min,
        args.kv_len_max,
        args.seed,
        query_lens=args.query_lens,
    )

    print("batch (query len, kv len) = ", seq_lens)

    main(
        seq_lens=seq_lens,
        num_heads=(args.num_query_heads, args.num_kv_heads),
        head_size=args.head_size,
        sliding_window=args.sliding_window,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        use_sink=args.use_sink,
        enable_kv_split=args.enable_kv_split,
        isa=args.isa
        if args.isa is not None
        else get_attn_isa(
            args.block_size,
            STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            args.head_size,
            args.kv_cache_dtype,
        ),
        kv_cache_dtype=args.kv_cache_dtype,
        seed=args.seed,
        iters=args.iters,
        benchmark_scheduler=args.benchmark_scheduler,
    )
