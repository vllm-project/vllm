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
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend, _get_attn_isa


def get_attn_isa(
    block_size: int | None = None,
    dtype: torch.dtype | None = None,
):
    if block_size and dtype:
        return _get_attn_isa(dtype, block_size)
    else:
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            return "neon"
        elif torch._C._cpu._is_amx_tile_supported():
            return "amx"
        else:
            return "vec"


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
    seed: int = 0,
    iters: int = 20,
) -> None:
    current_platform.seed_everything(seed)
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

    if isa is None:
        isa = get_attn_isa(block_size, dtype)

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

    # KV cache for CPU attention
    packed_key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
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
    )

    metadata = cpu_attn_get_scheduler_metadata(
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
    )

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
                sliding_window=window_size,
                block_table=block_tables,
                softcap=0,
                scheduler_metadata=metadata,
                s_aux=s_aux,
            )
            end_time = time.perf_counter_ns()
            times.append((end_time - start_time) / 1e6)
        return times

    # warmup
    run_benchmark(5)
    # benchmark
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


def generate_seq_lens(
    batch_size: int,
    q_len_min: int,
    q_len_max: int,
    kv_len_min: int,
    kv_len_max: int,
    seed: int = 0,
) -> list[tuple[int, int]]:
    assert 1 <= q_len_min <= q_len_max
    assert 1 <= kv_len_min <= kv_len_max
    assert kv_len_max >= q_len_min

    g = torch.Generator(device="cpu").manual_seed(seed)

    def rint(lo: int, hi: int) -> int:
        return torch.randint(lo, hi + 1, (1,), generator=g).item()

    seq_lens: list[tuple[int, int]] = []
    for _ in range(batch_size):
        # ensure q <= kv
        kv = rint(max(kv_len_min, q_len_min), kv_len_max)
        q = rint(q_len_min, min(q_len_max, kv))
        seq_lens.append((q, kv))

    return seq_lens


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the paged attention kernel.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--q-len-min", type=int, default=512)
    parser.add_argument("--q-len-max", type=int, default=512)
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
        "--isa", type=str, choices=["vec", "neon", "amx", "vec16"], default=None
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)

    args = parser.parse_args()
    print(args)

    seq_lens = generate_seq_lens(
        args.batch_size,
        args.q_len_min,
        args.q_len_max,
        args.kv_len_min,
        args.kv_len_max,
        args.seed,
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
        else get_attn_isa(args.block_size, STR_DTYPE_TO_TORCH_DTYPE[args.dtype]),
        seed=args.seed,
        iters=args.iters,
    )
