import argparse
import random
import time

import torch

from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
from flash_attn import flash_attn_func


@torch.inference_mode()
def main(
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_flash_attn: bool,
    num_iters: int,
    seed: int,
    do_profile: bool,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_lens = [seq_len] * num_seqs
    num_tokens = sum(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    qkv = torch.empty(num_tokens,
                      num_query_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    qkv.uniform_(-scale, scale)
    query, key, value = qkv.split(
        [num_query_heads, num_kv_heads, num_kv_heads], dim=1)

    def run_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if use_flash_attn:
                output = flash_attn_func(
                    query.unflatten(0, (num_seqs, max(seq_lens))),
                    key.unflatten(0, (num_seqs, max(seq_lens))),
                    value.unflatten(0, (num_seqs, max(seq_lens))),
                    softmax_scale=scale,
                    causal=True)
                output = output.reshape(num_seqs * max(seq_lens),
                                        num_query_heads, head_size)
            else:
                query_expanded = query
                key_expanded = key
                value_expanded = value
                num_queries_per_kv = num_query_heads // num_kv_heads
                if num_queries_per_kv > 1:
                    # Handle MQA and GQA
                    query_expanded = query_expanded.view(
                        query_expanded.shape[0], num_kv_heads,
                        num_queries_per_kv, query_expanded.shape[-1])
                    key_expanded = key[:, :,
                                       None, :].expand(key.shape[0],
                                                       num_kv_heads,
                                                       num_queries_per_kv,
                                                       key.shape[-1])
                    value_expanded = value[:, :, None, :].expand(
                        value.shape[0], num_kv_heads, num_queries_per_kv,
                        value.shape[-1])
                attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)
                output = xops.memory_efficient_attention_forward(
                    query_expanded.unsqueeze(0),
                    key_expanded.unsqueeze(0),
                    value_expanded.unsqueeze(0),
                    attn_bias=attn_bias,
                    p=0.0,
                    scale=scale,
                )
                output = output.squeeze(0)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=num_iters, profile=False)
    print(
        f"Average kernel running time: {latency * 1000000 / num_iters:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the multi query attention kernel.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=64)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16"],
                        default="half")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
    }
    main(
        num_seqs=args.batch_size,
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype=dtype_to_torch_dtype[args.dtype],
        use_flash_attn=args.use_flash_attn,
        num_iters=args.num_iters,
        seed=args.seed,
        do_profile=args.profile,
    )
