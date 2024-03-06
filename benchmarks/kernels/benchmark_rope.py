from typing import Optional

import argparse
import torch
import nvtx
from itertools import accumulate
from vllm.model_executor.layers.rotary_embedding import get_rope


def benchmark_rope_kernels_multi_lora(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    scaling_factors = [1, 2, 4, 8]
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "type": "linear",
        "factor": tuple(scaling_factors)
    })
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    offset_map = torch.tensor(
        list(
            accumulate([0] + [
                max_position * scaling_factor * 2
                for scaling_factor in scaling_factors[:-1]
            ])))
    query_types = torch.randint(0,
                                len(scaling_factors), (batch_size, seq_len),
                                device=device)
    query_offsets = offset_map[query_types].flatten()

    torch.cuda.synchronize()
    with nvtx.annotate("batched", color="green"):
        rope.forward(positions, query, key, query_offsets)
    torch.cuda.synchronize()

    queries = [query[query_types == i] for i in range(len(scaling_factors))]
    keys = [key[query_types == i] for i in range(len(scaling_factors))]
    packed_qk = zip(queries, keys)
    torch.cuda.synchronize()
    with nvtx.annotate("non-batched", color="yellow"):
        for query, key in packed_qk:
            # the value here is actually wrong because we don't pass any offsets
            # but we are only interested in the time it takes to execute the kernel
            rope.forward(positions, query, key)
    torch.cuda.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the rottery embedding kernels.")
    parser.add_argument("--is-neox-style", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--rottery-dim", type=int, choices=[16, 32], default=32)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["cuda:0", "cuda:1"], default="cuda:0")
    args = parser.parse_args()
    print(args)

    benchmark_rope_kernels_multi_lora(
        is_neox_style=args.is_neox_style,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_size=args.head_size,
        rotary_dim=args.rottery_dim,
        dtype=getattr(torch, args.dtype),
        seed=args.seed,
        device=args.device,
    )
