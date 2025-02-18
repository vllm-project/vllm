# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate
from typing import List, Optional

import nvtx
import torch

from vllm.model_executor.layers.rotary_embedding import (RotaryEmbedding,
                                                         get_rope)
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser


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
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    # silulating serving 4 LoRAs
    scaling_factors = [1, 2, 4, 8]
    # batched RoPE can take multiple scaling factors
    batched_rope = get_rope(head_size, rotary_dim, max_position, base,
                            is_neox_style, {
                                "rope_type": "linear",
                                "factor": tuple(scaling_factors)
                            })
    # non-batched RoPE takes only one scaling factor, we create multiple
    # instances to simulate the same behavior
    non_batched_ropes: List[RotaryEmbedding] = []
    for scaling_factor in scaling_factors:
        non_batched_ropes.append(
            get_rope(head_size, rotary_dim, max_position, base, is_neox_style,
                     {
                         "rope_type": "linear",
                         "factor": (scaling_factor, )
                     }))

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    # create query offsets for batched RoPE, we concat multiple kv cache
    # together and each query needs to find the right kv cache of its type
    offset_map = torch.tensor(
        list(
            accumulate([0] + [
                max_position * scaling_factor * 2
                for scaling_factor in scaling_factors[:-1]
            ])))
    query_types = torch.randint(0,
                                len(scaling_factors), (batch_size, seq_len),
                                device=device)
    # map query types to offsets
    query_offsets = offset_map[query_types]
    # the kernel takes flattened offsets
    flatten_offsets = query_offsets.flatten()

    # batched queries of the same type together for non-batched RoPE
    queries = [query[query_types == i] for i in range(len(scaling_factors))]
    keys = [key[query_types == i] for i in range(len(scaling_factors))]
    packed_qkr = zip(queries, keys, non_batched_ropes)
    # synchronize before start timing
    torch.cuda.synchronize()
    with nvtx.annotate("non-batched", color="yellow"):
        for q, k, r in packed_qkr:
            r.forward(positions, q, k)
    torch.cuda.synchronize()
    with nvtx.annotate("batched", color="green"):
        batched_rope.forward(positions, query, key, flatten_offsets)
    torch.cuda.synchronize()


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the rotary embedding kernels.")
    parser.add_argument("--is-neox-style", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 120, 128, 192, 256],
                        default=128)
    parser.add_argument("--rotary-dim", type=int, choices=[16, 32], default=32)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["bfloat16", "float"],
                        default="float")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        type=str,
                        choices=["cuda:0", "cuda:1"],
                        default="cuda:0")
    args = parser.parse_args()
    print(args)

    benchmark_rope_kernels_multi_lora(
        is_neox_style=args.is_neox_style,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_size=args.head_size,
        rotary_dim=args.rotary_dim,
        dtype=getattr(torch, args.dtype),
        seed=args.seed,
        device=args.device,
    )
