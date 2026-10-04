# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed


@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    kv_lora_rank: int,
    pe_dim: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    strided: bool,
    device: str = "cuda",
) -> float:
    """Return the median latency in microseconds for one cache write."""
    set_random_seed(42)
    torch.set_default_device(device)

    entry_size = kv_lora_rank + pe_dim
    if strided:
        # The MLA layer passes k_pe as the rope slice of the latent projection.
        latent = torch.randn(num_tokens, entry_size, dtype=dtype)
        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype)
        k_pe = latent[:, kv_lora_rank:]
    else:
        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype)
        k_pe = torch.randn(num_tokens, pe_dim, dtype=dtype)

    num_slots = block_size * num_blocks
    if num_tokens > num_slots:
        raise ValueError("num_tokens cannot exceed the total number of cache slots")
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long
    )
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(num_blocks, block_size, entry_size, dtype=cache_dtype)
    scale = torch.tensor(0.1, dtype=torch.float32)

    def fn():
        ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

    return triton.testing.do_bench_cudagraph(fn, rep=200, return_mode="median") * 1e3


def main(args):
    rows = []
    for exp in range(0, 17):
        n_tok = 2**exp
        lat = run_benchmark(
            num_tokens=n_tok,
            kv_lora_rank=args.kv_lora_rank,
            pe_dim=args.pe_dim,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            kv_cache_dtype=args.kv_cache_dtype,
            strided=args.strided,
        )
        rows.append([n_tok, f"{lat:.3f}"])

    print(
        f"concat_and_cache_mla: dtype={args.dtype} kv_cache_dtype="
        f"{args.kv_cache_dtype} kv_lora_rank={args.kv_lora_rank} "
        f"pe_dim={args.pe_dim} strided={args.strided}"
    )
    print(tabulate(rows, headers=["num_tokens", "latency (us)"]))


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--pe-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument(
        "--kv-cache-dtype", type=str, choices=["auto", "fp8"], default="auto"
    )
    parser.add_argument(
        "--strided",
        action="store_true",
        help="pass k_pe as a slice of the latent projection, as the model does",
    )
    args = parser.parse_args()
    main(args)
