import argparse
import random
import time
from typing import Optional, Union, Tuple, List
import math
import numpy as np
import torch

from vllm_flash_attn import flash_attn_with_kvcache
# from flash_attn import flash_attn_with_kvcache

NUM_BLOCKS = 256
PARTITION_SIZE = 512

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    """Make a padded tensor of a 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    padded_x = np.zeros([len(x), max_len], dtype=np.int32) + pad
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, :len(blocktb)] = blocktb
    return torch.tensor(padded_x, dtype=dtype, device=device)


def get_kv_cache_torch_dtype(
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    assert cache_dtype != "fp8"
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5
    key_caches, value_caches = [], []
    for _ in range(num_layers):
        key_value_cache = torch.empty(size=key_value_cache_shape,
                                      dtype=torch_dtype,
                                      device=device)
        key_value_cache.uniform_(-scale, scale)
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches


def create_kv_caches_with_random_flash_non_page(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    assert cache_dtype != "fp8"
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    key_value_cache_shape = (2, batch_size, seq_len, num_heads, head_size)
    scale = head_size**-0.5
    key_caches, value_caches = [], []
    for _ in range(num_layers):
        key_value_cache = torch.empty(size=key_value_cache_shape,
                                      dtype=torch_dtype,
                                      device=device)
        key_value_cache.uniform_(-scale, scale)
        key_caches.append(key_value_cache[0])
        value_caches.append(key_value_cache[1])
    return key_caches, value_caches


def num_splits_heuristic(batch_nheads_mblocks: int,
                         num_SMs: int,
                         num_n_blocks: int,
                         max_splits: int = 128) -> int:
    if (batch_nheads_mblocks >= 0.8 * num_SMs):
        return 1
    max_splits = min(max_splits, num_SMs, num_n_blocks)
    max_efficiency = 0.0
    efficiency = []

    def ceildiv(a, b):
        return (a + b - 1) // b

    def is_split_eligible(num_splits: int, num_n_blocks: int) -> bool:
        return num_splits == 1 or ceildiv(num_n_blocks, num_splits) != ceildiv(
            num_n_blocks, num_splits - 1)

    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits, num_n_blocks):
            # print(f"num_splits = {num_splits} not eligible, eff = 0.0\n")
            efficiency.append(0.0)
        else:
            n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs
            eff = n_waves / math.ceil(n_waves)
            # print(f"num_splits = {num_splits}, eff = {eff}\n")
            if eff > max_efficiency:
                max_efficiency = eff
            efficiency.append(eff)

    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits, num_n_blocks):
            continue
        if efficiency[num_splits - 1] >= 0.85 * max_efficiency:
            print(f"num_splits chosen = {num_splits}\n")
            return num_splits


# this function is used to determine the number of splits for the flash attention kernel
# copy from flash-attention/csrc/flash_attn/flash_api.cpp
def determine_num_splits(batch_size: int, num_heads: int, head_size: int,
                         seqlen_q: int, seqlen_k: int, num_SMs: int) -> int:
    block_n = 256 if head_size <= 64 else (128 if head_size <= 128 else 64)
    num_n_blocks = (seqlen_k + block_n - 1) // block_n
    num_m_blocks = (seqlen_q + 64 - 1) // 64

    print("batch_size: ", batch_size)
    print("num_heads: ", num_heads)
    print("head_size: ", head_size)
    print("seqlen_q: ", seqlen_q)
    print("seqlen_k: ", seqlen_k)
    print("num_SMs: ", num_SMs)
    print("block_n: ", block_n)
    print("num_n_blocks: ", num_n_blocks)
    print("num_m_blocks: ", num_m_blocks)

    num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,
                                      num_SMs * 2, num_n_blocks)
    return num_splits


@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
    kv_cache_dtype: Optional[str] = None,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=device)

    seq_lens = [seq_len for _ in range(num_seqs)]
    max_seq_len = max(seq_lens)
    print(f"max_seq_len = {max_seq_len}")

    cache_batch_idx = list(range(num_seqs))
    cache_batch_idx = torch.tensor(cache_batch_idx,
                                   dtype=torch.int32,
                                   device=device)

    block_table_lens = [(seq_len + block_size - 1) // block_size
                        for seq_len in seq_lens]

    seq_lens = torch.tensor(seq_lens, dtype=torch.int, device=device)

    # Create the block tables.
    # max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for i in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(block_table_lens[i])
        ]
        block_tables.append(block_table)

    max_block_table_len = max(len(block_table) for block_table in block_tables)
    print(f"max_block_table_len = {max_block_table_len}")

    block_tables = make_tensor_with_pad(
        block_tables,
        max_len=max_block_table_len,
        pad=0,
        dtype=torch.int,
        device=device,
    )
    # block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)

    # Create the KV cache.
    num_splits = 0
    if version == "flash-page":
        # num_splits = determine_num_splits(num_seqs, num_kv_heads, head_size, 1, max_seq_len, 78)
        key_caches, value_caches = create_kv_caches_with_random_flash(
            NUM_BLOCKS,
            block_size,
            1,
            num_kv_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            device=device)
    elif version == "flash-non-page":
        cache_max_seq_len = seq_len
        # In my case, each sequence has a max length of 131072, so I set the cache_max_seq_len to 131072 to preallocate the cache
        cache_max_seq_len = 131072

        key_caches, value_caches = create_kv_caches_with_random_flash_non_page(
            num_seqs,
            cache_max_seq_len,
            1,
            num_kv_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            device=device)

    else:
        raise ValueError(f"Invalid version: {version}")
    key_cache, value_cache = key_caches[0], value_caches[0]
    print(f"key_cache.shape = {key_cache.shape}")
    print(f"value_cache.shape = {value_cache.shape}")

    print(f"num_splits = {num_splits}")

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        # Using default kv_scale
        kv_scale = 1.0

        for _ in range(num_iters):
            if version == "flash-page":
                flash_attn_with_kvcache(
                    q=query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    cache_seqlens=seq_lens,
                    block_table=block_tables,
                    softmax_scale=scale,
                    causal=True,
                    alibi_slopes=alibi_slopes,
                    # out=output[num_prefill_tokens:].unsqueeze(1),
                    num_splits=0,
                )
            elif version == "flash-non-page":
                # print("num_splits = ", num_splits)
                flash_attn_with_kvcache(
                    q=query.unsqueeze(1),
                    k_cache=key_cache[:, :max_seq_len],
                    v_cache=value_cache[:, :max_seq_len],
                    cache_seqlens=seq_lens,
                    # cache_batch_idx=cache_batch_idx,
                    softmax_scale=scale,
                    causal=True,
                    alibi_slopes=alibi_slopes,
                    # out=output[num_prefill_tokens:].unsqueeze(1),
                    num_splits=0,
                    # cached_seqlen_k=max_seq_len,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # for i in range(1, 129):
    #     num_splits = i
    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["flash-page", "flash-non-page"],
                        default="flash-page")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 192, 256],
                        default=128)
    # In my other case, the block size is 4096, so I want to test this situation
    parser.add_argument("--block-size",
                        type=int,
                        choices=[16, 32, 4096],
                        default=4096)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help="Data type for kv cache storage. If 'auto', will use model "
        "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
        "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)")
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        version=args.version,
        num_seqs=args.batch_size,
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        kv_cache_dtype=args.kv_cache_dtype,
    )
