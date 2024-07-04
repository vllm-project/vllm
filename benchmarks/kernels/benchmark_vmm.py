import argparse
import random
import time
from typing import Optional, Union, Tuple, List
import math
import numpy as np
import torch

NUM_BLOCKS = 256
PARTITION_SIZE = 512

_2MB = 1 << 21

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

TORCH_DTYPE_TO_STR_DTYPE = {
    torch.double: "double",
    torch.float: "float",
    torch.float64: "float64",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.half: "half",
    torch.bfloat16: "bfloat16",
    torch.int: "int",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
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


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


@torch.inference_mode()
def main(
    version: str,
    op: str,
    m: int,  # rows
    n: int,  # 2MB num of each row
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    n_bytes = n * _2MB
    n_size = n_bytes // _get_dtype_size(dtype)
    random_mat_A = torch.randn(m, n_size, dtype=dtype, device=device)
    print("random_mat_A.data_ptr():", random_mat_A.data_ptr())

    if version == "base":
        mat_A = random_mat_A
        print("mat_A.data_ptr():", mat_A.data_ptr())
    elif version == "vmm":
        from vllm import _vmm_ops as vmm
        allocator = vmm.CacheAllocator()
        A_ptr = vmm.CacheDevicePtr()

        status = allocator.reserve_cache_ptr(A_ptr, m * n)
        if status != 0:
            raise RuntimeError(f"Failed to reserve cache memory: {status}")

        rows_allocated_nums = [0 for _ in range(m)]
        for _ in range(n):
            for row in range(m):
                offset = row * n_bytes + rows_allocated_nums[row] * _2MB
                status = allocator.alloc_cache_ptr(A_ptr, 1, offset)
                if status != 0:
                    raise RuntimeError(
                        f"Failed to allocate cache memory: {status}")
                rows_allocated_nums[row] += 1

        dtype_str = TORCH_DTYPE_TO_STR_DTYPE[dtype]
        print("dtype_str:", dtype_str)
        mat_A = vmm.wrap_cache_ptr_to_tensor(A_ptr, dtype_str, (m, n_size))

        mat_A.copy_(random_mat_A)
        print("mat_A.data_ptr():", mat_A.data_ptr())

    else:
        raise ValueError(f"Invalid version: {version}")

    vec_b = torch.randn(n_size, dtype=dtype, device=device)

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if op == "gemv":
                torch.matmul(mat_A, vec_b)
            elif op == "relu":
                torch.relu(mat_A)
            else:
                raise ValueError(f"Invalid op: {op}")

        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)
    # print("output:", output)

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
                        choices=["base", "vmm"],
                        default="base")
    parser.add_argument("--op",
                        type=str,
                        choices=["gemv", "relu"],
                        default="gemv")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(args)

    main(
        version=args.version,
        op=args.op,
        m=args.m,
        n=args.n,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
    )
